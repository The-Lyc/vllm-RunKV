# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for RunKV KV cache offloading implementation.

Test hierarchy:
1. Unit tests for PagedBlockMapper class
2. Integration tests for GPU model runner with offloading
3. End-to-end tests with actual model inference
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu_model_runner import PagedBlockMapper
# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="RunKV offloading tests require CUDA"
)

BLOCK_SIZE = 16
NUM_GPU_STAGING_BLOCKS = 32  # Small GPU staging buffer
NUM_CPU_BLOCKS = 128  # Larger CPU cache
DEVICE = torch.device("cuda:0")

def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_logical_block_table(
    seq_lens: list[int],
    *,
    block_size: int,
    block_id_base: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a ragged logical block table similar to vLLM scheduler output.

    In vLLM, "logical block IDs" are global page-like identifiers allocated from
    a shared pool (i.e., not per-request namespaces). To mimic that, we allocate
    IDs from a single increasing counter and assign them in a block-index-major
    order (interleaving requests), which resembles how allocations can intermix
    in real batching.
    """
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must be a non-empty list of positive ints")

    num_blocks_per_req = [_cdiv(seq_len, block_size) for seq_len in seq_lens]
    max_num_blocks_per_req = max(num_blocks_per_req)

    logical_block_table = np.full(
        (len(seq_lens), max_num_blocks_per_req), -1, dtype=np.int32
    )
    next_id = int(block_id_base)
    for block_col in range(max_num_blocks_per_req):
        for req_idx, num_blocks in enumerate(num_blocks_per_req):
            if block_col >= num_blocks:
                continue
            logical_block_table[req_idx, block_col] = next_id
            next_id += 1

    return (
        logical_block_table,
        np.asarray(num_blocks_per_req, dtype=np.int32),
        max_num_blocks_per_req,
    )


def _make_prefill_like_schedule(
    seq_lens: list[int],
    *,
    chunk_size: int = 32,
    seed: int = 0,
    shuffle_reqs_per_round: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a (req_idx, position) schedule resembling chunked prefill.

    Real schedulers often interleave *requests* in a batch, but within a given
    request the scheduled token positions are typically contiguous segments.
    This helper builds a round-robin schedule of contiguous chunks:
      round 0: req0[0:chunk], req1[0:chunk], ...
      round 1: req0[chunk:2*chunk], req1[chunk:2*chunk], ...
      ...

    Args:
      seq_lens: Prompt lengths per request.
      chunk_size: Contiguous segment length per scheduling round.
      seed: RNG seed (only used when shuffle_reqs_per_round=True).
      shuffle_reqs_per_round: If True, shuffle request order each round to
        emulate dynamic batching, while keeping intra-request contiguity.
    """
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must be a non-empty list of positive ints")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rng = np.random.default_rng(seed)
    num_reqs = len(seq_lens)
    max_len = max(seq_lens)

    req_indices_out: list[np.ndarray] = []
    positions_out: list[np.ndarray] = []

    for start in range(0, max_len, chunk_size):
        req_order = np.arange(num_reqs, dtype=np.int32)
        if shuffle_reqs_per_round:
            req_order = rng.permutation(req_order).astype(np.int32, copy=False)

        for req_idx in req_order.tolist():
            seq_len = seq_lens[req_idx]
            if start >= seq_len:
                continue
            end = min(start + chunk_size, seq_len)
            positions = np.arange(start, end, dtype=np.int32)
            req_indices = np.full(positions.shape[0], req_idx, dtype=np.int32)
            req_indices_out.append(req_indices)
            positions_out.append(positions)

    return np.concatenate(req_indices_out), np.concatenate(positions_out)


class TestPagedBlockMapper:
    """Unit tests for the PagedBlockMapper class."""
    
    @pytest.fixture
    def setup_mapper(self):
        """Create a PagedBlockMapper with mock buffers."""
        
        
        # Create GPU staging buffers (small)
        # Shape: [num_staging_blocks, 2, num_kv_heads, head_size]
        # Using simplified shape for testing: [num_blocks, 64]
        gpu_buffers = {
            0: torch.zeros(NUM_GPU_STAGING_BLOCKS, 64, device=DEVICE),
            1: torch.zeros(NUM_GPU_STAGING_BLOCKS, 64, device=DEVICE),
            2: torch.zeros(NUM_GPU_STAGING_BLOCKS, 64, device=DEVICE),
        }
        
        # Create CPU caches (larger)
        cpu_caches = {
            "layer.0": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
            "layer.1": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
            "layer.2": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
            "layer.3": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
            "layer.4": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
            "layer.5": torch.zeros(NUM_CPU_BLOCKS, 64, pin_memory=True),
        }
        
        mapper = PagedBlockMapper(
            block_size=BLOCK_SIZE,
            gpu_buffers=gpu_buffers,
            cpu_caches_per_layer=cpu_caches,
            device=DEVICE,
        )
        
        return mapper, gpu_buffers, cpu_caches
    
    def test_initialization(self, setup_mapper):
        """Test PagedBlockMapper initialization."""
        mapper, gpu_buffers, cpu_caches = setup_mapper
        
        assert mapper.block_size == BLOCK_SIZE
        assert mapper.capacity == NUM_GPU_STAGING_BLOCKS
        assert mapper.num_buffers == 3
        assert len(mapper.mapping) == 0
        assert len(mapper.dirty_blocks) == 0
    
    def test_assign_slots_basic(self, setup_mapper):
        """Test basic slot assignment."""
        mapper, _, _ = setup_mapper
        
        logical_ids = [10, 20, 30, 40, 50]
        mapper._assign_slots(logical_ids)
        
        # Check all logical IDs are mapped
        assert len(mapper.mapping) == len(logical_ids)
        for lid in logical_ids:
            assert lid in mapper.mapping
            assert 0 <= mapper.mapping[lid] < mapper.capacity
        
        # Check no duplicate slots
        slots = list(mapper.mapping.values())
        assert len(slots) == len(set(slots))
    
    def test_assign_slots_capacity_exceeded(self, setup_mapper):
        """Test that exceeding capacity raises error."""
        mapper, _, _ = setup_mapper
        
        # Try to assign more blocks than capacity
        logical_ids = list(range(NUM_GPU_STAGING_BLOCKS + 10))
        
        with pytest.raises(RuntimeError, match="staging buffer only has"):
            mapper._assign_slots(logical_ids)
    
    def test_prepare_step_basic(self, setup_mapper):
        """Test prepare_step with inference-like (interleaved) prefill inputs."""
        mapper, _, _ = setup_mapper

        # Mixed prompt lengths; interleaving mimics real batching behavior.
        seq_lens = [7, 35, 74, 123]
        num_reqs = len(seq_lens)
        logical_block_table, num_blocks_per_row, max_blocks = _build_logical_block_table(
            seq_lens, block_size=BLOCK_SIZE, block_id_base=100
        )
        req_indices, positions = _make_prefill_like_schedule(
            seq_lens, chunk_size=32, seed=0, shuffle_reqs_per_round=True
        )

        physical_table, slot_mapping, dirty_blocks = mapper.prepare_step(
            logical_block_table=logical_block_table,
            num_blocks_per_row=num_blocks_per_row,
            req_indices=req_indices,
            positions=positions,
            max_num_blocks_per_req=max_blocks,
            num_reqs=num_reqs,
        )
        
        # Verify physical block table shape
        assert physical_table.shape == (num_reqs, max_blocks)
        assert physical_table.device == DEVICE
        
        # Verify slot mapping shape
        assert slot_mapping.shape == (positions.shape[0],)

        # Prefill touches every allocated block at least once.
        expected_dirty_blocks: set[int] = set()
        for req_row, num_blocks in enumerate(num_blocks_per_row.tolist()):
            expected_dirty_blocks.update(
                int(bid) for bid in logical_block_table[req_row, :num_blocks]
            )
        assert dirty_blocks == expected_dirty_blocks

        # Verify slot_mapping follows convention:
        # slot = staging_slot * block_size + offset_in_block.
        slot_mapping_cpu = slot_mapping.cpu().numpy()
        for token_idx in range(positions.shape[0]):
            slot_value = int(slot_mapping_cpu[token_idx])
            assert slot_value >= 0

            req_row = int(req_indices[token_idx])
            pos = int(positions[token_idx])
            block_idx = pos // BLOCK_SIZE
            expected_offset = pos % BLOCK_SIZE

            staging_slot = slot_value // BLOCK_SIZE
            offset = slot_value % BLOCK_SIZE
            assert offset == expected_offset

            logical_id = int(logical_block_table[req_row, block_idx])
            assert staging_slot == mapper.mapping[logical_id]

    def test_prepare_step_decode_like_long_context(self, setup_mapper):
        """Test decode-like step with long contexts (full-history staging)."""
        mapper, _, _ = setup_mapper

        # Decode typically schedules 1 token per request, but attention needs
        # the full history, so staging can involve many blocks.
        seq_lens = [256, 128, 96]  # blocks: 16 + 8 + 6 = 30 <= capacity (32)
        num_reqs = len(seq_lens)
        logical_block_table, num_blocks_per_row, max_blocks = _build_logical_block_table(
            seq_lens, block_size=BLOCK_SIZE, block_id_base=10_000
        )

        req_indices = np.arange(num_reqs, dtype=np.int32)
        positions = np.asarray([seq_len - 1 for seq_len in seq_lens], dtype=np.int32)

        physical_table, slot_mapping, dirty_blocks = mapper.prepare_step(
            logical_block_table=logical_block_table,
            num_blocks_per_row=num_blocks_per_row,
            req_indices=req_indices,
            positions=positions,
            max_num_blocks_per_req=max_blocks,
            num_reqs=num_reqs,
        )

        # All blocks across all requests must be staged (full-history attention).
        expected_num_unique_blocks = int(num_blocks_per_row.sum())
        assert len(mapper.mapping) == expected_num_unique_blocks

        # Dirty blocks are just the blocks containing the newly-decoded tokens.
        expected_dirty_blocks = set()
        for req_row, pos in enumerate(positions.tolist()):
            blk_idx = pos // BLOCK_SIZE
            expected_dirty_blocks.add(int(logical_block_table[req_row, blk_idx]))
        assert dirty_blocks == expected_dirty_blocks

        # Slot mapping should point to the last token's block+offset for each req.
        slot_mapping_cpu = slot_mapping.cpu().numpy()
        physical_table_cpu = physical_table.cpu().numpy()
        for req_row in range(num_reqs):
            slot_value = int(slot_mapping_cpu[req_row])
            staging_slot = slot_value // BLOCK_SIZE
            offset = slot_value % BLOCK_SIZE
            assert offset == int(positions[req_row] % BLOCK_SIZE)

            blk_idx = int(positions[req_row] // BLOCK_SIZE)
            logical_id = int(logical_block_table[req_row, blk_idx])
            assert staging_slot == mapper.mapping[logical_id]

            # The physical table should map every allocated logical block.
            row_blocks = int(num_blocks_per_row[req_row])
            for col in range(row_blocks):
                logical_bid = int(logical_block_table[req_row, col])
                assert int(physical_table_cpu[req_row, col]) == mapper.mapping[
                    logical_bid
                ]
    
    def test_prepare_step_mapping_consistency(self, setup_mapper):
        """Test that physical block table uses staging slots correctly."""
        mapper, _, _ = setup_mapper
        
        num_reqs = 1
        max_blocks = 3
        logical_block_table = np.array([[100, 200, 300]], dtype=np.int32)
        num_blocks_per_row = np.array([3], dtype=np.int32)
        req_indices = np.array([0], dtype=np.int32)
        positions = np.array([0], dtype=np.int32)
        
        physical_table, _, _ = mapper.prepare_step(
            logical_block_table=logical_block_table,
            num_blocks_per_row=num_blocks_per_row,
            req_indices=req_indices,
            positions=positions,
            max_num_blocks_per_req=max_blocks,
            num_reqs=num_reqs,
        )
        
        # Verify physical table entries are staging slots
        for col in range(3):
            logical_id = logical_block_table[0, col]
            staging_slot = physical_table[0, col].item()
            # The staging slot should match the mapping
            assert staging_slot == mapper.mapping[logical_id]
            # Staging slots should be in valid range
            assert 0 <= staging_slot < mapper.capacity
    
    def test_load_and_flush_layer(self, setup_mapper):
        """Test load_layer and flush_layer data transfer."""
        mapper, gpu_buffers, cpu_caches = setup_mapper
        
        # Put some data in CPU cache
        cpu_caches["layer.0"][10] = torch.ones(64)
        cpu_caches["layer.0"][20] = torch.ones(64) * 2
        
        # Setup mapping manually
        mapper.mapping = {10: 0, 20: 1}
        
        # Load layer 0
        buffer_idx = mapper.load_layer("layer.0", layer_idx=0)
        
        # Verify buffer selection (ring buffer)
        assert buffer_idx == 0  # layer_idx % num_buffers
        
        # Verify data was copied to GPU
        assert torch.allclose(gpu_buffers[0][0].cpu(), cpu_caches["layer.0"][10])
        assert torch.allclose(gpu_buffers[0][1].cpu(), cpu_caches["layer.0"][20])
        
        # Modify GPU data (simulate attention computation)
        gpu_buffers[0][0] = torch.ones(64, device=DEVICE) * 5
        
        # Flush dirty block back to CPU
        dirty_blocks = {10}
        mapper.flush_layer("layer.0", layer_idx=0, dirty_blocks=dirty_blocks)
        
        # Verify data was copied back to CPU
        assert torch.allclose(cpu_caches["layer.0"][10], torch.ones(64) * 5)
        # Block 20 should be unchanged (not dirty)
        assert torch.allclose(cpu_caches["layer.0"][20], torch.ones(64) * 2)
    
    def test_ring_buffer_selection(self, setup_mapper):
        """Test that layers use ring buffers correctly."""
        mapper, _, _ = setup_mapper
        
        mapper.mapping = {0: 0}  # Dummy mapping
        
        assert mapper.load_layer("layer.0", layer_idx=0) == 0
        assert mapper.load_layer("layer.1", layer_idx=1) == 1
        assert mapper.load_layer("layer.2", layer_idx=2) == 2
        assert mapper.load_layer("layer.3", layer_idx=3) == 0  # Wraps around
        assert mapper.load_layer("layer.4", layer_idx=4) == 1
    
    def test_empty_mapping(self, setup_mapper):
        """Test behavior with empty mapping."""
        mapper, _, _ = setup_mapper
        
        # No mapping set
        buffer_idx = mapper.load_layer("layer.0", layer_idx=0)
        assert buffer_idx == 0
        
        # Flush with empty dirty_blocks should be no-op
        mapper.flush_layer("layer.0", layer_idx=0, dirty_blocks=set())


class TestRunKVOffloadConfig:
    """Tests for RunKVOffloadConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig
        
        config = RunKVOffloadConfig()
        
        assert config.enabled is False
        assert config.num_device_buffers == 3
        assert config.max_staging_blocks is None
        assert config.gpu_memory_fraction == 0.1
        assert config.enable_async_prefetch is True
        assert config.enable_async_offload is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig
        
        config = RunKVOffloadConfig(
            enabled=True,
            num_device_buffers=4,
            max_staging_blocks=256,
            gpu_memory_fraction=0.2,
            enable_async_prefetch=False,
        )
        
        assert config.enabled is True
        assert config.num_device_buffers == 4
        assert config.max_staging_blocks == 256
        assert config.gpu_memory_fraction == 0.2
        assert config.enable_async_prefetch is False


class TestSlotMappingConvention:
    """Tests to verify slot_mapping follows vLLM convention."""
    
    @pytest.fixture
    def setup_mapper(self):
        """Create a simple PagedBlockMapper."""
        from vllm.v1.worker.gpu_model_runner import PagedBlockMapper
        
        gpu_buffers = {0: torch.zeros(64, 32, device=DEVICE)}
        cpu_caches = {"layer.0": torch.zeros(256, 32, pin_memory=True)}
        
        return PagedBlockMapper(
            block_size=BLOCK_SIZE,
            gpu_buffers=gpu_buffers,
            cpu_caches_per_layer=cpu_caches,
            device=DEVICE,
        )
    
    def test_slot_mapping_decode(self, setup_mapper):
        """Test that slot_mapping can be decoded correctly."""
        mapper = setup_mapper
        
        # Setup: 1 request at position 35 (block 2, offset 3)
        logical_block_table = np.array([[10, 11, 12, 13]], dtype=np.int32)
        num_blocks_per_row = np.array([4], dtype=np.int32)
        req_indices = np.array([0], dtype=np.int32)
        positions = np.array([35], dtype=np.int32)  # block=35//16=2, offset=35%16=3
        
        _, slot_mapping, _ = mapper.prepare_step(
            logical_block_table=logical_block_table,
            num_blocks_per_row=num_blocks_per_row,
            req_indices=req_indices,
            positions=positions,
            max_num_blocks_per_req=4,
            num_reqs=1,
        )
        
        # Decode slot_mapping
        slot_value = slot_mapping[0].item()
        staging_slot = slot_value // BLOCK_SIZE
        offset = slot_value % BLOCK_SIZE
        
        # Offset should match position
        assert offset == 3  # 35 % 16 = 3
        
        # staging_slot should be the mapped value for logical block 12
        expected_staging = mapper.mapping[12]
        assert staging_slot == expected_staging

# class TestEndToEnd:
#     """End-to-end tests with actual model (requires GPU memory)."""
    
#     @pytest.mark.slow
#     @pytest.mark.skipif(
#         not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3,
#         reason="Requires at least 8GB GPU memory"
#     )
#     def inference_with_offloading(self):
#         """Test that model inference works with KV offloading enabled."""
#         from vllm import LLM, SamplingParams
        
#         # Use a small model for testing
#         llm = LLM(
#             model="facebook/opt-125m",
#             dtype="float16",
#             max_model_len=256,
#             # Enable RunKV offloading
#             # Note: This depends on how the config is exposed in vLLM API
#             # You may need to adjust based on actual integration
#         )
        
#         sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
        
#         # Run a simple inference
#         outputs = llm.generate(["Hello, my name is"], sampling_params)
        
#         assert len(outputs) == 1
#         assert len(outputs[0].outputs[0].text) > 0
        
#         # Run multiple inferences to test caching
#         for _ in range(3):
#             outputs = llm.generate(["Hello, my name is"], sampling_params)
#             assert len(outputs) == 1


class TestDataTransferCorrectness:
    """Tests for CPU<->GPU data transfer correctness."""
    
    @pytest.fixture
    def setup_full_mapper(self):
        """Create PagedBlockMapper with realistic KV cache shapes."""
        from vllm.v1.worker.gpu_model_runner import PagedBlockMapper
        
        num_kv_heads = 4
        head_size = 64
        # Shape: [num_blocks, 2, num_kv_heads, block_size, head_size]
        # 2 is for key and value
        kv_shape = (2, num_kv_heads, BLOCK_SIZE, head_size)
        
        gpu_buffers = {
            0: torch.zeros(NUM_GPU_STAGING_BLOCKS, *kv_shape, device=DEVICE),
            1: torch.zeros(NUM_GPU_STAGING_BLOCKS, *kv_shape, device=DEVICE),
        }
        
        cpu_caches = {
            "model.layers.0.self_attn": torch.zeros(NUM_CPU_BLOCKS, *kv_shape, pin_memory=True),
            "model.layers.1.self_attn": torch.zeros(NUM_CPU_BLOCKS, *kv_shape, pin_memory=True),
        }
        
        return PagedBlockMapper(
            block_size=BLOCK_SIZE,
            gpu_buffers=gpu_buffers,
            cpu_caches_per_layer=cpu_caches,
            device=DEVICE,
        )
    
    def test_data_integrity_round_trip(self, setup_full_mapper):
        """Test that data survives CPU->GPU->CPU round trip."""
        mapper = setup_full_mapper
        layer_name = "model.layers.0.self_attn"
        
        # Put unique data in CPU cache (full KV block).
        # Shape matches one logical KV block: [2, num_kv_heads, block_size, head_size]
        original_data = torch.randn(
            *mapper.cpu_caches_per_layer[layer_name].shape[1:], device="cpu"
        )
        cpu_cache = mapper.cpu_caches_per_layer[layer_name]
        cpu_cache[42].copy_(original_data)
        
        # Setup mapping and load
        mapper.mapping = {42: 5}
        mapper.load_layer(layer_name, layer_idx=0)
        
        # Verify GPU has correct data
        gpu_data = mapper.gpu_buffers[0][5].cpu()
        assert torch.allclose(gpu_data, original_data)
        
        # Modify on GPU
        modified_data = original_data * 2 + 1
        mapper.gpu_buffers[0][5].copy_(modified_data.to(DEVICE))
        
        # Flush back to CPU
        mapper.flush_layer(layer_name, layer_idx=0, dirty_blocks={42})
        
        # Verify CPU has modified data
        final_cpu = cpu_cache[42]
        assert torch.allclose(final_cpu, modified_data)


# Run with: pytest tests/v1/kv_offload/test_runkv_offload.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
