# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.layer_recompute import LayerRecomputeManager
from vllm.v1.worker.opt_dynamic_replay import (
    LayerReplayPlan,
    OPTDynamicReplayRuntime,
)


class _DummyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = nn.Identity()


class _DummyLlamaModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            layers=nn.ModuleList([_DummyDecoderLayer() for _ in range(num_layers)])
        )


def _make_manager(
    *,
    num_layers: int = 2,
    num_blocks: int = 16,
    block_size: int = 4,
    hidden_size: int = 3,
    io_prefix_blocks: list[int] | None = None,
    layer_recompute_mode: str = "io_hidden_states",
) -> LayerRecomputeManager:
    io_prefix_blocks = io_prefix_blocks or [1] * num_layers
    kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=io_prefix_blocks,
        layer_recompute_mode=layer_recompute_mode,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )
    return LayerRecomputeManager(
        device=torch.device("cpu"),
        kv_offload_config=kv_offload_config,
        kv_cache_config=kv_cache_config,
        hidden_size=hidden_size,
        block_size=block_size,
        num_layers=num_layers,
        model=_DummyLlamaModel(num_layers),
        dtype=torch.float32,
    )


def test_begin_step_resets_owner_changed_blocks() -> None:
    manager = _make_manager()

    manager.logical_id_owner_req_id[3] = "req-old"
    manager.cpu_block_positions[3, 0] = 123
    manager.cpu_block_valid_lens[3] = 2
    manager.cpu_block_positions[7, 0] = 77
    manager.cpu_block_valid_lens[7] = 1

    manager.begin_step(
        req_ids=["req-new"],
        req_indices_np=np.array([0], dtype=np.int32),
        positions_np=np.array([0], dtype=np.int32),
        logical_ids_np=np.array([3], dtype=np.int32),
        block_offsets_np=np.array([0], dtype=np.int32),
        logical_block_table_np=np.array([[3, 4, -1]], dtype=np.int32),
        num_blocks_per_row=np.array([2], dtype=np.int32),
    )

    assert manager.logical_id_owner_req_id[3] == "req-new"
    assert manager.logical_id_owner_req_id[4] == "req-new"
    assert torch.all(manager.cpu_block_positions[3] == -1)
    assert manager.cpu_block_valid_lens[3].item() == 0
    assert torch.all(manager.cpu_block_positions[4] == -1)
    assert manager.cpu_block_valid_lens[4].item() == 0

    # Unrelated block should remain unchanged.
    assert manager.cpu_block_positions[7, 0].item() == 77
    assert manager.cpu_block_valid_lens[7].item() == 1


def test_layernorm_hook_and_sync_store_suffix_tokens() -> None:
    manager = _make_manager(io_prefix_blocks=[1, 1], block_size=4, hidden_size=3)
    dummy_runner = SimpleNamespace()
    manager.register_layernorm_hooks(dummy_runner)

    manager.begin_step(
        req_ids=["req-1"],
        req_indices_np=np.array([0, 0, 0], dtype=np.int32),
        positions_np=np.array([0, 4, 5], dtype=np.int32),
        logical_ids_np=np.array([2, 2, 2], dtype=np.int32),
        block_offsets_np=np.array([0, 0, 1], dtype=np.int32),
        logical_block_table_np=np.array([[2, -1]], dtype=np.int32),
        num_blocks_per_row=np.array([1], dtype=np.int32),
    )

    hs = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32
    )
    _ = manager.model.model.layers[0].input_layernorm(hs)

    manager.sync_hs_d2h()

    stored = manager.cpu_attn_inputs_by_layer[0]
    assert torch.allclose(stored[2, 0], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(stored[2, 1], torch.tensor([3.0, 3.0, 3.0]))
    assert manager.cpu_block_positions[2, 0].item() == 4
    assert manager.cpu_block_positions[2, 1].item() == 5
    assert manager.cpu_block_valid_lens[2].item() == 2

    manager.remove_layernorm_hooks()


def test_capture_layer_input_d2h_and_load_cpu_fill_roundtrip() -> None:
    manager = _make_manager(io_prefix_blocks=[1, 1], block_size=4, hidden_size=3)

    manager.begin_step(
        req_ids=["req-1"],
        req_indices_np=np.array([0, 0, 0], dtype=np.int32),
        positions_np=np.array([4, 5, 6], dtype=np.int32),
        logical_ids_np=np.array([2, 2, 2], dtype=np.int32),
        block_offsets_np=np.array([0, 1, 2], dtype=np.int32),
        logical_block_table_np=np.array([[1, 2, -1]], dtype=np.int32),
        num_blocks_per_row=np.array([2], dtype=np.int32),
    )

    hs = torch.tensor(
        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=torch.float32
    )
    manager.capture_layer_input_d2h(
        layer_idx=1,
        hidden_states=hs,
        req_indices=np.array([0, 0, 0], dtype=np.int32),
        positions=np.array([4, 5, 6], dtype=np.int32),
    )

    manager.sync_hs_d2h()

    stored = manager.cpu_layer_inputs_by_layer[1]
    assert torch.allclose(stored[2, 0], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(stored[2, 1], torch.tensor([3.0, 3.0, 3.0]))
    assert torch.allclose(stored[2, 2], torch.tensor([4.0, 4.0, 4.0]))

    hs_gpu = manager.load_cpu_fill_h2d(
        layer_idx=1,
        cpu_fill_positions=np.array([4, 6], dtype=np.int32),
        cpu_fill_logical_ids=np.array([2, 2], dtype=np.int32),
        cpu_fill_block_offsets=np.array([0, 2], dtype=np.int32),
    )

    assert torch.allclose(
        hs_gpu,
        torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=torch.float32),
    )
    assert manager.sync_cpu_fill_h2d(1) is hs_gpu


def test_register_layernorm_hooks_is_noop_for_dynamic_mode() -> None:
    manager = _make_manager(layer_recompute_mode="prev_layer_output_dynamic")
    dummy_runner = SimpleNamespace()

    manager.register_layernorm_hooks(dummy_runner)

    assert manager._layernorm_hook_handles == []


def test_compute_skip_block_ids_is_suffix_only_by_block_index() -> None:
    manager = _make_manager(num_blocks=32, block_size=16)

    # Request has 4 blocks in this step: block_idx 0,1 are IO prefix; 2,3 are suffix.
    manager.begin_step(
        req_ids=["req-1"],
        req_indices_np=np.array([0], dtype=np.int32),
        positions_np=np.array([63], dtype=np.int32),
        logical_ids_np=np.array([13], dtype=np.int32),
        block_offsets_np=np.array([15], dtype=np.int32),
        logical_block_table_np=np.array([[10, 11, 12, 13]], dtype=np.int32),
        num_blocks_per_row=np.array([4], dtype=np.int32),
    )

    mapper = SimpleNamespace(mapping={10: 0, 11: 1, 12: 2, 13: 3}, block_size=16)
    skip = manager.compute_skip_block_ids_for_layer(
        layer_idx=0,
        gid=0,
        mapper=mapper,
        dirty_blocks={13},  # dirty should not affect skip decision.
        io_prefix_blocks=[2],
    )

    assert skip == {12, 13}


def test_normalize_io_prefix_blocks_broadcasts_single_value() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[8],
    )

    normalized = runner._normalize_layer_recompute_io_prefix_blocks(4)
    assert normalized == [8, 8, 8, 8]


def test_normalize_io_prefix_blocks_rejects_length_mismatch() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[4, 8],
    )

    with pytest.raises(ValueError, match="length 1 or match num_layers"):
        runner._normalize_layer_recompute_io_prefix_blocks(3)


def test_normalize_io_prefix_blocks_rejects_negative_values() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[4, -1, 6],
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        runner._normalize_layer_recompute_io_prefix_blocks(3)


def _make_dynamic_mode_runner(
    *,
    model_type: str = "opt",
    do_layer_norm_before: bool = True,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    dcp: int = 1,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    cascade_attn_enabled: bool = False,
    use_ubatching: bool = False,
) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = True
    runner.layer_recompute_enabled = False
    runner.layer_recompute_manager = None
    runner.replay_plan_provider = None
    runner.kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[4],
        layer_recompute_mode="prev_layer_output_dynamic",
    )
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type=model_type,
            do_layer_norm_before=do_layer_norm_before,
            hidden_size=16,
        )
    )
    runner.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        decode_context_parallel_size=dcp,
        use_ubatching=use_ubatching,
    )
    runner.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    runner.cascade_attn_enabled = cascade_attn_enabled
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.model = SimpleNamespace(
        model=SimpleNamespace(
            decoder=SimpleNamespace(
                layers=nn.ModuleList([nn.Identity(), nn.Identity()])
            )
        )
    )
    return runner


def _make_dynamic_plan(
    *,
    cpu_fill_token_count: int = 0,
    cpu_fill_positions: list[int] | None = None,
    cpu_fill_logical_ids: list[int] | None = None,
    cpu_fill_block_offsets: list[int] | None = None,
    gpu_reuse_slice_per_req: list[tuple[int, int]] | None = None,
    replay_token_count: int = 0,
    scheduled_token_count: int = 2,
    num_actual_tokens: int = 2,
) -> LayerReplayPlan:
    cpu_fill_positions = cpu_fill_positions or []
    cpu_fill_logical_ids = cpu_fill_logical_ids or []
    cpu_fill_block_offsets = cpu_fill_block_offsets or []
    gpu_reuse_slice_per_req = gpu_reuse_slice_per_req or [(0, 0)]
    return LayerReplayPlan(
        kv_replay_start_per_req=np.array([0], dtype=np.int32),
        computed_lens_per_req=np.array([4], dtype=np.int32),
        prev_gpu_start_per_req=np.array([4], dtype=np.int32),
        cpu_fill_token_count=cpu_fill_token_count,
        gpu_reuse_token_count=0,
        replay_token_count=replay_token_count,
        scheduled_token_count=scheduled_token_count,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(replay_token_count + scheduled_token_count, 1),
        query_start_loc=torch.tensor([0, num_actual_tokens], dtype=torch.int32),
        slot_mapping=torch.arange(num_actual_tokens, dtype=torch.int64),
        combined_replay_indices=torch.arange(replay_token_count, dtype=torch.int64),
        combined_scheduled_indices=torch.arange(
            replay_token_count, num_actual_tokens, dtype=torch.int64
        ),
        cpu_fill_positions=np.asarray(cpu_fill_positions, dtype=np.int32),
        cpu_fill_logical_ids=np.asarray(cpu_fill_logical_ids, dtype=np.int32),
        cpu_fill_block_offsets=np.asarray(cpu_fill_block_offsets, dtype=np.int32),
        gpu_reuse_slice_per_req=gpu_reuse_slice_per_req,
    )


def test_validate_prev_layer_output_dynamic_mode_accepts_phase1_config() -> None:
    runner = _make_dynamic_mode_runner()

    runner._validate_prev_layer_output_dynamic_mode(
        SimpleNamespace(kv_cache_groups=[object()])
    )


def test_validate_prev_layer_output_dynamic_mode_rejects_non_opt() -> None:
    runner = _make_dynamic_mode_runner(model_type="llama")

    with pytest.raises(ValueError, match="supports only OPT models"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


def test_validate_prev_layer_output_dynamic_mode_rejects_post_ln_opt() -> None:
    runner = _make_dynamic_mode_runner(do_layer_norm_before=False)

    with pytest.raises(ValueError, match="supports only pre-LN OPT models"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


@pytest.mark.parametrize(
    ("tp", "pp", "dp", "dcp"),
    [(2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)],
)
def test_validate_prev_layer_output_dynamic_mode_rejects_parallelism(
    tp: int, pp: int, dp: int, dcp: int
) -> None:
    runner = _make_dynamic_mode_runner(tp=tp, pp=pp, dp=dp, dcp=dcp)

    with pytest.raises(ValueError, match="requires single-device execution"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


def test_validate_prev_layer_output_dynamic_mode_rejects_multi_kv_group() -> None:
    runner = _make_dynamic_mode_runner()

    with pytest.raises(ValueError, match="supports exactly one KV cache group"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object(), object()])
        )


def test_validate_prev_layer_output_dynamic_mode_rejects_cudagraph() -> None:
    runner = _make_dynamic_mode_runner(cudagraph_mode=CUDAGraphMode.PIECEWISE)

    with pytest.raises(ValueError, match="requires cudagraph_mode=NONE"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


def test_validate_prev_layer_output_dynamic_mode_rejects_cascade_attention() -> None:
    runner = _make_dynamic_mode_runner(cascade_attn_enabled=True)

    with pytest.raises(ValueError, match="requires cascade attention to be disabled"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


def test_validate_prev_layer_output_dynamic_mode_rejects_ubatching() -> None:
    runner = _make_dynamic_mode_runner(use_ubatching=True)

    with pytest.raises(ValueError, match="requires ubatching to be disabled"):
        runner._validate_prev_layer_output_dynamic_mode(
            SimpleNamespace(kv_cache_groups=[object()])
        )


def test_dynamic_mode_initializes_manager_and_plan_provider() -> None:
    runner = _make_dynamic_mode_runner()

    runner._maybe_init_layer_recompute_manager(
        SimpleNamespace(kv_cache_groups=[object()], num_blocks=16),
        group_block_sizes=[16],
    )

    assert runner.layer_recompute_enabled is True
    assert runner.layer_recompute_manager is not None
    assert runner.replay_plan_provider is not None


def test_prepare_dynamic_replay_runtime_builds_layer0_plan_and_prefetches() -> None:
    runner = _make_dynamic_mode_runner()
    runner.layer_recompute_enabled = True
    runner.layer_recompute_manager = Mock()
    runner.replay_plan_provider = Mock()
    runner._runkv_num_layers = 2
    runner._runkv_layer_info = [("layer.0", 0, 0), ("layer.1", 1, 0)]
    runner._lr_req_indices_np = np.array([0, 0], dtype=np.int32)
    runner._lr_positions_np = np.array([4, 5], dtype=np.int32)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([4], dtype=np.int32),
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(
                    block_size=16,
                    get_numpy_array=lambda: np.array([[10, 11]], dtype=np.int32),
                )
            ]
        ),
    )
    runner.seq_lens = SimpleNamespace(
        np=np.array([6], dtype=np.int32),
        cpu=torch.tensor([6], dtype=torch.int32),
    )
    runner.paged_block_tables = [torch.zeros((1, 2), dtype=torch.int32)]
    runner.paged_dirty_blocks = [set()]
    mapper = Mock()
    mapper.mapping = {10: 0, 11: 1}
    runner.paged_block_mappers = [mapper]
    layer0_plan = _make_dynamic_plan(
        cpu_fill_token_count=2,
        cpu_fill_positions=[0, 1],
        cpu_fill_logical_ids=[10, 10],
        cpu_fill_block_offsets=[0, 1],
        replay_token_count=2,
        num_actual_tokens=4,
    )
    runner._build_dynamic_layer_plan = Mock(return_value=layer0_plan)
    layer0_metadata = {"layer.0.attn": object()}
    runner._build_layer_attn_metadata = Mock(return_value=layer0_metadata)
    runner.layer_recompute_manager.cpu_layer_inputs_by_layer = [torch.empty(1, 1, 1)]
    runner.layer_recompute_manager.compute_skip_block_ids_for_layer.return_value = {11}

    runtime = runner._prepare_dynamic_replay_runtime(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
    )

    assert isinstance(runtime, OPTDynamicReplayRuntime)
    assert runtime.current_layer_plan(0) is layer0_plan
    assert runtime.current_layer_metadata(0) is layer0_metadata
    np.testing.assert_array_equal(runtime.scheduled_req_indices, np.array([0, 0]))
    np.testing.assert_array_equal(runtime.scheduled_positions, np.array([4, 5]))
    mapper.load_layer_async.assert_called_once_with("layer.0", 0, skip_block_ids={11})
    runner.layer_recompute_manager.load_cpu_fill_h2d_async.assert_called_once_with(
        layer_idx=0,
        cpu_fill_positions=layer0_plan.cpu_fill_positions,
        cpu_fill_logical_ids=layer0_plan.cpu_fill_logical_ids,
        cpu_fill_block_offsets=layer0_plan.cpu_fill_block_offsets,
    )


def test_runkv_pre_hook_dynamic_mode_builds_next_plan_and_overlaps_loads() -> None:
    runner = _make_dynamic_mode_runner()
    runner.layer_recompute_enabled = True
    runner.kv_offload_config.layer_recompute_io_prefix_blocks = [1, 1]
    runner.layer_recompute_manager = Mock()
    runner._lr_num_reqs = 1
    runner._lr_num_scheduled_tokens_np = np.array([2], dtype=np.int32)
    runner.seq_lens = SimpleNamespace(
        np=np.array([6], dtype=np.int32),
        cpu=torch.tensor([6], dtype=torch.int32),
    )
    runner.paged_dirty_blocks = [set(), set()]
    runner.paged_block_tables = [
        torch.zeros((1, 2), dtype=torch.int32),
        torch.zeros((1, 2), dtype=torch.int32),
    ]
    mapper0 = Mock()
    mapper1 = Mock()
    runner.paged_block_mappers = [mapper0, mapper1]
    runner._get_next_layer_info = Mock(return_value=("layer.1", 1, 1))

    current_plan = _make_dynamic_plan()
    next_plan = _make_dynamic_plan(
        cpu_fill_token_count=2,
        cpu_fill_positions=[2, 3],
        cpu_fill_logical_ids=[12, 12],
        cpu_fill_block_offsets=[0, 1],
        replay_token_count=2,
        num_actual_tokens=4,
    )
    next_metadata = {"layer.1.attn": object()}
    runner._build_dynamic_layer_plan = Mock(return_value=next_plan)
    runner._build_layer_attn_metadata = Mock(return_value=next_metadata)
    runner.layer_recompute_manager.compute_skip_block_ids_for_layer.return_value = {12}

    runtime = OPTDynamicReplayRuntime(
        num_layers=2,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=Mock(),
        layer_recompute_manager=runner.layer_recompute_manager,
    )
    runtime.set_layer_plan(0, current_plan)
    runtime.set_layer_metadata(0, {"layer.0.attn": object()})
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"base": object()},
        virtual_engine=0,
        layer_recompute_runtime=runtime,
    )

    events: list[str] = []

    def _record_skip(**kwargs) -> set[int]:
        events.append(f"skip_{kwargs['layer_idx']}")
        return {12}

    runner.layer_recompute_manager.sync_cpu_fill_h2d.side_effect = (
        lambda layer_idx: events.append(f"sync_hs_{layer_idx}")
    )
    runner.layer_recompute_manager.load_cpu_fill_h2d_async.side_effect = (
        lambda **kwargs: events.append(f"prefetch_hs_{kwargs['layer_idx']}")
    )
    runner.layer_recompute_manager.compute_skip_block_ids_for_layer.side_effect = (
        _record_skip
    )
    mapper0.sync_load_layer.side_effect = lambda layer_idx: events.append(
        f"sync_kv_{layer_idx}"
    )
    mapper1.load_layer_async.side_effect = lambda *args, **kwargs: events.append(
        "prefetch_kv_1"
    )

    with override_forward_context(forward_context):
        runner._runkv_pre_hook(
            module=nn.Identity(),
            inputs=(),
            layer_name="layer.0",
            layer_idx=0,
            gid=0,
        )

    assert runtime.current_layer_plan(1) is next_plan
    assert runtime.current_layer_metadata(1) is next_metadata
    assert events == [
        "sync_kv_0",
        "sync_hs_0",
        "skip_1",
        "prefetch_kv_1",
        "prefetch_hs_1",
    ]
    runner._build_dynamic_layer_plan.assert_called_once_with(
        layer_idx=1,
        gid=1,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
        prev_layer_plan=current_plan,
    )
    runner._build_layer_attn_metadata.assert_called_once()


def test_prepare_layer_recompute_step_metadata_caches_arrays_and_calls_manager():
    class _DummyBlockTable:
        def __init__(self, table: np.ndarray, block_size: int):
            self._table = table
            self.block_size = block_size
            self.num_blocks_per_row = np.array(
                [table.shape[1]] * table.shape[0], dtype=np.int32
            )

        def get_numpy_array(self) -> np.ndarray:
            return self._table

    logical_table = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    bt = _DummyBlockTable(logical_table, block_size=16)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = True
    runner.layer_recompute_enabled = True
    runner.layer_recompute_manager = Mock()
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(block_tables=[bt]),
        req_ids=["req-0", "req-1"],
    )

    # Keep positions contiguous within each request, matching _prepare_inputs()
    # invariants (positions = num_computed_tokens + arange(num_sched)).
    req_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    positions = np.array([16, 17, 32, 33], dtype=np.int32)
    runner._prepare_layer_recompute_step_metadata(
        req_indices=req_indices,
        positions_np=positions,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([2, 2], dtype=np.int32),
    )

    assert np.array_equal(runner._lr_req_indices_np, req_indices)
    assert np.array_equal(runner._lr_positions_np, positions)
    assert np.array_equal(runner._lr_logical_ids_np, np.array([11, 11, 22, 22]))
    assert np.array_equal(runner._lr_block_offsets_np, np.array([0, 1, 0, 1]))
    assert runner._lr_block_size == 16
    assert runner._lr_num_reqs == 2
    assert np.array_equal(runner._lr_num_scheduled_tokens_np, np.array([2, 2]))

    call_kwargs = runner.layer_recompute_manager.begin_step.call_args.kwargs
    assert call_kwargs["req_ids"] == ["req-0", "req-1"]
    assert np.array_equal(call_kwargs["logical_ids_np"], np.array([11, 11, 22, 22]))
    assert np.array_equal(call_kwargs["block_offsets_np"], np.array([0, 1, 0, 1]))


def test_prepare_layer_recompute_step_metadata_clears_cache_when_disabled():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = False
    runner.layer_recompute_enabled = False
    runner.layer_recompute_manager = Mock()
    runner._lr_req_indices_np = np.array([1], dtype=np.int32)
    runner._lr_positions_np = np.array([2], dtype=np.int32)
    runner._lr_logical_ids_np = np.array([3], dtype=np.int32)
    runner._lr_block_offsets_np = np.array([4], dtype=np.int32)
    runner._lr_block_size = 16
    runner._lr_num_reqs = 1
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(block_tables=[]),
        req_ids=[],
    )

    runner._prepare_layer_recompute_step_metadata(
        req_indices=np.array([0], dtype=np.int32),
        positions_np=np.array([0], dtype=np.int32),
        num_reqs=0,
        num_scheduled_tokens_np=np.array([], dtype=np.int32),
    )

    assert runner._lr_req_indices_np is None
    assert runner._lr_positions_np is None
    assert runner._lr_logical_ids_np is None
    assert runner._lr_block_offsets_np is None
    assert runner._lr_block_size == 0
    assert runner._lr_num_reqs == 0
    assert runner._lr_num_scheduled_tokens_np is None
    runner.layer_recompute_manager.begin_step.assert_not_called()


def test_runkv_pre_hook_layer_recompute_pipeline_order_and_skip_forwarding():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.layer_recompute_enabled = True
    runner.kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[2, 2],
    )
    runner.paged_dirty_blocks = [{101}, {202}]

    mapper0 = Mock()
    mapper1 = Mock()
    runner.paged_block_mappers = [mapper0, mapper1]
    runner._get_next_layer_info = Mock(return_value=("layer.1", 1, 1))

    events: list[str] = []
    mapper0.load_layer_async.side_effect = lambda *args, **kwargs: events.append(
        "load_current"
    )
    mapper0.sync_load_layer.side_effect = lambda *args, **kwargs: events.append(
        "sync_current"
    )
    mapper1.load_layer_async.side_effect = lambda *args, **kwargs: events.append(
        "prefetch_next"
    )

    manager = Mock()
    skip_current = {10, 11}
    skip_next = {20}
    manager.compute_skip_block_ids_for_layer.side_effect = [skip_current, skip_next]
    manager.prefetch_recompute_inputs_for_layer.side_effect = (
        lambda *args, **kwargs: events.append(
            "prefetch_hs_current" if kwargs["layer_idx"] == 0 else "prefetch_hs_next"
        )
    )
    manager.recompute_kv_for_layer.side_effect = lambda *args, **kwargs: events.append(
        "recompute_current"
    )
    runner.layer_recompute_manager = manager

    runner._runkv_pre_hook(
        module=nn.Identity(),
        inputs=(),
        layer_name="layer.0",
        layer_idx=0,
        gid=0,
    )

    assert events == [
        "load_current",
        "prefetch_hs_current",
        "sync_current",
        "prefetch_next",
        "prefetch_hs_next",
        "recompute_current",
    ]
    mapper0.load_layer_async.assert_called_once_with(
        "layer.0", 0, skip_block_ids=skip_current
    )
    mapper1.load_layer_async.assert_called_once_with(
        "layer.1", 1, skip_block_ids=skip_next
    )
    mapper0.sync_load_layer.assert_called_once_with(0)
    assert manager.compute_skip_block_ids_for_layer.call_count == 2
    manager.prefetch_recompute_inputs_for_layer.assert_has_calls(
        [
            call(
                layer_idx=0,
                layer_name="layer.0",
                gid=0,
                mapper=mapper0,
                skip_block_ids=skip_current,
            ),
            call(
                layer_idx=1,
                layer_name="layer.1",
                gid=1,
                mapper=mapper1,
                skip_block_ids=skip_next,
            ),
        ]
    )
    manager.recompute_kv_for_layer.assert_called_once()


def test_runkv_pre_hook_without_layer_recompute_keeps_original_behavior():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.layer_recompute_enabled = False
    runner.layer_recompute_manager = Mock()
    runner.kv_offload_config = RunKVOffloadConfig(enabled=True)
    runner.paged_dirty_blocks = [set(), set()]

    mapper0 = Mock()
    mapper1 = Mock()
    runner.paged_block_mappers = [mapper0, mapper1]
    runner._get_next_layer_info = Mock(return_value=("layer.1", 1, 1))

    runner._runkv_pre_hook(
        module=nn.Identity(),
        inputs=(),
        layer_name="layer.0",
        layer_idx=0,
        gid=0,
    )

    mapper0.load_layer_async.assert_called_once_with("layer.0", 0, skip_block_ids=None)
    mapper0.sync_load_layer.assert_called_once_with(0)
    mapper1.load_layer_async.assert_called_once_with("layer.1", 1, skip_block_ids=None)
    runner.layer_recompute_manager.compute_skip_block_ids_for_layer.assert_not_called()
    runner.layer_recompute_manager.prefetch_recompute_inputs_for_layer.assert_not_called()
    runner.layer_recompute_manager.recompute_kv_for_layer.assert_not_called()


def test_sync_runkv_step_end_state_syncs_offload_and_hs_when_enabled() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = True
    runner.layer_recompute_enabled = True
    runner._sync_all_runkv_offloads = Mock()
    runner.layer_recompute_manager = Mock()

    runner._sync_runkv_step_end_state()

    runner._sync_all_runkv_offloads.assert_called_once_with()
    runner.layer_recompute_manager.sync_hs_d2h.assert_called_once_with()


def test_sync_runkv_step_end_state_skips_hs_sync_when_recompute_disabled() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = True
    runner.layer_recompute_enabled = False
    runner._sync_all_runkv_offloads = Mock()
    runner.layer_recompute_manager = Mock()

    runner._sync_runkv_step_end_state()

    runner._sync_all_runkv_offloads.assert_called_once_with()
    runner.layer_recompute_manager.sync_hs_d2h.assert_not_called()


def test_sync_runkv_step_end_state_is_noop_when_runkv_disabled() -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_runkv = False
    runner.layer_recompute_enabled = True
    runner._sync_all_runkv_offloads = Mock()
    runner.layer_recompute_manager = Mock()

    runner._sync_runkv_step_end_state()

    runner._sync_all_runkv_offloads.assert_not_called()
    runner.layer_recompute_manager.sync_hs_d2h.assert_not_called()
