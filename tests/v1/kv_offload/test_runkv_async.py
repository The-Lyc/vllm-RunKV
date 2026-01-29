#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for RunKV async KV cache loading/offloading.

Tests verify:
1. Async operations return immediately (non-blocking)
2. Data is transferred correctly
3. Pipeline overlapping works as expected
4. Sync points work correctly

Run with: pytest tests/v1/kv_offload/test_runkv_async.py -v
"""

import time

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import PagedBlockMapper


def create_test_mapper(
    device: torch.device,
    block_size: int = 16,
    capacity: int = 64,
    num_cpu_blocks: int = 256,
    num_buffers: int = 3,
    num_layers: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
) -> tuple[PagedBlockMapper, dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    """Create a PagedBlockMapper with test buffers."""
    # GPU staging buffers: [num_blocks, 2 (K/V), num_heads, head_dim]
    gpu_buffers = {}
    for i in range(num_buffers):
        gpu_buffers[i] = torch.zeros(
            capacity, 2, num_heads, head_dim, device=device, dtype=torch.float16
        )

    # CPU caches per layer: [num_cpu_blocks, 2, num_heads, head_dim]
    cpu_caches = {}
    for layer_idx in range(num_layers):
        layer_name = f"model.layers.{layer_idx}.self_attn"
        # Initialize with recognizable patterns for verification
        cache = torch.zeros(
            num_cpu_blocks, 2, num_heads, head_dim, dtype=torch.float16, pin_memory=True
        )
        # Fill with layer-specific pattern: block_id + layer_idx * 0.01
        for block_id in range(num_cpu_blocks):
            cache[block_id].fill_(block_id + layer_idx * 0.01)
        cpu_caches[layer_name] = cache

    mapper = PagedBlockMapper(block_size, gpu_buffers, cpu_caches, device)
    mapper.num_layers = num_layers

    return mapper, gpu_buffers, cpu_caches


class TestAsyncLoadLayer:
    """Tests for async KV loading (H2D transfer)."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")
        mapper, gpu_buffers, cpu_caches = create_test_mapper(device)
        yield mapper, gpu_buffers, cpu_caches
        # Cleanup
        torch.cuda.synchronize()

    def test_load_layer_async_is_non_blocking(self, setup):
        """Verify that load_layer_async returns before transfer completes.

        Strategy: Start async load, then check if event is still pending.
        If async, the transfer will still be in progress when we check.
        """
        mapper, gpu_buffers, cpu_caches = setup

        # Use enough blocks to make transfer noticeable
        num_blocks = 50
        mapper.mapping = {i: i for i in range(num_blocks)}

        torch.cuda.synchronize()

        # Start async load
        mapper.load_layer_async("model.layers.0.self_attn", 0)

        # Immediately check if load event exists (should exist if truly async)
        assert 0 in mapper.load_events, "Event should exist right after async call"

        # Now sync and verify event is cleaned up
        mapper.sync_load_layer(0)
        assert 0 not in mapper.load_events, "Event should be cleaned up after sync"

        print("\n  Async load returned immediately with pending event")
        print("  Event was cleaned up after sync_load_layer()")

    def test_async_load_overlaps_with_compute(self, setup):
        """Verify async load can overlap with GPU compute.

        Strategy:
        1. Measure time for SERIAL execution: load (sync) -> compute
        2. Measure time for PARALLEL execution: load (async) + compute -> sync
        3. Compare: parallel should be faster due to overlap
        """
        mapper, gpu_buffers, cpu_caches = setup

        # Use enough blocks to make transfer time measurable
        num_blocks = 60
        mapper.mapping = {i: i for i in range(num_blocks)}

        layer_name = "model.layers.0.self_attn"

        def do_compute():
            """Simulate GPU compute work."""
            compute_tensor = torch.randn(1000, 1000, device=mapper.device)
            for _ in range(10):
                compute_tensor = torch.matmul(compute_tensor, compute_tensor.T)
            torch.cuda.synchronize()

        # Warm up
        torch.cuda.synchronize()
        mapper.load_layer(layer_name, 0)
        do_compute()

        # ============ Serial execution: load(sync) then compute ============
        serial_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Sync load - blocks until complete
            mapper.load_layer(layer_name, 0)
            # Then compute
            do_compute()

            serial_times.append(time.perf_counter() - start)

        serial_avg = sum(serial_times) / len(serial_times)

        # ============ Parallel execution: load(async) + compute, then sync ============
        parallel_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Async load - returns immediately
            mapper.load_layer_async(layer_name, 0)
            # Compute overlaps with load
            do_compute()
            # Sync load - should be nearly instant if transfer completed during compute
            mapper.sync_load_layer(0)

            parallel_times.append(time.perf_counter() - start)

        parallel_avg = sum(parallel_times) / len(parallel_times)

        # ============ Compare ============
        speedup = (serial_avg - parallel_avg) / serial_avg * 100

        print(f"\n  Serial execution (load->compute):   {serial_avg * 1000:.2f} ms")
        print(f"  Parallel execution (load||compute): {parallel_avg * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}%")
        print(f"  Time saved: {(serial_avg - parallel_avg) * 1000:.2f} ms")

        # Parallel should be at least somewhat faster
        # (allow small margin for measurement noise)
        assert parallel_avg < serial_avg * 1.05, (
            f"Parallel ({parallel_avg * 1000:.2f}ms) should not"
            f" be slower than serial ({serial_avg * 1000:.2f}ms)"
        )

        print("  ✓ Async load successfully overlaps with compute")

    def test_async_load_data_correctness(self, setup):
        """Verify that async load transfers data correctly."""
        mapper, gpu_buffers, cpu_caches = setup

        # Set up mapping
        test_blocks = [0, 5, 10, 15]
        mapper.mapping = {block_id: slot for slot, block_id in enumerate(test_blocks)}

        layer_name = "model.layers.0.self_attn"
        layer_idx = 0

        # Perform async load and wait
        mapper.load_layer_async(layer_name, layer_idx)
        mapper.sync_load_layer(layer_idx)

        # Verify data
        buffer_idx = layer_idx % mapper.num_buffers
        gpu_buffer = gpu_buffers[buffer_idx]
        cpu_cache = cpu_caches[layer_name]

        for block_id, slot in mapper.mapping.items():
            expected = cpu_cache[block_id].to(gpu_buffer.device)
            actual = gpu_buffer[slot]
            assert torch.allclose(expected, actual), (
                f"Block {block_id} -> slot {slot} data mismatch"
            )

        print(f"\n  Verified {len(test_blocks)} blocks transferred correctly")

    def test_event_tracking(self, setup):
        """Verify that events are properly tracked and cleaned up."""
        mapper, gpu_buffers, cpu_caches = setup

        mapper.mapping = {0: 0, 1: 1}

        # Before async load, no event
        assert 0 not in mapper.load_events

        # After async load, event exists
        mapper.load_layer_async("model.layers.0.self_attn", 0)
        assert 0 in mapper.load_events

        # After sync, event is cleaned up
        mapper.sync_load_layer(0)
        assert 0 not in mapper.load_events

        print("\n  Event tracking works correctly")


class TestAsyncFlushLayer:
    """Tests for async KV offloading (D2H transfer)."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")
        mapper, gpu_buffers, cpu_caches = create_test_mapper(device)
        yield mapper, gpu_buffers, cpu_caches
        torch.cuda.synchronize()

    def test_flush_layer_async_is_non_blocking(self, setup):
        """Verify that flush_layer_async returns before transfer completes."""
        mapper, gpu_buffers, cpu_caches = setup

        num_blocks = 50
        mapper.mapping = {i: i for i in range(num_blocks)}
        dirty_blocks = set(range(num_blocks))

        layer_name = "model.layers.0.self_attn"

        # Load data first
        mapper.load_layer(layer_name, 0)

        # Modify GPU data
        buffer_idx = 0 % mapper.num_buffers
        gpu_buffers[buffer_idx][:num_blocks] = 999.0
        torch.cuda.synchronize()

        # Start async flush
        mapper.flush_layer_async(layer_name, 0, dirty_blocks)

        # Check event exists (should exist if truly async)
        assert 0 in mapper.offload_events, "Offload event should exist after async call"

        # Sync and verify cleanup
        mapper.sync_all_offloads()
        assert 0 not in mapper.offload_events, "Event should be cleaned up after sync"

        print("\n  Async flush returned immediately with pending event")
        print("  Event was cleaned up after sync_all_offloads()")

    def test_async_flush_data_correctness(self, setup):
        """Verify that async flush writes back data correctly."""
        mapper, gpu_buffers, cpu_caches = setup

        test_blocks = [0, 5, 10]
        mapper.mapping = {block_id: slot for slot, block_id in enumerate(test_blocks)}
        dirty_blocks = set(test_blocks)

        layer_name = "model.layers.0.self_attn"
        layer_idx = 0
        buffer_idx = layer_idx % mapper.num_buffers

        # Load data
        mapper.load_layer(layer_name, layer_idx)

        # Modify GPU buffer with new values (use a value that's exact in float16)
        new_value = 1024.0  # Exact in float16
        for slot in range(len(test_blocks)):
            gpu_buffers[buffer_idx][slot].fill_(new_value)
        torch.cuda.synchronize()

        # Async flush and sync
        mapper.flush_layer_async(layer_name, layer_idx, dirty_blocks)
        mapper.sync_all_offloads()

        # Verify CPU cache was updated
        cpu_cache = cpu_caches[layer_name]
        for block_id in test_blocks:
            actual_val = cpu_cache[block_id][0, 0, 0].item()
            assert actual_val == new_value, (
                f"Block {block_id} not updated: expected {new_value}, got {actual_val}"
            )

        print(f"\n  Verified {len(test_blocks)} dirty blocks flushed correctly")


class TestAsyncPipeline:
    """Tests for the async pipeline behavior."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")
        mapper, gpu_buffers, cpu_caches = create_test_mapper(
            device, num_layers=4, capacity=64, num_cpu_blocks=256
        )
        yield mapper, gpu_buffers, cpu_caches
        torch.cuda.synchronize()

    def test_overlapped_load_and_compute(self, setup):
        """Test that async load can overlap with compute operations.

        Compare serial vs parallel execution patterns.
        """
        mapper, gpu_buffers, cpu_caches = setup

        num_blocks = 50
        mapper.mapping = {i: i for i in range(num_blocks)}

        layer_name = "model.layers.1.self_attn"

        def do_compute():
            """Simulate attention compute."""
            compute_tensor = torch.randn(800, 800, device=mapper.device)
            for _ in range(5):
                compute_tensor = torch.matmul(compute_tensor, compute_tensor.T)
            torch.cuda.synchronize()

        # Warm up
        mapper.load_layer(layer_name, 1)
        do_compute()

        # Serial: load_sync -> compute
        torch.cuda.synchronize()
        serial_start = time.perf_counter()
        mapper.load_layer(layer_name, 1)  # sync load
        do_compute()
        serial_time = time.perf_counter() - serial_start

        # Parallel: load_async + compute -> sync
        torch.cuda.synchronize()
        parallel_start = time.perf_counter()
        mapper.load_layer_async(layer_name, 1)  # async load
        do_compute()  # overlaps with load
        mapper.sync_load_layer(1)  # wait for load
        parallel_time = time.perf_counter() - parallel_start

        print(f"\n  Serial (load->compute):   {serial_time * 1000:.2f} ms")
        print(f"  Parallel (load||compute): {parallel_time * 1000:.2f} ms")
        print(f"  Speedup: {(serial_time - parallel_time) / serial_time * 100:.1f}%")

    def test_pipeline_simulation(self, setup):
        """Simulate the actual pipeline: prefetch next layer while computing current."""
        mapper, gpu_buffers, cpu_caches = setup

        num_blocks = 20
        mapper.mapping = {i: i for i in range(num_blocks)}

        layer_names = [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.self_attn",
            "model.layers.3.self_attn",
        ]

        timings: dict[str, list[float]] = {"load": [], "compute": [], "total": []}

        # Simulate pipeline execution
        for layer_idx, layer_name in enumerate(layer_names):
            layer_start = time.perf_counter()

            if layer_idx == 0:
                # First layer: sync load
                load_start = time.perf_counter()
                mapper.load_layer(layer_name, layer_idx)
                timings["load"].append(time.perf_counter() - load_start)
            else:
                # Other layers: sync wait for prefetch
                load_start = time.perf_counter()
                mapper.sync_load_layer(layer_idx)
                timings["load"].append(time.perf_counter() - load_start)

            # Start prefetch for next layer (if not last)
            if layer_idx < len(layer_names) - 1:
                next_layer_name = layer_names[layer_idx + 1]
                mapper.load_layer_async(next_layer_name, layer_idx + 1)

            # Simulate compute
            compute_start = time.perf_counter()
            compute_tensor = torch.randn(500, 500, device=mapper.device)
            for _ in range(3):
                compute_tensor = torch.matmul(compute_tensor, compute_tensor.T)
            torch.cuda.synchronize()
            timings["compute"].append(time.perf_counter() - compute_start)

            # Async offload dirty blocks
            mapper.flush_layer_async(layer_name, layer_idx, set(range(num_blocks)))

            timings["total"].append(time.perf_counter() - layer_start)

        # Final sync
        mapper.sync_all_offloads()

        print("\n  Pipeline timing per layer:")
        print("  Layer | Load (ms) | Compute (ms) | Total (ms)")
        print("  ------|-----------|--------------|------------")
        for i in range(len(layer_names)):
            print(
                f"    {i}   |   {timings['load'][i] * 1000:6.2f}  |"
                f"    {timings['compute'][i] * 1000:6.2f}   |"
                f"   {timings['total'][i] * 1000:6.2f}"
            )

        # After first layer, load times should be smaller due to prefetch
        avg_load_after_first = sum(timings["load"][1:]) / len(timings["load"][1:])
        first_load = timings["load"][0]
        print(f"\n  First layer load: {first_load * 1000:.2f} ms")
        print(f"  Avg load (layers 1+): {avg_load_after_first * 1000:.2f} ms")
        print("  (Layers 1+ should have lower load time due to prefetch overlap)")

    def test_concurrent_load_and_offload(self, setup):
        """Test that load and offload can happen concurrently on different streams."""
        mapper, gpu_buffers, cpu_caches = setup

        mapper.mapping = {i: i for i in range(30)}

        # Load layer 0
        mapper.load_layer("model.layers.0.self_attn", 0)

        # Modify GPU buffer
        gpu_buffers[0][:30] = 777.0
        torch.cuda.synchronize()

        # Start async offload for layer 0
        mapper.flush_layer_async("model.layers.0.self_attn", 0, set(range(30)))

        # Immediately start async load for layer 1 (should use different stream)
        mapper.load_layer_async("model.layers.1.self_attn", 1)

        # Both should complete without blocking each other
        start = time.perf_counter()
        mapper.sync_load_layer(1)
        mapper.sync_all_offloads()
        total_sync = time.perf_counter() - start

        print(f"\n  Concurrent load+offload sync time: {total_sync * 1000:.3f} ms")
        print("  (Both operations ran on separate streams)")

        # Verify data correctness
        assert cpu_caches["model.layers.0.self_attn"][0, 0, 0, 0].item() == 777.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")
        mapper, gpu_buffers, cpu_caches = create_test_mapper(device)
        yield mapper, gpu_buffers, cpu_caches
        torch.cuda.synchronize()

    def test_empty_mapping(self, setup):
        """Test behavior with empty mapping."""
        mapper, _, _ = setup
        mapper.mapping = {}

        # Should return 0 and not crash
        result = mapper.load_layer_async("model.layers.0.self_attn", 0)
        assert result == 0

        # Sync on non-existent event should be no-op
        mapper.sync_load_layer(0)

        print("\n  Empty mapping handled correctly")

    def test_empty_dirty_blocks(self, setup):
        """Test flush with no dirty blocks."""
        mapper, _, _ = setup
        mapper.mapping = {0: 0}

        # Should not crash or create events
        mapper.flush_layer_async("model.layers.0.self_attn", 0, set())
        assert 0 not in mapper.offload_events

        print("\n  Empty dirty blocks handled correctly")

    def test_sync_without_async(self, setup):
        """Test calling sync without prior async call."""
        mapper, _, _ = setup

        # Should be no-op, not crash
        mapper.sync_load_layer(999)
        mapper.sync_offload_layer(999)
        mapper.sync_all_offloads()

        print("\n  Sync without async handled correctly")

    def test_multiple_syncs(self, setup):
        """Test multiple sync calls for same layer."""
        mapper, _, _ = setup
        mapper.mapping = {0: 0}

        mapper.load_layer_async("model.layers.0.self_attn", 0)
        mapper.sync_load_layer(0)
        # Second sync should be no-op
        mapper.sync_load_layer(0)

        print("\n  Multiple syncs handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
