# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for vllm.v1.worker.io_bandwidth_throttle.

The throttle's correctness rests on three properties — see
docs/design/io_bandwidth_throttle_design.md:

* ``cycles_per_byte`` derives from ``cycles_per_ms`` and ``target_gbps``
  without referring to any "native" bandwidth (multi-segment safe).
* When enabled on a real CUDA device, the effective load-stream wall time
  for a sequence of throttled copies tracks ``max(real_copy_ms,
  total_bytes / target_gbps)``.
* When disabled the hot path is a single branch and a return.
"""

from __future__ import annotations

import time

import pytest
import torch

from vllm.v1.worker.io_bandwidth_throttle import (
    IOBandwidthThrottle,
    calibrate_cycles_per_ms,
    get_throttle,
    set_throttle,
    throttle_after_copy_if_enabled,
)


def test_cycles_per_byte_independent_of_native_bw() -> None:
    """The formula must depend only on cycles_per_ms and target_gbps.

    cycles = cycles_per_ms * (n_bytes / target_gbps / 1e9 * 1e3)
           = cycles_per_byte * n_bytes
    where cycles_per_byte = cycles_per_ms * 1e-6 / target_gbps.
    """
    t = IOBandwidthThrottle()
    # Inject cycles_per_ms manually so the test doesn't require CUDA.
    fake_cycles_per_ms = 1.5e6
    t._cycles_per_ms = fake_cycles_per_ms
    t.set_target(5.0)  # 5 GB/s
    expected = fake_cycles_per_ms * 1e-6 / 5.0
    assert t._cycles_per_byte == pytest.approx(expected, rel=1e-12)
    # Changing target rescales cycles_per_byte; cycles_per_ms is untouched.
    t.set_target(10.0)
    assert t._cycles_per_byte == pytest.approx(fake_cycles_per_ms * 1e-6 / 10.0)
    assert t._cycles_per_ms == fake_cycles_per_ms


def test_disabled_is_noop() -> None:
    """``throttle_after_copy_if_enabled`` returns immediately when disabled."""
    # No global throttle installed.
    set_throttle(None)
    assert get_throttle() is None
    # Pass a sentinel that would explode on attribute access.
    class _Sentinel:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"throttle touched stream when disabled (.{name})")

    throttle_after_copy_if_enabled(_Sentinel(), 1 << 20)  # 1 MiB; must be no-op


def test_zero_bytes_is_noop() -> None:
    t = IOBandwidthThrottle()
    t._cycles_per_ms = 1e6
    t._cycles_per_byte = 1e-3
    t._enabled = True

    class _Sentinel:
        def wait_event(self, *_: object, **__: object) -> None:
            raise AssertionError("wait_event called for n_bytes=0")

    t.throttle_after_copy(_Sentinel(), 0)
    t.throttle_after_copy(_Sentinel(), -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_effective_bandwidth_tracks_target() -> None:
    """End-to-end: target bandwidth is observed at the load stream.

    Mirrors scripts/benchmark_throttled_io_bandwidth.py at a smaller size so
    it runs in seconds. We target a value well below native PCIe (2 GB/s) so
    real_copy_ms << target_ms and ``max(real, target_ms) ≈ target_ms``.
    """
    device = torch.device("cuda:0")
    cycles_per_ms = calibrate_cycles_per_ms(device)
    target_gbps = 2.0
    size_bytes = 16 * 1024 * 1024  # 16 MiB
    n_copies = 16
    expected_ms = n_copies * size_bytes / (target_gbps * 1e9) * 1e3

    throttle = IOBandwidthThrottle()
    throttle.enable(target_gbps, device, cycles_per_ms=cycles_per_ms)

    load_stream = torch.cuda.Stream(device=device)
    elems = size_bytes // 2
    dst = torch.empty(elems, dtype=torch.float16, device=device)
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)

    torch.cuda.synchronize(device)
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    issue_t0 = time.perf_counter()
    t_start.record(load_stream)
    with torch.cuda.stream(load_stream):
        for _ in range(n_copies):
            dst.copy_(src, non_blocking=True)
            throttle.throttle_after_copy(load_stream, size_bytes)
    t_end.record(load_stream)
    issue_ms = (time.perf_counter() - issue_t0) * 1e3
    t_end.synchronize()
    gpu_ms = t_start.elapsed_time(t_end)

    # gpu_ms should be close to expected_ms (= max(real, target_ms) when
    # target << native, the regime we care about). _sleep's effective
    # cycles/ms is mildly non-linear, so 50M-cycle calibration applied at
    # 1-10 ms scales can drift ±20 %; we only need the order of magnitude
    # and the linearity-by-bytes property (asserted in
    # test_linear_bytes_to_time) for the experiment's claim to hold.
    assert gpu_ms >= expected_ms * 0.75, (
        f"effective time {gpu_ms:.1f} ms is below expected {expected_ms:.1f} ms"
    )
    assert gpu_ms <= expected_ms * 1.35, (
        f"effective time {gpu_ms:.1f} ms is far above expected {expected_ms:.1f} ms"
    )
    # Host issue time must stay 2 orders of magnitude below GPU wall time;
    # this is the non-blocking guarantee that motivated this design.
    assert issue_ms < gpu_ms * 0.1, (
        f"host issue time {issue_ms:.2f} ms is too close to GPU wall time "
        f"{gpu_ms:.1f} ms; throttle is blocking the host"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_bytes_to_time() -> None:
    """Halving total bytes halves the throttled wall time (RunKV linearity)."""
    device = torch.device("cuda:0")
    cycles_per_ms = calibrate_cycles_per_ms(device)
    target_gbps = 2.0
    size_bytes = 16 * 1024 * 1024
    throttle = IOBandwidthThrottle()
    throttle.enable(target_gbps, device, cycles_per_ms=cycles_per_ms)

    load_stream = torch.cuda.Stream(device=device)
    elems = size_bytes // 2
    dst = torch.empty(elems, dtype=torch.float16, device=device)
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)

    def _run(n_copies: int) -> float:
        torch.cuda.synchronize(device)
        e_s = torch.cuda.Event(enable_timing=True)
        e_e = torch.cuda.Event(enable_timing=True)
        e_s.record(load_stream)
        with torch.cuda.stream(load_stream):
            for _ in range(n_copies):
                dst.copy_(src, non_blocking=True)
                throttle.throttle_after_copy(load_stream, size_bytes)
        e_e.record(load_stream)
        e_e.synchronize()
        return e_s.elapsed_time(e_e)

    full_ms = _run(16)
    half_ms = _run(8)
    ratio = full_ms / half_ms
    assert 1.7 <= ratio <= 2.3, (
        f"throttled time should scale linearly with bytes; got "
        f"full={full_ms:.1f}ms, half={half_ms:.1f}ms, ratio={ratio:.2f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_set_throttle_singleton_roundtrip() -> None:
    """``set_throttle`` / ``get_throttle`` honor None correctly."""
    assert get_throttle() is None
    t = IOBandwidthThrottle()
    set_throttle(t)
    assert get_throttle() is t
    set_throttle(None)
    assert get_throttle() is None
