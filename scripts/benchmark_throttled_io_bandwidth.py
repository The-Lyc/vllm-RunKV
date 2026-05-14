#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate stream-side IO bandwidth throttling.

Throttle method: after each ``cudaMemcpyAsync`` on the IO stream, enqueue a
GPU-side spin kernel (``torch.cuda._sleep``) of length

    sleep_time = max(0, size / target_bw - size / native_bw)

so that the IO stream's wall-clock throughput is clamped to ``target_bw``
while the host thread and any other stream remain unaffected.

Three checks:
  1. Effective IO bandwidth tracks target_bw across a sweep.
  2. Host-side issue time stays << GPU wall time (no host blocking).
  3. A compute stream overlaps with the throttled IO stream
     (concurrent ~ max(io, compute), not their sum).
"""

from __future__ import annotations

import argparse
import time

import torch


def calibrate_cycles_per_ms(device: torch.device, samples: int = 5) -> float:
    """Estimate how many GPU cycles ``torch.cuda._sleep`` consumes per ms."""
    stream = torch.cuda.Stream(device=device)
    test_cycles = 50_000_000
    measured = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            torch.cuda._sleep(test_cycles)
            end.record(stream)
        end.synchronize()
        measured.append(start.elapsed_time(end))
    measured.sort()
    median_ms = measured[len(measured) // 2]
    return test_cycles / median_ms


def measure_native_h2d_bandwidth(
    size_bytes: int, device: torch.device, iters: int = 16
) -> float:
    elems = size_bytes // 2  # float16
    d = torch.empty(elems, dtype=torch.float16, device=device)
    h = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    stream = torch.cuda.current_stream()
    for _ in range(2):
        d.copy_(h, non_blocking=True)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        d.copy_(h, non_blocking=True)
    end.record(stream)
    end.synchronize()
    ms = start.elapsed_time(end)
    return (size_bytes * iters) / (ms * 1e-3) / 1e9


def _enqueue_throttled_copies(
    *,
    io_stream: torch.cuda.Stream,
    dst: torch.Tensor,
    src: torch.Tensor,
    n_copies: int,
    sleep_cycles: int,
) -> None:
    with torch.cuda.stream(io_stream):
        for _ in range(n_copies):
            dst.copy_(src, non_blocking=True)
            if sleep_cycles > 0:
                torch.cuda._sleep(sleep_cycles)


def test_bandwidth_tracking(
    *,
    target_gbps: float,
    size_bytes: int,
    n_copies: int,
    cycles_per_ms: float,
    native_gbps: float,
    device: torch.device,
) -> dict:
    elems = size_bytes // 2
    dst = torch.empty(elems, dtype=torch.float16, device=device)
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    io_stream = torch.cuda.Stream(device=device)

    target_ms = size_bytes / (target_gbps * 1e9) * 1e3
    native_ms = size_bytes / (native_gbps * 1e9) * 1e3
    sleep_ms = max(0.0, target_ms - native_ms)
    sleep_cycles = int(cycles_per_ms * sleep_ms)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record(io_stream)
    issue_t0 = time.perf_counter()
    _enqueue_throttled_copies(
        io_stream=io_stream,
        dst=dst,
        src=src,
        n_copies=n_copies,
        sleep_cycles=sleep_cycles,
    )
    end_evt.record(io_stream)
    issue_ms = (time.perf_counter() - issue_t0) * 1e3
    end_evt.synchronize()
    gpu_ms = start_evt.elapsed_time(end_evt)
    effective_gbps = (size_bytes * n_copies) / (gpu_ms * 1e-3) / 1e9
    return {
        "target_gbps": target_gbps,
        "effective_gbps": effective_gbps,
        "gpu_ms": gpu_ms,
        "host_issue_ms": issue_ms,
        "sleep_per_copy_ms": sleep_ms,
        "ratio": effective_gbps / target_gbps,
    }


def test_compute_overlap(
    *,
    target_gbps: float,
    size_bytes: int,
    n_copies: int,
    cycles_per_ms: float,
    native_gbps: float,
    device: torch.device,
    matmul_size: int = 4096,
) -> dict:
    elems = size_bytes // 2
    dst = torch.empty(elems, dtype=torch.float16, device=device)
    src = torch.empty(elems, dtype=torch.float16, pin_memory=True)
    a = torch.randn(matmul_size, matmul_size, dtype=torch.float16, device=device)
    b = torch.randn(matmul_size, matmul_size, dtype=torch.float16, device=device)

    io_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    target_ms = size_bytes / (target_gbps * 1e9) * 1e3
    native_ms = size_bytes / (native_gbps * 1e9) * 1e3
    sleep_ms = max(0.0, target_ms - native_ms)
    sleep_cycles = int(cycles_per_ms * sleep_ms)

    # Warmup matmul kernel selection
    with torch.cuda.stream(compute_stream):
        for _ in range(3):
            _ = a @ b
    torch.cuda.synchronize()

    # IO alone
    io_evt_s = torch.cuda.Event(enable_timing=True)
    io_evt_e = torch.cuda.Event(enable_timing=True)
    io_evt_s.record(io_stream)
    _enqueue_throttled_copies(
        io_stream=io_stream, dst=dst, src=src,
        n_copies=n_copies, sleep_cycles=sleep_cycles,
    )
    io_evt_e.record(io_stream)
    io_evt_e.synchronize()
    io_alone_ms = io_evt_s.elapsed_time(io_evt_e)

    # Compute alone
    c_evt_s = torch.cuda.Event(enable_timing=True)
    c_evt_e = torch.cuda.Event(enable_timing=True)
    c_evt_s.record(compute_stream)
    with torch.cuda.stream(compute_stream):
        c = a
        for _ in range(n_copies):
            c = c @ b
    c_evt_e.record(compute_stream)
    c_evt_e.synchronize()
    compute_alone_ms = c_evt_s.elapsed_time(c_evt_e)

    # Concurrent — measure with host perf_counter (events across streams
    # cannot be elapsed_time'd directly)
    torch.cuda.synchronize()
    host_t0 = time.perf_counter()
    issue_t0 = time.perf_counter()
    _enqueue_throttled_copies(
        io_stream=io_stream, dst=dst, src=src,
        n_copies=n_copies, sleep_cycles=sleep_cycles,
    )
    with torch.cuda.stream(compute_stream):
        c = a
        for _ in range(n_copies):
            c = c @ b
    issue_ms = (time.perf_counter() - issue_t0) * 1e3
    torch.cuda.synchronize()
    concurrent_ms = (time.perf_counter() - host_t0) * 1e3

    return {
        "target_gbps": target_gbps,
        "io_alone_ms": io_alone_ms,
        "compute_alone_ms": compute_alone_ms,
        "concurrent_ms": concurrent_ms,
        "sum_ms": io_alone_ms + compute_alone_ms,
        "overlap_speedup": (io_alone_ms + compute_alone_ms) / concurrent_ms,
        "host_issue_ms": issue_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size-mb", type=float, default=64.0)
    ap.add_argument("--n-copies", type=int, default=32)
    ap.add_argument(
        "--targets-gbps", type=float, nargs="+",
        default=[2.0, 5.0, 10.0, 20.0],
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    size_bytes = int(args.size_mb * 1024 * 1024)

    print("[calibrate] sleep cycles/ms ...")
    cycles_per_ms = calibrate_cycles_per_ms(device)
    print(f"  cycles_per_ms = {cycles_per_ms:.3e}")
    native_gbps = measure_native_h2d_bandwidth(size_bytes, device)
    print(f"  native h2d BW = {native_gbps:.2f} GB/s")
    print(f"  per copy = {args.size_mb:.1f} MB, n_copies = {args.n_copies}\n")

    print("=== Test 1: effective bandwidth vs target ===")
    print(f"  {'target':>8} {'effective':>10} {'ratio':>6} "
          f"{'gpu_ms':>8} {'host_issue_ms':>14} {'sleep_per_copy_ms':>18}")
    for tgt in args.targets_gbps:
        r = test_bandwidth_tracking(
            target_gbps=tgt, size_bytes=size_bytes, n_copies=args.n_copies,
            cycles_per_ms=cycles_per_ms, native_gbps=native_gbps, device=device,
        )
        print(f"  {r['target_gbps']:>8.2f} {r['effective_gbps']:>10.2f} "
              f"{r['ratio']:>6.2f} {r['gpu_ms']:>8.1f} "
              f"{r['host_issue_ms']:>14.2f} {r['sleep_per_copy_ms']:>18.3f}")

    print("\n=== Test 2: compute/IO overlap under throttle ===")
    print(f"  {'target':>8} {'io_alone':>10} {'comp_alone':>11} "
          f"{'concurrent':>11} {'sum':>8} {'speedup':>8} "
          f"{'host_issue_ms':>14}")
    for tgt in args.targets_gbps:
        r = test_compute_overlap(
            target_gbps=tgt, size_bytes=size_bytes, n_copies=args.n_copies,
            cycles_per_ms=cycles_per_ms, native_gbps=native_gbps, device=device,
        )
        print(f"  {r['target_gbps']:>8.2f} {r['io_alone_ms']:>10.1f} "
              f"{r['compute_alone_ms']:>11.1f} {r['concurrent_ms']:>11.1f} "
              f"{r['sum_ms']:>8.1f} {r['overlap_speedup']:>8.2f} "
              f"{r['host_issue_ms']:>14.2f}")

    print("\nInterpretation:")
    print("  Test 1 pass: ratio (effective/target) ~1.0 for all targets and")
    print("    host_issue_ms << gpu_ms (host not blocked by GPU sleep).")
    print("  Test 2 pass: speedup > 1.0 (concurrent < sum), meaning compute")
    print("    overlaps with the throttled IO stream.")


if __name__ == "__main__":
    main()
