#!/usr/bin/env python3
"""Benchmark PCIe contention when CPU simultaneously transfers data to/from two GPUs.

Measures H2D / D2H / bidirectional bandwidth for each GPU in isolation, then
repeats both transfers concurrently and reports the per-GPU bandwidth ratio
(concurrent / solo).  A ratio < 1 indicates PCIe bandwidth contention.

Topology notes
--------------
If both GPUs sit behind the same PCIe switch or share the same root-port
uplink, the total upstream bandwidth is capped and each GPU will see
degradation.  If each GPU has a dedicated root-port lane the ratio should
stay near 1.0.

Usage examples
--------------
# Default: sweep all direction combos for cuda:0 and cuda:1
    python scripts/benchmark_dual_gpu_pcie_contention.py

# Only H2D, 512 MB buffer, 5-second runs
    python scripts/benchmark_dual_gpu_pcie_contention.py \\
        --directions h2d --buffer-mb 512 --runtime-s 5

# Test three GPU pairs with custom devices
    python scripts/benchmark_dual_gpu_pcie_contention.py \\
        --gpu0 cuda:0 --gpu1 cuda:2
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Direction = Literal["h2d", "d2h", "bidirectional"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    """Bandwidth result for one GPU in one run (solo or concurrent)."""
    device: str
    direction: Direction
    buffer_mb: int
    iterations: int
    bytes_total: int
    duration_s: float
    gbps: float


@dataclass
class SoloResult:
    gpu0: SingleResult
    gpu1: SingleResult


@dataclass
class ConcurrentResult:
    direction_gpu0: Direction
    direction_gpu1: Direction
    gpu0: SingleResult
    gpu1: SingleResult
    gpu0_ratio_vs_solo: float | None
    gpu1_ratio_vs_solo: float | None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PCIe contention when CPU simultaneously transfers to two GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu0", default="cuda:0", help="First CUDA device.")
    parser.add_argument("--gpu1", default="cuda:1", help="Second CUDA device.")
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["h2d", "d2h", "bidirectional"],
        choices=["h2d", "d2h", "bidirectional"],
        help="Transfer directions to benchmark for each GPU.",
    )
    parser.add_argument(
        "--buffer-mb",
        type=int,
        default=256,
        help="Pinned host buffer size in MB (per GPU).",
    )
    parser.add_argument(
        "--runtime-s",
        type=float,
        default=5.0,
        help="How long to run each transfer loop (seconds).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations before timing starts.",
    )
    parser.add_argument(
        "--output-dir",
        default="exp_results/dual_gpu_pcie_contention",
        help="Directory to write results.json and summary.md.",
    )
    parser.add_argument(
        "--per-iter-sync",
        action="store_true",
        help=(
            "Re-sync both workers before every iteration via a Barrier, "
            "so both GPUs submit DMA commands at the exact same moment. "
            "More accurate contention measurement at the cost of ~100us overhead per iteration."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core transfer worker
# ---------------------------------------------------------------------------

def _worker(
    device_str: str,
    direction: Direction,
    buffer_mb: int,
    warmup_iters: int,
    barrier: threading.Barrier,
    result_queue: "queue.Queue[SingleResult | Exception]",
    runtime_s: float,
    per_iter_sync: bool = False,
) -> None:
    """Run in a background thread; signals barrier when ready to start timing.

    per_iter_sync: if True, call barrier.wait() before every iteration so that
    both GPU workers submit their DMA commands at the same moment, ensuring
    their transfer windows are strictly aligned throughout the run.
    """
    try:
        import torch

        device = torch.device(device_str)
        torch.cuda.set_device(device)
        numel = buffer_mb * 1024 * 1024

        host_src = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        host_dst = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        gpu_buf = torch.empty(numel, dtype=torch.uint8, device=device)
        host_src.fill_(0xAB)
        gpu_buf.fill_(0xCD)

        stream = torch.cuda.Stream(device=device)

        def _copy_once():
            with torch.cuda.stream(stream):
                if direction in ("h2d", "bidirectional"):
                    gpu_buf.copy_(host_src, non_blocking=True)
                if direction in ("d2h", "bidirectional"):
                    host_dst.copy_(gpu_buf, non_blocking=True)
            stream.synchronize()

        bytes_per_iter = numel * (2 if direction == "bidirectional" else 1)

        # Warmup
        for _ in range(warmup_iters):
            _copy_once()

        # All threads reach the barrier together → simultaneous start
        barrier.wait()

        iterations = 0
        bytes_total = 0
        start = time.perf_counter()
        deadline = start + runtime_s
        while time.perf_counter() < deadline:
            if per_iter_sync:
                # Re-sync before every iteration: both workers submit DMA at
                # the same moment, keeping transfer windows strictly aligned.
                barrier.wait()
            _copy_once()
            iterations += 1
            bytes_total += bytes_per_iter
        elapsed = max(time.perf_counter() - start, 1e-9)

        result_queue.put(SingleResult(
            device=device_str,
            direction=direction,
            buffer_mb=buffer_mb,
            iterations=iterations,
            bytes_total=bytes_total,
            duration_s=elapsed,
            gbps=bytes_total / elapsed / 1e9,
        ))
    except Exception as exc:
        result_queue.put(exc)


def _run_single(
    device_str: str,
    direction: Direction,
    buffer_mb: int,
    warmup_iters: int,
    runtime_s: float,
) -> SingleResult:
    """Run transfer on one GPU (solo, no barrier needed)."""
    # Reuse _worker but with a 1-party barrier so it fires immediately.
    # per_iter_sync is always False for solo runs: a 1-party barrier would
    # still add ~50us overhead per iteration with no benefit.
    q: queue.Queue[SingleResult | Exception] = queue.Queue()
    barrier = threading.Barrier(1)
    t = threading.Thread(
        target=_worker,
        args=(device_str, direction, buffer_mb, warmup_iters, barrier, q, runtime_s, False),
        daemon=True,
    )
    t.start()
    t.join()
    result = q.get_nowait()
    if isinstance(result, Exception):
        raise result
    return result


def _run_concurrent(
    gpu0: str,
    gpu1: str,
    direction0: Direction,
    direction1: Direction,
    buffer_mb: int,
    warmup_iters: int,
    runtime_s: float,
    per_iter_sync: bool = False,
) -> tuple[SingleResult, SingleResult]:
    """Run transfers on both GPUs simultaneously using a 2-party barrier."""
    q: queue.Queue[SingleResult | Exception] = queue.Queue()
    # When per_iter_sync=True the barrier is reused every iteration, so its
    # party count must still be 2 (one per worker thread).
    barrier = threading.Barrier(2)

    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu0, direction0, buffer_mb, warmup_iters, barrier, q, runtime_s, per_iter_sync),
            daemon=True,
        ),
        threading.Thread(
            target=_worker,
            args=(gpu1, direction1, buffer_mb, warmup_iters, barrier, q, runtime_s, per_iter_sync),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results: list[SingleResult] = []
    while not q.empty():
        item = q.get_nowait()
        if isinstance(item, Exception):
            raise item
        results.append(item)

    # Order: gpu0 first, gpu1 second
    by_device = {r.device: r for r in results}
    return by_device[gpu0], by_device[gpu1]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(out_dir: Path, data: dict) -> None:
    lines = [
        "# Dual-GPU PCIe Contention Benchmark",
        "",
        "## Solo Bandwidth",
        "",
        "| GPU | Direction | GB/s | Iterations |",
        "|---|---|---:|---:|",
    ]
    for key, r in data["solo"].items():
        lines.append(
            f"| {r['device']} | {r['direction']} | {r['gbps']:.3f} | {r['iterations']} |"
        )

    lines.extend([
        "",
        "## Concurrent Bandwidth (both GPUs running simultaneously)",
        "",
        "| GPU0 dir | GPU1 dir | GPU0 GB/s | GPU0/solo | GPU1 GB/s | GPU1/solo |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for item in data["concurrent"]:
        g0r = item["gpu0_ratio_vs_solo"]
        g1r = item["gpu1_ratio_vs_solo"]
        lines.append(
            f"| {item['direction_gpu0']} | {item['direction_gpu1']} "
            f"| {item['gpu0']['gbps']:.3f} | {g0r:.3f if g0r is not None else 'n/a'} "
            f"| {item['gpu1']['gbps']:.3f} | {g1r:.3f if g1r is not None else 'n/a'} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _fmt_ratio(r: float | None) -> str:
    if r is None:
        return "  n/a "
    color = "↓" if r < 0.97 else "↑" if r > 1.02 else "≈"
    return f"{r:.3f}{color}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required.")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(
            f"This benchmark requires at least 2 GPUs, found {num_gpus}. "
            "Use --gpu0 and --gpu1 to specify devices."
        )

    gpu0, gpu1 = args.gpu0, args.gpu1
    if gpu0 == gpu1:
        raise ValueError("--gpu0 and --gpu1 must be different devices.")

    p0 = torch.cuda.get_device_properties(gpu0)
    p1 = torch.cuda.get_device_properties(gpu1)
    print(f"[device] GPU0 ({gpu0}): {p0.name}")
    print(f"[device] GPU1 ({gpu1}): {p1.name}")
    print(f"[config] directions={args.directions}  buffer={args.buffer_mb} MB  runtime={args.runtime_s}s")
    print()

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Solo runs: one GPU at a time
    # ------------------------------------------------------------------
    print("=== Solo runs ===")
    solo: dict[str, dict[str, SingleResult]] = {gpu0: {}, gpu1: {}}

    for direction in args.directions:
        for dev in (gpu0, gpu1):
            print(f"  [solo/{dev}/{direction}] ...", end="", flush=True)
            r = _run_single(dev, direction, args.buffer_mb, args.warmup_iters, args.runtime_s)
            solo[dev][direction] = r
            print(f" {r.gbps:.3f} GB/s  ({r.iterations} iters)")

    print()

    # ------------------------------------------------------------------
    # Concurrent runs: both GPUs simultaneously, same direction pairing
    # ------------------------------------------------------------------
    print("=== Concurrent runs ===")
    concurrent_results: list[ConcurrentResult] = []

    for dir0 in args.directions:
        for dir1 in args.directions:
            print(f"  [concurrent] GPU0:{dir0} + GPU1:{dir1} ...", end="", flush=True)
            r0, r1 = _run_concurrent(
                gpu0, gpu1, dir0, dir1,
                args.buffer_mb, args.warmup_iters, args.runtime_s,
                per_iter_sync=args.per_iter_sync,
            )
            baseline0 = solo[gpu0].get(dir0)
            baseline1 = solo[gpu1].get(dir1)
            ratio0 = r0.gbps / baseline0.gbps if baseline0 and baseline0.gbps > 0 else None
            ratio1 = r1.gbps / baseline1.gbps if baseline1 and baseline1.gbps > 0 else None
            cr = ConcurrentResult(
                direction_gpu0=dir0,
                direction_gpu1=dir1,
                gpu0=r0,
                gpu1=r1,
                gpu0_ratio_vs_solo=ratio0,
                gpu1_ratio_vs_solo=ratio1,
            )
            concurrent_results.append(cr)
            print(
                f" GPU0={r0.gbps:.3f} GB/s (×{_fmt_ratio(ratio0)})"
                f"  GPU1={r1.gbps:.3f} GB/s (×{_fmt_ratio(ratio1)})"
            )

    print()

    # ------------------------------------------------------------------
    # Serialize & write outputs
    # ------------------------------------------------------------------
    solo_flat = {
        f"{dev}/{direction}": asdict(result)
        for dev, directions in solo.items()
        for direction, result in directions.items()
    }

    output = {
        "gpu0": gpu0,
        "gpu1": gpu1,
        "gpu0_name": p0.name,
        "gpu1_name": p1.name,
        "args": vars(args),
        "solo": solo_flat,
        "concurrent": [
            {
                "direction_gpu0": cr.direction_gpu0,
                "direction_gpu1": cr.direction_gpu1,
                "gpu0": asdict(cr.gpu0),
                "gpu1": asdict(cr.gpu1),
                "gpu0_ratio_vs_solo": cr.gpu0_ratio_vs_solo,
                "gpu1_ratio_vs_solo": cr.gpu1_ratio_vs_solo,
            }
            for cr in concurrent_results
        ],
    }

    (out_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    _write_summary(out_dir, output)

    print(f"[done] wrote {out_dir / 'results.json'}")
    print(f"[done] wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
