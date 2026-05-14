#!/usr/bin/env python3
"""Benchmark GPU <-> CPU (PCIe) memory copy bandwidth.

Measures H2D, D2H, and bidirectional transfer throughput across a range of
buffer sizes, with both pinned and pageable host memory.

Usage examples
--------------
# Quick run with default settings:
    python scripts/benchmark_gpu_cpu_bandwidth.py

# Custom buffer sizes and longer runtime:
    python scripts/benchmark_gpu_cpu_bandwidth.py --buffer-sizes 64 256 1024 --runtime-s 5

# Single direction, single size:
    python scripts/benchmark_gpu_cpu_bandwidth.py --directions h2d --buffer-sizes 512 --skip-pageable
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Direction = Literal["h2d", "d2h", "bidirectional"]
MemType = Literal["pinned", "pageable"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    direction: Direction
    mem_type: MemType
    buffer_mb: int
    num_streams: int
    iterations: int
    bytes_total: int
    duration_s: float
    gbps: float


@dataclass
class SweepResult:
    direction: Direction
    mem_type: MemType
    num_streams: int
    results: list[TransferResult]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_DEFAULT_SIZES = [1, 4, 16, 64, 128, 256, 512, 1024]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU <-> CPU (PCIe) memory copy bandwidth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--buffer-sizes",
        nargs="+",
        type=int,
        default=_DEFAULT_SIZES,
        metavar="MB",
        help="List of buffer sizes in MB to sweep over.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["h2d", "d2h", "bidirectional"],
        choices=["h2d", "d2h", "bidirectional"],
        help="Transfer directions to benchmark.",
    )
    parser.add_argument(
        "--mem-types",
        nargs="+",
        default=["pinned", "pageable"],
        choices=["pinned", "pageable"],
        dest="mem_types",
        help="Host memory types to benchmark.",
    )
    parser.add_argument(
        "--skip-pageable",
        action="store_true",
        help="Shortcut to skip pageable memory tests (only run pinned).",
    )
    parser.add_argument(
        "--num-streams",
        nargs="+",
        type=int,
        default=[1],
        metavar="N",
        help="Number of concurrent CUDA streams to use.",
    )
    parser.add_argument(
        "--runtime-s",
        type=float,
        default=3.0,
        help="How long to run each transfer loop (seconds).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations before timing starts.",
    )
    parser.add_argument(
        "--gpu-device",
        default="cuda:0",
        help="CUDA device string.",
    )
    parser.add_argument(
        "--output-dir",
        default="exp_results/gpu_cpu_bandwidth",
        help="Directory to write results.json and summary.md.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _alloc_host(numel: int, pinned: bool):
    import torch
    if pinned:
        return torch.empty(numel, dtype=torch.uint8, pin_memory=True)
    else:
        return torch.empty(numel, dtype=torch.uint8)


def _run_transfer(
    direction: Direction,
    mem_type: MemType,
    buffer_mb: int,
    num_streams: int,
    runtime_s: float,
    warmup_iters: int,
    device,
) -> TransferResult:
    import torch

    numel = buffer_mb * 1024 * 1024
    pinned = mem_type == "pinned"

    # Allocate one buffer per stream so streams can overlap
    host_srcs = [_alloc_host(numel, pinned) for _ in range(num_streams)]
    host_dsts = [_alloc_host(numel, pinned) for _ in range(num_streams)]
    gpu_bufs = [torch.empty(numel, dtype=torch.uint8, device=device) for _ in range(num_streams)]
    for i in range(num_streams):
        host_srcs[i].fill_(0xAB)
        gpu_bufs[i].fill_(0xCD)

    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    def _do_copies():
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                if direction in ("h2d", "bidirectional"):
                    gpu_bufs[i].copy_(host_srcs[i], non_blocking=True)
                if direction in ("d2h", "bidirectional"):
                    host_dsts[i].copy_(gpu_bufs[i], non_blocking=True)
        for s in streams:
            s.synchronize()

    # Warmup
    for _ in range(warmup_iters):
        _do_copies()

    bytes_per_iter = 0
    for _ in range(num_streams):
        if direction == "h2d":
            bytes_per_iter += numel
        elif direction == "d2h":
            bytes_per_iter += numel
        else:  # bidirectional
            bytes_per_iter += numel * 2

    iterations = 0
    bytes_total = 0
    start = time.perf_counter()
    deadline = start + runtime_s
    while time.perf_counter() < deadline:
        _do_copies()
        iterations += 1
        bytes_total += bytes_per_iter
    elapsed = max(time.perf_counter() - start, 1e-9)

    return TransferResult(
        direction=direction,
        mem_type=mem_type,
        buffer_mb=buffer_mb,
        num_streams=num_streams,
        iterations=iterations,
        bytes_total=bytes_total,
        duration_s=elapsed,
        gbps=bytes_total / elapsed / 1e9,
    )


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def _write_summary(out_dir: Path, all_results: list[dict]) -> None:
    lines = [
        "# GPU <-> CPU PCIe Bandwidth Benchmark",
        "",
        "| Direction | Mem Type | Streams | Buffer (MB) | GB/s | Iterations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['direction']} | {r['mem_type']} | {r['num_streams']} "
            f"| {r['buffer_mb']} | {r['gbps']:.3f} | {r['iterations']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _print_row(result: TransferResult) -> None:
    tag = f"[{result.direction}/{result.mem_type}/s{result.num_streams}/{result.buffer_mb}MB]"
    print(f"{tag:<42} {result.gbps:7.3f} GB/s  ({result.iterations} iters, {result.duration_s:.2f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if args.skip_pageable and "pageable" in args.mem_types:
        args.mem_types = [m for m in args.mem_types if m != "pageable"]

    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required. Install it with: pip install torch")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. This benchmark requires a GPU.")

    device = torch.device(args.gpu_device)
    torch.cuda.set_device(device)

    # Print device info
    props = torch.cuda.get_device_properties(device)
    print(f"[device] {props.name}  (CUDA {torch.version.cuda})")
    print(f"[config] directions={args.directions}  mem_types={args.mem_types}")
    print(f"         buffer_sizes={args.buffer_sizes} MB  streams={args.num_streams}")
    print(f"         runtime={args.runtime_s}s  warmup={args.warmup_iters} iters")
    print()

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[TransferResult] = []

    for num_streams in args.num_streams:
        for mem_type in args.mem_types:
            for direction in args.directions:
                for buffer_mb in args.buffer_sizes:
                    result = _run_transfer(
                        direction=direction,
                        mem_type=mem_type,
                        buffer_mb=buffer_mb,
                        num_streams=num_streams,
                        runtime_s=args.runtime_s,
                        warmup_iters=args.warmup_iters,
                        device=device,
                    )
                    _print_row(result)
                    all_results.append(result)

    result_dicts = [asdict(r) for r in all_results]
    output = {
        "device": props.name,
        "cuda_version": torch.version.cuda,
        "args": vars(args),
        "results": result_dicts,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    _write_summary(out_dir, result_dicts)

    print()
    print(f"[done] wrote {out_dir / 'results.json'}")
    print(f"[done] wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
