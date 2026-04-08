#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark: index_select gather vs. multi-segment DMA.

This script loads block distribution snapshots captured at runtime by
setting RUNKV_CAPTURE_BLOCK_DIST=1 during inference, then:

1. **Analyses** the contiguous-segment distribution of ``load_ids`` within
   the CPU pinned cache.  For each snapshot it sorts the logical block IDs
   that need to be transferred (all blocks minus skip blocks), finds runs of
   consecutive IDs (contiguous segments), and reports segment count and sizes.

2. **Benchmarks** two H2D copy strategies on synthetic KV data that matches
   the captured tensor shapes:
     a) *Gather-then-DMA* (current): ``torch.index_select`` to gather
        scattered blocks into a contiguous pinned staging buffer, then a
        single ``cudaMemcpyAsync``.
     b) *Multi-segment DMA* (proposed): for each contiguous segment in the
        CPU cache, issue a separate ``cudaMemcpyAsync`` directly from the
        source slice — no CPU gather step.

Usage
-----
  # Step 1: run inference with capture enabled
  RUNKV_CAPTURE_BLOCK_DIST=1 python <your_inference_script.py> ...

  # Step 2: run this benchmark on the captured data
  python examples/offline_inference/bench_dma_segments.py \
      --snapshot runkv_block_dist/block_dist_<pid>.json \
      [--warmup 10] [--iters 50] [--layers 0,5,10]

  # Or run with synthetic (random) block distributions:
  python examples/offline_inference/bench_dma_segments.py \
      --synthetic \
      --num-blocks 256 --skip-ratio 0.3 \
      --cpu-shape 512,2,16,8,128 \
      [--warmup 10] [--iters 50]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------


@dataclass
class SegmentInfo:
    """A contiguous run of logical block IDs."""

    start: int  # first logical block ID in the segment
    length: int  # number of consecutive blocks


def find_contiguous_segments(ids: list[int]) -> list[SegmentInfo]:
    """Given a sorted list of logical block IDs, return contiguous segments."""
    if not ids:
        return []
    segments: list[SegmentInfo] = []
    start = ids[0]
    length = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1] + 1:
            length += 1
        else:
            segments.append(SegmentInfo(start=start, length=length))
            start = ids[i]
            length = 1
    segments.append(SegmentInfo(start=start, length=length))
    return segments


def print_segment_analysis(
    records: list[dict],
    layer_filter: set[int] | None = None,
) -> None:
    """Print segment distribution statistics for captured records."""
    print("\n" + "=" * 72)
    print("Contiguous Segment Analysis")
    print("=" * 72)

    all_seg_counts: list[int] = []
    all_seg_lengths: list[int] = []
    all_load_counts: list[int] = []

    for i, rec in enumerate(records):
        if layer_filter and rec["layer_idx"] not in layer_filter:
            continue
        load_ids = sorted(rec["load_ids"])
        segments = find_contiguous_segments(load_ids)
        seg_lengths = [s.length for s in segments]

        all_seg_counts.append(len(segments))
        all_seg_lengths.extend(seg_lengths)
        all_load_counts.append(len(load_ids))

        if len(records) <= 20 or i < 5 or i >= len(records) - 3:
            print(
                f"\n  Record {i:>3d} | layer={rec['layer_idx']:>2d} | "
                f"total={rec['total_blocks']:>4d} | "
                f"skip={rec['skip_count']:>4d} | "
                f"load={rec['load_count']:>4d} | "
                f"segments={len(segments):>3d}"
            )
            if seg_lengths:
                print(
                    f"    seg lengths: min={min(seg_lengths)} "
                    f"max={max(seg_lengths)} "
                    f"mean={sum(seg_lengths) / len(seg_lengths):.1f} "
                    f"median={sorted(seg_lengths)[len(seg_lengths) // 2]}"
                )
                if len(seg_lengths) <= 15:
                    print(f"    all segments: {seg_lengths}")
        elif i == 5:
            print("    ... (middle records omitted) ...")

    if not all_seg_counts:
        print("  No records to analyse.")
        return

    print("\n" + "-" * 72)
    print("Aggregate Statistics")
    print("-" * 72)
    n = len(all_seg_counts)
    print(f"  Records analysed        : {n}")
    print(f"  Avg load blocks/record  : {sum(all_load_counts) / n:.1f}")
    print(f"  Avg segments/record     : {sum(all_seg_counts) / n:.1f}")
    print(f"  Median segments/record  : {sorted(all_seg_counts)[n // 2]}")
    print(f"  Max segments/record     : {max(all_seg_counts)}")
    if all_seg_lengths:
        print(
            f"  Avg segment length      : "
            f"{sum(all_seg_lengths) / len(all_seg_lengths):.1f}"
        )
        print(
            f"  Median segment length   : "
            f"{sorted(all_seg_lengths)[len(all_seg_lengths) // 2]}"
        )
        print(f"  Min segment length      : {min(all_seg_lengths)}")
        print(f"  Max segment length      : {max(all_seg_lengths)}")

    # Distribution of segment counts
    from collections import Counter

    count_dist = Counter(all_seg_counts)
    print("\n  Segment count distribution:")
    for k in sorted(count_dist):
        pct = count_dist[k] / n * 100
        bar = "#" * int(pct / 2)
        print(f"    {k:>3d} segments: {count_dist[k]:>4d} records ({pct:>5.1f}%) {bar}")


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def bench_gather_then_dma(
    cpu_cache: torch.Tensor,
    staging_pinned: torch.Tensor,
    gpu_buffer: torch.Tensor,
    src_indices: torch.Tensor,
    copy_count: int,
    blocks_dim: int,
    warmup: int,
    iters: int,
    stream: torch.cuda.Stream,
) -> tuple[float, float]:
    """Benchmark: index_select gather + single DMA.

    Returns (avg_total_us, avg_gather_us).
    """
    for _ in range(warmup):
        if blocks_dim == 0:
            torch.index_select(
                cpu_cache, 0, src_indices, out=staging_pinned[:copy_count]
            )
            with torch.cuda.stream(stream):
                gpu_buffer[:copy_count].copy_(
                    staging_pinned[:copy_count], non_blocking=True
                )
            stream.synchronize()
        else:
            for outer in range(cpu_cache.shape[0]):
                torch.index_select(
                    cpu_cache[outer],
                    0,
                    src_indices,
                    out=staging_pinned[outer, :copy_count],
                )
            with torch.cuda.stream(stream):
                for outer in range(cpu_cache.shape[0]):
                    gpu_buffer[outer, :copy_count].copy_(
                        staging_pinned[outer, :copy_count], non_blocking=True
                    )
            stream.synchronize()

    total_times = []
    gather_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if blocks_dim == 0:
            torch.index_select(
                cpu_cache, 0, src_indices, out=staging_pinned[:copy_count]
            )
            t_gather = time.perf_counter()
            with torch.cuda.stream(stream):
                gpu_buffer[:copy_count].copy_(
                    staging_pinned[:copy_count], non_blocking=True
                )
            stream.synchronize()
        else:
            for outer in range(cpu_cache.shape[0]):
                torch.index_select(
                    cpu_cache[outer],
                    0,
                    src_indices,
                    out=staging_pinned[outer, :copy_count],
                )
            t_gather = time.perf_counter()
            with torch.cuda.stream(stream):
                for outer in range(cpu_cache.shape[0]):
                    gpu_buffer[outer, :copy_count].copy_(
                        staging_pinned[outer, :copy_count], non_blocking=True
                    )
            stream.synchronize()

        t_end = time.perf_counter()
        total_times.append((t_end - t0) * 1e6)
        gather_times.append((t_gather - t0) * 1e6)

    return (
        sum(total_times) / iters,
        sum(gather_times) / iters,
    )


def bench_multi_segment_dma(
    cpu_cache: torch.Tensor,
    gpu_buffer: torch.Tensor,
    segments: list[SegmentInfo],
    blocks_dim: int,
    warmup: int,
    iters: int,
    stream: torch.cuda.Stream,
) -> float:
    """Benchmark: multiple DMA copies, one per contiguous segment.

    Each segment copies directly from the contiguous slice of the CPU pinned
    cache to a contiguous region in the GPU buffer. No CPU gather needed.

    Returns avg_total_us.
    """
    # Pre-compute destination offsets: segments are packed contiguously
    # in the GPU buffer (slot 0, 1, 2, ...).
    dst_offsets: list[int] = []
    off = 0
    for seg in segments:
        dst_offsets.append(off)
        off += seg.length

    for _ in range(warmup):
        with torch.cuda.stream(stream):
            if blocks_dim == 0:
                for seg, dst_off in zip(segments, dst_offsets):
                    gpu_buffer[dst_off : dst_off + seg.length].copy_(
                        cpu_cache[seg.start : seg.start + seg.length],
                        non_blocking=True,
                    )
            else:
                for outer in range(cpu_cache.shape[0]):
                    for seg, dst_off in zip(segments, dst_offsets):
                        gpu_buffer[outer, dst_off : dst_off + seg.length].copy_(
                            cpu_cache[outer, seg.start : seg.start + seg.length],
                            non_blocking=True,
                        )
        stream.synchronize()

    total_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            if blocks_dim == 0:
                for seg, dst_off in zip(segments, dst_offsets):
                    gpu_buffer[dst_off : dst_off + seg.length].copy_(
                        cpu_cache[seg.start : seg.start + seg.length],
                        non_blocking=True,
                    )
            else:
                for outer in range(cpu_cache.shape[0]):
                    for seg, dst_off in zip(segments, dst_offsets):
                        gpu_buffer[outer, dst_off : dst_off + seg.length].copy_(
                            cpu_cache[outer, seg.start : seg.start + seg.length],
                            non_blocking=True,
                        )
        stream.synchronize()
        total_times.append((time.perf_counter() - t0) * 1e6)

    return sum(total_times) / iters


# ---------------------------------------------------------------------------
# Build tensors
# ---------------------------------------------------------------------------


def make_tensors(
    cpu_shape: list[int],
    dtype_str: str,
    blocks_dim: int,
    num_load_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create cpu_cache (pinned), staging_pinned, and gpu_buffer."""
    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    cpu_cache = torch.randn(cpu_shape, dtype=dtype).pin_memory()

    # GPU buffer and staging: same shape but replace blocks dim size with
    # num_load_blocks (or at least the original size for safety).
    gpu_shape = list(cpu_shape)
    gpu_shape[blocks_dim] = max(num_load_blocks, gpu_shape[blocks_dim])
    gpu_buffer = torch.empty(gpu_shape, dtype=dtype, device="cuda")

    staging_shape = list(cpu_shape)
    staging_shape[blocks_dim] = max(num_load_blocks, staging_shape[blocks_dim])
    staging_pinned = torch.empty(staging_shape, dtype=dtype).pin_memory()

    return cpu_cache, staging_pinned, gpu_buffer


# ---------------------------------------------------------------------------
# Run benchmark on a single record
# ---------------------------------------------------------------------------


def run_benchmark_for_record(
    rec: dict,
    warmup: int,
    iters: int,
    stream: torch.cuda.Stream,
) -> dict:
    """Run both strategies on one captured record. Return results dict."""
    load_ids = sorted(rec["load_ids"])
    if not load_ids:
        return {"layer_idx": rec["layer_idx"], "skipped": True}

    segments = find_contiguous_segments(load_ids)
    cpu_shape = rec["cpu_cache_shape"]
    dtype_str = rec.get("cpu_cache_dtype", "float16")
    blocks_dim = rec.get("blocks_dim", 0)

    cpu_cache, staging_pinned, gpu_buffer = make_tensors(
        cpu_shape,
        dtype_str,
        blocks_dim,
        len(load_ids),
    )

    src_indices = torch.tensor(load_ids, dtype=torch.int64)

    avg_total_gather, avg_gather_only = bench_gather_then_dma(
        cpu_cache,
        staging_pinned,
        gpu_buffer,
        src_indices,
        len(load_ids),
        blocks_dim,
        warmup,
        iters,
        stream,
    )

    avg_multi_seg = bench_multi_segment_dma(
        cpu_cache,
        gpu_buffer,
        segments,
        blocks_dim,
        warmup,
        iters,
        stream,
    )

    return {
        "layer_idx": rec["layer_idx"],
        "total_blocks": rec["total_blocks"],
        "skip_count": rec["skip_count"],
        "load_count": len(load_ids),
        "num_segments": len(segments),
        "seg_lengths": [s.length for s in segments],
        "gather_dma_us": avg_total_gather,
        "gather_only_us": avg_gather_only,
        "multi_seg_dma_us": avg_multi_seg,
        "speedup": avg_total_gather / max(avg_multi_seg, 1e-9),
    }


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------


def generate_synthetic_records(
    num_blocks: int,
    skip_ratio: float,
    cpu_shape: list[int],
    dtype_str: str = "float16",
    blocks_dim: int = 0,
    num_records: int = 10,
) -> list[dict]:
    """Generate synthetic block distribution records for benchmarking."""
    import random

    records = []
    for i in range(num_records):
        all_ids = list(range(num_blocks))
        # Simulate skip blocks: pick blocks from the tail (higher indices)
        # as in real replay (skip = recompute suffix)
        skip_count = int(num_blocks * skip_ratio)
        # Random skip pattern: some from the middle, some from the tail
        skip_ids = sorted(random.sample(all_ids, skip_count))
        skip_set = set(skip_ids)
        load_ids = sorted(lid for lid in all_ids if lid not in skip_set)
        shape = list(cpu_shape)
        shape[blocks_dim] = max(shape[blocks_dim], num_blocks)
        records.append(
            {
                "layer_idx": i % 32,
                "all_logical_ids": all_ids,
                "skip_ids": skip_ids,
                "load_ids": load_ids,
                "total_blocks": num_blocks,
                "skip_count": skip_count,
                "load_count": len(load_ids),
                "cpu_cache_shape": shape,
                "cpu_cache_dtype": dtype_str,
                "blocks_dim": blocks_dim,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark index_select gather vs. multi-segment DMA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to block_dist_<pid>.json captured during inference",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic (random) block distributions instead of snapshot",
    )
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--skip-ratio", type=float, default=0.3)
    parser.add_argument(
        "--cpu-shape",
        type=str,
        default="512,2,16,8,128",
        help="CPU cache shape as comma-separated ints (synthetic mode)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for synthetic tensors",
    )
    parser.add_argument("--blocks-dim", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to filter (e.g., '0,5,10')",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=20,
        help="Max records to benchmark (skip remainder after analysis)",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only print segment analysis, skip benchmarking",
    )
    args = parser.parse_args()

    layer_filter = None
    if args.layers:
        layer_filter = set(int(x) for x in args.layers.split(","))

    # Load or generate records
    if args.snapshot:
        with open(args.snapshot) as f:
            data = json.load(f)
        records = data["records"]
        print(f"Loaded {len(records)} records from {args.snapshot}")
        print(
            f"  block_size={data.get('block_size')} "
            f"blocks_dim={data.get('blocks_dim')} "
            f"capacity={data.get('capacity')}"
        )
        # Propagate blocks_dim from the top-level if not in each record
        top_blocks_dim = data.get("blocks_dim", 0)
        for rec in records:
            rec.setdefault("blocks_dim", top_blocks_dim)
    elif args.synthetic:
        cpu_shape = [int(x) for x in args.cpu_shape.split(",")]
        records = generate_synthetic_records(
            num_blocks=args.num_blocks,
            skip_ratio=args.skip_ratio,
            cpu_shape=cpu_shape,
            dtype_str=args.dtype,
            blocks_dim=args.blocks_dim,
        )
        print(
            f"Generated {len(records)} synthetic records "
            f"(blocks={args.num_blocks}, skip_ratio={args.skip_ratio})"
        )
    else:
        # Auto-discover snapshot files
        dist_dir = os.environ.get(
            "RUNKV_CAPTURE_BLOCK_DIST_DIR",
            os.path.join(os.getcwd(), "runkv_block_dist"),
        )
        candidates = []
        if os.path.isdir(dist_dir):
            candidates = sorted(f for f in os.listdir(dist_dir) if f.endswith(".json"))
        if not candidates:
            print("No snapshot found. Use --snapshot <path> or --synthetic.")
            print(f"  Looked in: {dist_dir}")
            return
        snap_path = os.path.join(dist_dir, candidates[-1])
        print(f"Auto-discovered snapshot: {snap_path}")
        with open(snap_path) as f:
            data = json.load(f)
        records = data["records"]
        top_blocks_dim = data.get("blocks_dim", 0)
        for rec in records:
            rec.setdefault("blocks_dim", top_blocks_dim)

    # --- Segment analysis (always) ---
    print_segment_analysis(records, layer_filter)

    if args.analysis_only:
        return

    # --- Benchmark ---
    if not torch.cuda.is_available():
        print("\nCUDA not available — skipping benchmark.")
        return

    stream = torch.cuda.Stream()
    bench_records = records
    if layer_filter:
        bench_records = [r for r in records if r["layer_idx"] in layer_filter]
    bench_records = bench_records[: args.max_records]

    print(f"\n{'=' * 72}")
    print(f"DMA Copy Benchmark  (warmup={args.warmup}, iters={args.iters})")
    print(f"{'=' * 72}")
    print(
        f"{'rec':>4s} {'layer':>5s} {'load':>5s} {'skip':>5s} "
        f"{'segs':>5s} {'gather+DMA':>11s} {'gather':>11s} "
        f"{'multi-seg':>11s} {'speedup':>8s}"
    )
    print("-" * 72)

    results = []
    for i, rec in enumerate(bench_records):
        res = run_benchmark_for_record(rec, args.warmup, args.iters, stream)
        results.append(res)
        if res.get("skipped"):
            print(f"{i:>4d} {'skip':>5s} (no blocks to load)")
            continue
        print(
            f"{i:>4d} {res['layer_idx']:>5d} {res['load_count']:>5d} "
            f"{res['skip_count']:>5d} {res['num_segments']:>5d} "
            f"{res['gather_dma_us']:>9.1f}us {res['gather_only_us']:>9.1f}us "
            f"{res['multi_seg_dma_us']:>9.1f}us {res['speedup']:>7.2f}x"
        )

    # Summary
    valid = [r for r in results if not r.get("skipped")]
    if valid:
        avg_speedup = sum(r["speedup"] for r in valid) / len(valid)
        avg_gather = sum(r["gather_dma_us"] for r in valid) / len(valid)
        avg_multi = sum(r["multi_seg_dma_us"] for r in valid) / len(valid)
        avg_gather_only = sum(r["gather_only_us"] for r in valid) / len(valid)
        print("-" * 72)
        print(
            f"{'AVG':>4s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} "
            f"{avg_gather:>9.1f}us {avg_gather_only:>9.1f}us "
            f"{avg_multi:>9.1f}us {avg_speedup:>7.2f}x"
        )
        print(
            f"\nConclusion: multi-segment DMA is "
            f"{'FASTER' if avg_speedup > 1.0 else 'SLOWER'} "
            f"by {avg_speedup:.2f}x on average."
        )
        print(
            f"  CPU gather alone takes {avg_gather_only:.1f}us avg "
            f"({avg_gather_only / max(avg_gather, 1e-9) * 100:.0f}% of "
            f"gather+DMA)."
        )


if __name__ == "__main__":
    main()
