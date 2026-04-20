#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract clean per-layer GPU compute time from nsys sqlite exports.

The attention pre-hook fires INSIDE layer(), so the CUDA-event-based
``pure_fwd`` measurement includes CPU-induced GPU bubbles.  This script
bypasses that by:

  1. Finding each ``runkv:layer_compute:L{X}`` NVTX range (CPU timestamps).
  2. Mapping CUDA runtime launch calls within the range → GPU kernels via
     ``correlationId``.
  3. Summing **compute-stream** kernel durations → *kernel_active_ms*
     (pure GPU work, excluding all bubbles).
  4. Taking the GPU wall-clock span (first kernel start → last kernel end)
     → *gpu_span_ms* (includes DMA wait + prehook bubble).

The difference ``gpu_span_ms − kernel_active_ms`` is the total GPU idle
time within the layer (DMA wait + CPU-induced bubble).

Usage::

    python tools/extract_clean_layer_timing.py \\
        --runkv-sqlite  exp_results/sqlite/runkv_20260417.sqlite \\
        --tight-sqlite  exp_results/sqlite/tightllm_20260417.sqlite \\
        --output        exp_results/analysis/per_layer/clean_layer_timing.txt
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerStepRecord:
    layer_idx: int
    step_idx: int
    # CPU-side NVTX durations (ms)
    layer_compute_cpu_ms: float = 0.0
    prehook_cpu_ms: float = 0.0
    # GPU-side kernel measurements (ms)
    kernel_active_ms: float = 0.0  # sum of individual kernel durations
    gpu_span_ms: float = 0.0  # first kernel start → last kernel end
    gpu_bubble_ms: float = 0.0  # span - active
    num_kernels: int = 0


@dataclass
class MethodData:
    label: str
    records: dict[int, list[LayerStepRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )


def extract_from_sqlite(
    db_path: str, label: str, compute_stream_id: int = 7
) -> MethodData:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    data = MethodData(label=label)

    # ── 1. Collect layer_compute NVTX ranges grouped by layer ────────────
    c.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE text LIKE 'runkv:layer_compute:L%'
          AND end IS NOT NULL
        ORDER BY text, start
    """)
    layer_compute_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            layer_compute_ranges[int(m.group(1))].append((start, end))

    # ── 2. Collect pre_hook NVTX ranges grouped by layer ─────────────────
    c.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE text LIKE 'runkv_recompute:pre_hook:%'
          AND end IS NOT NULL
        ORDER BY text, start
    """)
    prehook_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            prehook_ranges[int(m.group(1))].append((start, end))

    # ── 3. For each layer × step, find GPU kernels ───────────────────────
    num_layers = max(layer_compute_ranges.keys()) + 1 if layer_compute_ranges else 0
    num_steps = (
        max(len(v) for v in layer_compute_ranges.values())
        if layer_compute_ranges
        else 0
    )

    # Pre-fetch all runtime launch calls and kernels for efficiency
    # Build a mapping: correlationId → kernel info
    c.execute("""
        SELECT correlationId, start, end, streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)
    kernel_by_corr: dict[int, tuple[int, int, int]] = {}
    for corr_id, k_start, k_end, stream_id in c.fetchall():
        kernel_by_corr[corr_id] = (k_start, k_end, stream_id)

    # Build sorted list of runtime launch calls
    c.execute("""
        SELECT r.correlationId, r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%Launch%'
        ORDER BY r.start
    """)
    all_launches = c.fetchall()  # [(correlationId, cpu_start), ...]

    # Binary search helper
    import bisect

    launch_cpu_starts = [x[1] for x in all_launches]
    launch_corr_ids = [x[0] for x in all_launches]

    for layer_idx in sorted(layer_compute_ranges.keys()):
        lc_list = layer_compute_ranges[layer_idx]
        ph_list = prehook_ranges.get(layer_idx, [])

        for step_idx, (lc_start, lc_end) in enumerate(lc_list):
            rec = LayerStepRecord(layer_idx=layer_idx, step_idx=step_idx)
            rec.layer_compute_cpu_ms = (lc_end - lc_start) / 1e6

            # Match prehook
            if step_idx < len(ph_list):
                ph_s, ph_e = ph_list[step_idx]
                if lc_start <= ph_s <= lc_end:
                    rec.prehook_cpu_ms = (ph_e - ph_s) / 1e6

            # Find launches within [lc_start, lc_end] using binary search
            lo = bisect.bisect_left(launch_cpu_starts, lc_start)
            hi = bisect.bisect_right(launch_cpu_starts, lc_end)

            compute_kernels = []
            for i in range(lo, hi):
                corr_id = launch_corr_ids[i]
                if corr_id in kernel_by_corr:
                    k_start, k_end, stream_id = kernel_by_corr[corr_id]
                    if stream_id == compute_stream_id:
                        compute_kernels.append((k_start, k_end))

            if compute_kernels:
                rec.num_kernels = len(compute_kernels)
                rec.kernel_active_ms = sum(
                    (ke - ks) / 1e6 for ks, ke in compute_kernels
                )
                gpu_start = min(ks for ks, _ in compute_kernels)
                gpu_end = max(ke for _, ke in compute_kernels)
                rec.gpu_span_ms = (gpu_end - gpu_start) / 1e6
                rec.gpu_bubble_ms = rec.gpu_span_ms - rec.kernel_active_ms

            data.records[layer_idx].append(rec)

    conn.close()
    return data


def _stat(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, p50, p95)."""
    if not values:
        return (0.0, 0.0, 0.0)
    s = sorted(values)
    return (
        statistics.mean(s),
        s[len(s) // 2],
        s[int(len(s) * 0.95)],
    )


def print_comparison(runkv: MethodData, tight: MethodData, out_path: str | None):
    lines: list[str] = []

    def pr(s: str = ""):
        lines.append(s)

    all_layers = sorted(set(runkv.records.keys()) | set(tight.records.keys()))

    # ═══════════════════════════════════════════════════════════════════
    # Section 1: Pre-hook CPU duration
    # ═══════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 78)
    pr("  PRE-HOOK CPU DURATION per layer (ms)")
    pr("=" * 78)
    pr("  This is CPU-side wall time of the attention pre-hook.")
    pr(
        "  It includes: event.synchronize() wait + plan build + metadata build + DMA launch"
    )
    pr(
        f"{'layer':<8} {'r_mean':>8} {'r_p50':>8} {'t_mean':>8} {'t_p50':>8} {'delta':>8}"
    )
    pr("-" * 48)

    r_ph_all, t_ph_all = [], []
    for li in all_layers:
        r_vals = [r.prehook_cpu_ms for r in runkv.records.get(li, [])]
        t_vals = [r.prehook_cpu_ms for r in tight.records.get(li, [])]
        r_m, r_p50, _ = _stat(r_vals)
        t_m, t_p50, _ = _stat(t_vals)
        pr(
            f"L{li:<7} {r_m:>8.2f} {r_p50:>8.2f} {t_m:>8.2f} {t_p50:>8.2f} {r_m - t_m:>+8.2f}"
        )
        if li > 0:
            r_ph_all.extend(r_vals)
            t_ph_all.extend(t_vals)

    pr(f"\n  RunKV  L1+ avg prehook: {statistics.mean(r_ph_all):.2f}ms")
    pr(f"  Tight  L1+ avg prehook: {statistics.mean(t_ph_all):.2f}ms")

    # ═══════════════════════════════════════════════════════════════════
    # Section 2: Clean GPU compute (kernel active time)
    # ═══════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 78)
    pr("  CLEAN GPU COMPUTE per layer (kernel_active_ms)")
    pr("=" * 78)
    pr("  Sum of individual CUDA kernel durations on compute stream.")
    pr("  Excludes all GPU bubbles (prehook CPU overhead + DMA wait).")
    pr(
        f"{'layer':<8} {'r_active':>10} {'r_span':>10} {'r_bubble':>10}"
        f" {'t_active':>10} {'t_span':>10} {'t_bubble':>10} {'act_delta':>10}"
    )
    pr("-" * 78)

    r_act_all, t_act_all = [], []
    r_bub_all, t_bub_all = [], []
    for li in all_layers:
        r_act = [r.kernel_active_ms for r in runkv.records.get(li, [])]
        r_spn = [r.gpu_span_ms for r in runkv.records.get(li, [])]
        r_bub = [r.gpu_bubble_ms for r in runkv.records.get(li, [])]
        t_act = [r.kernel_active_ms for r in tight.records.get(li, [])]
        t_spn = [r.gpu_span_ms for r in tight.records.get(li, [])]
        t_bub = [r.gpu_bubble_ms for r in tight.records.get(li, [])]

        ra, _, _ = _stat(r_act)
        rs, _, _ = _stat(r_spn)
        rb, _, _ = _stat(r_bub)
        ta, _, _ = _stat(t_act)
        ts, _, _ = _stat(t_spn)
        tb, _, _ = _stat(t_bub)

        pr(
            f"L{li:<7} {ra:>10.2f} {rs:>10.2f} {rb:>10.2f}"
            f" {ta:>10.2f} {ts:>10.2f} {tb:>10.2f} {ra - ta:>+10.2f}"
        )
        if li > 0:
            r_act_all.extend(r_act)
            t_act_all.extend(t_act)
            r_bub_all.extend(r_bub)
            t_bub_all.extend(t_bub)

    pr(
        f"\n  RunKV  L1+ avg kernel_active: {statistics.mean(r_act_all):.2f}ms"
        f"  avg bubble: {statistics.mean(r_bub_all):.2f}ms"
    )
    pr(
        f"  Tight  L1+ avg kernel_active: {statistics.mean(t_act_all):.2f}ms"
        f"  avg bubble: {statistics.mean(t_bub_all):.2f}ms"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Section 3: Per-token efficiency (clean)
    # ═══════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 78)
    pr("  CLEAN PER-TOKEN EFFICIENCY (kernel_active / num_kernels)")
    pr("=" * 78)
    pr("  kernel_active / actual_tokens  →  ms per 1000 tokens")
    pr("  If both methods have same kernel speed, this should scale linearly.")

    # Use the token counts from the JSONL data if available
    # For now, use average kernel counts as a proxy
    r_kern_counts = [
        statistics.mean([r.num_kernels for r in runkv.records.get(li, [])])
        for li in all_layers
        if li > 0
    ]
    t_kern_counts = [
        statistics.mean([r.num_kernels for r in tight.records.get(li, [])])
        for li in all_layers
        if li > 0
    ]

    pr(f"\n  RunKV  avg kernels/layer: {statistics.mean(r_kern_counts):.1f}")
    pr(f"  Tight  avg kernels/layer: {statistics.mean(t_kern_counts):.1f}")

    # ═══════════════════════════════════════════════════════════════════
    # Section 4: GPU bubble decomposition
    # ═══════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 78)
    pr("  GPU BUBBLE ANALYSIS")
    pr("=" * 78)
    pr("  bubble = gpu_span - kernel_active")
    pr("  bubble includes: DMA wait_event stall + prehook CPU-induced idle")
    pr(
        f"{'layer':<8} {'r_bubble':>10} {'t_bubble':>10} {'delta':>10}"
        f" {'r_prehook':>10} {'t_prehook':>10}"
    )
    pr("-" * 58)

    for li in all_layers:
        rb = (
            statistics.mean([r.gpu_bubble_ms for r in runkv.records.get(li, [])])
            if runkv.records.get(li)
            else 0
        )
        tb = (
            statistics.mean([r.gpu_bubble_ms for r in tight.records.get(li, [])])
            if tight.records.get(li)
            else 0
        )
        rp = (
            statistics.mean([r.prehook_cpu_ms for r in runkv.records.get(li, [])])
            if runkv.records.get(li)
            else 0
        )
        tp = (
            statistics.mean([r.prehook_cpu_ms for r in tight.records.get(li, [])])
            if tight.records.get(li)
            else 0
        )
        pr(
            f"L{li:<7} {rb:>10.2f} {tb:>10.2f} {rb - tb:>+10.2f}"
            f" {rp:>10.2f} {tp:>10.2f}"
        )

    pr(
        f"\n  RunKV  L1+ avg bubble: {statistics.mean(r_bub_all):.2f}ms"
        f"  avg prehook CPU: {statistics.mean(r_ph_all):.2f}ms"
    )
    pr(
        f"  Tight  L1+ avg bubble: {statistics.mean(t_bub_all):.2f}ms"
        f"  avg prehook CPU: {statistics.mean(t_ph_all):.2f}ms"
    )
    pr(
        f"\n  → RunKV bubble overhead vs TightLLM: "
        f"+{statistics.mean(r_bub_all) - statistics.mean(t_bub_all):.2f}ms/layer"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Section 5: IO timing (from nsys memcpy on load stream)
    # ═══════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 78)
    pr("  NOTE ON IO TIMING")
    pr("=" * 78)
    pr("  IO duration (load_ready - load_start) is measured via CUDA events")
    pr("  on the load_stream.  The pre-hook does NOT affect load_stream timing.")
    pr("  → IO measurements from the JSONL profiler are already clean.")

    output = "\n".join(lines)
    print(output)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output + "\n")
        print(f"\nWritten to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runkv-sqlite", required=True)
    parser.add_argument("--tight-sqlite", required=True)
    parser.add_argument(
        "--compute-stream",
        type=int,
        default=7,
        help="CUDA stream ID for the compute stream (default: 7)",
    )
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    print("Extracting RunKV kernel data...")
    runkv = extract_from_sqlite(args.runkv_sqlite, "RunKV", args.compute_stream)
    print("Extracting TightLLM kernel data...")
    tight = extract_from_sqlite(args.tight_sqlite, "TightLLM", args.compute_stream)
    print_comparison(runkv, tight, args.output)


if __name__ == "__main__":
    main()
