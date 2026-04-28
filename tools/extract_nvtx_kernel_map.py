#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Extract per-layer kernel breakdown from nsys sqlite exports.

For each ``runkv:layer_compute:L{X}`` NVTX range, finds all CUDA kernel
launches that occurred within that CPU time window, maps them to GPU
kernels via correlationId, and classifies each kernel into a phase:

  LayerNorm → QKV_proj → KV_cache → Attention → O_proj → Residual
  → LayerNorm → FFN_up → Activation → FFN_down → Residual

Usage:
  python tools/extract_nvtx_kernel_map.py \\
      --sqlite exp_results/sqlite/runkv_20260420.sqlite \\
      --step 2 \\
      --layers 0 1 2

  # All steps, all layers, save CSV:
  python tools/extract_nvtx_kernel_map.py \\
      --sqlite exp_results/sqlite/runkv_20260420.sqlite \\
      --output exp_results/analysis/kernel_map.csv

  # Compare runkv vs tightllm:
  python tools/extract_nvtx_kernel_map.py \\
      --sqlite exp_results/sqlite/runkv_20260420.sqlite \\
      --sqlite2 exp_results/sqlite/tightllm_20260420.sqlite \\
      --step 5
"""
from __future__ import annotations

import argparse
import bisect
import csv
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Kernel classification rules ─────────────────────────────────────────────
# Order matters: first match wins.  Patterns are matched against shortName.
KERNEL_PHASE_RULES: list[tuple[str, str]] = [
    # KV cache
    (r"reshape_and_cache", "KV_cache_write"),
    (r"batch_copy_blocks", "KV_block_copy"),
    # Attention
    (r"flash_fwd", "FlashAttn"),
    (r"splitKreduce", "FlashAttn_reduce"),
    (r"fmha", "Attention"),
    # LayerNorm
    (r"layer_norm", "LayerNorm"),
    (r"rms_norm", "RMSNorm"),
    # GEMM (cannot distinguish QKV/O/FFN by kernel name alone;
    #        we use position within the layer to refine below)
    (r"gemm|xmma_gemm|cutlass|Kernel2", "GEMM"),
    # Activations
    (r"silu|gelu|relu|swiglu|tanh", "Activation"),
    # Elementwise / residual
    (r"vectorized_elementwise|unrolled_elementwise|elementwise_kernel",
     "Elementwise"),
    # Index / gather / scatter (replay assembly)
    (r"index_elementwise|gather|scatter|indexSelect|CatArray", "Index/Gather"),
    # Reduce
    (r"reduce_kernel", "Reduce"),
]

_PHASE_PATTERNS = [(re.compile(p, re.IGNORECASE), label)
                   for p, label in KERNEL_PHASE_RULES]

# OPT layer forward order (for GEMM disambiguation):
#   LayerNorm → GEMM(QKV) → KV_cache → Attention → GEMM(O_proj) → Residual
#   → LayerNorm → GEMM(FFN_up) → Activation → GEMM(FFN_down) → Residual
_GEMM_SEQUENCE = [
    "GEMM:QKV_proj",
    "GEMM:O_proj",
    "GEMM:FFN_up",
    "GEMM:FFN_down",
]


def classify_kernel(name: str) -> str:
    for rx, label in _PHASE_PATTERNS:
        if rx.search(name):
            return label
    return "Other"


def refine_gemm_labels(kernels: list[dict]) -> None:
    """Disambiguate GEMM kernels by their order within the layer.

    OPT model has exactly 4 GEMMs per layer:
      1st = QKV projection
      2nd = output projection
      3rd = FFN up (fc1)
      4th = FFN down (fc2)
    """
    gemm_idx = 0
    for k in kernels:
        if k["phase"] == "GEMM":
            if gemm_idx < len(_GEMM_SEQUENCE):
                k["phase"] = _GEMM_SEQUENCE[gemm_idx]
            else:
                k["phase"] = f"GEMM:extra_{gemm_idx}"
            gemm_idx += 1


# ── Data extraction ─────────────────────────────────────────────────────────

def _nvtx_text_sql():
    return "COALESCE(n.text, s.value)"


def _nvtx_from():
    return "NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id"


@dataclass
class LayerKernelData:
    label: str  # e.g. "runkv" or "tightllm"
    step_idx: int
    layer_idx: int
    nvtx_start_ns: int
    nvtx_end_ns: int
    nvtx_dur_ms: float
    prehook_dur_ms: float | None  # pre_hook NVTX if found
    kernels: list[dict] = field(default_factory=list)

    @property
    def compute_stream_dur_ms(self) -> float:
        return sum(k["dur_ms"] for k in self.kernels if k["stream"] == 7)

    @property
    def kernel_summary(self) -> dict[str, float]:
        """Sum durations by phase."""
        result: dict[str, float] = defaultdict(float)
        for k in self.kernels:
            result[k["phase"]] += k["dur_ms"]
        return dict(result)


def extract_layer_kernels(
    db_path: str,
    label: str = "runkv",
    step_filter: int | None = None,
    layer_filter: list[int] | None = None,
    compute_stream_id: int = 7,
) -> list[LayerKernelData]:
    """Extract kernels for each layer_compute NVTX range."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ── 1. Collect layer_compute NVTX ranges ──────────────────────────────
    lc_query = f"""
        SELECT {_nvtx_text_sql()} as t, n.start, n.end
        FROM {_nvtx_from()}
        WHERE {_nvtx_text_sql()} LIKE 'runkv:layer_compute:L%'
          AND n.end IS NOT NULL
        ORDER BY n.start
    """
    lc_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.execute(lc_query).fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            lc_ranges[int(m.group(1))].append((start, end))

    if not lc_ranges:
        # Fallback: try without runkv prefix (for tightllm which shares the same NVTX)
        conn.close()
        return []

    # ── 2. Collect pre_hook NVTX ranges ──────────────────────────────────
    ph_query = f"""
        SELECT {_nvtx_text_sql()} as t, n.start, n.end
        FROM {_nvtx_from()}
        WHERE {_nvtx_text_sql()} LIKE 'runkv_recompute:pre_hook:%'
          AND n.end IS NOT NULL
        ORDER BY n.start
    """
    ph_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.execute(ph_query).fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            ph_ranges[int(m.group(1))].append((start, end))

    # ── 3. Pre-fetch all kernel launches and kernel records ──────────────
    c.execute("""
        SELECT r.correlationId, r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%Launch%'
        ORDER BY r.start
    """)
    all_launches = c.fetchall()
    launch_cpu_starts = [x[1] for x in all_launches]
    launch_corr_ids = [x[0] for x in all_launches]

    # Build kernel lookup by correlationId
    c.execute("""
        SELECT k.correlationId, k.start, k.end, k.streamId,
               s.value,
               k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
    """)
    kernel_by_corr: dict[int, tuple] = {}
    for row in c.fetchall():
        kernel_by_corr[row[0]] = row[1:]  # (start, end, streamId, name, gX,gY,gZ, bX,bY,bZ)

    conn.close()

    # ── 4. For each layer × step, extract kernels ────────────────────────
    results: list[LayerKernelData] = []
    num_steps = max(len(v) for v in lc_ranges.values()) if lc_ranges else 0

    for layer_idx in sorted(lc_ranges):
        if layer_filter is not None and layer_idx not in layer_filter:
            continue

        for step_idx, (lc_start, lc_end) in enumerate(lc_ranges[layer_idx]):
            if step_filter is not None and step_idx != step_filter:
                continue

            # Pre-hook duration
            prehook_dur = None
            ph_list = ph_ranges.get(layer_idx, [])
            if step_idx < len(ph_list):
                ph_s, ph_e = ph_list[step_idx]
                prehook_dur = (ph_e - ph_s) / 1e6

            # Find launches within [lc_start, lc_end]
            lo = bisect.bisect_left(launch_cpu_starts, lc_start)
            hi = bisect.bisect_right(launch_cpu_starts, lc_end)

            kernels = []
            for i in range(lo, hi):
                corr_id = launch_corr_ids[i]
                if corr_id not in kernel_by_corr:
                    continue
                kdata = kernel_by_corr[corr_id]
                k_start, k_end, stream_id, name = kdata[0], kdata[1], kdata[2], kdata[3]
                grid = f"{kdata[4]}x{kdata[5]}x{kdata[6]}"
                block = f"{kdata[7]}x{kdata[8]}x{kdata[9]}"

                phase = classify_kernel(name)
                kernels.append({
                    "name": name,
                    "phase": phase,
                    "stream": stream_id,
                    "start_ns": k_start,
                    "end_ns": k_end,
                    "dur_ms": (k_end - k_start) / 1e6,
                    "dur_us": (k_end - k_start) / 1e3,
                    "grid": grid,
                    "block": block,
                    "launch_cpu_ns": launch_cpu_starts[i],
                    "in_prehook": (
                        prehook_dur is not None
                        and step_idx < len(ph_list)
                        and ph_list[step_idx][0] <= launch_cpu_starts[i] <= ph_list[step_idx][1]
                    ),
                })

            # Refine GEMM labels by position (only for compute stream kernels)
            compute_kernels = [k for k in kernels if k["stream"] == compute_stream_id]
            refine_gemm_labels(compute_kernels)

            data = LayerKernelData(
                label=label,
                step_idx=step_idx,
                layer_idx=layer_idx,
                nvtx_start_ns=lc_start,
                nvtx_end_ns=lc_end,
                nvtx_dur_ms=(lc_end - lc_start) / 1e6,
                prehook_dur_ms=prehook_dur,
                kernels=kernels,
            )
            results.append(data)

    return results


# ── Output formatting ───────────────────────────────────────────────────────

def print_layer_detail(data: LayerKernelData) -> None:
    """Print detailed kernel list for one layer-step."""
    print(f"\n{'═' * 90}")
    print(f"  [{data.label}] Step {data.step_idx}  Layer L{data.layer_idx}")
    print(f"  NVTX layer_compute: {data.nvtx_dur_ms:.3f}ms")
    if data.prehook_dur_ms is not None:
        print(f"  NVTX pre_hook:      {data.prehook_dur_ms:.3f}ms")
    print(f"  Compute-stream kernel total: {data.compute_stream_dur_ms:.3f}ms")
    print(f"{'═' * 90}")

    print(f"  {'#':>3s}  {'Phase':<20s}  {'Stream':>6s}  {'Duration':>10s}  "
          f"{'Grid':<14s}  {'Block':<12s}  {'InHook':>6s}  Kernel Name")
    print(f"  {'─' * 3}  {'─' * 20}  {'─' * 6}  {'─' * 10}  "
          f"{'─' * 14}  {'─' * 12}  {'─' * 6}  {'─' * 40}")

    for i, k in enumerate(data.kernels):
        hook_marker = "  ✓" if k["in_prehook"] else ""
        dur_str = f"{k['dur_us']:.1f}µs" if k["dur_us"] < 1000 else f"{k['dur_ms']:.3f}ms"
        print(f"  {i:3d}  {k['phase']:<20s}  {k['stream']:6d}  {dur_str:>10s}  "
              f"{k['grid']:<14s}  {k['block']:<12s}  {hook_marker:>6s}  {k['name']}")

    # Phase summary
    summary = data.kernel_summary
    print(f"\n  Phase summary (compute stream only):")
    for phase, dur in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"    {phase:<25s}  {dur:.3f}ms")


def print_step_overview(results: list[LayerKernelData], step: int) -> None:
    """Print compact per-layer phase breakdown for one step."""
    step_data = [d for d in results if d.step_idx == step]
    if not step_data:
        print(f"  No data for step {step}")
        return

    step_data.sort(key=lambda d: d.layer_idx)

    # Collect all phase names
    all_phases: list[str] = []
    for d in step_data:
        for phase in d.kernel_summary:
            if phase not in all_phases:
                all_phases.append(phase)

    # Preferred order
    preferred = [
        "LayerNorm", "RMSNorm",
        "GEMM:QKV_proj", "KV_cache_write", "FlashAttn", "FlashAttn_reduce",
        "GEMM:O_proj", "Elementwise",
        "GEMM:FFN_up", "Activation", "GEMM:FFN_down",
        "Index/Gather", "KV_block_copy", "Reduce", "Other",
    ]
    ordered_phases = [p for p in preferred if p in all_phases]
    ordered_phases += [p for p in all_phases if p not in ordered_phases]

    label = step_data[0].label
    print(f"\n{'═' * 120}")
    print(f"  [{label}] Step {step} — Per-layer kernel phase durations (ms)")
    print(f"{'═' * 120}")

    # Header
    col_w = 10
    header = f"  {'Layer':<8s}  {'Total':>{col_w}s}  {'PreHook':>{col_w}s}"
    for p in ordered_phases:
        # Shorten phase name for column
        short = p.replace("GEMM:", "").replace("Flash", "FA_")
        header += f"  {short:>{col_w}s}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for d in step_data:
        summary = d.kernel_summary
        ph_str = f"{d.prehook_dur_ms:.2f}" if d.prehook_dur_ms is not None else "-"
        row = f"  L{d.layer_idx:<6d}  {d.nvtx_dur_ms:>{col_w}.3f}  {ph_str:>{col_w}s}"
        for p in ordered_phases:
            v = summary.get(p, 0)
            row += f"  {v:>{col_w}.3f}" if v > 0 else f"  {'-':>{col_w}s}"
        print(row)

    # Totals
    total_row = f"  {'TOTAL':<8s}  {sum(d.nvtx_dur_ms for d in step_data):>{col_w}.3f}  "
    total_row += f"{sum(d.prehook_dur_ms or 0 for d in step_data):>{col_w}.2f}"
    for p in ordered_phases:
        v = sum(d.kernel_summary.get(p, 0) for d in step_data)
        total_row += f"  {v:>{col_w}.3f}" if v > 0 else f"  {'-':>{col_w}s}"
    print("  " + "─" * (len(header) - 2))
    print(total_row)


def save_csv(results: list[LayerKernelData], output: str) -> None:
    """Save flat kernel list to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "step", "layer", "kernel_idx", "phase", "kernel_name",
            "stream", "dur_us", "grid", "block", "in_prehook",
            "layer_nvtx_ms", "prehook_ms",
        ])
        for d in results:
            for i, k in enumerate(d.kernels):
                writer.writerow([
                    d.label, d.step_idx, d.layer_idx, i, k["phase"], k["name"],
                    k["stream"], f"{k['dur_us']:.1f}", k["grid"], k["block"],
                    k["in_prehook"], f"{d.nvtx_dur_ms:.3f}",
                    f"{d.prehook_dur_ms:.3f}" if d.prehook_dur_ms is not None else "",
                ])
    print(f"\n[saved] {path}  ({sum(len(d.kernels) for d in results)} kernel records)")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract NVTX→kernel mapping from nsys sqlite"
    )
    ap.add_argument("--sqlite", required=True, help="nsys sqlite export (primary)")
    ap.add_argument("--sqlite2", default="", help="Second sqlite for comparison")
    ap.add_argument("--label", default="runkv", help="Label for primary sqlite")
    ap.add_argument("--label2", default="tightllm", help="Label for second sqlite")
    ap.add_argument("--step", type=int, default=None,
                    help="Show only this step index (0-based). Default: show overview of step 2.")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Show only these layer indices")
    ap.add_argument("--detail", action="store_true",
                    help="Show detailed per-kernel listing (verbose)")
    ap.add_argument("--compute-stream", type=int, default=7)
    ap.add_argument("--output", default="", help="Save CSV to this path")
    args = ap.parse_args()

    step = args.step if args.step is not None else 2  # Default to step 2 (skip warmup)

    print(f"Extracting kernel data from {args.sqlite} ...")
    results = extract_layer_kernels(
        args.sqlite,
        label=args.label,
        step_filter=step if not args.detail else args.step,
        layer_filter=args.layers,
        compute_stream_id=args.compute_stream,
    )
    print(f"  → {len(results)} layer-step records, "
          f"{sum(len(d.kernels) for d in results)} total kernel launches")

    if args.sqlite2:
        print(f"Extracting kernel data from {args.sqlite2} ...")
        results2 = extract_layer_kernels(
            args.sqlite2,
            label=args.label2,
            step_filter=step if not args.detail else args.step,
            layer_filter=args.layers,
            compute_stream_id=args.compute_stream,
        )
        print(f"  → {len(results2)} layer-step records, "
              f"{sum(len(d.kernels) for d in results2)} total kernel launches")
        results.extend(results2)

    if not results:
        print("No data found. Check that the sqlite has runkv:layer_compute:L* NVTX ranges.")
        return

    if args.detail:
        for d in results:
            print_layer_detail(d)
    else:
        # Group by label and print step overview
        by_label = defaultdict(list)
        for d in results:
            by_label[d.label].append(d)
        for label in sorted(by_label):
            print_step_overview(by_label[label], step)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
