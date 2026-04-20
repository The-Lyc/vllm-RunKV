#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-layer compute vs IO timing analysis for RunKV and TightLLM.

Data sources
------------
1. opt_component_mfu JSONL  — GPU-side timestamps via CUDA events
     compute_end_ms_from_anchor[L]: end of full layer L (attn+FFN) on GPU
     load_ready_ms_from_anchor[L]:  end of H2D DMA for layer L on load_stream

2. nsys sqlite              — CPU-side NVTX + GPU memcpy activity
     NVTX_EVENTS:            per-layer prehook sub-ranges (CPU wall time)
     CUPTI_ACTIVITY_KIND_MEMCPY: actual H2D DMA transfer durations (GPU time)

What we can derive
------------------
From existing data (approximate):
  layer_total_gpu[L] = compute_end[L] - compute_end[L-1]
    = prehook_stall_if_cpu_slower[L] + h2d_wait_if_dma_slow[L]
      + attn[L] + FFN[L]

  h2d_dma_dur[L]: from CUPTI MEMCPY whose end-time matches load_ready_abs[L]
    where load_ready_abs[L] = fwd_start_abs + load_ready_ms[L] / 1000

  prehook_nvtx_sum[L]: sum of h2d_sync + imbalance + build_plan +
                        build_meta + skip_ids + schedule_io from NVTX
    (schedule_io NVTX available after adding runkv:prehook:schedule_io:L*)

With new NVTX runkv:layer_compute:L* (added in opt.py):
  layer_compute_gpu[L]: CPU-side NVTX range that brackets the full
    layer forward. In nsys the CUDA kernels within this range show the
    true GPU active time for attn+FFN of layer L.

Usage
-----
# Approximate analysis from existing data:
python tools/analyze_per_layer_timing.py \\
    --runkv-mfu    exp_results/opt_feedback_observation/opt_component_mfu_*.jsonl \\
    --runkv-sqlite exp_results/sqlite/runkv-opt-2.7b_context=4k_bs=32_decode=32.sqlite \\
    --tightllm-sqlite exp_results/sqlite/tightllm-opt-2.7b_context=4k_bs=32_decode=32.sqlite \\
    --output-dir   exp_results/analysis/per_layer/

# After re-running with new NVTX (runkv:layer_compute:L*, runkv:prehook:schedule_io:L*):
# Pass the new sqlite — the script auto-detects which ranges are available.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np

    HAS_NP = True
except ImportError:
    HAS_NP = False

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════


def load_mfu_flat(paths: list[str]) -> list[dict]:
    """Load flat JSONL (one record per step×layer) from opt_component_mfu."""
    records: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [warn] not found: {p}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "layers" in rec:
                    # step-level nested record — expand
                    for lr in rec["layers"]:
                        flat = dict(lr)
                        flat.setdefault("step", rec.get("step"))
                        flat.setdefault("rank", rec.get("rank"))
                        records.append(flat)
                else:
                    records.append(rec)
    return records


def load_nvtx(sqlite_path: str) -> list[dict]:
    if not Path(sqlite_path).exists():
        return []
    db = sqlite3.connect(sqlite_path)
    try:
        rows = db.execute(
            "SELECT text, start, end FROM NVTX_EVENTS "
            "WHERE text IS NOT NULL AND end IS NOT NULL"
        ).fetchall()
    finally:
        db.close()
    return [
        {"text": r[0], "start_ns": r[1], "end_ns": r[2], "dur_ms": (r[2] - r[1]) / 1e6}
        for r in rows
    ]


def extract_kernel_timing_from_sqlite(
    sqlite_path: str,
    compute_stream_id: int = 7,
) -> dict[int, dict[str, list[float]]]:
    """Extract clean per-layer GPU kernel timing from nsys sqlite.

    For each ``runkv:layer_compute:L{X}`` NVTX range (one per step×layer):
      1. Find CUDA runtime launch calls within the CPU time window.
      2. Map to GPU kernels via ``correlationId``.
      3. Sum compute-stream kernel durations → ``kernel_active_ms``.
      4. GPU wall-clock span (first→last kernel) → ``gpu_span_ms``.
      5. ``gpu_bubble_ms = gpu_span - kernel_active``.
      6. Matched ``pre_hook`` NVTX duration → ``prehook_cpu_ms``.

    Returns ``{layer_idx: {"kernel_active": [...], "gpu_span": [...],
                           "gpu_bubble": [...], "prehook_cpu": [...]}}``.
    """
    import bisect

    if not Path(sqlite_path).exists():
        return {}
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # ── 1. layer_compute NVTX ranges ────────────────────────────────────
    c.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
        WHERE text LIKE 'runkv:layer_compute:L%' AND end IS NOT NULL
        ORDER BY text, start
    """)
    lc_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            lc_ranges[int(m.group(1))].append((start, end))

    if not lc_ranges:
        conn.close()
        return {}

    # ── 2. pre_hook NVTX ranges ─────────────────────────────────────────
    c.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
        WHERE text LIKE 'runkv_recompute:pre_hook:%' AND end IS NOT NULL
        ORDER BY text, start
    """)
    ph_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in c.fetchall():
        m = re.search(r":L(\d+)$", text)
        if m:
            ph_ranges[int(m.group(1))].append((start, end))

    # ── 3. Pre-fetch kernel and runtime launch data ─────────────────────
    c.execute(
        "SELECT correlationId, start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )
    kernel_by_corr: dict[int, tuple[int, int, int]] = {}
    for corr_id, k_start, k_end, stream_id in c.fetchall():
        kernel_by_corr[corr_id] = (k_start, k_end, stream_id)

    c.execute("""
        SELECT r.correlationId, r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%Launch%'
        ORDER BY r.start
    """)
    all_launches = c.fetchall()
    conn.close()

    launch_cpu_starts = [x[1] for x in all_launches]
    launch_corr_ids = [x[0] for x in all_launches]

    # ── 4. For each layer × step, find compute-stream kernels ───────────
    result: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "kernel_active": [],
            "gpu_span": [],
            "gpu_bubble": [],
            "prehook_cpu": [],
        }
    )

    for layer_idx in sorted(lc_ranges):
        ph_list = ph_ranges.get(layer_idx, [])
        for step_idx, (lc_start, lc_end) in enumerate(lc_ranges[layer_idx]):
            # Pre-hook CPU duration
            prehook_ms = 0.0
            if step_idx < len(ph_list):
                ph_s, ph_e = ph_list[step_idx]
                if lc_start <= ph_s <= lc_end:
                    prehook_ms = (ph_e - ph_s) / 1e6

            # Find launches within [lc_start, lc_end]
            lo = bisect.bisect_left(launch_cpu_starts, lc_start)
            hi = bisect.bisect_right(launch_cpu_starts, lc_end)
            compute_kernels = []
            for i in range(lo, hi):
                corr_id = launch_corr_ids[i]
                if corr_id in kernel_by_corr:
                    k_start, k_end, sid = kernel_by_corr[corr_id]
                    if sid == compute_stream_id:
                        compute_kernels.append((k_start, k_end))

            if compute_kernels:
                active = sum((ke - ks) / 1e6 for ks, ke in compute_kernels)
                gpu_s = min(ks for ks, _ in compute_kernels)
                gpu_e = max(ke for _, ke in compute_kernels)
                span = (gpu_e - gpu_s) / 1e6
            else:
                active = 0.0
                span = 0.0

            result[layer_idx]["kernel_active"].append(active)
            result[layer_idx]["gpu_span"].append(span)
            result[layer_idx]["gpu_bubble"].append(span - active)
            result[layer_idx]["prehook_cpu"].append(prehook_ms)

    return dict(result)


def load_memcpy_h2d(sqlite_path: str, stream_id: int | None = None) -> list[dict]:
    """Load H2D (copyKind=1) memcpy records from nsys sqlite."""
    if not Path(sqlite_path).exists():
        return []
    db = sqlite3.connect(sqlite_path)
    try:
        if stream_id is not None:
            rows = db.execute(
                "SELECT start, end, bytes, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE copyKind=1 AND streamId=? ORDER BY start",
                (stream_id,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT start, end, bytes, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE copyKind=1 ORDER BY start"
            ).fetchall()
    finally:
        db.close()
    return [
        {
            "start_ns": r[0],
            "end_ns": r[1],
            "bytes": r[2],
            "stream_id": r[3],
            "dur_ms": (r[1] - r[0]) / 1e6,
        }
        for r in rows
    ]


def detect_h2d_load_stream(sqlite_path: str) -> int | None:
    """Heuristically detect the H2D load stream (largest avg transfer size)."""
    if not Path(sqlite_path).exists():
        return None
    db = sqlite3.connect(sqlite_path)
    try:
        rows = db.execute(
            "SELECT streamId, AVG(bytes), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE copyKind=1 GROUP BY streamId ORDER BY AVG(bytes) DESC LIMIT 1"
        ).fetchall()
    finally:
        db.close()
    return int(rows[0][0]) if rows else None


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer compute delta (from compute_end GPU events)
# ══════════════════════════════════════════════════════════════════════════════


def compute_layer_deltas(
    flat: list[dict],
    skip_steps: int = 1,
) -> dict[int, list[float]]:
    """
    Compute per-layer GPU time as delta between consecutive compute_end events.

    layer_total_gpu[L] = compute_end[L] - compute_end[L-1]
    = (prehook stall if CPU slower than prev FFN)
      + (DMA wait if H2D not ready)
      + attn[L] + FFN[L]

    Returns {layer_idx: [measurements across steps]}.
    """
    # Group by step
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in flat:
        if r.get("step", 0) >= skip_steps:
            by_step[r["step"]].append(r)

    deltas: dict[int, list[float]] = defaultdict(list)
    for step, layers in sorted(by_step.items()):
        layers_s = sorted(layers, key=lambda x: x.get("layer_idx", 0))
        prev_end: float | None = None
        for lr in layers_s:
            ce = lr.get("compute_end_ms_from_anchor")
            li = lr.get("layer_idx")
            if ce is None or li is None:
                prev_end = ce
                continue
            if prev_end is not None:
                deltas[li].append(ce - prev_end)
            prev_end = ce
    return dict(deltas)


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer imbalance (from imbalance_ms in JSONL)
# ══════════════════════════════════════════════════════════════════════════════


def collect_imbalance(
    flat: list[dict],
    skip_steps: int = 1,
) -> dict[int, list[float]]:
    """Collect imbalance_ms per layer across steps.

    Returns {layer_idx: [imbalance_ms values across steps]}.
    """
    result: dict[int, list[float]] = defaultdict(list)
    for r in flat:
        if r.get("step", 0) < skip_steps:
            continue
        li = r.get("layer_idx")
        imb = r.get("imbalance_ms")
        if li is not None and imb is not None:
            result[li].append(float(imb))
    return dict(result)


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer compute_end vs load_ready (from JSONL GPU events)
# ══════════════════════════════════════════════════════════════════════════════


def collect_compute_and_io_events(
    flat: list[dict],
    skip_steps: int = 1,
) -> dict[str, dict[int, list[float]]]:
    """Collect all per-layer timing events across steps.

    Returns dict with keys:
      compute_start, forward_start, compute_end, load_start,
      load_ready, kv_ready, hs_ready
    Each value is {layer_idx: [values across steps]}.
    """
    result: dict[str, dict[int, list[float]]] = {
        "compute_start": defaultdict(list),
        "forward_start": defaultdict(list),
        "compute_end": defaultdict(list),
        "load_start": defaultdict(list),
        "load_ready": defaultdict(list),
        "kv_ready": defaultdict(list),
        "hs_ready": defaultdict(list),
    }
    field_map = {
        "compute_start_ms_from_anchor": "compute_start",
        "forward_start_ms_from_anchor": "forward_start",
        "compute_end_ms_from_anchor": "compute_end",
        "load_start_ms_from_anchor": "load_start",
        "load_ready_ms_from_anchor": "load_ready",
        "kv_ready_ms_from_anchor": "kv_ready",
        "hs_ready_ms_from_anchor": "hs_ready",
    }
    for r in flat:
        if r.get("step", 0) < skip_steps:
            continue
        li = r.get("layer_idx")
        if li is None:
            continue
        for field, key in field_map.items():
            v = r.get(field)
            if v is not None:
                result[key][li].append(float(v))
    return {k: dict(v) for k, v in result.items()}


def collect_per_step_layer_timing(
    flat: list[dict],
    skip_steps: int = 1,
) -> dict[str, dict[int, dict[int, float]]]:
    """Collect per-step × per-layer timing for compute, IO, and imbalance.

    Returns ``{metric: {step: {layer_idx: value_ms}}}``.

    Metrics:
      - ``compute_dur``: compute_end[L] - compute_end[L-1]  (layer GPU time)
      - ``io_dur``: load_ready[L] - load_start[L]  (H2D transfer duration)
      - ``imbalance``: imbalance_ms from JSONL
      - ``compute_end``: raw compute_end_ms_from_anchor
      - ``load_ready``: raw load_ready_ms_from_anchor
    """
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in flat:
        step = r.get("step", 0)
        if step >= skip_steps:
            by_step[step].append(r)

    result: dict[str, dict[int, dict[int, float]]] = {
        "compute_dur": defaultdict(dict),
        "io_dur": defaultdict(dict),
        "imbalance": defaultdict(dict),
        "compute_end": defaultdict(dict),
        "load_ready": defaultdict(dict),
    }

    for step, records in sorted(by_step.items()):
        layers_s = sorted(records, key=lambda x: x.get("layer_idx", 0))
        prev_end: float | None = None
        for lr in layers_s:
            li = lr.get("layer_idx")
            if li is None:
                continue
            ce = lr.get("compute_end_ms_from_anchor")
            if ce is not None:
                result["compute_end"][step][li] = ce
                if prev_end is not None:
                    result["compute_dur"][step][li] = ce - prev_end
                prev_end = ce

            ls = lr.get("load_start_ms_from_anchor")
            lready = lr.get("load_ready_ms_from_anchor")
            if lready is not None:
                result["load_ready"][step][li] = lready
            if ls is not None and lready is not None:
                result["io_dur"][step][li] = lready - ls

            imb = lr.get("imbalance_ms")
            if imb is not None:
                result["imbalance"][step][li] = float(imb)

    return {k: dict(v) for k, v in result.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer H2D DMA timing from CUPTI + compute_end anchor correlation
# ══════════════════════════════════════════════════════════════════════════════


def _find_forward_anchors(nvtx: list[dict]) -> list[float]:
    """
    Return list of absolute step start times (ns) from 'gpu_model_runner: forward'.
    """
    anchors = []
    for r in nvtx:
        if r["text"] == "gpu_model_runner: forward":
            anchors.append(float(r["start_ns"]))
    return sorted(anchors)


def correlate_dma_to_layers(
    flat: list[dict],
    nvtx: list[dict],
    memcpy_h2d: list[dict],
    skip_steps: int = 1,
    tol_ns: float = 2e6,  # 2 ms tolerance for matching
) -> dict[int, list[float]]:
    """
    Match each layer's load_ready_ms_from_anchor to an H2D MEMCPY end time,
    giving us the actual H2D DMA transfer duration per layer.

    Strategy:
      1. Get step anchor (forward() start) from NVTX in absolute ns.
      2. For each (step, layer), compute load_ready_abs_ns = anchor_ns + load_ready_ms * 1e6.
      3. Find the MEMCPY on the H2D load stream whose end_ns is closest to load_ready_abs_ns.
      4. That MEMCPY's duration = actual DMA time for that layer.

    Returns {layer_idx: [dma_dur_ms across steps]}.
    """
    anchors = _find_forward_anchors(nvtx)
    if not anchors or not memcpy_h2d:
        return {}

    # Build memcpy end-time lookup (sorted for bisect)
    import bisect

    memcpy_ends = sorted(m["end_ns"] for m in memcpy_h2d)
    memcpy_by_end = {m["end_ns"]: m for m in memcpy_h2d}

    # Group flat by step
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in flat:
        if (
            r.get("step", 0) >= skip_steps
            and r.get("load_ready_ms_from_anchor") is not None
        ):
            by_step[r["step"]].append(r)

    dma_durs: dict[int, list[float]] = defaultdict(list)
    step_list = sorted(by_step.keys())

    for step_i, step in enumerate(step_list):
        if step_i >= len(anchors):
            break
        anchor_ns = (
            anchors[step_i + skip_steps]
            if step_i + skip_steps < len(anchors)
            else anchors[step_i]
        )
        # Use step index to pick anchor (steps start at skip_steps)
        # Actually we need anchor[step] not anchor[step_i]
        # Re-index: step -> anchor by position
        # anchors are sorted by time; steps are sequential → step N uses anchors[N]
        if step < len(anchors):
            anchor_ns = anchors[step]
        else:
            continue

        for lr in by_step[step]:
            load_ready_ms = lr["load_ready_ms_from_anchor"]
            layer_idx = lr.get("layer_idx")
            if layer_idx is None:
                continue
            load_ready_abs_ns = anchor_ns + load_ready_ms * 1e6

            # Find closest MEMCPY end
            pos = bisect.bisect_left(memcpy_ends, load_ready_abs_ns - tol_ns)
            best_match = None
            best_delta = float("inf")
            for idx in range(max(0, pos - 2), min(len(memcpy_ends), pos + 5)):
                end_ns = memcpy_ends[idx]
                delta = abs(end_ns - load_ready_abs_ns)
                if delta < best_delta:
                    best_delta = delta
                    best_match = memcpy_by_end[end_ns]

            if best_match is not None and best_delta < tol_ns:
                dma_durs[layer_idx].append(best_match["dur_ms"])

    return dict(dma_durs)


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer prehook overhead from NVTX
# ══════════════════════════════════════════════════════════════════════════════

_PREHOOK_NVTX_PREFIXES = [
    ("h2d_sync", "runkv:h2d_sync:L"),
    ("imbalance", "runkv:prehook:imbalance:L"),
    ("build_plan", "runkv:prehook:build_plan:L"),
    ("build_meta", "runkv:prehook:build_metadata:L"),
    ("skip_ids", "runkv:prehook:skip_ids:L"),
    ("schedule_io", "runkv:prehook:schedule_io:L"),  # available after patch
]

# Layer-level compute NVTX (available after adding to opt.py)
_LAYER_COMPUTE_PREFIX = "runkv:layer_compute:L"


def extract_prehook_by_layer(
    nvtx: list[dict],
) -> dict[str, dict[int, float]]:
    """
    For each prehook sub-range, compute avg duration per layer.

    Returns {segment: {layer_idx: avg_ms}}.
    """
    result: dict[str, dict[int, list[float]]] = {
        seg: defaultdict(list) for seg, _ in _PREHOOK_NVTX_PREFIXES
    }
    for r in nvtx:
        for seg, prefix in _PREHOOK_NVTX_PREFIXES:
            if r["text"].startswith(prefix):
                try:
                    layer_idx = int(r["text"][len(prefix) :])
                    result[seg][layer_idx].append(r["dur_ms"])
                except ValueError:
                    pass
    return {
        seg: {k: sum(v) / len(v) for k, v in d.items()} for seg, d in result.items()
    }


def extract_layer_compute_nvtx(nvtx: list[dict]) -> dict[int, float]:
    """
    Extract avg duration of runkv:layer_compute:L* NVTX ranges.
    Available only after patching opt.py.
    """
    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in nvtx:
        if r["text"].startswith(_LAYER_COMPUTE_PREFIX):
            try:
                layer_idx = int(r["text"][len(_LAYER_COMPUTE_PREFIX) :])
                by_layer[layer_idx].append(r["dur_ms"])
            except ValueError:
                pass
    return {k: sum(v) / len(v) for k, v in by_layer.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    s = sorted(vals)
    n = len(s)
    return {"n": n, "mean": sum(s) / n, "p50": s[n // 2], "p95": s[int(n * 0.95)]}


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════


class Report:
    def __init__(self, out_dir: Path) -> None:
        self.lines: list[str] = []
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def h1(self, t: str) -> None:
        sep = "═" * 72
        self.lines += ["", sep, f"  {t}", sep]

    def h2(self, t: str) -> None:
        self.lines += ["", f"── {t} {'─' * max(0, 68 - len(t))}"]

    def row(self, line: str) -> None:
        self.lines.append(line)

    def table(
        self, headers: list[str], rows: list[list[Any]], col_width: int = 13
    ) -> None:
        fmt = "  ".join(f"{{:<{col_width}}}" for _ in headers)
        self.lines.append(fmt.format(*headers))
        self.lines.append("  ".join("-" * col_width for _ in headers))
        for r in rows:
            self.lines.append(fmt.format(*[str(x) for x in r]))

    def save(self, name: str = "per_layer_summary.txt") -> Path:
        out = self.out_dir / name
        text = "\n".join(self.lines) + "\n"
        out.write_text(text)
        print(text)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis sections
# ══════════════════════════════════════════════════════════════════════════════


def analyze_layer_total_gpu(
    rpt: Report,
    runkv_deltas: dict[int, list[float]],
    tightllm_deltas: dict[int, list[float]],
) -> None:
    rpt.h1("PER-LAYER TOTAL GPU TIME  (compute_end[L] − compute_end[L−1], ms)")
    rpt.row("  Includes: prehook CPU stall + H2D wait + attn[L] + FFN[L].")
    rpt.row("  NOTE: prehook CPU work on host is overlapped with prev FFN on GPU.")
    rpt.row("  If prehook > prev_FFN, GPU stalls → layer total inflates.")

    all_layers = sorted(set(runkv_deltas) | set(tightllm_deltas))
    headers = ["layer", "runkv_mean", "runkv_p95", "tight_mean", "tight_p95", "delta"]
    rows = []
    for L in all_layers:
        rv = runkv_deltas.get(L, [])
        tv = tightllm_deltas.get(L, [])
        r_m = _avg(rv)
        t_m = _avg(tv)
        rows.append(
            [
                f"L{L}",
                f"{r_m:.2f}" if rv else "-",
                f"{_stats(rv)['p95']:.2f}" if rv else "-",
                f"{t_m:.2f}" if tv else "-",
                f"{_stats(tv)['p95']:.2f}" if tv else "-",
                f"{r_m - t_m:+.2f}" if (rv and tv) else "-",
            ]
        )
    rpt.table(headers, rows)

    # Summary
    if runkv_deltas and tightllm_deltas:
        r_sum = sum(_avg(v) for v in runkv_deltas.values())
        t_sum = sum(_avg(v) for v in tightllm_deltas.values())
        rpt.row(
            f"\n  Sum over all layers: RunKV={r_sum:.1f}ms  TightLLM={t_sum:.1f}ms"
            f"  delta={r_sum - t_sum:+.1f}ms"
        )


def analyze_h2d_dma(
    rpt: Report,
    runkv_dma: dict[int, list[float]],
    tightllm_dma: dict[int, list[float]],
) -> None:
    rpt.h1("PER-LAYER H2D DMA TIME  (CUPTI MEMCPY duration, correlated via load_ready)")
    rpt.row("  Actual GPU-side transfer time on load_stream per layer.")
    rpt.row(
        "  Both methods use same DMA path → should be similar if same blocks prefetched."
    )

    all_layers = sorted(set(runkv_dma) | set(tightllm_dma))
    if not all_layers:
        rpt.row("  (no data — correlation failed or sqlite not provided)")
        return

    headers = ["layer", "runkv_dma_ms", "tight_dma_ms", "delta_ms", "r_n", "t_n"]
    rows = []
    for L in all_layers:
        rv = runkv_dma.get(L, [])
        tv = tightllm_dma.get(L, [])
        r_m = _avg(rv)
        t_m = _avg(tv)
        rows.append(
            [
                f"L{L}",
                f"{r_m:.3f}" if rv else "-",
                f"{t_m:.3f}" if tv else "-",
                f"{r_m - t_m:+.3f}" if (rv and tv) else "-",
                str(len(rv)),
                str(len(tv)),
            ]
        )
    rpt.table(headers, rows)


def analyze_prehook_per_layer(
    rpt: Report,
    label: str,
    prehook: dict[str, dict[int, float]],
) -> None:
    rpt.h2(f"{label} prehook sub-range avg duration per layer (ms, from NVTX)")
    segments = [s for s, _ in _PREHOOK_NVTX_PREFIXES if prehook.get(s)]
    if not segments:
        rpt.row("  (no prehook NVTX data found)")
        return
    all_layers = sorted({L for seg in segments for L in prehook[seg]})
    headers = ["layer"] + segments + ["total"]
    rows = []
    for L in all_layers:
        row = [f"L{L}"]
        total = 0.0
        for seg in segments:
            v = prehook[seg].get(L, float("nan"))
            row.append(f"{v:.3f}" if v == v else "-")
            if v == v:
                total += v
        row.append(f"{total:.3f}")
        rows.append(row)
    rpt.table(headers, rows, col_width=12)

    # Sum per segment
    rpt.row("\n  Per-segment sum over all layers:")
    grand = 0.0
    for seg in segments:
        s = sum(prehook[seg].values())
        grand += s
        rpt.row(f"    {seg:15s} {s:.2f} ms/step")
    rpt.row(f"    {'TOTAL':15s} {grand:.2f} ms/step")


def analyze_imbalance(
    rpt: Report,
    runkv_imb: dict[int, list[float]],
    tightllm_imb: dict[int, list[float]],
) -> None:
    rpt.h1("PER-LAYER IMBALANCE  (imbalance_ms from CUDA events)")
    rpt.row("  imbalance = compute_end[L] − load_ready[L+1]")
    rpt.row("  negative → IO finished well before compute (IO has slack)")
    rpt.row("  positive → compute finished before IO (GPU stalls waiting for DMA)")
    rpt.row("  closer to 0 → better overlap between compute and IO")

    all_layers = sorted(set(runkv_imb) | set(tightllm_imb))
    if not all_layers:
        rpt.row("  (no imbalance data)")
        return

    headers = [
        "layer",
        "runkv_mean",
        "runkv_p50",
        "runkv_p95",
        "tight_mean",
        "tight_p50",
        "tight_p95",
        "delta_mean",
    ]
    rows = []
    for L in all_layers:
        rv = runkv_imb.get(L, [])
        tv = tightllm_imb.get(L, [])
        rs = _stats(rv)
        ts = _stats(tv)
        rows.append(
            [
                f"L{L}",
                f"{rs['mean']:.2f}" if rv else "-",
                f"{rs['p50']:.2f}" if rv else "-",
                f"{rs['p95']:.2f}" if rv else "-",
                f"{ts['mean']:.2f}" if tv else "-",
                f"{ts['p50']:.2f}" if tv else "-",
                f"{ts['p95']:.2f}" if tv else "-",
                f"{rs['mean'] - ts['mean']:+.2f}" if (rv and tv) else "-",
            ]
        )
    rpt.table(headers, rows)

    # Absolute imbalance table
    rpt.h2("Absolute imbalance |imbalance_ms| (sign-independent balance quality)")
    rpt.row("  Lower = better overlap between compute and IO.")
    abs_headers = [
        "layer",
        "runkv |mean|",
        "runkv |p50|",
        "runkv |p95|",
        "tight |mean|",
        "tight |p50|",
        "tight |p95|",
        "delta |mean|",
    ]
    abs_rows = []
    for L in all_layers:
        rv = [abs(x) for x in runkv_imb.get(L, [])]
        tv = [abs(x) for x in tightllm_imb.get(L, [])]
        rs = _stats(rv)
        ts = _stats(tv)
        abs_rows.append(
            [
                f"L{L}",
                f"{rs['mean']:.2f}" if rv else "-",
                f"{rs['p50']:.2f}" if rv else "-",
                f"{rs['p95']:.2f}" if rv else "-",
                f"{ts['mean']:.2f}" if tv else "-",
                f"{ts['p50']:.2f}" if tv else "-",
                f"{ts['p95']:.2f}" if tv else "-",
                f"{rs['mean'] - ts['mean']:+.2f}" if (rv and tv) else "-",
            ]
        )
    rpt.table(abs_headers, abs_rows)

    # Summary
    if runkv_imb and tightllm_imb:
        r_mean_all = _avg([_avg(v) for v in runkv_imb.values()])
        t_mean_all = _avg([_avg(v) for v in tightllm_imb.values()])
        r_abs_mean = _avg([_avg([abs(x) for x in v]) for v in runkv_imb.values()])
        t_abs_mean = _avg([_avg([abs(x) for x in v]) for v in tightllm_imb.values()])
        r_pos = sum(1 for v in runkv_imb.values() if _avg(v) > 0)
        t_pos = sum(1 for v in tightllm_imb.values() if _avg(v) > 0)
        rpt.row(
            f"\n  Signed avg imbalance:    RunKV={r_mean_all:.2f}ms  "
            f"TightLLM={t_mean_all:.2f}ms"
        )
        rpt.row(
            f"  Absolute avg |imbalance|: RunKV={r_abs_mean:.2f}ms  "
            f"TightLLM={t_abs_mean:.2f}ms"
        )
        rpt.row(
            f"  Layers with positive mean (GPU stall): "
            f"RunKV={r_pos}/{len(runkv_imb)}  "
            f"TightLLM={t_pos}/{len(tightllm_imb)}"
        )


def analyze_compute_vs_io(
    rpt: Report,
    runkv_ev: dict[str, dict[int, list[float]]],
    tightllm_ev: dict[str, dict[int, list[float]]],
) -> None:
    runkv_cs = runkv_ev["compute_start"]
    runkv_ce = runkv_ev["compute_end"]
    runkv_ls = runkv_ev["load_start"]
    runkv_lr = runkv_ev["load_ready"]
    tightllm_cs = tightllm_ev["compute_start"]
    tightllm_ce = tightllm_ev["compute_end"]
    tightllm_ls = tightllm_ev["load_start"]
    tightllm_lr = tightllm_ev["load_ready"]

    # ── Section 1: Absolute event timelines ─────────────────────────────
    rpt.h1("PER-LAYER EVENT TIMELINES  (absolute ms from step anchor)")
    rpt.row("  compute_start/end[L]: layer L forward on compute stream")
    rpt.row("  load_start/ready[L]:  H2D DMA for layer L on load stream")

    all_layers = sorted(
        set(runkv_ce) | set(runkv_lr) | set(tightllm_ce) | set(tightllm_lr)
    )
    if not all_layers:
        rpt.row("  (no data)")
        return

    headers = ["layer", "r_cs", "r_ce", "r_ls", "r_lr", "t_cs", "t_ce", "t_ls", "t_lr"]
    rows = []
    for L in all_layers:
        rows.append(
            [
                f"L{L}",
                f"{_avg(runkv_cs[L]):.1f}" if runkv_cs.get(L) else "-",
                f"{_avg(runkv_ce[L]):.1f}" if runkv_ce.get(L) else "-",
                f"{_avg(runkv_ls[L]):.1f}" if runkv_ls.get(L) else "-",
                f"{_avg(runkv_lr[L]):.1f}" if runkv_lr.get(L) else "-",
                f"{_avg(tightllm_cs[L]):.1f}" if tightllm_cs.get(L) else "-",
                f"{_avg(tightllm_ce[L]):.1f}" if tightllm_ce.get(L) else "-",
                f"{_avg(tightllm_ls[L]):.1f}" if tightllm_ls.get(L) else "-",
                f"{_avg(tightllm_lr[L]):.1f}" if tightllm_lr.get(L) else "-",
            ]
        )
    rpt.table(headers, rows, col_width=10)

    # ── Section 2: Per-layer durations (compute and IO) ─────────────────
    rpt.h1("PER-LAYER DURATIONS  (compute_dur = end − start, io_dur = ready − start)")
    rpt.row("  compute_dur[L]: true GPU compute time (attn + FFN), no prehook")
    rpt.row("  io_dur[L]:      true H2D DMA transfer time on load stream")
    has_start_data = bool(runkv_cs or tightllm_cs or runkv_ls or tightllm_ls)
    if not has_start_data:
        rpt.row("  ⚠ compute_start / load_start not available in current data.")
        rpt.row("  Re-run profiling after this code change to get accurate durations.")
        rpt.row("  Falling back to compute_end deltas (includes prehook overhead).")

    headers2 = [
        "layer",
        "r_comp_dur",
        "r_io_dur",
        "t_comp_dur",
        "t_io_dur",
        "comp_delta",
        "io_delta",
    ]
    rows2 = []
    for L in all_layers:
        # compute duration
        r_cd = (
            _avg(runkv_ce[L]) - _avg(runkv_cs[L])
            if runkv_ce.get(L) and runkv_cs.get(L)
            else float("nan")
        )
        t_cd = (
            _avg(tightllm_ce[L]) - _avg(tightllm_cs[L])
            if tightllm_ce.get(L) and tightllm_cs.get(L)
            else float("nan")
        )
        # io duration
        r_id = (
            _avg(runkv_lr[L]) - _avg(runkv_ls[L])
            if runkv_lr.get(L) and runkv_ls.get(L)
            else float("nan")
        )
        t_id = (
            _avg(tightllm_lr[L]) - _avg(tightllm_ls[L])
            if tightllm_lr.get(L) and tightllm_ls.get(L)
            else float("nan")
        )
        rows2.append(
            [
                f"L{L}",
                f"{r_cd:.2f}" if r_cd == r_cd else "-",
                f"{r_id:.2f}" if r_id == r_id else "-",
                f"{t_cd:.2f}" if t_cd == t_cd else "-",
                f"{t_id:.2f}" if t_id == t_id else "-",
                f"{r_cd - t_cd:+.2f}" if (r_cd == r_cd and t_cd == t_cd) else "-",
                f"{r_id - t_id:+.2f}" if (r_id == r_id and t_id == t_id) else "-",
            ]
        )
    rpt.table(headers2, rows2)

    # Summary
    r_comp_durs = [
        _avg(runkv_ce[L]) - _avg(runkv_cs[L])
        for L in all_layers
        if runkv_ce.get(L) and runkv_cs.get(L)
    ]
    t_comp_durs = [
        _avg(tightllm_ce[L]) - _avg(tightllm_cs[L])
        for L in all_layers
        if tightllm_ce.get(L) and tightllm_cs.get(L)
    ]
    r_io_durs = [
        _avg(runkv_lr[L]) - _avg(runkv_ls[L])
        for L in all_layers
        if runkv_lr.get(L) and runkv_ls.get(L)
    ]
    t_io_durs = [
        _avg(tightllm_lr[L]) - _avg(tightllm_ls[L])
        for L in all_layers
        if tightllm_lr.get(L) and tightllm_ls.get(L)
    ]
    if r_comp_durs:
        rpt.row(
            f"\n  RunKV:    avg compute={_avg(r_comp_durs):.2f}ms  "
            f"sum={sum(r_comp_durs):.1f}ms"
        )
    if t_comp_durs:
        rpt.row(
            f"  TightLLM: avg compute={_avg(t_comp_durs):.2f}ms  "
            f"sum={sum(t_comp_durs):.1f}ms"
        )
    if r_io_durs:
        rpt.row(
            f"  RunKV:    avg IO={_avg(r_io_durs):.2f}ms  sum={sum(r_io_durs):.1f}ms"
        )
    if t_io_durs:
        rpt.row(
            f"  TightLLM: avg IO={_avg(t_io_durs):.2f}ms  sum={sum(t_io_durs):.1f}ms"
        )

    # ── Section 3: Compute decomposition (assembly overhead vs pure forward)
    runkv_fs = runkv_ev["forward_start"]
    tightllm_fs = tightllm_ev["forward_start"]
    has_fwd_start = bool(runkv_fs or tightllm_fs)

    if has_fwd_start:
        rpt.h1("COMPUTE DECOMPOSITION  (assembly_overhead + pure_forward)")
        rpt.row("  assembly = forward_start − compute_start")
        rpt.row("    = cpu_fill H2D sync wait + tensor scatter/gather")
        rpt.row("  pure_fwd = compute_end − forward_start")
        rpt.row("    = layer(hidden_states) + index_select = attn + FFN")

        headers3 = [
            "layer",
            "r_asm",
            "r_fwd",
            "r_total",
            "t_asm",
            "t_fwd",
            "t_total",
            "asm_delta",
            "fwd_delta",
        ]
        rows3 = []
        for L in all_layers:
            r_asm = (
                _avg(runkv_fs[L]) - _avg(runkv_cs[L])
                if runkv_fs.get(L) and runkv_cs.get(L)
                else float("nan")
            )
            r_fwd = (
                _avg(runkv_ce[L]) - _avg(runkv_fs[L])
                if runkv_ce.get(L) and runkv_fs.get(L)
                else float("nan")
            )
            r_tot = (
                r_asm + r_fwd if (r_asm == r_asm and r_fwd == r_fwd) else float("nan")
            )
            t_asm = (
                _avg(tightllm_fs[L]) - _avg(tightllm_cs[L])
                if tightllm_fs.get(L) and tightllm_cs.get(L)
                else float("nan")
            )
            t_fwd = (
                _avg(tightllm_ce[L]) - _avg(tightllm_fs[L])
                if tightllm_ce.get(L) and tightllm_fs.get(L)
                else float("nan")
            )
            t_tot = (
                t_asm + t_fwd if (t_asm == t_asm and t_fwd == t_fwd) else float("nan")
            )
            rows3.append(
                [
                    f"L{L}",
                    f"{r_asm:.2f}" if r_asm == r_asm else "-",
                    f"{r_fwd:.2f}" if r_fwd == r_fwd else "-",
                    f"{r_tot:.2f}" if r_tot == r_tot else "-",
                    f"{t_asm:.2f}" if t_asm == t_asm else "-",
                    f"{t_fwd:.2f}" if t_fwd == t_fwd else "-",
                    f"{t_tot:.2f}" if t_tot == t_tot else "-",
                    f"{r_asm - t_asm:+.2f}"
                    if (r_asm == r_asm and t_asm == t_asm)
                    else "-",
                    f"{r_fwd - t_fwd:+.2f}"
                    if (r_fwd == r_fwd and t_fwd == t_fwd)
                    else "-",
                ]
            )
        rpt.table(headers3, rows3)

        # Summary
        r_asms = [
            _avg(runkv_fs[L]) - _avg(runkv_cs[L])
            for L in all_layers
            if runkv_fs.get(L) and runkv_cs.get(L)
        ]
        t_asms = [
            _avg(tightllm_fs[L]) - _avg(tightllm_cs[L])
            for L in all_layers
            if tightllm_fs.get(L) and tightllm_cs.get(L)
        ]
        r_fwds = [
            _avg(runkv_ce[L]) - _avg(runkv_fs[L])
            for L in all_layers
            if runkv_ce.get(L) and runkv_fs.get(L)
        ]
        t_fwds = [
            _avg(tightllm_ce[L]) - _avg(tightllm_fs[L])
            for L in all_layers
            if tightllm_ce.get(L) and tightllm_fs.get(L)
        ]
        if r_asms:
            rpt.row(
                f"\n  RunKV:    avg assembly={_avg(r_asms):.2f}ms  "
                f"avg pure_fwd={_avg(r_fwds):.2f}ms"
            )
        if t_asms:
            rpt.row(
                f"  TightLLM: avg assembly={_avg(t_asms):.2f}ms  "
                f"avg pure_fwd={_avg(t_fwds):.2f}ms"
            )
        if r_asms and t_asms:
            asm_delta = _avg(r_asms) - _avg(t_asms)
            fwd_delta = _avg(r_fwds) - _avg(t_fwds)
            rpt.row(
                f"  Delta:    assembly={asm_delta:+.2f}ms  pure_fwd={fwd_delta:+.2f}ms"
            )
            denom = abs(asm_delta) + abs(fwd_delta)
            if denom > 0:
                rpt.row(
                    f"  → assembly overhead explains "
                    f"{abs(asm_delta) / denom * 100:.0f}% "
                    f"of the compute_dur difference"
                )
            else:
                rpt.row("  → assembly and pure_fwd deltas are both zero")

    # ── Section 4: IO decomposition (KV DMA vs HS DMA) ─────────────────
    runkv_kv = runkv_ev["kv_ready"]
    runkv_hs = runkv_ev["hs_ready"]
    tightllm_kv = tightllm_ev["kv_ready"]
    tightllm_hs = tightllm_ev["hs_ready"]
    has_io_split = bool(runkv_kv or runkv_hs or tightllm_kv or tightllm_hs)

    if has_io_split:
        rpt.h1("IO DECOMPOSITION  (KV DMA vs Hidden-States DMA)")
        rpt.row("  kv_dur  = kv_ready − load_start    (KV block DMA on load_stream)")
        rpt.row(
            "  hs_dur  = hs_ready − load_start    (hidden states DMA on hs_h2d_stream)"
        )
        rpt.row("  io_dur  = max(kv_ready, hs_ready) − load_start")
        rpt.row("  bottleneck: whichever stream finishes last determines io_dur")

        headers4 = [
            "layer",
            "r_kv_dur",
            "r_hs_dur",
            "r_btlnk",
            "t_kv_dur",
            "t_hs_dur",
            "t_btlnk",
        ]
        rows4 = []
        for L in all_layers:
            r_kd = (
                _avg(runkv_kv[L]) - _avg(runkv_ls[L])
                if runkv_kv.get(L) and runkv_ls.get(L)
                else float("nan")
            )
            r_hd = (
                _avg(runkv_hs[L]) - _avg(runkv_ls[L])
                if runkv_hs.get(L) and runkv_ls.get(L)
                else float("nan")
            )
            r_bt = (
                "HS"
                if (r_hd == r_hd and r_kd == r_kd and r_hd > r_kd)
                else "KV"
                if (r_hd == r_hd and r_kd == r_kd)
                else "-"
            )
            t_kd = (
                _avg(tightllm_kv[L]) - _avg(tightllm_ls[L])
                if tightllm_kv.get(L) and tightllm_ls.get(L)
                else float("nan")
            )
            t_hd = (
                _avg(tightllm_hs[L]) - _avg(tightllm_ls[L])
                if tightllm_hs.get(L) and tightllm_ls.get(L)
                else float("nan")
            )
            t_bt = (
                "HS"
                if (t_hd == t_hd and t_kd == t_kd and t_hd > t_kd)
                else "KV"
                if (t_hd == t_hd and t_kd == t_kd)
                else "-"
            )
            rows4.append(
                [
                    f"L{L}",
                    f"{r_kd:.2f}" if r_kd == r_kd else "-",
                    f"{r_hd:.2f}" if r_hd == r_hd else "-",
                    r_bt,
                    f"{t_kd:.2f}" if t_kd == t_kd else "-",
                    f"{t_hd:.2f}" if t_hd == t_hd else "-",
                    t_bt,
                ]
            )
        rpt.table(headers4, rows4)


def collect_replay_stats(
    flat: list[dict],
    skip_steps: int = 1,
) -> dict[int, dict[str, list[float]]]:
    """Collect replay_token_count, num_actual_tokens, num_tokens per layer.

    Returns {layer_idx: {"replay": [...], "actual": [...], "sched": [...],
                         "cpu_fill": [...], "gpu_reuse": [...]}}.
    """
    result: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "replay": [],
            "actual": [],
            "sched": [],
            "cpu_fill": [],
            "gpu_reuse": [],
        }
    )
    for r in flat:
        if r.get("step", 0) < skip_steps:
            continue
        li = r.get("layer_idx")
        if li is None:
            continue
        rt = r.get("replay_token_count")
        at = r.get("num_actual_tokens")
        st = r.get("num_tokens")
        cf = r.get("cpu_fill_token_count")
        gr = r.get("gpu_reuse_token_count")
        if rt is not None:
            result[li]["replay"].append(float(rt))
        if at is not None:
            result[li]["actual"].append(float(at))
        if st is not None:
            result[li]["sched"].append(float(st))
        if cf is not None:
            result[li]["cpu_fill"].append(float(cf))
        if gr is not None:
            result[li]["gpu_reuse"].append(float(gr))
    return dict(result)


def analyze_replay_tokens(
    rpt: Report,
    runkv_rs: dict[int, dict[str, list[float]]],
    tightllm_rs: dict[int, dict[str, list[float]]],
    runkv_deltas: dict[int, list[float]],
    tightllm_deltas: dict[int, list[float]],
) -> None:
    rpt.h1("PER-LAYER REPLAY TOKEN STATISTICS")
    rpt.row("  replay:  tokens recomputed from CPU hidden states")
    rpt.row("  actual:  total tokens fed to layer(hidden_states) = sched + replay")
    rpt.row("  sched:   tokens scheduled by the scheduler (new + decode)")
    rpt.row("  ratio:   replay / actual")

    all_layers = sorted(set(runkv_rs) | set(tightllm_rs))
    if not all_layers:
        rpt.row("  (no data)")
        return

    headers = [
        "layer",
        "r_replay",
        "r_actual",
        "r_ratio",
        "t_replay",
        "t_actual",
        "t_ratio",
        "tok_delta%",
    ]
    rows = []
    for L in all_layers:
        rv = runkv_rs.get(L, {})
        tv = tightllm_rs.get(L, {})
        r_rep = _avg(rv.get("replay", []))
        r_act = _avg(rv.get("actual", []))
        t_rep = _avg(tv.get("replay", []))
        t_act = _avg(tv.get("actual", []))
        r_ratio = r_rep / r_act if r_act > 0 else 0
        t_ratio = t_rep / t_act if t_act > 0 else 0
        tok_pct = ((r_act - t_act) / t_act * 100) if t_act > 0 else float("nan")
        rows.append(
            [
                f"L{L}",
                f"{r_rep:.0f}" if rv.get("replay") else "-",
                f"{r_act:.0f}" if rv.get("actual") else "-",
                f"{r_ratio:.3f}" if rv.get("actual") else "-",
                f"{t_rep:.0f}" if tv.get("replay") else "-",
                f"{t_act:.0f}" if tv.get("actual") else "-",
                f"{t_ratio:.3f}" if tv.get("actual") else "-",
                f"{tok_pct:+.1f}%" if tok_pct == tok_pct else "-",
            ]
        )
    rpt.table(headers, rows)

    # Summary + efficiency analysis
    r_total_rep = sum(
        _avg(runkv_rs[L]["replay"]) for L in all_layers if runkv_rs.get(L)
    )
    t_total_rep = sum(
        _avg(tightllm_rs[L]["replay"]) for L in all_layers if tightllm_rs.get(L)
    )
    r_total_act = sum(
        _avg(runkv_rs[L]["actual"]) for L in all_layers if runkv_rs.get(L)
    )
    t_total_act = sum(
        _avg(tightllm_rs[L]["actual"]) for L in all_layers if tightllm_rs.get(L)
    )

    rpt.row(
        f"\n  Total replay tokens/step: RunKV={r_total_rep:.0f}  "
        f"TightLLM={t_total_rep:.0f}  "
        f"delta={r_total_rep - t_total_rep:+.0f} "
        f"({(r_total_rep - t_total_rep) / t_total_rep * 100:+.1f}%)"
        if t_total_rep > 0
        else ""
    )
    rpt.row(
        f"  Total actual tokens/step: RunKV={r_total_act:.0f}  "
        f"TightLLM={t_total_act:.0f}  "
        f"delta={r_total_act - t_total_act:+.0f} "
        f"({(r_total_act - t_total_act) / t_total_act * 100:+.1f}%)"
        if t_total_act > 0
        else ""
    )

    # CPU-fill vs GPU-reuse breakdown
    has_cf = any(runkv_rs.get(L, {}).get("cpu_fill") for L in all_layers)
    has_cf_t = any(tightllm_rs.get(L, {}).get("cpu_fill") for L in all_layers)
    if has_cf or has_cf_t:
        rpt.h2("Replay token breakdown: cpu_fill vs gpu_reuse")
        rpt.row("  cpu_fill:   hidden states loaded from CPU (H2D on hs_h2d_stream)")
        rpt.row("  gpu_reuse:  hidden states reused from previous layer's GPU output")
        rpt.row("  More cpu_fill → more H2D DMA → longer assembly overhead")
        cf_headers = [
            "layer",
            "r_cf",
            "r_gr",
            "r_cf%",
            "t_cf",
            "t_gr",
            "t_cf%",
            "cf_delta",
        ]
        cf_rows = []
        for L in all_layers:
            rv = runkv_rs.get(L, {})
            tv = tightllm_rs.get(L, {})
            r_cf = _avg(rv.get("cpu_fill", []))
            r_gr = _avg(rv.get("gpu_reuse", []))
            r_rep = _avg(rv.get("replay", []))
            t_cf = _avg(tv.get("cpu_fill", []))
            t_gr = _avg(tv.get("gpu_reuse", []))
            t_rep = _avg(tv.get("replay", []))
            r_pct = r_cf / r_rep * 100 if r_rep > 0 else 0
            t_pct = t_cf / t_rep * 100 if t_rep > 0 else 0
            cf_rows.append(
                [
                    f"L{L}",
                    f"{r_cf:.0f}" if rv.get("cpu_fill") else "-",
                    f"{r_gr:.0f}" if rv.get("gpu_reuse") else "-",
                    f"{r_pct:.1f}%" if rv.get("cpu_fill") else "-",
                    f"{t_cf:.0f}" if tv.get("cpu_fill") else "-",
                    f"{t_gr:.0f}" if tv.get("gpu_reuse") else "-",
                    f"{t_pct:.1f}%" if tv.get("cpu_fill") else "-",
                    f"{r_cf - t_cf:+.0f}"
                    if (rv.get("cpu_fill") and tv.get("cpu_fill"))
                    else "-",
                ]
            )
        rpt.table(cf_headers, cf_rows)

    # Efficiency: ms per 1000 actual tokens
    if runkv_deltas and tightllm_deltas:
        rpt.h2("Compute efficiency: layer_total_gpu / actual_tokens")
        rpt.row(
            "  ms per 1000 actual tokens — normalizes for different replay budgets."
        )
        rpt.row("  If RunKV and TightLLM have the same pure compute speed,")
        rpt.row("  this metric should be equal. Differences reveal prehook overhead.")
        eff_headers = ["layer", "r_ms/1kt", "t_ms/1kt", "delta", "overhead%"]
        eff_rows = []
        for L in all_layers:
            r_dt = _avg(runkv_deltas.get(L, []))
            t_dt = _avg(tightllm_deltas.get(L, []))
            r_act = _avg(runkv_rs.get(L, {}).get("actual", []))
            t_act = _avg(tightllm_rs.get(L, {}).get("actual", []))
            r_eff = r_dt / r_act * 1000 if r_act > 0 else float("nan")
            t_eff = t_dt / t_act * 1000 if t_act > 0 else float("nan")
            ovh = (
                (r_eff - t_eff) / t_eff * 100
                if (r_eff == r_eff and t_eff == t_eff and t_eff > 0)
                else float("nan")
            )
            eff_rows.append(
                [
                    f"L{L}",
                    f"{r_eff:.2f}" if r_eff == r_eff else "-",
                    f"{t_eff:.2f}" if t_eff == t_eff else "-",
                    f"{r_eff - t_eff:+.2f}"
                    if (r_eff == r_eff and t_eff == t_eff)
                    else "-",
                    f"{ovh:+.1f}%" if ovh == ovh else "-",
                ]
            )
        rpt.table(eff_headers, eff_rows)

        r_effs = [
            _avg(runkv_deltas.get(L, []))
            / _avg(runkv_rs.get(L, {}).get("actual", [1]))
            * 1000
            for L in all_layers
            if runkv_deltas.get(L) and runkv_rs.get(L, {}).get("actual")
        ]
        t_effs = [
            _avg(tightllm_deltas.get(L, []))
            / _avg(tightllm_rs.get(L, {}).get("actual", [1]))
            * 1000
            for L in all_layers
            if tightllm_deltas.get(L) and tightllm_rs.get(L, {}).get("actual")
        ]
        if r_effs and t_effs:
            r_avg_eff = _avg(r_effs)
            t_avg_eff = _avg(t_effs)
            rpt.row(
                f"\n  Avg efficiency: RunKV={r_avg_eff:.2f}  TightLLM={t_avg_eff:.2f}  "
                f"overhead={r_avg_eff - t_avg_eff:+.2f} ms/1kt "
                f"({(r_avg_eff - t_avg_eff) / t_avg_eff * 100:+.1f}%)"
            )
            rpt.row("  overhead > 0 means RunKV spends more wall-time per token.")
            rpt.row(
                "  This residual overhead ≈ prehook cost (feedback plan + IO scheduling)."
            )


def analyze_layer_compute_nvtx(
    rpt: Report,
    runkv_lc: dict[int, float],
    tightllm_lc: dict[int, float],
) -> None:
    """Analyze layer_compute NVTX durations (CPU-side wall time of each layer)."""
    rpt.h1("LAYER COMPUTE NVTX  (CPU-side wall time per layer, ms)")
    if not runkv_lc and not tightllm_lc:
        rpt.row("  NOT AVAILABLE — need runkv:layer_compute:L* NVTX ranges.")
        rpt.row("  Patch opt.py to emit these ranges, then re-profile with nsys.")
        return

    rpt.row("  Average CPU wall time of each layer's forward pass (from NVTX).")
    rpt.row("  Includes prehook CPU time + GPU kernel launches + sync waits.")

    all_layers = sorted(set(runkv_lc) | set(tightllm_lc))
    headers = ["layer", "runkv_ms", "tight_ms", "delta"]
    rows = []
    for L in all_layers:
        rv = runkv_lc.get(L, float("nan"))
        tv = tightllm_lc.get(L, float("nan"))
        delta = rv - tv if (rv == rv and tv == tv) else float("nan")
        rows.append(
            [
                f"L{L}",
                f"{rv:.2f}" if rv == rv else "-",
                f"{tv:.2f}" if tv == tv else "-",
                f"{delta:+.2f}" if delta == delta else "-",
            ]
        )
    rpt.table(headers, rows)

    r_vals = [v for v in runkv_lc.values() if v == v]
    t_vals = [v for v in tightllm_lc.values() if v == v]
    if r_vals:
        rpt.row(f"\n  RunKV    avg layer_compute: {_avg(r_vals):.2f}ms")
    if t_vals:
        rpt.row(f"  TightLLM avg layer_compute: {_avg(t_vals):.2f}ms")
    if r_vals and t_vals:
        rpt.row(f"  Delta: {_avg(r_vals) - _avg(t_vals):+.2f}ms")


def analyze_clean_compute(
    rpt: Report,
    runkv_kt: dict[int, dict[str, list[float]]],
    tightllm_kt: dict[int, dict[str, list[float]]],
    runkv_rs: dict[int, dict[str, list[float]]],
    tightllm_rs: dict[int, dict[str, list[float]]],
) -> None:
    """Analyze clean GPU compute time derived from nsys kernel-level data."""
    rpt.h1("CLEAN GPU COMPUTE  (from nsys kernel-level analysis)")
    if not runkv_kt and not tightllm_kt:
        rpt.row("  NOT AVAILABLE — pass --runkv-sqlite / --tightllm-sqlite with")
        rpt.row("  nsys exports that contain runkv:layer_compute NVTX + kernel data.")
        return

    rpt.row("  kernel_active: sum of CUDA kernel durations on compute stream")
    rpt.row("    = pure GPU work (attn + FFN + index_select), no bubbles")
    rpt.row("  gpu_bubble: gpu_span − kernel_active")
    rpt.row("    = GPU idle time from prehook CPU overhead + DMA wait")
    rpt.row("  prehook_cpu: CPU wall time of the attention pre-hook")

    all_layers = sorted(set(runkv_kt) | set(tightllm_kt))
    # ── Table 1: kernel_active, bubble, prehook ─────────────────────────
    headers = [
        "layer",
        "r_active",
        "r_bubble",
        "r_prehook",
        "t_active",
        "t_bubble",
        "t_prehook",
        "act_delta",
    ]
    rows = []
    for L in all_layers:
        ra = _avg(runkv_kt[L]["kernel_active"]) if runkv_kt.get(L) else float("nan")
        rb = _avg(runkv_kt[L]["gpu_bubble"]) if runkv_kt.get(L) else float("nan")
        rp = _avg(runkv_kt[L]["prehook_cpu"]) if runkv_kt.get(L) else float("nan")
        ta = (
            _avg(tightllm_kt[L]["kernel_active"])
            if tightllm_kt.get(L)
            else float("nan")
        )
        tb = _avg(tightllm_kt[L]["gpu_bubble"]) if tightllm_kt.get(L) else float("nan")
        tp = _avg(tightllm_kt[L]["prehook_cpu"]) if tightllm_kt.get(L) else float("nan")
        rows.append(
            [
                f"L{L}",
                f"{ra:.2f}" if ra == ra else "-",
                f"{rb:.2f}" if rb == rb else "-",
                f"{rp:.2f}" if rp == rp else "-",
                f"{ta:.2f}" if ta == ta else "-",
                f"{tb:.2f}" if tb == tb else "-",
                f"{tp:.2f}" if tp == tp else "-",
                f"{ra - ta:+.2f}" if (ra == ra and ta == ta) else "-",
            ]
        )
    rpt.table(headers, rows)

    # Summary
    r_acts = [_avg(runkv_kt[L]["kernel_active"]) for L in all_layers if runkv_kt.get(L)]
    t_acts = [
        _avg(tightllm_kt[L]["kernel_active"]) for L in all_layers if tightllm_kt.get(L)
    ]
    r_bubs = [_avg(runkv_kt[L]["gpu_bubble"]) for L in all_layers if runkv_kt.get(L)]
    t_bubs = [
        _avg(tightllm_kt[L]["gpu_bubble"]) for L in all_layers if tightllm_kt.get(L)
    ]
    r_phs = [_avg(runkv_kt[L]["prehook_cpu"]) for L in all_layers if runkv_kt.get(L)]
    t_phs = [
        _avg(tightllm_kt[L]["prehook_cpu"]) for L in all_layers if tightllm_kt.get(L)
    ]
    if r_acts:
        rpt.row(
            f"\n  RunKV:    avg kernel_active={_avg(r_acts):.2f}ms  "
            f"avg bubble={_avg(r_bubs):.2f}ms  avg prehook_cpu={_avg(r_phs):.2f}ms"
        )
    if t_acts:
        rpt.row(
            f"  TightLLM: avg kernel_active={_avg(t_acts):.2f}ms  "
            f"avg bubble={_avg(t_bubs):.2f}ms  avg prehook_cpu={_avg(t_phs):.2f}ms"
        )
    if r_acts and t_acts:
        act_delta = _avg(r_acts) - _avg(t_acts)
        bub_delta = _avg(r_bubs) - _avg(t_bubs)
        rpt.row(
            f"  Delta:    kernel_active={act_delta:+.2f}ms  bubble={bub_delta:+.2f}ms"
        )

    # ── Table 2: clean per-token efficiency ─────────────────────────────
    has_replay = bool(runkv_rs or tightllm_rs)
    if has_replay:
        rpt.h2("Clean per-token efficiency: kernel_active / actual_tokens")
        rpt.row("  ms per 1000 actual tokens — using kernel-level GPU time.")
        rpt.row("  Excludes all prehook bubble. Should be equal if both methods")
        rpt.row("  have the same kernel speed per token.")
        eff_headers = ["layer", "r_ms/1kt", "t_ms/1kt", "delta", "overhead%"]
        eff_rows = []
        for L in all_layers:
            r_act_t = _avg(runkv_rs.get(L, {}).get("actual", []))
            t_act_t = _avg(tightllm_rs.get(L, {}).get("actual", []))
            ra = _avg(runkv_kt[L]["kernel_active"]) if runkv_kt.get(L) else float("nan")
            ta = (
                _avg(tightllm_kt[L]["kernel_active"])
                if tightllm_kt.get(L)
                else float("nan")
            )
            r_eff = ra / r_act_t * 1000 if (ra == ra and r_act_t > 0) else float("nan")
            t_eff = ta / t_act_t * 1000 if (ta == ta and t_act_t > 0) else float("nan")
            ovh = (
                (r_eff - t_eff) / t_eff * 100
                if (r_eff == r_eff and t_eff == t_eff and t_eff > 0)
                else float("nan")
            )
            eff_rows.append(
                [
                    f"L{L}",
                    f"{r_eff:.2f}" if r_eff == r_eff else "-",
                    f"{t_eff:.2f}" if t_eff == t_eff else "-",
                    f"{r_eff - t_eff:+.2f}"
                    if (r_eff == r_eff and t_eff == t_eff)
                    else "-",
                    f"{ovh:+.1f}%" if ovh == ovh else "-",
                ]
            )
        rpt.table(eff_headers, eff_rows)

        r_effs = [
            _avg(runkv_kt[L]["kernel_active"]) / _avg(runkv_rs[L]["actual"]) * 1000
            for L in all_layers
            if runkv_kt.get(L)
            and runkv_rs.get(L, {}).get("actual")
            and _avg(runkv_rs[L]["actual"]) > 0
        ]
        t_effs = [
            _avg(tightllm_kt[L]["kernel_active"])
            / _avg(tightllm_rs[L]["actual"])
            * 1000
            for L in all_layers
            if tightllm_kt.get(L)
            and tightllm_rs.get(L, {}).get("actual")
            and _avg(tightllm_rs[L]["actual"]) > 0
        ]
        if r_effs and t_effs:
            r_avg = _avg(r_effs)
            t_avg = _avg(t_effs)
            rpt.row(
                f"\n  Clean avg efficiency: RunKV={r_avg:.2f}  TightLLM={t_avg:.2f}  "
                f"delta={r_avg - t_avg:+.2f} ms/1kt "
                f"({(r_avg - t_avg) / t_avg * 100:+.1f}%)"
            )
            rpt.row(
                "  This should be close to the token-count difference "
                "(~10%), confirming equal kernel speed."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════


def _nan(v: float) -> float:
    return v if v == v else 0.0


def plot_imbalance(
    runkv_imb: dict[int, list[float]],
    tightllm_imb: dict[int, list[float]],
    out_dir: Path,
) -> None:
    """Plot per-layer imbalance comparison."""
    if not HAS_MPL:
        return
    all_layers = sorted(set(runkv_imb) | set(tightllm_imb))
    if not all_layers:
        return

    x = list(range(len(all_layers)))
    labels = [f"L{l}" for l in all_layers]

    r_means = [_avg(runkv_imb.get(l, [])) for l in all_layers]
    t_means = [_avg(tightllm_imb.get(l, [])) for l in all_layers]

    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.35
    if any(v == v for v in r_means):
        ax.bar(
            [i - w / 2 for i in x],
            [_nan(v) for v in r_means],
            w,
            color="#4C72B0",
            alpha=0.8,
            label="RunKV",
        )
    if any(v == v for v in t_means):
        ax.bar(
            [i + w / 2 for i in x],
            [_nan(v) for v in t_means],
            w,
            color="#DD8452",
            alpha=0.8,
            label="TightLLM",
        )

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("imbalance (ms)")
    ax.set_title(
        "Per-layer imbalance: compute_end[L] − load_ready[L+1]\n"
        "negative = IO ahead (slack)  |  positive = GPU stall (bubble)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "layer_imbalance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_compute_vs_io(
    runkv_ev: dict[str, dict[int, list[float]]],
    tightllm_ev: dict[str, dict[int, list[float]]],
    out_dir: Path,
) -> None:
    """Plot per-layer compute and IO durations and timelines."""
    if not HAS_MPL:
        return
    runkv_cs, runkv_ce = runkv_ev["compute_start"], runkv_ev["compute_end"]
    runkv_ls, runkv_lr = runkv_ev["load_start"], runkv_ev["load_ready"]
    tightllm_cs, tightllm_ce = tightllm_ev["compute_start"], tightllm_ev["compute_end"]
    tightllm_ls, tightllm_lr = tightllm_ev["load_start"], tightllm_ev["load_ready"]

    all_layers = sorted(
        set(runkv_ce) | set(runkv_lr) | set(tightllm_ce) | set(tightllm_lr)
    )
    if not all_layers:
        return

    x = list(range(len(all_layers)))
    labels = [f"L{l}" for l in all_layers]

    has_start = bool(runkv_cs or tightllm_cs)

    # ── Figure: per-layer compute and IO durations ─────────────────────
    if has_start:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top: compute duration
        ax = axes[0]
        for cs_d, ce_d, style, color, lbl in [
            (runkv_cs, runkv_ce, "o-", "#4C72B0", "RunKV"),
            (tightllm_cs, tightllm_ce, "s-", "#DD8452", "TightLLM"),
        ]:
            vals = [
                _avg(ce_d.get(l, [])) - _avg(cs_d.get(l, []))
                if (ce_d.get(l) and cs_d.get(l))
                else float("nan")
                for l in all_layers
            ]
            if any(v == v for v in vals):
                ax.plot(x, vals, style, color=color, lw=1.5, ms=4, label=lbl)
        ax.set_ylabel("ms")
        ax.set_title(
            "Per-layer compute duration (compute_end − compute_start)\n"
            "Pure GPU time: attn + FFN, no prehook overhead"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom: IO duration
        ax = axes[1]
        for ls_d, lr_d, style, color, lbl in [
            (runkv_ls, runkv_lr, "o-", "#4C72B0", "RunKV"),
            (tightllm_ls, tightllm_lr, "s-", "#DD8452", "TightLLM"),
        ]:
            vals = [
                _avg(lr_d.get(l, [])) - _avg(ls_d.get(l, []))
                if (lr_d.get(l) and ls_d.get(l))
                else float("nan")
                for l in all_layers
            ]
            if any(v == v for v in vals):
                ax.plot(x, vals, style, color=color, lw=1.5, ms=4, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("ms")
        ax.set_title(
            "Per-layer IO duration (load_ready − load_start)\n"
            "Actual H2D DMA transfer time on load stream"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = out_dir / "layer_compute_io_durations.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] {out}")

    # ── Figure: absolute timelines + gap ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    for data, style, label in [
        (runkv_ce, "o-", "RunKV compute_end"),
        (runkv_lr, "o--", "RunKV load_ready"),
        (tightllm_ce, "s-", "TightLLM compute_end"),
        (tightllm_lr, "s--", "TightLLM load_ready"),
    ]:
        vals = [_avg(data.get(l, [])) for l in all_layers]
        if any(v == v for v in vals):
            color = "#4C72B0" if "RunKV" in label else "#DD8452"
            ax.plot(
                x,
                vals,
                style,
                color=color,
                lw=1.5,
                ms=4,
                label=label,
                alpha=0.6 if "load_ready" in label else 1.0,
            )
    if has_start:
        for data, style, label in [
            (runkv_cs, "o:", "RunKV compute_start"),
            (tightllm_cs, "s:", "TightLLM compute_start"),
        ]:
            vals = [_avg(data.get(l, [])) for l in all_layers]
            if any(v == v for v in vals):
                color = "#4C72B0" if "RunKV" in label else "#DD8452"
                ax.plot(x, vals, style, color=color, lw=1, ms=3, label=label, alpha=0.4)
    ax.set_ylabel("ms from step anchor")
    ax.set_title("Per-layer event timelines (absolute)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    r_gaps = [
        _avg(runkv_ce.get(l, [])) - _avg(runkv_lr.get(l, []))
        if (runkv_ce.get(l) and runkv_lr.get(l))
        else float("nan")
        for l in all_layers
    ]
    t_gaps = [
        _avg(tightllm_ce.get(l, [])) - _avg(tightllm_lr.get(l, []))
        if (tightllm_ce.get(l) and tightllm_lr.get(l))
        else float("nan")
        for l in all_layers
    ]
    if any(v == v for v in r_gaps):
        ax.plot(x, r_gaps, "o-", color="#4C72B0", lw=1.5, ms=4, label="RunKV gap")
    if any(v == v for v in t_gaps):
        ax.plot(x, t_gaps, "s-", color="#DD8452", lw=1.5, ms=4, label="TightLLM gap")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("gap (ms)")
    ax.set_title(
        "compute_end[L] − load_ready[L]: positive = IO ahead, negative = GPU stall"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "layer_compute_vs_io.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_replay_tokens(
    runkv_rs: dict[int, dict[str, list[float]]],
    tightllm_rs: dict[int, dict[str, list[float]]],
    runkv_deltas: dict[int, list[float]],
    tightllm_deltas: dict[int, list[float]],
    out_dir: Path,
) -> None:
    """Plot per-layer replay token counts and compute efficiency."""
    if not HAS_MPL:
        return
    all_layers = sorted(set(runkv_rs) | set(tightllm_rs))
    if not all_layers:
        return

    x = list(range(len(all_layers)))
    labels = [f"L{l}" for l in all_layers]

    # ── Figure 1: replay token counts and actual tokens ───────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    r_replay = [_avg(runkv_rs.get(l, {}).get("replay", [])) for l in all_layers]
    t_replay = [_avg(tightllm_rs.get(l, {}).get("replay", [])) for l in all_layers]
    w = 0.35
    if any(v == v for v in r_replay):
        ax.bar(
            [i - w / 2 for i in x],
            [_nan(v) for v in r_replay],
            w,
            color="#4C72B0",
            alpha=0.8,
            label="RunKV replay",
        )
    if any(v == v for v in t_replay):
        ax.bar(
            [i + w / 2 for i in x],
            [_nan(v) for v in t_replay],
            w,
            color="#DD8452",
            alpha=0.8,
            label="TightLLM replay",
        )
    ax.set_ylabel("tokens")
    ax.set_title(
        "Per-layer replay token count\n(tokens recomputed from CPU hidden states)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    r_actual = [_avg(runkv_rs.get(l, {}).get("actual", [])) for l in all_layers]
    t_actual = [_avg(tightllm_rs.get(l, {}).get("actual", [])) for l in all_layers]
    if any(v == v for v in r_actual):
        ax.plot(x, r_actual, "o-", color="#4C72B0", lw=1.5, ms=4, label="RunKV actual")
    if any(v == v for v in t_actual):
        ax.plot(
            x, t_actual, "s-", color="#DD8452", lw=1.5, ms=4, label="TightLLM actual"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("tokens")
    ax.set_title("Per-layer actual token count (sched + replay)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "layer_replay_tokens.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")

    # ── Figure 2: compute efficiency (ms per 1000 actual tokens) ──────────
    if runkv_deltas and tightllm_deltas:
        r_eff = [
            _avg(runkv_deltas.get(l, []))
            / _avg(runkv_rs.get(l, {}).get("actual", [1]))
            * 1000
            if (runkv_deltas.get(l) and runkv_rs.get(l, {}).get("actual"))
            else float("nan")
            for l in all_layers
        ]
        t_eff = [
            _avg(tightllm_deltas.get(l, []))
            / _avg(tightllm_rs.get(l, {}).get("actual", [1]))
            * 1000
            if (tightllm_deltas.get(l) and tightllm_rs.get(l, {}).get("actual"))
            else float("nan")
            for l in all_layers
        ]
        if any(v == v for v in r_eff) or any(v == v for v in t_eff):
            fig, ax = plt.subplots(figsize=(14, 4))
            if any(v == v for v in r_eff):
                ax.plot(x, r_eff, "o-", color="#4C72B0", lw=1.5, ms=4, label="RunKV")
            if any(v == v for v in t_eff):
                ax.plot(x, t_eff, "s-", color="#DD8452", lw=1.5, ms=4, label="TightLLM")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_ylabel("ms / 1000 tokens")
            ax.set_title(
                "Compute efficiency: layer_total_gpu / actual_tokens\n"
                "Equal → same speed; RunKV higher → prehook overhead dominant"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = out_dir / "layer_compute_efficiency.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  [plot] {out}")


def plot_clean_compute(
    runkv_kt: dict[int, dict[str, list[float]]],
    tightllm_kt: dict[int, dict[str, list[float]]],
    out_dir: Path,
) -> None:
    """Plot clean GPU compute timing: kernel_active, gpu_bubble, prehook_cpu."""
    if not HAS_MPL:
        return
    all_layers = sorted(set(runkv_kt) | set(tightllm_kt))
    if not all_layers:
        return

    x = list(range(len(all_layers)))
    labels = [f"L{l}" for l in all_layers]

    # ── Figure 1: kernel_active comparison (bar chart) ────────────────────
    r_active = [
        _avg(runkv_kt[l]["kernel_active"]) if runkv_kt.get(l) else 0 for l in all_layers
    ]
    t_active = [
        _avg(tightllm_kt[l]["kernel_active"]) if tightllm_kt.get(l) else 0
        for l in all_layers
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    w = 0.35

    ax = axes[0]
    if any(v > 0 for v in r_active):
        ax.bar(
            [i - w / 2 for i in x],
            r_active,
            w,
            color="#4C72B0",
            alpha=0.8,
            label="RunKV",
        )
    if any(v > 0 for v in t_active):
        ax.bar(
            [i + w / 2 for i in x],
            t_active,
            w,
            color="#DD8452",
            alpha=0.8,
            label="TightLLM",
        )
    ax.set_ylabel("ms")
    ax.set_title(
        "Clean GPU compute: kernel_active per layer\n"
        "(sum of CUDA kernel durations on compute stream — no bubbles)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── Bottom: gpu_bubble comparison ─────────────────────────────────────
    r_bubble = [
        _avg(runkv_kt[l]["gpu_bubble"]) if runkv_kt.get(l) else 0 for l in all_layers
    ]
    t_bubble = [
        _avg(tightllm_kt[l]["gpu_bubble"]) if tightllm_kt.get(l) else 0
        for l in all_layers
    ]

    ax = axes[1]
    if any(v > 0 for v in r_bubble):
        ax.bar(
            [i - w / 2 for i in x],
            r_bubble,
            w,
            color="#4C72B0",
            alpha=0.8,
            label="RunKV",
        )
    if any(v > 0 for v in t_bubble):
        ax.bar(
            [i + w / 2 for i in x],
            t_bubble,
            w,
            color="#DD8452",
            alpha=0.8,
            label="TightLLM",
        )
    ax.set_xticks(x[::2])
    ax.set_xticklabels(labels[::2], rotation=90, fontsize=7)
    ax.set_ylabel("ms")
    ax.set_title(
        "GPU bubble per layer: gpu_span − kernel_active\n"
        "(idle time from prehook CPU overhead + DMA wait)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = out_dir / "layer_clean_compute.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")

    # ── Figure 2: stacked bar — kernel_active + bubble = gpu_span ─────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, label, kt, color_act, color_bub in [
        (axes[0], "RunKV", runkv_kt, "#4C72B0", "#a6bddb"),
        (axes[1], "TightLLM", tightllm_kt, "#DD8452", "#fdbe85"),
    ]:
        if not kt:
            ax.set_title(f"{label}: no kernel timing data")
            continue
        act_vals = [
            _avg(kt[l]["kernel_active"]) if kt.get(l) else 0 for l in all_layers
        ]
        bub_vals = [_avg(kt[l]["gpu_bubble"]) if kt.get(l) else 0 for l in all_layers]
        ph_vals = [_avg(kt[l]["prehook_cpu"]) if kt.get(l) else 0 for l in all_layers]

        ax.bar(x, act_vals, color=color_act, alpha=0.85, label="kernel_active")
        ax.bar(
            x, bub_vals, bottom=act_vals, color=color_bub, alpha=0.7, label="gpu_bubble"
        )
        ax.plot(x, ph_vals, "k--", lw=1.2, ms=3, label="prehook_cpu")

        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=90, fontsize=7)
        ax.set_title(f"{label}: kernel_active + bubble = gpu_span")
        ax.set_ylabel("ms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    out = out_dir / "layer_clean_compute_stacked.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_per_step_layer_lines(
    runkv_psl: dict[str, dict[int, dict[int, float]]],
    tightllm_psl: dict[str, dict[int, dict[int, float]]],
    out_dir: Path,
) -> None:
    """Plot per-step × per-layer line charts for compute, IO, and imbalance.

    For each metric, produces one figure per method (RunKV / TightLLM) with
    x-axis = step, y-axis = time (ms), and one line per layer.  Also produces
    a combined heatmap view.
    """
    if not HAS_MPL:
        return

    metrics_info = [
        ("compute_dur", "Compute duration (ms)", "compute_end[L]−compute_end[L−1]"),
        ("io_dur", "IO duration (ms)", "load_ready[L]−load_start[L]"),
        ("imbalance", "Imbalance (ms)", "compute_end[L]−load_ready[L+1]"),
    ]

    for metric, ylabel, description in metrics_info:
        for label, psl, cmap_base in [
            ("RunKV", runkv_psl, "Blues"),
            ("TightLLM", tightllm_psl, "Oranges"),
        ]:
            data = psl.get(metric, {})
            if not data:
                continue

            steps = sorted(data.keys())
            all_layers = sorted({l for step_data in data.values() for l in step_data})
            if not steps or not all_layers:
                continue

            # Build 2D array: rows = layers, cols = steps
            import numpy as _np

            mat = _np.full((len(all_layers), len(steps)), _np.nan)
            for si, step in enumerate(steps):
                for li, layer in enumerate(all_layers):
                    v = data.get(step, {}).get(layer)
                    if v is not None:
                        mat[li, si] = v

            # ── Line chart: one line per layer ─────────────────────────
            fig, ax = plt.subplots(figsize=(16, 6))
            cmap = plt.get_cmap("tab20" if len(all_layers) <= 20 else "viridis")
            for li, layer in enumerate(all_layers):
                color = cmap(li / max(len(all_layers) - 1, 1))
                vals = mat[li, :]
                # Only plot if we have some non-nan values
                valid = ~_np.isnan(vals)
                if valid.any():
                    ax.plot(
                        _np.array(steps)[valid],
                        vals[valid],
                        color=color,
                        lw=0.8,
                        alpha=0.7,
                        label=f"L{layer}" if len(all_layers) <= 32 else None,
                    )

            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: {description} per step per layer")
            if len(all_layers) <= 32:
                ax.legend(
                    fontsize=6,
                    ncol=min(4, (len(all_layers) + 7) // 8),
                    loc="upper right",
                )
            ax.grid(True, alpha=0.3)
            if metric == "imbalance":
                ax.axhline(0, color="k", lw=0.8, ls="--")
            fig.tight_layout()
            tag = label.lower().replace(" ", "_")
            out = out_dir / f"step_layer_{metric}_{tag}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  [plot] {out}")

            # ── Heatmap: x=step, y=layer ──────────────────────────────
            fig, ax = plt.subplots(figsize=(16, max(4, len(all_layers) * 0.25)))
            if metric == "imbalance":
                # Diverging colormap centered at 0
                abs_max = _np.nanmax(_np.abs(mat)) if not _np.all(_np.isnan(mat)) else 1
                im = ax.imshow(
                    mat,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-abs_max,
                    vmax=abs_max,
                    interpolation="nearest",
                )
            else:
                im = ax.imshow(
                    mat,
                    aspect="auto",
                    cmap=cmap_base,
                    interpolation="nearest",
                )
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(ylabel, fontsize=9)

            # Tick labels
            n_steps = len(steps)
            if n_steps > 40:
                step_ticks = list(range(0, n_steps, max(1, n_steps // 20)))
            else:
                step_ticks = list(range(n_steps))
            ax.set_xticks(step_ticks)
            ax.set_xticklabels(
                [str(steps[i]) for i in step_ticks], fontsize=7, rotation=45
            )
            ax.set_yticks(range(len(all_layers)))
            ax.set_yticklabels([f"L{l}" for l in all_layers], fontsize=7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Layer")
            ax.set_title(f"{label}: {description} heatmap (step × layer)")
            fig.tight_layout()
            out = out_dir / f"step_layer_{metric}_{tag}_heatmap.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  [plot] {out}")

        # ── Combined overlay: RunKV vs TightLLM mean ± std per step ──
        r_data = runkv_psl.get(metric, {})
        t_data = tightllm_psl.get(metric, {})
        if not r_data and not t_data:
            continue

        fig, ax = plt.subplots(figsize=(16, 5))
        for data, lbl, color in [
            (r_data, "RunKV", "#4C72B0"),
            (t_data, "TightLLM", "#DD8452"),
        ]:
            if not data:
                continue
            import numpy as _np

            steps_s = sorted(data.keys())
            means = []
            stds = []
            for step in steps_s:
                vals = [v for v in data[step].values() if v == v]
                if vals:
                    means.append(sum(vals) / len(vals))
                    stds.append(
                        (sum((x - means[-1]) ** 2 for x in vals) / len(vals)) ** 0.5
                    )
                else:
                    means.append(float("nan"))
                    stds.append(0)
            means_a = _np.array(means)
            stds_a = _np.array(stds)
            valid = ~_np.isnan(means_a)
            steps_a = _np.array(steps_s)
            ax.plot(
                steps_a[valid],
                means_a[valid],
                color=color,
                lw=1.5,
                label=f"{lbl} mean",
            )
            ax.fill_between(
                steps_a[valid],
                (means_a - stds_a)[valid],
                (means_a + stds_a)[valid],
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{description}: cross-layer mean ± std per step")
        if metric == "imbalance":
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"step_layer_{metric}_combined.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] {out}")


def plot_layer_timeline(
    runkv_deltas: dict[int, list[float]],
    tightllm_deltas: dict[int, list[float]],
    runkv_dma: dict[int, list[float]],
    tightllm_dma: dict[int, list[float]],
    runkv_prehook: dict[str, dict[int, float]],
    tightllm_prehook: dict[str, dict[int, float]],
    runkv_lc: dict[int, float],
    tightllm_lc: dict[int, float],
    out_dir: Path,
) -> None:
    if not HAS_MPL:
        return

    all_layers = sorted(
        set(runkv_deltas) | set(tightllm_deltas) | set(runkv_dma) | set(tightllm_dma)
    )
    if not all_layers:
        return

    x = list(range(len(all_layers)))
    labels = [f"L{l}" for l in all_layers]

    # ── Figure 1: per-layer total GPU time ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    r_totals = [_avg(runkv_deltas.get(l, [])) for l in all_layers]
    t_totals = [_avg(tightllm_deltas.get(l, [])) for l in all_layers]
    if any(v == v for v in r_totals):
        ax.plot(x, r_totals, "o-", color="#4C72B0", lw=1.5, ms=4, label="RunKV total")
    if any(v == v for v in t_totals):
        ax.plot(
            x, t_totals, "s-", color="#DD8452", lw=1.5, ms=4, label="TightLLM total"
        )
    ax.set_xticks(x[::2])
    ax.set_xticklabels(labels[::2], rotation=90, fontsize=7)
    ax.set_ylabel("ms")
    ax.set_title(
        "Per-layer GPU time: compute_end[L] − compute_end[L−1]\n"
        "(includes prehook CPU stall + H2D wait + attn + FFN)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "layer_total_gpu_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")

    # ── Figure 2: per-layer H2D DMA vs prehook breakdown stacked bar ────────
    segments = [
        s
        for s, _ in _PREHOOK_NVTX_PREFIXES
        if runkv_prehook.get(s) or tightllm_prehook.get(s)
    ]
    colors = {
        "h2d_sync": "#e6b8a2",
        "imbalance": "#b5cde0",
        "build_plan": "#c44e52",
        "build_meta": "#8172b2",
        "skip_ids": "#55a868",
        "schedule_io": "#937860",
    }
    dma_color_r = "#4c72b0"
    dma_color_t = "#dd8452"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, label, deltas, dma, prehook in [
        (axes[0], "RunKV", runkv_deltas, runkv_dma, runkv_prehook),
        (axes[1], "TightLLM", tightllm_deltas, tightllm_dma, tightllm_prehook),
    ]:
        if not deltas and not dma:
            ax.set_title(f"{label}: no data")
            continue

        dma_vals = [_avg(dma.get(l, [])) for l in all_layers]
        bar_bottom = [0.0] * len(all_layers)

        # DMA bar
        ax.bar(
            x,
            dma_vals,
            bottom=bar_bottom,
            color=dma_color_r if "RunKV" in label else dma_color_t,
            alpha=0.7,
            label="H2D DMA",
        )
        bar_bottom = [b + d for b, d in zip(bar_bottom, dma_vals)]

        # Prehook segments
        for seg in segments:
            seg_vals = [prehook.get(seg, {}).get(l, 0) for l in all_layers]
            ax.bar(
                x,
                seg_vals,
                bottom=bar_bottom,
                color=colors.get(seg, "#aaa"),
                alpha=0.85,
                label=seg,
            )
            bar_bottom = [b + v for b, v in zip(bar_bottom, seg_vals)]

        # If layer_compute NVTX available, add a line for it
        lc = runkv_lc if "RunKV" in label else tightllm_lc
        if lc:
            lc_vals = [lc.get(l, float("nan")) for l in all_layers]
            ax2 = ax.twinx()
            ax2.plot(x, lc_vals, "k--", lw=1.5, ms=3, label="layer_compute NVTX")
            ax2.set_ylabel("layer_compute NVTX (ms)", fontsize=8)
            ax2.legend(loc="upper right", fontsize=7)

        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=90, fontsize=7)
        ax.set_title(f"{label}: per-layer prehook + H2D stacked")
        ax.set_ylabel("ms")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    out = out_dir / "layer_prehook_dma_stacked.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")

    # ── Figure 3: GPU bubble estimate ───────────────────────────────────────
    # Estimate: if prehook_sum[L] > FFN[L-1], GPU stalls for the difference.
    # We don't have FFN directly, but we can estimate:
    # If layer_compute NVTX is available: FFN ≈ layer_compute - attn
    # Otherwise: use total_delta - prehook_sum as proxy for (stall + attn + FFN)
    if runkv_prehook.get("build_plan") and tightllm_prehook.get("build_plan"):
        fig, ax = plt.subplots(figsize=(14, 4))
        for label, deltas, prehook, color in [
            ("RunKV", runkv_deltas, runkv_prehook, "#4C72B0"),
            ("TightLLM", tightllm_deltas, tightllm_prehook, "#DD8452"),
        ]:
            prehook_segs = [s for s, _ in _PREHOOK_NVTX_PREFIXES if prehook.get(s)]
            prehook_sum = [
                sum(prehook.get(seg, {}).get(l, 0) for seg in prehook_segs)
                for l in all_layers
            ]
            total = [_avg(deltas.get(l, [])) for l in all_layers]
            # residual = total - prehook = stall_time + actual compute (attn+FFN)
            residual = [t - p for t, p in zip(total, prehook_sum)]
            ax.plot(
                x,
                prehook_sum,
                "--",
                color=color,
                lw=1.2,
                alpha=0.7,
                label=f"{label} prehook sum",
            )
            ax.plot(
                x,
                residual,
                "-",
                color=color,
                lw=2,
                label=f"{label} residual (stall+compute)",
            )

        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=90, fontsize=7)
        ax.set_ylabel("ms")
        ax.set_title(
            "Prehook sum vs residual per layer\n"
            "residual = layer_total − prehook = GPU stall + attn + FFN"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / "layer_prehook_vs_residual.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def _glob_expand(paths: list[str]) -> list[str]:
    import glob as _g

    out = []
    for p in paths:
        e = _g.glob(p)
        out.extend(e if e else [p])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer compute vs IO timing analysis")
    ap.add_argument("--runkv-mfu", nargs="*", default=[])
    ap.add_argument("--tightllm-mfu", nargs="*", default=[])
    ap.add_argument("--runkv-sqlite", default="")
    ap.add_argument("--tightllm-sqlite", default="")
    ap.add_argument("--output-dir", default="exp_results/analysis/per_layer")
    ap.add_argument("--skip-warmup-steps", type=int, default=1)
    ap.add_argument(
        "--compute-stream",
        type=int,
        default=7,
        help="CUDA stream ID for the compute stream (default: 7)",
    )
    ap.add_argument(
        "--dma-tol-ms",
        type=float,
        default=2.0,
        help="Tolerance (ms) for MEMCPY end-time matching (default 2.0)",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    rpt = Report(out_dir)
    rpt.h1("Per-Layer Compute vs IO Timing Analysis")
    skip = args.skip_warmup_steps

    print("\nLoading data...")
    runkv_flat = load_mfu_flat(_glob_expand(args.runkv_mfu))
    tightllm_flat = load_mfu_flat(_glob_expand(args.tightllm_mfu))
    runkv_flat = [r for r in runkv_flat if r.get("step", 0) >= skip]
    tightllm_flat = [r for r in tightllm_flat if r.get("step", 0) >= skip]

    runkv_nvtx = load_nvtx(args.runkv_sqlite) if args.runkv_sqlite else []
    tightllm_nvtx = load_nvtx(args.tightllm_sqlite) if args.tightllm_sqlite else []

    # Detect H2D load stream
    r_stream = detect_h2d_load_stream(args.runkv_sqlite) if args.runkv_sqlite else None
    t_stream = (
        detect_h2d_load_stream(args.tightllm_sqlite) if args.tightllm_sqlite else None
    )
    runkv_memcpy = (
        load_memcpy_h2d(args.runkv_sqlite, r_stream) if args.runkv_sqlite else []
    )
    tightllm_memcpy = (
        load_memcpy_h2d(args.tightllm_sqlite, t_stream) if args.tightllm_sqlite else []
    )

    # ── Extract clean kernel timing from sqlite ──────────────────────────────
    runkv_kt = (
        extract_kernel_timing_from_sqlite(args.runkv_sqlite, args.compute_stream)
        if args.runkv_sqlite
        else {}
    )
    tightllm_kt = (
        extract_kernel_timing_from_sqlite(args.tightllm_sqlite, args.compute_stream)
        if args.tightllm_sqlite
        else {}
    )

    print(f"  runkv    flat records : {len(runkv_flat)}")
    print(f"  tightllm flat records : {len(tightllm_flat)}")
    print(f"  runkv    nvtx ranges  : {len(runkv_nvtx)}")
    print(f"  tightllm nvtx ranges  : {len(tightllm_nvtx)}")
    print(f"  runkv    H2D memcpy   : {len(runkv_memcpy)} (stream={r_stream})")
    print(f"  tightllm H2D memcpy   : {len(tightllm_memcpy)} (stream={t_stream})")
    print(f"  runkv    kernel timing layers : {len(runkv_kt)}")
    print(f"  tightllm kernel timing layers : {len(tightllm_kt)}")

    # ── Collect imbalance from JSONL ─────────────────────────────────────────
    runkv_imb = collect_imbalance(runkv_flat, skip)
    tightllm_imb = collect_imbalance(tightllm_flat, skip)
    print(f"  runkv    imbalance layers : {len(runkv_imb)}")
    print(f"  tightllm imbalance layers : {len(tightllm_imb)}")

    # ── Collect all timing events from JSONL ─────────────────────────────────
    runkv_ev = collect_compute_and_io_events(runkv_flat, skip)
    tightllm_ev = collect_compute_and_io_events(tightllm_flat, skip)
    print(f"  runkv    compute_start layers : {len(runkv_ev['compute_start'])}")
    print(f"  runkv    load_start layers    : {len(runkv_ev['load_start'])}")

    # ── Compute deltas from GPU events ────────────────────────────────────────
    runkv_deltas = compute_layer_deltas(runkv_flat, skip)
    tightllm_deltas = compute_layer_deltas(tightllm_flat, skip)

    # ── Correlate DMA durations ───────────────────────────────────────────────
    tol_ns = args.dma_tol_ms * 1e6
    runkv_dma = (
        correlate_dma_to_layers(runkv_flat, runkv_nvtx, runkv_memcpy, skip, tol_ns)
        if runkv_nvtx and runkv_memcpy
        else {}
    )
    tightllm_dma = (
        correlate_dma_to_layers(
            tightllm_flat, tightllm_nvtx, tightllm_memcpy, skip, tol_ns
        )
        if tightllm_nvtx and tightllm_memcpy
        else {}
    )
    print(f"  runkv    DMA correlated layers : {len(runkv_dma)}")
    print(f"  tightllm DMA correlated layers : {len(tightllm_dma)}")

    # ── NVTX prehook and layer_compute ───────────────────────────────────────
    runkv_prehook = extract_prehook_by_layer(runkv_nvtx)
    tightllm_prehook = extract_prehook_by_layer(tightllm_nvtx)
    runkv_lc = extract_layer_compute_nvtx(runkv_nvtx)
    tightllm_lc = extract_layer_compute_nvtx(tightllm_nvtx)

    have_schedule_io_r = bool(runkv_prehook.get("schedule_io"))
    have_schedule_io_t = bool(tightllm_prehook.get("schedule_io"))
    have_layer_compute_r = bool(runkv_lc)
    have_layer_compute_t = bool(tightllm_lc)
    have_kernel_timing_r = bool(runkv_kt)
    have_kernel_timing_t = bool(tightllm_kt)

    rpt.h2("Data availability")
    rpt.row(
        f"  runkv  schedule_io NVTX : {'YES' if have_schedule_io_r else 'NO  ← run after patching gpu_model_runner.py'}"
    )
    rpt.row(f"  tight  schedule_io NVTX : {'YES' if have_schedule_io_t else 'NO'}")
    rpt.row(
        f"  runkv  layer_compute NVTX: {'YES' if have_layer_compute_r else 'NO  ← run after patching opt.py'}"
    )
    rpt.row(f"  tight  layer_compute NVTX: {'YES' if have_layer_compute_t else 'NO'}")
    rpt.row(
        f"  runkv  kernel timing     : {'YES (' + str(len(runkv_kt)) + ' layers)' if have_kernel_timing_r else 'NO  ← need layer_compute NVTX + kernel data'}"
    )
    rpt.row(
        f"  tight  kernel timing     : {'YES (' + str(len(tightllm_kt)) + ' layers)' if have_kernel_timing_t else 'NO  ← need layer_compute NVTX + kernel data'}"
    )

    # ── Collect replay token stats from JSONL ───────────────────────────────
    runkv_rs = collect_replay_stats(runkv_flat, skip)
    tightllm_rs = collect_replay_stats(tightllm_flat, skip)
    print(f"  runkv    replay stat layers : {len(runkv_rs)}")
    print(f"  tightllm replay stat layers : {len(tightllm_rs)}")

    # ── Analyses ─────────────────────────────────────────────────────────────
    analyze_imbalance(rpt, runkv_imb, tightllm_imb)
    analyze_compute_vs_io(rpt, runkv_ev, tightllm_ev)
    analyze_layer_total_gpu(rpt, runkv_deltas, tightllm_deltas)
    analyze_h2d_dma(rpt, runkv_dma, tightllm_dma)
    analyze_replay_tokens(rpt, runkv_rs, tightllm_rs, runkv_deltas, tightllm_deltas)

    if runkv_nvtx:
        analyze_prehook_per_layer(rpt, "RunKV", runkv_prehook)
    if tightllm_nvtx:
        analyze_prehook_per_layer(rpt, "TightLLM", tightllm_prehook)

    analyze_layer_compute_nvtx(rpt, runkv_lc, tightllm_lc)

    analyze_clean_compute(rpt, runkv_kt, tightllm_kt, runkv_rs, tightllm_rs)

    # ── Interpretation guide ─────────────────────────────────────────────────
    rpt.h1("HOW TO READ THESE NUMBERS")
    rpt.row("")
    rpt.row("  layer_total_gpu[L] = compute_end[L] - compute_end[L-1]")
    rpt.row("  = prehook_CPU_stall[L] + H2D_wait[L] + attn[L] + FFN[L]")
    rpt.row("")
    rpt.row("  prehook_CPU_stall[L] > 0 when:")
    rpt.row("    prehook_duration[L] > FFN[L-1] GPU time")
    rpt.row("    → GPU finishes FFN[L-1] but CPU hasn't issued h2d_sync yet")
    rpt.row("    → GPU sits idle waiting for Python to return from prehook")
    rpt.row("")
    rpt.row("  H2D_wait[L] > 0 when:")
    rpt.row("    DMA for L not done when compute stream reaches h2d_sync:L")
    rpt.row("    = positive imbalance (compute faster than transfer)")
    rpt.row("")
    rpt.row("  With runkv:layer_compute:L* NVTX (new):")
    rpt.row("    nsys shows CUDA kernels inside this range = true GPU compute")
    rpt.row("    Kernels OUTSIDE it but inside forward() = prehook stall bubbles")
    rpt.row("")
    rpt.row("  With runkv:prehook:schedule_io:L* NVTX (new):")
    rpt.row("    prehook fully decomposed: no more 'misc' residual")

    summary_path = rpt.save()
    print(f"\n[done] {summary_path}")

    # ── Collect per-step × per-layer timing ──────────────────────────────────
    runkv_psl = collect_per_step_layer_timing(runkv_flat, skip)
    tightllm_psl = collect_per_step_layer_timing(tightllm_flat, skip)
    print(
        f"  runkv    per-step-layer metrics : {list(k for k, v in runkv_psl.items() if v)}"
    )
    print(
        f"  tightllm per-step-layer metrics : {list(k for k, v in tightllm_psl.items() if v)}"
    )

    plot_imbalance(runkv_imb, tightllm_imb, out_dir)
    plot_compute_vs_io(runkv_ev, tightllm_ev, out_dir)
    plot_replay_tokens(runkv_rs, tightllm_rs, runkv_deltas, tightllm_deltas, out_dir)
    plot_clean_compute(runkv_kt, tightllm_kt, out_dir)
    plot_per_step_layer_lines(runkv_psl, tightllm_psl, out_dir)
    plot_layer_timeline(
        runkv_deltas,
        tightllm_deltas,
        runkv_dma,
        tightllm_dma,
        runkv_prehook,
        tightllm_prehook,
        runkv_lc,
        tightllm_lc,
        out_dir,
    )


if __name__ == "__main__":
    main()
