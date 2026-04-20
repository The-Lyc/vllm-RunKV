#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Analyze RunKV vs TightLLM performance using:
  1. Prehook timing JSONL  (RUNKV_PREHOOK_TIMING=1 output)
  2. Component MFU JSONL   (opt_replay_component_mfu.py output)
  3. nsys sqlite files     (nsys profile --export sqlite ...)

Usage examples
--------------
# JSONL only (no sqlite):
python tools/analyze_runkv_vs_tightllm.py \
    --runkv-prehook  runkv_prehook_timing/prehook_timing_*.jsonl \
    --runkv-mfu      exp_results/opt_feedback_observation/opt_component_mfu_1000_*.jsonl \
    --tightllm-mfu   exp_results/tightllm_observation/opt_component_mfu_*.jsonl

# Full comparison with nsys sqlite:
python tools/analyze_runkv_vs_tightllm.py \
    --runkv-sqlite   exp_results/sqlite/runkv-opt-2.7b_context=4k_bs=32_decode=32.sqlite \
    --tightllm-sqlite exp_results/sqlite/tightllm-opt-2.7b_context=4k_bs=32_decode=32.sqlite \
    --runkv-prehook  runkv_prehook_timing/prehook_timing_*.jsonl \
    --output-dir     exp_results/analysis/

Output
------
  analysis_summary.txt   — text report
  prehook_breakdown.png  — segment breakdown bar chart
  build_plan_per_layer.png — per-layer build_plan cost (key difference)
  forward_duration.png   — forward pass CDF comparison
  imbalance_dist.png     — imbalance distribution
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════


def _load_jsonl_lines(paths: list[str]) -> list[dict]:
    records = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [warn] not found: {p}", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_prehook_timing(paths: list[str]) -> list[dict]:
    """Load RUNKV_PREHOOK_TIMING=1 JSONL records (one per step)."""
    return _load_jsonl_lines(paths)


def load_mfu_jsonl(paths: list[str]) -> tuple[list[dict], list[dict]]:
    """Load opt_component_mfu JSONL.

    Detects format automatically:
    - If line has a top-level 'layers' key  → step-level record (nested)
    - Otherwise                             → flat record (one per layer)

    Returns (step_records, flat_records).  Either list may be empty if the
    corresponding format is absent from the given files.
    """
    step_records: list[dict] = []
    flat_records: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [warn] not found: {p}", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "layers" in rec:
                    step_records.append(rec)
                    # Also expand layers to flat
                    for lr in rec.get("layers", []):
                        flat_rec = dict(lr)
                        flat_rec.setdefault("step", rec.get("step"))
                        flat_rec.setdefault("num_reqs", rec.get("num_reqs"))
                        flat_rec.setdefault("rank", rec.get("rank"))
                        flat_records.append(flat_rec)
                else:
                    flat_records.append(rec)
    return step_records, flat_records


def load_sqlite_nvtx(path: str) -> list[dict]:
    """Load all non-null NVTX ranges from an nsys sqlite file."""
    if not Path(path).exists():
        print(f"  [warn] sqlite not found: {path}", file=sys.stderr)
        return []
    db = sqlite3.connect(path)
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


# ══════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ══════════════════════════════════════════════════════════════════════════════


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p5": s[int(n * 0.05)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "min": s[0],
        "max": s[-1],
    }


def nvtx_group_stats(
    records: list[dict],
    pattern: str,
) -> dict[str, dict]:
    """Group NVTX records whose text matches `pattern` (regex).

    Returns {text: stats_dict}.
    """
    rx = re.compile(pattern)
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if rx.search(r["text"]):
            groups[r["text"]].append(r["dur_ms"])
    return {k: _stats(v) for k, v in groups.items()}


def nvtx_prefix_aggregate(
    records: list[dict],
    prefix: str,
) -> dict[str, float]:
    """Sum avg duration over all ranges whose text starts with `prefix`.

    Returns {text: avg_ms}.
    """
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["text"].startswith(prefix):
            totals[r["text"]].append(r["dur_ms"])
    return {k: sum(v) / len(v) for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Analysis sections
# ══════════════════════════════════════════════════════════════════════════════


class Report:
    def __init__(self, out_dir: Path) -> None:
        self.lines: list[str] = []
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def h1(self, title: str) -> None:
        sep = "═" * 72
        self.lines += ["", sep, f"  {title}", sep]

    def h2(self, title: str) -> None:
        self.lines += ["", f"── {title} {'─' * max(0, 68 - len(title))}"]

    def row(self, line: str) -> None:
        self.lines.append(line)

    def table(
        self, headers: list[str], rows: list[list[Any]], col_width: int = 14
    ) -> None:
        fmt = "  ".join(f"{{:<{col_width}}}" for _ in headers)
        self.lines.append(fmt.format(*headers))
        self.lines.append("  ".join("-" * col_width for _ in headers))
        for r in rows:
            self.lines.append(fmt.format(*[str(x) for x in r]))

    def save(self, filename: str = "analysis_summary.txt") -> Path:
        out = self.out_dir / filename
        text = "\n".join(self.lines) + "\n"
        out.write_text(text)
        print(text)
        return out


# ─── Section 1: Prehook timing breakdown ──────────────────────────────────────

PREHOOK_SEGMENTS = [
    ("sync_wait_ms", "sync_wait"),
    ("imbalance_ms", "imbalance"),
    ("build_plan_ms", "build_plan"),
    ("build_meta_ms", "build_meta"),
    ("skip_ids_ms", "skip_ids"),
    ("schedule_io_ms", "schedule_io"),
    ("misc_ms", "misc"),
]


def analyze_prehook(
    rpt: Report,
    runkv_prehook: list[dict],
    tightllm_prehook: list[dict],
) -> None:
    rpt.h1("PREHOOK TIMING BREAKDOWN  (ms per step, CPU wall-clock)")

    def _table_for(label: str, records: list[dict]) -> None:
        if not records:
            rpt.row(f"  {label}: no data")
            return
        # Skip step 0 (warmup often anomalous)
        warm = [r for r in records if r.get("step", 0) > 0]
        if not warm:
            warm = records
        rpt.h2(f"{label}  (n={len(warm)} steps, skipping step 0)")
        headers = ["segment", "mean_ms", "p50_ms", "p95_ms", "total%"]
        total_means = sum(
            sum(r.get(k, 0) for r in warm) / len(warm) for k, _ in PREHOOK_SEGMENTS
        )
        rows = []
        for key, name in PREHOOK_SEGMENTS:
            vals = [r.get(key, 0) for r in warm]
            st = _stats(vals)
            pct = f"{100 * st['mean'] / total_means:.1f}%" if total_means else "-"
            rows.append(
                [
                    name,
                    f"{st['mean']:.2f}",
                    f"{st['median']:.2f}",
                    f"{st['p95']:.2f}",
                    pct,
                ]
            )
        # total row
        totals = [r.get("total_ms", 0) for r in warm]
        st_tot = _stats(totals)
        rows.append(
            [
                "TOTAL",
                f"{st_tot['mean']:.2f}",
                f"{st_tot['median']:.2f}",
                f"{st_tot['p95']:.2f}",
                "100%",
            ]
        )
        rpt.table(headers, rows)

    _table_for("RunKV", runkv_prehook)
    _table_for("TightLLM", tightllm_prehook)

    # Side-by-side delta
    if runkv_prehook and tightllm_prehook:
        warm_r = [r for r in runkv_prehook if r.get("step", 0) > 0] or runkv_prehook
        warm_t = [
            r for r in tightllm_prehook if r.get("step", 0) > 0
        ] or tightllm_prehook
        rpt.h2("Delta: RunKV − TightLLM  (mean ms, positive = RunKV slower)")
        headers = ["segment", "runkv_ms", "tightllm_ms", "delta_ms"]
        rows = []
        for key, name in PREHOOK_SEGMENTS + [("total_ms", "TOTAL")]:
            r_mean = sum(r.get(key, 0) for r in warm_r) / len(warm_r)
            t_mean = sum(r.get(key, 0) for r in warm_t) / len(warm_t)
            rows.append(
                [name, f"{r_mean:.2f}", f"{t_mean:.2f}", f"{r_mean - t_mean:+.2f}"]
            )
        rpt.table(headers, rows)


def plot_prehook_breakdown(
    runkv_prehook: list[dict],
    tightllm_prehook: list[dict],
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    warm_r = [r for r in runkv_prehook if r.get("step", 0) > 0] or runkv_prehook
    warm_t = [r for r in tightllm_prehook if r.get("step", 0) > 0] or tightllm_prehook
    labels = [name for _, name in PREHOOK_SEGMENTS]
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#937860",
        "#DA8BC3",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, records, title in [
        (axes[0], warm_r, "RunKV"),
        (axes[1], warm_t, "TightLLM"),
    ]:
        if not records:
            ax.set_title(f"{title}: no data")
            continue
        means = [
            sum(r.get(k, 0) for r in records) / len(records)
            for k, _ in PREHOOK_SEGMENTS
        ]
        bars = ax.bar(labels, means, color=colors)
        ax.set_title(f"{title} prehook segment means")
        ax.set_ylabel("ms / step")
        ax.tick_params(axis="x", rotation=35)
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    out = out_dir / "prehook_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


# ─── Section 2: NVTX analysis from sqlite ─────────────────────────────────────


def analyze_nvtx(
    rpt: Report,
    runkv_nvtx: list[dict],
    tightllm_nvtx: list[dict],
) -> None:
    rpt.h1("NVTX RANGE ANALYSIS  (from nsys sqlite)")

    # Forward pass duration
    rpt.h2("Forward pass duration  (gpu_model_runner: forward)")
    for label, records in [("RunKV", runkv_nvtx), ("TightLLM", tightllm_nvtx)]:
        fwd = [r["dur_ms"] for r in records if r["text"] == "gpu_model_runner: forward"]
        if fwd:
            st = _stats(fwd)
            rpt.row(
                f"  {label:10s}  n={st['n']:4d}  mean={st['mean']:.1f}ms  "
                f"p50={st['median']:.1f}ms  p95={st['p95']:.1f}ms"
            )

    # build_plan per layer
    rpt.h2("build_plan per layer  (runkv:prehook:build_plan:L*)")
    rpt.row(
        "  KEY FINDING: TightLLM fast-paths identical plans; "
        "RunKV rebuilds every layer."
    )
    headers = ["layer", "runkv_mean_ms", "tightllm_mean_ms", "delta_ms"]
    r_bp = nvtx_prefix_aggregate(runkv_nvtx, "runkv:prehook:build_plan:L")
    t_bp = nvtx_prefix_aggregate(tightllm_nvtx, "runkv:prehook:build_plan:L")
    all_layers = sorted(
        set(r_bp.keys()) | set(t_bp.keys()),
        key=lambda x: int(x.rsplit("L", 1)[-1]),
    )
    rows = []
    for text in all_layers:
        layer_id = text.rsplit("L", 1)[-1]
        rv = r_bp.get(text, float("nan"))
        tv = t_bp.get(text, float("nan"))
        delta = rv - tv if (rv == rv and tv == tv) else float("nan")
        rows.append(
            [
                f"L{layer_id}",
                f"{rv:.3f}" if rv == rv else "-",
                f"{tv:.3f}" if tv == tv else "-",
                f"{delta:+.3f}" if delta == delta else "-",
            ]
        )
    rpt.table(headers, rows[:32])  # cap at 32 layers
    if all_layers:
        r_vals = [r_bp[k] for k in all_layers if k in r_bp]
        t_vals = [t_bp[k] for k in all_layers if k in t_bp]
        r_total = sum(r_vals)
        t_total = sum(t_vals)
        rpt.row("\n  build_plan sum over all layers:")
        rpt.row(f"    RunKV    = {r_total:.1f} ms/step")
        rpt.row(f"    TightLLM = {t_total:.1f} ms/step")
        rpt.row(
            f"    Delta    = {r_total - t_total:+.1f} ms/step  "
            f"← prehook overhead difference"
        )

    # Planner overhead (begin_step + per-layer controller/observe)
    rpt.h2("Planner overhead (begin_step + per-layer update)")
    planner_ranges = {
        "RunKV": [
            ("feedback:begin_step", "begin_step"),
            ("feedback:controller_update:L*", "controller_update/layer"),
        ],
        "TightLLM": [
            ("tightllm:begin_step", "begin_step"),
            ("tightllm:ilp_solve", "ilp_solve"),
            ("tightllm:observe_feedback:L*", "observe_feedback/layer"),
        ],
    }
    for label, nvtx, spec in [
        ("RunKV", runkv_nvtx, planner_ranges["RunKV"]),
        ("TightLLM", tightllm_nvtx, planner_ranges["TightLLM"]),
    ]:
        rpt.row(f"\n  {label}:")
        for pattern, desc in spec:
            rx = re.compile(pattern.replace("*", ".*").replace(":", r"\:"))
            vals = [r["dur_ms"] for r in nvtx if rx.search(r["text"])]
            if not vals:
                rpt.row(f"    {desc:35s}  no data")
                continue
            # Aggregate by unique text to get per-occurrence mean
            by_text: dict[str, list[float]] = defaultdict(list)
            for r in nvtx:
                if rx.search(r["text"]):
                    by_text[r["text"]].append(r["dur_ms"])
            per_occ = [sum(v) / len(v) for v in by_text.values()]
            # If multiple layers, sum gives total per step
            per_step = sum(per_occ)
            mean_per_range = sum(vals) / len(vals)
            rpt.row(
                f"    {desc:35s}  "
                f"mean/range={mean_per_range:.3f}ms  "
                f"total/step≈{per_step:.2f}ms  "
                f"(n_ranges={len(by_text)})"
            )

    # imbalance measurement cost
    rpt.h2("imbalance measurement cost (runkv:prehook:imbalance:L*)")
    for label, nvtx in [("RunKV", runkv_nvtx), ("TightLLM", tightllm_nvtx)]:
        imb = [
            r["dur_ms"]
            for r in nvtx
            if r["text"].startswith("runkv:prehook:imbalance:")
        ]
        if not imb:
            rpt.row(f"  {label}: no data")
            continue
        st = _stats(imb)
        # group by layer for per-step total
        by_text: dict[str, list[float]] = defaultdict(list)
        for r in nvtx:
            if r["text"].startswith("runkv:prehook:imbalance:"):
                by_text[r["text"]].append(r["dur_ms"])
        per_step = sum(sum(v) / len(v) for v in by_text.values())
        rpt.row(
            f"  {label:10s}  mean/layer={st['mean']:.3f}ms  "
            f"total/step≈{per_step:.2f}ms  (n_layers={len(by_text)})"
        )

    # H2D sync
    rpt.h2("H2D DMA sync (runkv:h2d_sync:L*)")
    for label, nvtx in [("RunKV", runkv_nvtx), ("TightLLM", tightllm_nvtx)]:
        h2d = [r["dur_ms"] for r in nvtx if r["text"].startswith("runkv:h2d_sync:")]
        if not h2d:
            rpt.row(f"  {label}: no data")
            continue
        st = _stats(h2d)
        rpt.row(
            f"  {label:10s}  mean={st['mean']:.3f}ms  "
            f"p95={st['p95']:.3f}ms  total_obs={st['n']}"
        )


def plot_build_plan_per_layer(
    runkv_nvtx: list[dict],
    tightllm_nvtx: list[dict],
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    def _extract(nvtx: list[dict]) -> dict[int, float]:
        by_text: dict[str, list[float]] = defaultdict(list)
        for r in nvtx:
            if r["text"].startswith("runkv:prehook:build_plan:L"):
                by_text[r["text"]].append(r["dur_ms"])
        result: dict[int, float] = {}
        for text, vals in by_text.items():
            layer = int(text.rsplit("L", 1)[-1])
            result[layer] = sum(vals) / len(vals)
        return result

    r_data = _extract(runkv_nvtx)
    t_data = _extract(tightllm_nvtx)
    all_layers = sorted(set(r_data) | set(t_data))
    if not all_layers:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    x = list(range(len(all_layers)))
    if r_data:
        r_vals = [r_data.get(l, 0) for l in all_layers]
        ax.bar(
            [xi - 0.2 for xi in x],
            r_vals,
            width=0.4,
            label="RunKV",
            color="#4C72B0",
            alpha=0.85,
        )
    if t_data:
        t_vals = [t_data.get(l, 0) for l in all_layers]
        ax.bar(
            [xi + 0.2 for xi in x],
            t_vals,
            width=0.4,
            label="TightLLM",
            color="#DD8452",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in all_layers], rotation=90, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("avg build_plan duration (ms)")
    ax.set_title(
        "build_plan cost per layer\n"
        "(TightLLM fast-paths identical plans → near-zero for L1+)"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    out = out_dir / "build_plan_per_layer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_forward_duration(
    runkv_nvtx: list[dict],
    tightllm_nvtx: list[dict],
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return
    r_fwd = sorted(
        r["dur_ms"] for r in runkv_nvtx if r["text"] == "gpu_model_runner: forward"
    )
    t_fwd = sorted(
        r["dur_ms"] for r in tightllm_nvtx if r["text"] == "gpu_model_runner: forward"
    )
    if not r_fwd and not t_fwd:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for vals, label, color in [
        (r_fwd, "RunKV", "#4C72B0"),
        (t_fwd, "TightLLM", "#DD8452"),
    ]:
        if vals:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(
                vals,
                cdf,
                label=f"{label} (mean={sum(vals) / len(vals):.0f}ms)",
                color=color,
                lw=2,
            )
    ax.set_xlabel("forward() duration (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Forward pass duration CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "forward_duration_cdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


# ─── Section 3: Imbalance from MFU JSONL ──────────────────────────────────────


def analyze_imbalance(
    rpt: Report,
    runkv_flat: list[dict],
    tightllm_flat: list[dict],
) -> None:
    rpt.h1("IMBALANCE DISTRIBUTION  (compute_end − load_ready, ms)")
    rpt.row("  Positive = transfer slower than compute (under-replaying).")
    rpt.row("  Negative = compute slower than transfer (over-replaying).")
    rpt.row("  Closer to 0 = better balance.")

    for label, flat in [("RunKV", runkv_flat), ("TightLLM", tightllm_flat)]:
        vals = [r["imbalance_ms"] for r in flat if r.get("imbalance_ms") is not None]
        if not vals:
            rpt.row(f"  {label}: no data")
            continue
        abs_vals = [abs(v) for v in vals]
        st = _stats(vals)
        st_abs = _stats(abs_vals)
        within_1ms = sum(1 for v in vals if abs(v) < 1) / len(vals)
        rpt.h2(f"{label}  (n={len(vals)} layer-steps)")
        rpt.row(
            f"  mean={st['mean']:+.2f}ms  median={st['median']:+.2f}ms  "
            f"p5={st['p5']:+.2f}ms  p95={st['p95']:+.2f}ms"
        )
        rpt.row(
            f"  abs_mean={st_abs['mean']:.2f}ms  abs_p95={st_abs['p95']:.2f}ms  "
            f"abs_max={st_abs['max']:.2f}ms"
        )
        rpt.row(f"  |imbalance|<1ms: {100 * within_1ms:.1f}%")

        # Controller action breakdown (RunKV only)
        actions: dict[str, int] = defaultdict(int)
        for r in flat:
            cu = r.get("controller_update")
            if cu and isinstance(cu, dict):
                actions[cu.get("action", "unknown")] += 1
        if actions:
            rpt.row(
                "  controller actions: "
                + ", ".join(f"{k}={v}" for k, v in sorted(actions.items()))
            )


def plot_imbalance(
    runkv_flat: list[dict],
    tightllm_flat: list[dict],
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return
    r_vals = [
        r["imbalance_ms"] for r in runkv_flat if r.get("imbalance_ms") is not None
    ]
    t_vals = [
        r["imbalance_ms"] for r in tightllm_flat if r.get("imbalance_ms") is not None
    ]
    if not r_vals and not t_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    # Histogram
    ax = axes[0]
    clip = 50  # clip extreme first-step values
    for vals, label, color in [
        (r_vals, "RunKV", "#4C72B0"),
        (t_vals, "TightLLM", "#DD8452"),
    ]:
        if vals:
            clipped = [max(-clip, min(clip, v)) for v in vals]
            ax.hist(
                clipped, bins=60, alpha=0.55, label=label, color=color, density=True
            )
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("imbalance_ms  (clipped to ±50)")
    ax.set_ylabel("density")
    ax.set_title("Imbalance distribution")
    ax.legend()

    # Abs imbalance CDF
    ax = axes[1]
    for vals, label, color in [
        (r_vals, "RunKV", "#4C72B0"),
        (t_vals, "TightLLM", "#DD8452"),
    ]:
        if vals:
            abs_s = sorted(abs(v) for v in vals)
            cdf = np.arange(1, len(abs_s) + 1) / len(abs_s)
            ax.plot(
                abs_s,
                cdf,
                label=f"{label} (abs_mean={sum(abs_s) / len(abs_s):.1f}ms)",
                color=color,
                lw=2,
            )
    ax.set_xlim(left=0)
    ax.set_xlabel("|imbalance| (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Abs imbalance CDF  (left=better balance)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "imbalance_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out}")


# ─── Section 4: schedule_io sub-timing ────────────────────────────────────────

SIO_SEGMENTS = [
    ("sio_segment_build_ms", "seg_build"),
    ("sio_mseg_dma_ms", "mseg_dma"),
    ("sio_event_ms", "event"),
    ("sio_cf_prepare_ms", "cf_prepare"),
    ("sio_cf_gather_ms", "cf_gather"),
    ("sio_cf_pin_ms", "cf_pin"),
    ("sio_cf_h2d_ms", "cf_h2d"),
    ("sio_misc_ms", "misc"),
]


def analyze_schedule_io(
    rpt: Report,
    runkv_prehook: list[dict],
    tightllm_prehook: list[dict],
) -> None:
    rpt.h1("SCHEDULE_IO SUB-TIMING  (ms per step, RunKV prehook)")
    for label, records in [("RunKV", runkv_prehook), ("TightLLM", tightllm_prehook)]:
        warm = [r for r in records if r.get("step", 0) > 0] or records
        if not warm:
            continue
        rpt.h2(label)
        headers = ["segment", "mean_ms", "p50_ms", "p95_ms"]
        rows = []
        for key, name in SIO_SEGMENTS:
            vals = [r.get(key, 0) for r in warm]
            if not any(vals):
                continue
            st = _stats(vals)
            rows.append(
                [name, f"{st['mean']:.3f}", f"{st['median']:.3f}", f"{st['p95']:.3f}"]
            )
        rpt.table(headers, rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _glob_expand(paths: list[str]) -> list[str]:
    """Expand glob patterns in the given list."""
    import glob as _glob

    out = []
    for p in paths:
        expanded = _glob.glob(p)
        out.extend(expanded if expanded else [p])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze RunKV vs TightLLM performance.")
    ap.add_argument(
        "--runkv-prehook",
        nargs="*",
        default=[],
        metavar="JSONL",
        help="RUNKV_PREHOOK_TIMING=1 JSONL files (RunKV)",
    )
    ap.add_argument(
        "--tightllm-prehook",
        nargs="*",
        default=[],
        metavar="JSONL",
        help="RUNKV_PREHOOK_TIMING=1 JSONL files (TightLLM)",
    )
    ap.add_argument(
        "--runkv-mfu",
        nargs="*",
        default=[],
        metavar="JSONL",
        help="opt_component_mfu JSONL files (RunKV, step or flat)",
    )
    ap.add_argument(
        "--tightllm-mfu",
        nargs="*",
        default=[],
        metavar="JSONL",
        help="opt_component_mfu JSONL files (TightLLM, step or flat)",
    )
    ap.add_argument(
        "--runkv-sqlite", default="", metavar="SQLITE", help="nsys sqlite for RunKV"
    )
    ap.add_argument(
        "--tightllm-sqlite",
        default="",
        metavar="SQLITE",
        help="nsys sqlite for TightLLM",
    )
    ap.add_argument(
        "--output-dir",
        default="exp_results/analysis",
        metavar="DIR",
        help="Directory for output files (default: exp_results/analysis)",
    )
    ap.add_argument(
        "--skip-warmup-steps",
        type=int,
        default=1,
        help="Skip first N steps as warmup (default: 1)",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    rpt = Report(out_dir)

    rpt.h1("RunKV vs TightLLM Performance Analysis")
    rpt.row(f"  output directory: {out_dir.resolve()}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")

    runkv_prehook_raw = load_prehook_timing(_glob_expand(args.runkv_prehook))
    tightllm_prehook_raw = load_prehook_timing(_glob_expand(args.tightllm_prehook))

    skip = args.skip_warmup_steps
    runkv_prehook = [r for r in runkv_prehook_raw if r.get("step", 0) >= skip]
    tightllm_prehook = [r for r in tightllm_prehook_raw if r.get("step", 0) >= skip]

    _, runkv_flat = load_mfu_jsonl(_glob_expand(args.runkv_mfu))
    _, tightllm_flat = load_mfu_jsonl(_glob_expand(args.tightllm_mfu))
    # Skip warmup in flat too
    runkv_flat = [r for r in runkv_flat if r.get("step", 0) >= skip]
    tightllm_flat = [r for r in tightllm_flat if r.get("step", 0) >= skip]

    runkv_nvtx = load_sqlite_nvtx(args.runkv_sqlite) if args.runkv_sqlite else []
    tightllm_nvtx = (
        load_sqlite_nvtx(args.tightllm_sqlite) if args.tightllm_sqlite else []
    )

    print(f"  runkv    prehook steps : {len(runkv_prehook)}")
    print(f"  tightllm prehook steps : {len(tightllm_prehook)}")
    print(f"  runkv    flat layers   : {len(runkv_flat)}")
    print(f"  tightllm flat layers   : {len(tightllm_flat)}")
    print(f"  runkv    nvtx ranges   : {len(runkv_nvtx)}")
    print(f"  tightllm nvtx ranges   : {len(tightllm_nvtx)}")

    # ── Analyses ───────────────────────────────────────────────────────────────
    if runkv_prehook or tightllm_prehook:
        analyze_prehook(rpt, runkv_prehook, tightllm_prehook)
        analyze_schedule_io(rpt, runkv_prehook, tightllm_prehook)
        plot_prehook_breakdown(runkv_prehook, tightllm_prehook, out_dir)

    if runkv_flat or tightllm_flat:
        analyze_imbalance(rpt, runkv_flat, tightllm_flat)
        plot_imbalance(runkv_flat, tightllm_flat, out_dir)

    if runkv_nvtx or tightllm_nvtx:
        analyze_nvtx(rpt, runkv_nvtx, tightllm_nvtx)
        plot_build_plan_per_layer(runkv_nvtx, tightllm_nvtx, out_dir)
        plot_forward_duration(runkv_nvtx, tightllm_nvtx, out_dir)

    # ── Summary diagnosis ──────────────────────────────────────────────────────
    rpt.h1("DIAGNOSIS SUMMARY")
    rpt.row("  Check the sections above for the specific numbers.")
    rpt.row("")
    rpt.row("  Expected root cause:")
    rpt.row("    RunKV rebuilds compute_layer_replay_plan_for_layer() for EVERY")
    rpt.row("    layer in each step (no fast-path when budget is stable).")
    rpt.row("    TightLLM short-circuits once plan reaches steady state:")
    rpt.row("      if not enable_feedback_correction")
    rpt.row("         and prev_layer_plan.cpu_fill_token_count == 0:")
    rpt.row("          return prev_layer_plan   # skip all numpy work")
    rpt.row("")
    rpt.row("  Fix: add a fast-path in FeedbackReplayPlanProvider.get_layer_plan")
    rpt.row("       that reuses prev_layer_plan when global_budget_blocks hasn't")
    rpt.row("       changed (i.e. controller was in deadband for this layer).")
    rpt.row("")
    rpt.row("  Secondary overhead candidates:")
    rpt.row("    1. observe_layer_feedback(): Newton/secant update per layer")
    rpt.row("       (check 'imbalance_ms' column in prehook timing)")
    rpt.row("    2. begin_step(): fingerprint tuple/hashing")
    rpt.row("       (check 'feedback:begin_step' NVTX range in sqlite)")

    summary_path = rpt.save("analysis_summary.txt")
    print(f"\n[done] report saved to {summary_path}")


if __name__ == "__main__":
    main()
