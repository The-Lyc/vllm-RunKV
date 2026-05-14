#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare per-layer IO duration reduction against replay_ratio.

Inputs (flat JSONL produced by opt_component_mfu hooks):
  - RunKV   (feedback controller on): IO contains KV + Hidden States transfers
  - TightLLM (observation only)     : IO contains KV transfers (ideal proxy)

Because `num_actual_tokens` itself depends on the replay configuration, the
raw IO duration cannot be compared directly across replay_ratio buckets.  We
report two views:

  (1) Raw per-layer reduction
        baseline_io_dur[L] := median of (load_ready - load_start)
                              over records with replay_ratio < ratio_band
        actual_reduction   := 1 - io_dur(r) / baseline_io_dur[L]

  (2) Token-normalised (per-token IO) reduction  ← primary metric
        io_per_tok          := io_dur / num_actual_tokens
        baseline_pt[L]      := median(io_per_tok) over baseline records
        predicted_io_per_tok(r) := baseline_pt[L] * (1 - replay_ratio)
        actual_reduction_pt := 1 - io_per_tok(r) / baseline_pt[L]

Additional fields:
  kv_io_dur           := kv_ready  - load_start         (KV portion only)
  hs_overhead         := load_ready - kv_ready          (RunKV-only)

Expected behaviour for view (2):
  * TightLLM: actual_reduction_pt ≈ replay_ratio (slope ≈ 1.0, intercept ≈ 0).
  * RunKV   : trend follows replay_ratio but biased lower because HS transfers
              do not shrink with replay_ratio (slope < 1.0).

Usage
-----
python tools/analyze_io_vs_replay_ratio.py \\
    --runkv-mfu    exp_results/opt_feedback_observation/opt_component_mfu_1000_20260512_1114.flat.jsonl \\
    --tightllm-mfu exp_results/tightllm_observation/opt_component_mfu_1000_20260512_1114.flat.jsonl \\
    --output-dir   exp_results/analysis/io_vs_replay/20260512_1114
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np

    HAS_NP = True
except ImportError:
    HAS_NP = False


# ─────────────────────────────────────────────────────────────────────────────
# Loading & extraction
# ─────────────────────────────────────────────────────────────────────────────

_FIELDS = (
    "step",
    "layer_idx",
    "load_start_ms_from_anchor",
    "load_ready_ms_from_anchor",
    "kv_ready_ms_from_anchor",
    "hs_ready_ms_from_anchor",
    "replay_ratio",
    "replay_token_count",
    "cpu_fill_token_count",
    "gpu_reuse_token_count",
    "num_actual_tokens",
    "num_tokens",
    "num_reqs",
)


def load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("load_start_ms_from_anchor") is None:
                continue
            if r.get("load_ready_ms_from_anchor") is None:
                continue
            if r.get("layer_idx") is None:
                continue
            rows.append(r)
    return rows


def derive(r: dict) -> dict:
    ls = r["load_start_ms_from_anchor"]
    lready = r["load_ready_ms_from_anchor"]
    kvr = r.get("kv_ready_ms_from_anchor")
    hsr = r.get("hs_ready_ms_from_anchor")
    io_dur = lready - ls
    kv_io_dur = (kvr - ls) if kvr is not None else None
    if hsr is not None and kvr is not None:
        hs_overhead = hsr - kvr
    elif kvr is not None:
        hs_overhead = lready - kvr  # load_ready == max(kv_ready, hs_ready)
    else:
        hs_overhead = None
    return {
        "io_dur": io_dur,
        "kv_io_dur": kv_io_dur,
        "hs_overhead": hs_overhead,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────────


def compute_baseline(
    rows: list[dict],
    ratio_band: float,
    warmup_steps: int,
    metric: str,
    per_token: bool = False,
) -> dict[int, float]:
    """Per-layer median of `metric` (or metric/num_actual_tokens) for records
    with replay_ratio < band."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if (r.get("step") or 0) < warmup_steps:
            continue
        rr = r.get("replay_ratio")
        if rr is None or rr >= ratio_band:
            continue
        d = derive(r)
        v = d[metric]
        if v is None:
            continue
        if per_token:
            tok = r.get("num_actual_tokens") or r.get("num_tokens")
            if not tok:
                continue
            v = v / tok
        buckets[r["layer_idx"]].append(v)
    return {li: statistics.median(vs) for li, vs in buckets.items() if vs}


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for y = slope*x + intercept (least squares).

    Fallback pure-Python implementation when numpy is unavailable.
    """
    if not xs or len(xs) < 2:
        return float("nan"), float("nan"), float("nan")
    if HAS_NP:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(slope), float(intercept), r2
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), my, float("nan")
    slope = num / den
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def safe_std(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_file(
    label: str,
    path: Path,
    ratio_band: float,
    warmup_steps: int,
    exclude_layer_zero: bool,
) -> dict:
    rows = load_records(path)
    if exclude_layer_zero:
        rows = [r for r in rows if r["layer_idx"] != 0]
    if not rows:
        raise RuntimeError(f"[{label}] no usable records in {path}")

    baseline_io = compute_baseline(rows, ratio_band, warmup_steps, "io_dur")
    baseline_kv = compute_baseline(rows, ratio_band, warmup_steps, "kv_io_dur")
    baseline_io_pt = compute_baseline(
        rows, ratio_band, warmup_steps, "io_dur", per_token=True)
    baseline_kv_pt = compute_baseline(
        rows, ratio_band, warmup_steps, "kv_io_dur", per_token=True)

    enriched: list[dict] = []
    for r in rows:
        if (r.get("step") or 0) < warmup_steps:
            continue
        d = derive(r)
        li = r["layer_idx"]
        rr = r.get("replay_ratio")
        tok = r.get("num_actual_tokens") or r.get("num_tokens") or 0
        b_io = baseline_io.get(li)
        b_kv = baseline_kv.get(li)
        b_io_pt = baseline_io_pt.get(li)
        b_kv_pt = baseline_kv_pt.get(li)
        io_per_tok = (d["io_dur"] / tok) if tok else None
        kv_per_tok = (
            (d["kv_io_dur"] / tok)
            if (tok and d["kv_io_dur"] is not None) else None
        )
        actual_io = (
            (1.0 - d["io_dur"] / b_io) if (b_io and b_io > 0) else float("nan")
        )
        actual_kv = (
            (1.0 - d["kv_io_dur"] / b_kv)
            if (b_kv and b_kv > 0 and d["kv_io_dur"] is not None)
            else float("nan")
        )
        actual_io_pt = (
            (1.0 - io_per_tok / b_io_pt)
            if (b_io_pt and b_io_pt > 0 and io_per_tok is not None)
            else float("nan")
        )
        actual_kv_pt = (
            (1.0 - kv_per_tok / b_kv_pt)
            if (b_kv_pt and b_kv_pt > 0 and kv_per_tok is not None)
            else float("nan")
        )
        predicted_io = (
            b_io_pt * tok * (1.0 - rr)
            if (b_io_pt is not None and rr is not None and tok) else float("nan")
        )
        expected = rr if rr is not None else float("nan")
        enriched.append(
            {
                "label": label,
                "step": r.get("step"),
                "layer_idx": li,
                "replay_ratio": rr,
                "num_reqs": r.get("num_reqs"),
                "num_tokens": r.get("num_tokens"),
                "num_actual_tokens": r.get("num_actual_tokens"),
                "load_start": r["load_start_ms_from_anchor"],
                "load_ready": r["load_ready_ms_from_anchor"],
                "kv_ready": r.get("kv_ready_ms_from_anchor"),
                "hs_ready": r.get("hs_ready_ms_from_anchor"),
                "io_dur": d["io_dur"],
                "kv_io_dur": d["kv_io_dur"],
                "hs_overhead": d["hs_overhead"],
                "io_per_tok": io_per_tok,
                "kv_per_tok": kv_per_tok,
                "baseline_io_dur": b_io,
                "baseline_kv_io_dur": b_kv,
                "baseline_io_per_tok": b_io_pt,
                "baseline_kv_per_tok": b_kv_pt,
                "predicted_io_dur": predicted_io,
                "actual_io_reduction": actual_io,
                "actual_kv_reduction": actual_kv,
                "actual_io_reduction_pt": actual_io_pt,
                "actual_kv_reduction_pt": actual_kv_pt,
                "expected_reduction": expected,
                "delta_io": (
                    (actual_io - expected)
                    if not (actual_io != actual_io or expected != expected)
                    else float("nan")
                ),
                "delta_kv": (
                    (actual_kv - expected)
                    if not (actual_kv != actual_kv or expected != expected)
                    else float("nan")
                ),
                "delta_io_pt": (
                    (actual_io_pt - expected)
                    if not (actual_io_pt != actual_io_pt or expected != expected)
                    else float("nan")
                ),
                "delta_kv_pt": (
                    (actual_kv_pt - expected)
                    if not (actual_kv_pt != actual_kv_pt or expected != expected)
                    else float("nan")
                ),
            }
        )

    # Global linear fit over records that have valid (expected, actual).
    def _xy(key_actual: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for e in enriched:
            x = e["expected_reduction"]
            y = e[key_actual]
            if x is None or y is None:
                continue
            if x != x or y != y:  # NaN check
                continue
            xs.append(x)
            ys.append(y)
        return xs, ys

    xs_io, ys_io = _xy("actual_io_reduction")
    xs_kv, ys_kv = _xy("actual_kv_reduction")
    xs_io_pt, ys_io_pt = _xy("actual_io_reduction_pt")
    xs_kv_pt, ys_kv_pt = _xy("actual_kv_reduction_pt")
    fit_io = linear_fit(xs_io, ys_io)
    fit_kv = linear_fit(xs_kv, ys_kv)
    fit_io_pt = linear_fit(xs_io_pt, ys_io_pt)
    fit_kv_pt = linear_fit(xs_kv_pt, ys_kv_pt)

    # Per-ratio bucket averages
    bucket_keys: dict[float, list[dict]] = defaultdict(list)
    for e in enriched:
        rr = e["expected_reduction"]
        if rr is None or rr != rr:
            continue
        # Round to 3 decimals to group identical ratios.
        bucket_keys[round(rr, 3)].append(e)
    def _good(seq):
        return [v for v in seq if v == v]

    bucket_rows = []
    for rr, items in sorted(bucket_keys.items()):
        ios = _good([it["actual_io_reduction"] for it in items])
        kvs = _good([it["actual_kv_reduction"] for it in items])
        ios_pt = _good([it["actual_io_reduction_pt"] for it in items])
        kvs_pt = _good([it["actual_kv_reduction_pt"] for it in items])
        bucket_rows.append(
            {
                "label": label,
                "replay_ratio": rr,
                "n": len(items),
                "expected": rr,
                "actual_io_mean": safe_mean(ios),
                "actual_io_std": safe_std(ios),
                "actual_kv_mean": safe_mean(kvs),
                "actual_kv_std": safe_std(kvs),
                "actual_io_pt_mean": safe_mean(ios_pt),
                "actual_io_pt_std": safe_std(ios_pt),
                "actual_kv_pt_mean": safe_mean(kvs_pt),
                "actual_kv_pt_std": safe_std(kvs_pt),
                "delta_io_mean": safe_mean([io - rr for io in ios]),
                "delta_kv_mean": safe_mean([kv - rr for kv in kvs]),
                "delta_io_pt_mean": safe_mean([io - rr for io in ios_pt]),
                "delta_kv_pt_mean": safe_mean([kv - rr for kv in kvs_pt]),
            }
        )

    return {
        "label": label,
        "path": str(path),
        "n_records": len(enriched),
        "baseline_io": baseline_io,
        "baseline_kv": baseline_kv,
        "baseline_io_pt": baseline_io_pt,
        "baseline_kv_pt": baseline_kv_pt,
        "records": enriched,
        "buckets": bucket_rows,
        "fit_io": fit_io,
        "fit_kv": fit_kv,
        "fit_io_pt": fit_io_pt,
        "fit_kv_pt": fit_kv_pt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────


def write_per_record_csv(path: Path, results: list[dict]) -> None:
    if not results:
        return
    cols = [
        "label", "step", "layer_idx", "replay_ratio", "num_reqs",
        "num_tokens", "num_actual_tokens",
        "load_start", "load_ready", "kv_ready", "hs_ready",
        "io_dur", "kv_io_dur", "hs_overhead",
        "io_per_tok", "kv_per_tok",
        "baseline_io_dur", "baseline_kv_io_dur",
        "baseline_io_per_tok", "baseline_kv_per_tok",
        "predicted_io_dur",
        "actual_io_reduction", "actual_kv_reduction",
        "actual_io_reduction_pt", "actual_kv_reduction_pt",
        "expected_reduction",
        "delta_io", "delta_kv", "delta_io_pt", "delta_kv_pt",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for res in results:
            for r in res["records"]:
                w.writerow({c: r.get(c) for c in cols})


def write_bucket_csv(path: Path, results: list[dict]) -> None:
    cols = [
        "label", "replay_ratio", "n", "expected",
        "actual_io_mean", "actual_io_std",
        "actual_kv_mean", "actual_kv_std",
        "actual_io_pt_mean", "actual_io_pt_std",
        "actual_kv_pt_mean", "actual_kv_pt_std",
        "delta_io_mean", "delta_kv_mean",
        "delta_io_pt_mean", "delta_kv_pt_mean",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for res in results:
            for b in res["buckets"]:
                w.writerow({c: b.get(c) for c in cols})


def write_summary(path: Path, results: list[dict], args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("IO reduction vs replay_ratio — summary")
    lines.append("=" * 78)
    lines.append(f"  ratio_band       = {args.ratio_band}")
    lines.append(f"  warmup_steps     = {args.warmup_steps}")
    lines.append(f"  exclude_layer_0  = {args.exclude_layer_zero}")
    lines.append("")
    for res in results:
        lines.append("-" * 78)
        lines.append(f"[{res['label']}]  {res['path']}")
        lines.append(f"  records used     : {res['n_records']}")
        n_layers_io = len(res["baseline_io"])
        n_layers_kv = len(res["baseline_kv"])
        if res["baseline_io"]:
            io_vals = list(res["baseline_io"].values())
            lines.append(
                f"  baseline_io_dur  : layers={n_layers_io}  "
                f"mean={safe_mean(io_vals):.3f} ms  "
                f"std={safe_std(io_vals):.3f} ms  "
                f"min={min(io_vals):.3f}  max={max(io_vals):.3f}"
            )
        else:
            lines.append(
                "  baseline_io_dur  : NO baseline records "
                "(no replay_ratio<band); reductions are NaN"
            )
        if res["baseline_kv"]:
            kv_vals = list(res["baseline_kv"].values())
            lines.append(
                f"  baseline_kv_dur  : layers={n_layers_kv}  "
                f"mean={safe_mean(kv_vals):.3f} ms  "
                f"std={safe_std(kv_vals):.3f} ms"
            )

        s_io, b_io, r2_io = res["fit_io"]
        s_kv, b_kv, r2_kv = res["fit_kv"]
        s_io_pt, b_io_pt, r2_io_pt = res["fit_io_pt"]
        s_kv_pt, b_kv_pt, r2_kv_pt = res["fit_kv_pt"]
        lines.append(
            f"  fit RAW   IO    : actual = {s_io:.3f} * expected + {b_io:.4f}   "
            f"R^2={r2_io:.3f}"
        )
        lines.append(
            f"  fit RAW   KV    : actual = {s_kv:.3f} * expected + {b_kv:.4f}   "
            f"R^2={r2_kv:.3f}"
        )
        lines.append(
            f"  fit PER-TOK IO  : actual = {s_io_pt:.3f} * expected + {b_io_pt:.4f}   "
            f"R^2={r2_io_pt:.3f}   (primary metric)"
        )
        lines.append(
            f"  fit PER-TOK KV  : actual = {s_kv_pt:.3f} * expected + {b_kv_pt:.4f}   "
            f"R^2={r2_kv_pt:.3f}"
        )
        lines.append("")
        lines.append("  per replay_ratio bucket (PRIMARY: per-token view):")
        lines.append(
            "    {:>8s}  {:>5s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
                "ratio", "n", "act_io_pt", "Δ_io_pt", "act_kv_pt", "Δ_kv_pt"
            )
        )
        for b in res["buckets"]:
            lines.append(
                "    {:>8.4f}  {:>5d}  {:>10.4f}  {:>+10.4f}  {:>10.4f}  {:>+10.4f}".format(
                    b["replay_ratio"], b["n"],
                    b["actual_io_pt_mean"], b["delta_io_pt_mean"],
                    b["actual_kv_pt_mean"], b["delta_kv_pt_mean"],
                )
            )
        lines.append("")

    # High-level verdict
    lines.append("=" * 78)
    lines.append("Verdict")
    lines.append("=" * 78)
    for res in results:
        s_io_pt, _, _ = res["fit_io_pt"]
        s_kv_pt, _, _ = res["fit_kv_pt"]
        within = "PASS" if (0.9 <= s_io_pt <= 1.1) else "BIAS"
        lines.append(
            f"  [{res['label']}] per-tok IO-slope={s_io_pt:.3f} (target 1.0; {within}), "
            f"per-tok KV-slope={s_kv_pt:.3f}"
        )
    lines.append("")
    lines.append(
        "  Interpretation: tightllm should yield IO-slope ≈ 1.0 (no HS overhead);"
    )
    lines.append(
        "  runkv IO-slope < 1.0 reflects fixed HS transfer time; the KV-slope "
        "should still be ≈ 1.0."
    )

    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_scatter(out: Path, results: list[dict]) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(6.4 * len(results), 5.4),
                             squeeze=False)
    for ax, res in zip(axes[0], results):
        xs_io, ys_io, xs_kv, ys_kv, layers = [], [], [], [], []
        for r in res["records"]:
            x = r["expected_reduction"]
            yi = r["actual_io_reduction_pt"]
            yk = r["actual_kv_reduction_pt"]
            if x is None or x != x:
                continue
            if yi == yi:
                xs_io.append(x); ys_io.append(yi); layers.append(r["layer_idx"])
            if yk == yk:
                xs_kv.append(x); ys_kv.append(yk)

        if xs_io:
            sc = ax.scatter(
                xs_io, ys_io, c=layers, cmap="viridis",
                s=14, alpha=0.55, label="actual (IO/tok)",
            )
            plt.colorbar(sc, ax=ax, label="layer_idx", shrink=0.85)
        if xs_kv:
            ax.scatter(
                xs_kv, ys_kv, facecolors="none", edgecolors="red",
                s=22, linewidths=0.8, label="actual (KV/tok)",
            )

        all_x = xs_io + xs_kv
        if all_x:
            lo = min(min(all_x), 0.0)
            hi = max(max(all_x), 1.0)
        else:
            lo, hi = 0.0, 1.0
        line = [lo, hi]
        ax.plot(line, line, "k--", lw=1.0, label="y = x (ideal)")

        s_io, b_io, r2_io = res["fit_io_pt"]
        if s_io == s_io:  # not NaN
            ax.plot(
                line,
                [s_io * lo + b_io, s_io * hi + b_io],
                "b-", lw=1.0,
                label=f"fit IO/tok: {s_io:.2f}x+{b_io:.2f} (R²={r2_io:.2f})",
            )
        s_kv, b_kv, r2_kv = res["fit_kv_pt"]
        if s_kv == s_kv:
            ax.plot(
                line,
                [s_kv * lo + b_kv, s_kv * hi + b_kv],
                "r-", lw=1.0, alpha=0.7,
                label=f"fit KV/tok: {s_kv:.2f}x+{b_kv:.2f} (R²={r2_kv:.2f})",
            )

        ax.set_title(res["label"])
        ax.set_xlabel("expected reduction = replay_ratio")
        ax.set_ylabel("actual reduction = 1 - io_per_tok / baseline_per_tok")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Actual IO reduction (per-token normalized) vs replay_ratio")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_layer_bars(out: Path, results: list[dict]) -> None:
    """For the most populated non-baseline ratio bucket per file,
    plot per-layer actual vs expected."""
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3.4 * len(results)),
                             squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        # pick most-populated bucket with ratio > 0.05
        candidates = [b for b in res["buckets"] if b["replay_ratio"] > 0.05]
        if not candidates:
            ax.set_title(f"{res['label']}: no replay_ratio>0.05 records")
            continue
        target = max(candidates, key=lambda b: b["n"])
        rr = target["replay_ratio"]
        per_layer_io: dict[int, list[float]] = defaultdict(list)
        per_layer_kv: dict[int, list[float]] = defaultdict(list)
        for r in res["records"]:
            if round((r["expected_reduction"] or -1), 3) != rr:
                continue
            if r["actual_io_reduction_pt"] == r["actual_io_reduction_pt"]:
                per_layer_io[r["layer_idx"]].append(r["actual_io_reduction_pt"])
            if r["actual_kv_reduction_pt"] == r["actual_kv_reduction_pt"]:
                per_layer_kv[r["layer_idx"]].append(r["actual_kv_reduction_pt"])
        layers = sorted(per_layer_io.keys())
        if not layers:
            ax.set_title(f"{res['label']}: no data for ratio={rr}")
            continue
        io_means = [safe_mean(per_layer_io[L]) for L in layers]
        kv_means = [safe_mean(per_layer_kv.get(L, [])) for L in layers]
        x = list(range(len(layers)))
        w = 0.4
        ax.bar([xi - w / 2 for xi in x], io_means, width=w,
               color="C0", label="actual IO/tok reduction")
        ax.bar([xi + w / 2 for xi in x], kv_means, width=w,
               color="C3", label="actual KV/tok reduction", alpha=0.7)
        ax.axhline(rr, color="k", ls="--", lw=1.0, label=f"expected = {rr}")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_xlabel("layer_idx")
        ax.set_ylabel("reduction")
        ax.set_title(f"{res['label']} (replay_ratio={rr}, n={target['n']})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_hs_overhead(out: Path, results: list[dict]) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for res in results:
        per_layer: dict[int, list[float]] = defaultdict(list)
        for r in res["records"]:
            v = r["hs_overhead"]
            if v is None:
                continue
            per_layer[r["layer_idx"]].append(v)
        if not per_layer:
            continue
        layers = sorted(per_layer.keys())
        means = [safe_mean(per_layer[L]) for L in layers]
        ax.plot(layers, means, marker="o", ms=3, lw=1.0, label=res["label"])
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("hs_overhead = load_ready - kv_ready  (ms)")
    ax.set_title("Hidden-state IO overhead per layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runkv-mfu",
        default="exp_results/opt_feedback_observation/"
                "opt_component_mfu_1000_20260512_1114.flat.jsonl",
    )
    p.add_argument(
        "--tightllm-mfu",
        default="exp_results/tightllm_observation/"
                "opt_component_mfu_1000_20260512_1114.flat.jsonl",
    )
    p.add_argument(
        "--output-dir",
        default="exp_results/analysis/io_vs_replay/latest",
    )
    p.add_argument("--ratio-band", type=float, default=0.02,
                   help="records with replay_ratio < band define the baseline")
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="skip the first N steps")
    p.add_argument("--exclude-layer-zero", action="store_true", default=True)
    p.add_argument("--include-layer-zero", dest="exclude_layer_zero",
                   action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for label, path_str in (("runkv", args.runkv_mfu),
                            ("tightllm", args.tightllm_mfu)):
        path = Path(path_str)
        if not path.exists():
            print(f"[warn] {label}: missing {path}, skipped")
            continue
        print(f"[{label}] loading {path}")
        res = analyze_file(
            label=label,
            path=path,
            ratio_band=args.ratio_band,
            warmup_steps=args.warmup_steps,
            exclude_layer_zero=args.exclude_layer_zero,
        )
        results.append(res)

    if not results:
        raise SystemExit("No inputs available.")

    write_per_record_csv(out_dir / "per_record.csv", results)
    write_bucket_csv(out_dir / "per_ratio_bucket.csv", results)
    write_summary(out_dir / "summary.txt", results, args)
    plot_scatter(out_dir / "scatter_actual_vs_expected.png", results)
    plot_per_layer_bars(out_dir / "per_layer_reduction_bars.png", results)
    plot_hs_overhead(out_dir / "hs_overhead_per_layer.png", results)

    # Console digest
    print()
    print((out_dir / "summary.txt").read_text())
    print(f"[done] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
