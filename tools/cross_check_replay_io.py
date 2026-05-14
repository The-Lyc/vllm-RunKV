#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-validate "did replay actually reduce IO?" across three runs.

Inputs are three flat.jsonl files emitted by OPTComponentMFUStepProfiler
(see vllm/v1/profiling/opt_component_mfu.py), produced by running the
observation script three times with different planner configs:

  dryrun   — DRY_RUN=1                   (planner observes but does not
                                          apply replay; gives baseline IO).
  runkv    — PLANNER=feedback, DRY_RUN=0 (feedback controller drives replay).
  tightllm — PLANNER=tightllm            (ILP-derived plans).

The script answers three questions, in this order:

  V1. Internal consistency. For each run, the directly-H2D'd KV token count
      (derived from plan + valid_lens) must equal the actually-enqueued
      block count × block_size, up to a per-request partial-tail-block
      slack of `block_size × num_reqs`.

  V2. Physical consistency. Stage 1 (HS H2D) and stage 2 (KV H2D) wall-clock
      time must be explainable by token-count × bytes / PCIe bandwidth.
      A linear regression of measured-ms vs theoretical-ms gives an
      effective PCIe BW; if a run's BW is far from the others or the
      regression slope is wildly off, that data is suspect.

  V3. Headline: does replay convert into IO savings?  For each (step,
      layer) tuple that exists in all three runs, emit one row aligning
      dryrun / runkv / tightllm direct_h2d_kv and stage 2 ms, plus the
      Δ vs dryrun. The expected Δ ms is
        Δ direct_h2d_kv × kv_bytes_per_token / PCIe_BW
      and the "efficiency" of replay → IO savings is the ratio between
      the measured Δ ms and the expected Δ ms — computed per record, not
      averaged. A value near 1.0 means replay genuinely freed up IO at
      that specific record; a value near 0 means replay was paid for in
      compute but did not unlock IO headroom there.

The per-(step, layer) table (validation 3) is the file the user
explicitly asked for: each record's replay-ratio, IO-token count, and
IO-time Δ across the three runs, with no averaging.

Outputs (under --output-dir):

  per_step_layer_summary.csv  headline table — one row per (step, layer).
  validation_1.txt            token-vs-block sanity per run.
  validation_2.csv            per-record ms-vs-theoretical-bytes scatter.
  validation_2_summary.txt    fitted PCIe BW per run, per stage.
  summary.md                  verdict / numbers in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# OPT-2.7B-8k defaults (MHA). Override via CLI for other models.
OPT_DEFAULTS = dict(
    hidden_size=2560,
    num_kv_heads=32,
    head_dim=80,
    dtype_size=2,
    block_size=16,
)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _load_flat(
    path: Path, warmup_steps: int, skip_layer_zero: bool
) -> list[dict[str, Any]]:
    """Load the flat jsonl, dropping warmup steps and (optionally) layer 0.

    layer_idx=0 is normally skipped because its IO is set up by the bootstrap
    path (not pre_hook), so its event timestamps lie outside the per-step
    anchor window and look like outliers.
    """
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("step", 0) < warmup_steps:
                continue
            if skip_layer_zero and r.get("layer_idx") == 0:
                continue
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# validation 1: token vs block
# ---------------------------------------------------------------------------


def validation_1(
    label: str, records: list[dict[str, Any]], block_size: int
) -> dict[str, Any]:
    """Per-record assertion: token-count derived from plan+valid_lens must
    match block-count × block_size, up to a `num_reqs × block_size` slack
    (partial tail block per request).
    """
    n_total = 0
    n_bad = 0
    worst_diff = 0
    worst_bound = 0
    worst_row: dict[str, Any] | None = None
    for r in records:
        tok = r.get("direct_h2d_kv_token_count")
        blk = r.get("load_layer_block_count")
        num_reqs = r.get("num_reqs")
        if tok is None or blk is None or num_reqs is None:
            continue
        n_total += 1
        diff = abs(int(tok) - int(blk) * block_size)
        bound = int(num_reqs) * block_size
        if diff > bound:
            n_bad += 1
            if diff > worst_diff:
                worst_diff = diff
                worst_bound = bound
                worst_row = {
                    "step": r.get("step"),
                    "layer_idx": r.get("layer_idx"),
                    "tok": int(tok),
                    "blk": int(blk),
                    "num_reqs": int(num_reqs),
                    "diff": diff,
                    "bound": bound,
                }
    return dict(
        label=label,
        n_total=n_total,
        n_bad=n_bad,
        worst_diff=worst_diff,
        worst_bound=worst_bound,
        worst_row=worst_row,
        passed=(n_bad == 0),
    )


# ---------------------------------------------------------------------------
# validation 2: bytes vs time
# ---------------------------------------------------------------------------


def _linreg_through_origin(xs: list[float], ys: list[float]) -> float:
    """Slope of y = k·x via least squares forced through 0. Returns 0.0 if no
    valid points. We force through 0 because both stages should take 0 ms when
    zero bytes are transferred."""
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        num += x * y
        den += x * x
    return num / den if den > 0 else 0.0


def validation_2(
    label: str,
    records: list[dict[str, Any]],
    kv_bytes_per_token: int,
    hs_bytes_per_token: int,
) -> dict[str, Any]:
    """Regress measured stage-{1,2} ms on theoretical bytes to recover an
    effective PCIe bandwidth. Stage 1 is HS-only (cpu_fill); stage 2 is
    KV-only (direct_h2d_kv).
    """
    stage2_x: list[float] = []  # bytes
    stage2_y: list[float] = []  # ms
    stage1_x: list[float] = []
    stage1_y: list[float] = []
    rows: list[dict[str, Any]] = []
    for r in records:
        # Stage 2 (KV): kv_ready - load_start
        kv_ready = r.get("kv_ready_ms_from_anchor")
        load_start = r.get("load_start_ms_from_anchor")
        dh = r.get("direct_h2d_kv_token_count")
        s2_ms = (
            kv_ready - load_start
            if kv_ready is not None and load_start is not None
            else None
        )
        s2_bytes = int(dh) * kv_bytes_per_token if dh is not None else None
        # Stage 1 (HS): hs_ready - cpu_fill_start
        hs_ready = r.get("hs_ready_ms_from_anchor")
        cf_start = r.get("cpu_fill_start_ms_from_anchor")
        cf = r.get("cpu_fill_token_count")
        s1_ms = (
            hs_ready - cf_start
            if hs_ready is not None and cf_start is not None
            else None
        )
        s1_bytes = int(cf) * hs_bytes_per_token if cf is not None else None

        if s2_ms is not None and s2_bytes is not None and s2_bytes > 0:
            stage2_x.append(float(s2_bytes))
            stage2_y.append(float(s2_ms))
        if s1_ms is not None and s1_bytes is not None and s1_bytes > 0:
            stage1_x.append(float(s1_bytes))
            stage1_y.append(float(s1_ms))
        rows.append(
            dict(
                label=label,
                step=r.get("step"),
                layer_idx=r.get("layer_idx"),
                stage1_ms=s1_ms,
                stage1_bytes=s1_bytes,
                stage2_ms=s2_ms,
                stage2_bytes=s2_bytes,
            )
        )

    # slope is ms / byte → effective bandwidth = 1/slope bytes/ms = 1e3/slope B/s.
    slope2 = _linreg_through_origin(stage2_x, stage2_y)
    slope1 = _linreg_through_origin(stage1_x, stage1_y)
    bw2 = 1e3 / slope2 / 1e9 if slope2 > 0 else None  # GB/s
    bw1 = 1e3 / slope1 / 1e9 if slope1 > 0 else None

    return dict(
        label=label,
        n_stage2=len(stage2_x),
        n_stage1=len(stage1_x),
        slope2_ms_per_byte=slope2,
        slope1_ms_per_byte=slope1,
        effective_pcie_bw_stage2_GBps=bw2,
        effective_pcie_bw_stage1_GBps=bw1,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# validation 3: per-(step, layer) Δ table — the headline file
# ---------------------------------------------------------------------------


def _extract_per_record(r: dict[str, Any]) -> dict[str, Any]:
    """Pull the numeric fields we care about out of a flat.jsonl record.

    Stage 1/2 ms are derived here so downstream is uniform across runs even
    when one of them has no cpu_fill (e.g. dryrun/tightllm in pure
    gpu_reuse mode → stage1_ms = None).
    """
    kv_ready = r.get("kv_ready_ms_from_anchor")
    load_start = r.get("load_start_ms_from_anchor")
    s2_ms = (
        kv_ready - load_start
        if kv_ready is not None and load_start is not None
        else None
    )
    hs_ready = r.get("hs_ready_ms_from_anchor")
    cf_start = r.get("cpu_fill_start_ms_from_anchor")
    s1_ms = (
        hs_ready - cf_start
        if hs_ready is not None and cf_start is not None
        else None
    )
    return {
        "replay_token_count": r.get("replay_token_count"),
        "direct_h2d_kv_token_count": r.get("direct_h2d_kv_token_count"),
        "cpu_fill_token_count": r.get("cpu_fill_token_count"),
        "gpu_reuse_token_count": r.get("gpu_reuse_token_count"),
        "load_layer_block_count": r.get("load_layer_block_count"),
        "history_token_count": r.get("history_token_count"),
        "kv_replay_fraction": r.get("kv_replay_fraction"),
        "replay_ratio_legacy": r.get("replay_ratio"),
        "num_reqs": r.get("num_reqs"),
        "num_actual_tokens": r.get("num_actual_tokens"),
        "num_tokens": r.get("num_tokens"),
        "stage1_ms": s1_ms,
        "stage2_ms": s2_ms,
    }


def _index_by_step_layer(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for r in records:
        step = r.get("step")
        layer = r.get("layer_idx")
        if step is None or layer is None:
            continue
        out[(int(step), int(layer))] = _extract_per_record(r)
    return out


def validation_3(
    dry: dict[tuple[int, int], dict[str, Any]],
    rk: dict[tuple[int, int], dict[str, Any]],
    tl: dict[tuple[int, int], dict[str, Any]],
    kv_bytes_per_token: int,
    pcie_bw_GBps: float,
) -> list[dict[str, Any]]:
    """Per-(step, layer) comparison. Each row aligns the same (step, layer)
    across three runs and computes Δ direct_h2d_kv and Δ stage2_ms relative
    to dryrun. The "efficiency" column (actual Δ ms / expected Δ ms) lets
    you see, per record, how much of the replay actually converted into
    wall-clock IO savings — no averaging across steps or layers.
    """
    keys = sorted(set(dry) | set(rk) | set(tl))
    pcie_bw_bytes_per_ms = pcie_bw_GBps * 1e9 / 1e3
    rows: list[dict[str, Any]] = []
    for key in keys:
        step, layer = key
        d = dry.get(key, {})
        k = rk.get(key, {})
        t = tl.get(key, {})

        def _columns_for(run_label: str, src: dict[str, Any]) -> dict[str, Any]:
            dh_dry = d.get("direct_h2d_kv_token_count")
            dh_run = src.get("direct_h2d_kv_token_count")
            s2_dry = d.get("stage2_ms")
            s2_run = src.get("stage2_ms")
            delta_dh = (
                dh_dry - dh_run if dh_dry is not None and dh_run is not None else None
            )
            delta_ms = (
                s2_dry - s2_run if s2_dry is not None and s2_run is not None else None
            )
            expected_delta_ms = (
                delta_dh * kv_bytes_per_token / pcie_bw_bytes_per_ms
                if delta_dh is not None
                else None
            )
            efficiency = (
                delta_ms / expected_delta_ms
                if delta_ms is not None
                and expected_delta_ms is not None
                and expected_delta_ms > 0
                else None
            )
            return {
                f"{run_label}_replay_tokens": src.get("replay_token_count"),
                f"{run_label}_direct_h2d_kv": dh_run,
                f"{run_label}_load_layer_blocks": src.get("load_layer_block_count"),
                f"{run_label}_cpu_fill_tokens": src.get("cpu_fill_token_count"),
                f"{run_label}_gpu_reuse_tokens": src.get("gpu_reuse_token_count"),
                f"{run_label}_kv_replay_fraction": src.get("kv_replay_fraction"),
                f"{run_label}_replay_ratio_legacy": src.get("replay_ratio_legacy"),
                f"{run_label}_stage1_ms": src.get("stage1_ms"),
                f"{run_label}_stage2_ms": s2_run,
                f"delta_direct_h2d_kv_{run_label}_vs_dryrun": delta_dh,
                f"delta_stage2_ms_{run_label}_vs_dryrun": delta_ms,
                f"expected_delta_stage2_ms_{run_label}": expected_delta_ms,
                f"io_savings_efficiency_{run_label}": efficiency,
            }

        # num_reqs / num_tokens come from whichever run has them — these
        # should match across runs for the same (step, layer); if they
        # don't, the runs aren't aligned (different batch composition).
        num_reqs = (
            d.get("num_reqs") or k.get("num_reqs") or t.get("num_reqs")
        )
        num_tokens = (
            d.get("num_tokens") or k.get("num_tokens") or t.get("num_tokens")
        )
        row: dict[str, Any] = {
            "step": step,
            "layer_idx": layer,
            "num_reqs": num_reqs,
            "num_tokens_scheduled": num_tokens,
            "dryrun_replay_tokens": d.get("replay_token_count"),
            "dryrun_direct_h2d_kv": d.get("direct_h2d_kv_token_count"),
            "dryrun_load_layer_blocks": d.get("load_layer_block_count"),
            "dryrun_cpu_fill_tokens": d.get("cpu_fill_token_count"),
            "dryrun_gpu_reuse_tokens": d.get("gpu_reuse_token_count"),
            "dryrun_kv_replay_fraction": d.get("kv_replay_fraction"),
            "dryrun_replay_ratio_legacy": d.get("replay_ratio_legacy"),
            "dryrun_stage1_ms": d.get("stage1_ms"),
            "dryrun_stage2_ms": d.get("stage2_ms"),
        }
        row.update(_columns_for("runkv", k))
        row.update(_columns_for("tightllm", t))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    # Stable column order: take keys from the widest row.
    cols: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                cols.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ---------------------------------------------------------------------------
# focused views: the three quantities the user wants to eyeball
#   - replay_tokens
#   - io_tokens   (= direct_h2d_kv)
#   - io_ms       (= stage2_ms = kv_ready - load_start)
# ---------------------------------------------------------------------------


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return 100.0 * num / den


def _build_delta_view(
    v3_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Narrow wide format focused on (replay, io_tokens, io_ms). One row per
    (step, layer). Columns are grouped by metric so adjacent columns are
    always comparable across the three runs."""
    out: list[dict[str, Any]] = []
    for r in v3_rows:
        io_tok_dry = r.get("dryrun_direct_h2d_kv")
        io_tok_rk = r.get("runkv_direct_h2d_kv")
        io_tok_tl = r.get("tightllm_direct_h2d_kv")
        io_ms_dry = r.get("dryrun_stage2_ms")
        io_ms_rk = r.get("runkv_stage2_ms")
        io_ms_tl = r.get("tightllm_stage2_ms")
        d_tok_rk = r.get("delta_direct_h2d_kv_runkv_vs_dryrun")
        d_tok_tl = r.get("delta_direct_h2d_kv_tightllm_vs_dryrun")
        d_ms_rk = r.get("delta_stage2_ms_runkv_vs_dryrun")
        d_ms_tl = r.get("delta_stage2_ms_tightllm_vs_dryrun")
        out.append(
            {
                "step": r["step"],
                "layer_idx": r["layer_idx"],
                # --- replay tokens (what each run forced into qkv_proj) ---
                "replay_dry": r.get("dryrun_replay_tokens"),
                "replay_rk": r.get("runkv_replay_tokens"),
                "replay_tl": r.get("tightllm_replay_tokens"),
                # --- IO tokens (KV that ACTUALLY went H2D) ---
                "io_tok_dry": io_tok_dry,
                "io_tok_rk": io_tok_rk,
                "io_tok_tl": io_tok_tl,
                "io_tok_saved_rk": d_tok_rk,
                "io_tok_saved_tl": d_tok_tl,
                "io_tok_saved_pct_rk": _pct(d_tok_rk, io_tok_dry),
                "io_tok_saved_pct_tl": _pct(d_tok_tl, io_tok_dry),
                # --- IO duration (stage 2 ms = kv_ready - load_start) ---
                "io_ms_dry": io_ms_dry,
                "io_ms_rk": io_ms_rk,
                "io_ms_tl": io_ms_tl,
                "io_ms_saved_rk": d_ms_rk,
                "io_ms_saved_tl": d_ms_tl,
                "io_ms_saved_pct_rk": _pct(d_ms_rk, io_ms_dry),
                "io_ms_saved_pct_tl": _pct(d_ms_tl, io_ms_dry),
                # --- efficiency = actual ms saved / expected ms saved ---
                "efficiency_rk": r.get("io_savings_efficiency_runkv"),
                "efficiency_tl": r.get("io_savings_efficiency_tightllm"),
            }
        )
    return out


def _build_per_step_totals(
    v3_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per step, summing layer-level quantities across all layers
    in that step. This is the natural "step-level" view that maps onto
    user-visible latency: the total IO tokens / IO ms per forward pass.
    """
    by_step: dict[int, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in v3_rows:
        s = by_step[int(r["step"])]
        s["replay_dry"].append(r.get("dryrun_replay_tokens"))
        s["replay_rk"].append(r.get("runkv_replay_tokens"))
        s["replay_tl"].append(r.get("tightllm_replay_tokens"))
        s["io_tok_dry"].append(r.get("dryrun_direct_h2d_kv"))
        s["io_tok_rk"].append(r.get("runkv_direct_h2d_kv"))
        s["io_tok_tl"].append(r.get("tightllm_direct_h2d_kv"))
        s["io_ms_dry"].append(r.get("dryrun_stage2_ms"))
        s["io_ms_rk"].append(r.get("runkv_stage2_ms"))
        s["io_ms_tl"].append(r.get("tightllm_stage2_ms"))

    def _sum(values: list) -> float | None:
        cleaned = [v for v in values if v is not None]
        return sum(cleaned) if cleaned else None

    out: list[dict[str, Any]] = []
    for step in sorted(by_step):
        b = by_step[step]
        sums = {k: _sum(v) for k, v in b.items()}
        d_tok_rk = (
            sums["io_tok_dry"] - sums["io_tok_rk"]
            if sums["io_tok_dry"] is not None and sums["io_tok_rk"] is not None
            else None
        )
        d_tok_tl = (
            sums["io_tok_dry"] - sums["io_tok_tl"]
            if sums["io_tok_dry"] is not None and sums["io_tok_tl"] is not None
            else None
        )
        d_ms_rk = (
            sums["io_ms_dry"] - sums["io_ms_rk"]
            if sums["io_ms_dry"] is not None and sums["io_ms_rk"] is not None
            else None
        )
        d_ms_tl = (
            sums["io_ms_dry"] - sums["io_ms_tl"]
            if sums["io_ms_dry"] is not None and sums["io_ms_tl"] is not None
            else None
        )
        out.append(
            {
                "step": step,
                "replay_dry_total": sums["replay_dry"],
                "replay_rk_total": sums["replay_rk"],
                "replay_tl_total": sums["replay_tl"],
                "io_tok_dry_total": sums["io_tok_dry"],
                "io_tok_rk_total": sums["io_tok_rk"],
                "io_tok_tl_total": sums["io_tok_tl"],
                "io_tok_saved_rk_total": d_tok_rk,
                "io_tok_saved_tl_total": d_tok_tl,
                "io_tok_saved_pct_rk": _pct(d_tok_rk, sums["io_tok_dry"]),
                "io_tok_saved_pct_tl": _pct(d_tok_tl, sums["io_tok_dry"]),
                "io_ms_dry_total": sums["io_ms_dry"],
                "io_ms_rk_total": sums["io_ms_rk"],
                "io_ms_tl_total": sums["io_ms_tl"],
                "io_ms_saved_rk_total": d_ms_rk,
                "io_ms_saved_tl_total": d_ms_tl,
                "io_ms_saved_pct_rk": _pct(d_ms_rk, sums["io_ms_dry"]),
                "io_ms_saved_pct_tl": _pct(d_ms_tl, sums["io_ms_dry"]),
            }
        )
    return out


def _write_delta_view_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    sample_step: int | None,
) -> None:
    """Render one sample step (all layers) as an aligned markdown table —
    intended for at-a-glance reading in an IDE. The full data lives in
    delta_view.csv.
    """
    if not rows:
        path.write_text("(no rows)\n")
        return
    if sample_step is None:
        sample_step = sorted({r["step"] for r in rows})[len(rows) // 2 // 31]
    subset = [r for r in rows if r["step"] == sample_step]
    headers = [
        ("layer", 5, "d"),
        ("replay_rk", 10, ".0f"),
        ("replay_tl", 10, ".0f"),
        ("io_tok_dry", 11, ".0f"),
        ("io_tok_rk", 10, ".0f"),
        ("io_tok_tl", 10, ".0f"),
        ("io_tok_Δrk", 11, ".0f"),
        ("io_tok_Δtl", 11, ".0f"),
        ("%save_rk", 9, ".1f"),
        ("%save_tl", 9, ".1f"),
        ("io_ms_dry", 10, ".2f"),
        ("io_ms_rk", 9, ".2f"),
        ("io_ms_tl", 9, ".2f"),
        ("io_ms_Δrk", 10, ".2f"),
        ("io_ms_Δtl", 10, ".2f"),
        ("%msave_rk", 10, ".1f"),
        ("%msave_tl", 10, ".1f"),
    ]
    col_keys = [
        "layer_idx",
        "replay_rk", "replay_tl",
        "io_tok_dry", "io_tok_rk", "io_tok_tl",
        "io_tok_saved_rk", "io_tok_saved_tl",
        "io_tok_saved_pct_rk", "io_tok_saved_pct_tl",
        "io_ms_dry", "io_ms_rk", "io_ms_tl",
        "io_ms_saved_rk", "io_ms_saved_tl",
        "io_ms_saved_pct_rk", "io_ms_saved_pct_tl",
    ]

    def _cell(value: Any, fmt: str, width: int) -> str:
        if value is None:
            s = "—"
        else:
            try:
                s = format(value, fmt)
            except (TypeError, ValueError):
                s = str(value)
        return s.rjust(width)

    with path.open("w") as f:
        f.write(f"# Sample view at step={sample_step} (all layers).\n")
        f.write(
            "# replay_rk/tl are RunKV/TightLLM replay tokens (dryrun is 0).\n"
            "# io_tok_* are direct_h2d_kv tokens; io_ms_* are stage 2 ms.\n"
            "# Δrk/Δtl = dryrun − run (positive = run saved that much).\n"
            "# %save = saved / dryrun × 100.\n\n"
        )
        # header
        f.write(
            " ".join(name.rjust(width) for name, width, _ in headers) + "\n"
        )
        f.write(
            " ".join("-" * width for _, width, _ in headers) + "\n"
        )
        for r in subset:
            cells = []
            for (_, width, fmt), key in zip(headers, col_keys):
                cells.append(_cell(r.get(key), fmt, width))
            f.write(" ".join(cells) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dryrun", required=True, type=Path,
                    help="flat.jsonl from DRY_RUN=1 run.")
    ap.add_argument("--runkv", required=True, type=Path,
                    help="flat.jsonl from PLANNER=feedback DRY_RUN=0 run.")
    ap.add_argument("--tightllm", required=True, type=Path,
                    help="flat.jsonl from PLANNER=tightllm run.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--warmup-steps", type=int, default=1,
                    help="Drop the first N steps from each run.")
    ap.add_argument("--include-layer-zero", action="store_true",
                    help="Do NOT skip layer 0 (default: skip it because its "
                         "IO is set up outside pre_hook and event "
                         "timestamps look like outliers).")
    ap.add_argument("--hidden-size", type=int, default=OPT_DEFAULTS["hidden_size"])
    ap.add_argument("--num-kv-heads", type=int, default=OPT_DEFAULTS["num_kv_heads"])
    ap.add_argument("--head-dim", type=int, default=OPT_DEFAULTS["head_dim"])
    ap.add_argument("--dtype-size", type=int, default=OPT_DEFAULTS["dtype_size"],
                    help="Bytes per element (2 for fp16/bf16, 4 for fp32).")
    ap.add_argument("--block-size", type=int, default=OPT_DEFAULTS["block_size"])
    ap.add_argument("--pcie-bw-GBps", type=float, default=None,
                    help="Effective PCIe bandwidth used to compute "
                         "expected Δ stage2 ms in validation 3. Default: "
                         "use stage-2 BW fitted from dryrun (validation 2).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # bytes per token:
    #   KV: 2 (K+V) × num_kv_heads × head_dim × dtype_size
    #   HS: hidden_size × dtype_size
    kv_bytes_per_token = 2 * args.num_kv_heads * args.head_dim * args.dtype_size
    hs_bytes_per_token = args.hidden_size * args.dtype_size

    skip_layer_zero = not args.include_layer_zero
    dry_recs = _load_flat(args.dryrun, args.warmup_steps, skip_layer_zero)
    rk_recs = _load_flat(args.runkv, args.warmup_steps, skip_layer_zero)
    tl_recs = _load_flat(args.tightllm, args.warmup_steps, skip_layer_zero)

    # --- Validation 1 ---
    v1 = [
        validation_1("dryrun", dry_recs, args.block_size),
        validation_1("runkv", rk_recs, args.block_size),
        validation_1("tightllm", tl_recs, args.block_size),
    ]
    with (args.output_dir / "validation_1.txt").open("w") as f:
        f.write("Validation 1 — token-count vs block-count internal consistency\n")
        f.write("Per record: |direct_h2d_kv_token_count − load_layer_block_count × "
                f"block_size| ≤ num_reqs × {args.block_size}\n\n")
        for v in v1:
            f.write(f"[{v['label']}] n_total={v['n_total']} n_bad={v['n_bad']} "
                    f"worst_diff={v['worst_diff']} worst_bound={v['worst_bound']}"
                    f" → {'PASS' if v['passed'] else 'FAIL'}\n")
            if v["worst_row"] is not None and not v["passed"]:
                f.write(f"  worst row: {json.dumps(v['worst_row'])}\n")
        f.write("\n")

    # --- Validation 2 ---
    v2 = [
        validation_2("dryrun", dry_recs, kv_bytes_per_token, hs_bytes_per_token),
        validation_2("runkv", rk_recs, kv_bytes_per_token, hs_bytes_per_token),
        validation_2("tightllm", tl_recs, kv_bytes_per_token, hs_bytes_per_token),
    ]
    all_scatter_rows: list[dict[str, Any]] = []
    for v in v2:
        all_scatter_rows.extend(v["rows"])
    _write_csv(args.output_dir / "validation_2.csv", all_scatter_rows)
    with (args.output_dir / "validation_2_summary.txt").open("w") as f:
        f.write("Validation 2 — measured ms regressed on theoretical bytes\n")
        f.write(f"KV bytes/token  = 2×{args.num_kv_heads}×{args.head_dim}"
                f"×{args.dtype_size} = {kv_bytes_per_token}\n")
        f.write(f"HS bytes/token  = {args.hidden_size}×{args.dtype_size}"
                f" = {hs_bytes_per_token}\n\n")
        for v in v2:
            f.write(f"[{v['label']}]\n")
            f.write(f"  n_stage2={v['n_stage2']} → "
                    f"effective PCIe BW (KV) = "
                    f"{_fmt(v['effective_pcie_bw_stage2_GBps'])} GB/s\n")
            f.write(f"  n_stage1={v['n_stage1']} → "
                    f"effective PCIe BW (HS) = "
                    f"{_fmt(v['effective_pcie_bw_stage1_GBps'])} GB/s\n")
        f.write("\n")

    # PCIe bandwidth used for V3 expected-Δ computation
    if args.pcie_bw_GBps is not None:
        pcie_bw_GBps = args.pcie_bw_GBps
        pcie_source = f"--pcie-bw-GBps={pcie_bw_GBps}"
    else:
        dryrun_bw = v2[0]["effective_pcie_bw_stage2_GBps"]
        pcie_bw_GBps = float(dryrun_bw) if dryrun_bw else 20.0
        pcie_source = (
            f"fitted from dryrun stage2 = {pcie_bw_GBps:.2f} GB/s"
            if dryrun_bw
            else "fallback default 20 GB/s"
        )

    # --- Validation 3 ---
    dry_idx = _index_by_step_layer(dry_recs)
    rk_idx = _index_by_step_layer(rk_recs)
    tl_idx = _index_by_step_layer(tl_recs)
    v3_rows = validation_3(
        dry_idx, rk_idx, tl_idx, kv_bytes_per_token, pcie_bw_GBps
    )
    _write_csv(args.output_dir / "per_step_layer_summary.csv", v3_rows)

    # Focused views: replay tokens / IO tokens / IO ms, with deltas and
    # percentage savings side-by-side. These are the intuitive companion
    # files to the full per_step_layer_summary.csv — same data, narrower
    # and grouped so each metric's triplet of run-values sits adjacent.
    delta_rows = _build_delta_view(v3_rows)
    _write_csv(args.output_dir / "delta_view.csv", delta_rows)
    step_rows = _build_per_step_totals(v3_rows)
    _write_csv(args.output_dir / "per_step_totals.csv", step_rows)
    if delta_rows:
        # Pick the median warm step for the markdown sample
        all_steps = sorted({r["step"] for r in delta_rows})
        sample_step = all_steps[len(all_steps) // 2]
        _write_delta_view_markdown(
            args.output_dir / "delta_view_sample.txt", delta_rows, sample_step
        )

    # Aggregate efficiency
    def _mean_no_none(key: str) -> float | None:
        vs = [r[key] for r in v3_rows if r.get(key) is not None]
        return statistics.fmean(vs) if vs else None

    summary_path = args.output_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write("# Replay → IO cross-check summary\n\n")
        f.write(f"PCIe bandwidth for expected-Δ: {pcie_source}.\n\n")
        f.write("## Validation 1 — token vs block consistency\n\n")
        for v in v1:
            verdict = "PASS" if v["passed"] else f"FAIL ({v['n_bad']} bad rows)"
            f.write(f"- **{v['label']}**: {verdict}\n")
        f.write("\n## Validation 2 — effective PCIe bandwidth (per stage)\n\n")
        f.write("| run | stage2 (KV) GB/s | stage1 (HS) GB/s |\n")
        f.write("|---|---|---|\n")
        for v in v2:
            f.write(
                f"| {v['label']} | {_fmt(v['effective_pcie_bw_stage2_GBps'])} "
                f"| {_fmt(v['effective_pcie_bw_stage1_GBps'])} |\n"
            )
        f.write(
            "\n## Validation 3 — replay vs IO savings (per record, then "
            "aggregated)\n\n"
        )
        f.write(
            "The raw per-(step, layer) deltas are in "
            "`per_step_layer_summary.csv` — one row per aligned record, no "
            "averaging. The table below is a single roll-up over that "
            "file (mean of the per-record values where non-null) just so "
            "the headline number is visible at a glance.\n\n"
            "Per-record column semantics (each is computed for one "
            "specific (step, layer) tuple):\n"
            "  Δ direct_h2d_kv  = dryrun.direct_h2d_kv − run.direct_h2d_kv\n"
            "  Δ stage2_ms      = dryrun.stage2_ms      − run.stage2_ms\n"
            "  expected Δ ms    = Δ direct_h2d_kv × kv_bytes_per_token "
            "/ PCIe_BW\n"
            "  efficiency       = Δ stage2_ms / expected Δ ms\n\n"
        )
        f.write("| run | mean Δ direct_h2d_kv | mean Δ stage2_ms | "
                "mean expected Δ ms | mean efficiency |\n")
        f.write("|---|---|---|---|---|\n")
        for label in ("runkv", "tightllm"):
            dh = _mean_no_none(f"delta_direct_h2d_kv_{label}_vs_dryrun")
            ms = _mean_no_none(f"delta_stage2_ms_{label}_vs_dryrun")
            exp = _mean_no_none(f"expected_delta_stage2_ms_{label}")
            eff = _mean_no_none(f"io_savings_efficiency_{label}")
            f.write(
                f"| {label} | {_fmt(dh, 1)} | {_fmt(ms, 3)} | "
                f"{_fmt(exp, 3)} | {_fmt(eff, 3)} |\n"
            )
        f.write(
            "\nInterpretation of per-record efficiency:\n"
            "- ≈ 1.0 → replay genuinely freed IO bandwidth at that "
            "specific (step, layer).\n"
            "- ≪ 1.0 → replay reduced token count but did not save "
            "wall-clock IO at that point; either PCIe was not the "
            "bottleneck or stage-1 (HS) overhead ate the gain.\n"
            "- > 1.0 → dryrun baseline was IO-bound past the linear "
            "regime at that record, or PCIe_BW estimate is too low.\n"
        )
        f.write(
            "\nFor the full per-record view (one row per (step, layer)),"
            " open `per_step_layer_summary.csv`. Columns:\n"
            "  step, layer_idx, num_reqs, num_tokens_scheduled,\n"
            "  dryrun_{replay_tokens, direct_h2d_kv, load_layer_blocks,\n"
            "    cpu_fill_tokens, gpu_reuse_tokens, kv_replay_fraction,\n"
            "    replay_ratio_legacy, stage1_ms, stage2_ms},\n"
            "  runkv_<same fields>,\n"
            "  delta_direct_h2d_kv_runkv_vs_dryrun, "
            "delta_stage2_ms_runkv_vs_dryrun,\n"
            "  expected_delta_stage2_ms_runkv, "
            "io_savings_efficiency_runkv,\n"
            "  tightllm_<same fields>, "
            "delta_*_tightllm_vs_dryrun, ...\n"
        )
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
