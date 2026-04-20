#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
nsys_compare.py — Compare TightLLM vs RunKV from nsys SQLite exports.

Usage:
  # Step 1: export .nsys-rep to SQLite (run once per report)
  nsys export --type sqlite -o tightllm.sqlite tightllm.nsys-rep
  nsys export --type sqlite -o runkv.sqlite    runkv.nsys-rep

  # Step 2: run this script
  python tools/nsys_compare.py tightllm.sqlite runkv.sqlite [--out results/]

What it extracts (all from NVTX ranges + CUDA memcpy events):
  1. Per-step total latency  (step anchor = feedback:begin_step / tightllm:begin_step)
  2. Per-layer prehook sub-components breakdown:
       imbalance | build_plan | build_metadata | skip_ids | (full prehook)
  3. Per-layer controller overhead:
       feedback:controller_update:L{N}  or  tightllm:observe_feedback:L{N}
  4. Per-layer attention compute:
       runkv:attention_compute:*:L{N}
  5. Per-layer imbalance VALUE (from the prehook — duration of the sync-wait
     proxy; NOT the ms value stored in Python, but you can infer it from the
     gap between DMA-done and compute-done CUDA events if you pass --cuda)
  6. Per-layer H2D DMA transfer time (CUDA MemcpyHtoD events in the load_stream
     that fall within each layer's time window)  [requires --cuda flag]

Output:
  - Console: summary table
  - CSV files (if --out is set): per-step and per-layer data for both runs
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# NVTX tag patterns we care about
# ---------------------------------------------------------------------------
STEP_MARKERS = {
    "runkv": "feedback:begin_step",
    "tightllm": "tightllm:begin_step",
}
# Universal fallback: present in both planners, exact range = one step's forward pass.
# Also used as the "fair" comparison baseline when planner-specific markers span
# beyond the forward pass (e.g. include Python scheduling overhead).
STEP_MARKER_FALLBACK = "gpu_model_runner: forward"

PREHOOK_SUBS = [
    "imbalance",
    "build_plan",
    "build_metadata",
    "skip_ids",
]

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _connect(path: str) -> sqlite3.Connection:
    if not Path(path).exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def _nvtx_df(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      start_ns, end_ns, duration_ns, text
    Handles both nsys 2023+ schema (NVTX_EVENTS with textId → StringIds)
    and older schemas where text is inline.
    """
    if not _has_table(con, "NVTX_EVENTS"):
        raise RuntimeError(
            "Table NVTX_EVENTS not found. "
            "Make sure you exported with: nsys export --type sqlite"
        )

    # Check if text is inline or via StringIds
    cols = {row[1] for row in con.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()}

    if "text" in cols:
        q = """
        SELECT start AS start_ns,
               end   AS end_ns,
               (end - start) AS duration_ns,
               text
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND text IS NOT NULL
        """
    elif "textId" in cols and _has_table(con, "StringIds"):
        q = """
        SELECT e.start AS start_ns,
               e.end   AS end_ns,
               (e.end - e.start) AS duration_ns,
               s.value AS text
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE e.end IS NOT NULL AND s.value IS NOT NULL
        """
    else:
        raise RuntimeError(
            "Cannot find NVTX text column. Schema columns: " + ", ".join(sorted(cols))
        )

    df = pd.read_sql_query(q, con)
    df["duration_us"] = df["duration_ns"] / 1e3
    return df


def _memcpy_df(con: sqlite3.Connection) -> pd.DataFrame | None:
    """All CUDA Memcpy events (both H2D and D2H) for DMA analysis."""
    tname = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not _has_table(con, tname):
        return None
    cols = {row[1] for row in con.execute(f"PRAGMA table_info({tname})").fetchall()}
    if "copyKind" not in cols:
        return None
    q = f"""
    SELECT start AS start_ns,
           end   AS end_ns,
           (end - start) AS duration_ns,
           bytes,
           copyKind
    FROM {tname}
    WHERE end IS NOT NULL
    """
    df = pd.read_sql_query(q, con)
    df["duration_us"] = df["duration_ns"] / 1e3
    return df


def _kernel_df(con: sqlite3.Connection) -> pd.DataFrame | None:
    """All CUDA kernel execution events."""
    tname = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not _has_table(con, tname):
        return None
    q = f"""
    SELECT start AS start_ns,
           end   AS end_ns,
           (end - start) AS duration_ns
    FROM {tname}
    WHERE end IS NOT NULL
    """
    df = pd.read_sql_query(q, con)
    df["duration_us"] = df["duration_ns"] / 1e3
    return df


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_layer_idx(text: str) -> int | None:
    """Extract L{N} from an NVTX tag string."""
    import re

    m = re.search(r":L(\d+)$", text)
    return int(m.group(1)) if m else None


def get_step_intervals(nvtx: pd.DataFrame, planner: str) -> pd.DataFrame:
    """
    Return DataFrame with columns [step, start_ns, end_ns, duration_us].

    Uses 'gpu_model_runner: forward' as the primary step marker for both planners —
    this covers exactly the forward pass and makes TightLLM vs RunKV apples-to-apples.
    Planner-specific begin_step markers span beyond the forward pass (include Python
    scheduling gaps) and would inflate step latency unfairly.

    Falls back to planner-specific markers only if the forward marker is absent.
    """
    # Primary: gpu_model_runner: forward (same for both planners, exact forward range)
    fb = (
        nvtx[nvtx["text"] == STEP_MARKER_FALLBACK]
        .copy()
        .sort_values("start_ns")
        .reset_index(drop=True)
    )
    if not fb.empty:
        rows = [
            {
                "step": i,
                "start_ns": row["start_ns"],
                "end_ns": row["end_ns"],
                "duration_us": row["duration_us"],
            }
            for i, row in fb.iterrows()
        ]
        return pd.DataFrame(rows)

    # Fallback: planner-specific begin_step (consecutive-start approach)
    marker = STEP_MARKERS[planner]
    steps = (
        nvtx[nvtx["text"] == marker]
        .copy()
        .sort_values("start_ns")
        .reset_index(drop=True)
    )
    if steps.empty:
        raise RuntimeError(
            f"Neither '{STEP_MARKER_FALLBACK}' nor '{marker}' NVTX markers found.\n"
            "Make sure VLLM_NVTX_SCOPES_FOR_PROFILING=1 was set during capture."
        )
    print(f"  NOTE: '{STEP_MARKER_FALLBACK}' not found, falling back to '{marker}'")
    rows = []
    for i, row in steps.iterrows():
        step_end = (
            steps.iloc[i + 1]["start_ns"] if i + 1 < len(steps) else row["end_ns"]
        )
        rows.append(
            {
                "step": i,
                "start_ns": row["start_ns"],
                "end_ns": step_end,
                "duration_us": (step_end - row["start_ns"]) / 1e3,
            }
        )
    return pd.DataFrame(rows)


def get_prehook_data(
    nvtx: pd.DataFrame, steps: pd.DataFrame, planner: str
) -> pd.DataFrame:
    """
    Per-layer, per-step prehook sub-component durations.
    Returns DataFrame with columns:
      step, layer, imbalance_us, build_plan_us, build_metadata_us, skip_ids_us,
      controller_us, attention_us
    """
    records = []

    for _, s_row in steps.iterrows():
        step = s_row["step"]
        s0, s1 = s_row["start_ns"], s_row["end_ns"]
        window = nvtx[(nvtx["start_ns"] >= s0) & (nvtx["end_ns"] <= s1)]

        # Discover all layer indices present in this step
        layer_idxs = set()
        for text in window["text"]:
            li = _extract_layer_idx(text)
            if li is not None:
                layer_idxs.add(li)

        for layer in sorted(layer_idxs):
            row: dict = {"step": step, "layer": layer}

            # prehook sub-components
            for sub in PREHOOK_SUBS:
                tag = f"runkv:prehook:{sub}:L{layer}"
                match = window[window["text"] == tag]
                row[f"{sub}_us"] = (
                    match["duration_us"].sum() if not match.empty else 0.0
                )

            # controller update
            if planner == "runkv":
                ctrl_tag = f"feedback:controller_update:L{layer}"
            else:
                ctrl_tag = f"tightllm:observe_feedback:L{layer}"
            ctrl = window[window["text"] == ctrl_tag]
            row["controller_us"] = ctrl["duration_us"].sum() if not ctrl.empty else 0.0

            # attention compute  (runkv:attention_compute:*:L{N})
            attn = window[
                window["text"].str.contains(
                    rf":attention_compute:.*:L{layer}$", regex=True, na=False
                )
            ]
            row["attention_us"] = attn["duration_us"].sum() if not attn.empty else 0.0

            # --- RunKV-specific: actual wait for DMA ready (= true imbalance penalty) ---
            wait = window[
                window["text"].str.contains(
                    rf":layer_wait_ready:.*:L{layer}$", regex=True, na=False
                )
            ]
            row["wait_ready_us"] = wait["duration_us"].sum() if not wait.empty else 0.0

            # H2D sync cost per layer
            h2d = window[window["text"] == f"runkv:h2d_sync:L{layer}"]
            row["h2d_sync_us"] = h2d["duration_us"].sum() if not h2d.empty else 0.0

            # DMA transfer duration (mseg_dma, any segment count)
            dma = window[
                window["text"].str.match(rf"runkv:mseg_dma:L{layer}:", na=False)
            ]
            row["mseg_dma_us"] = dma["duration_us"].sum() if not dma.empty else 0.0

            # D2H sync
            d2h = window[window["text"] == f"runkv:d2h_sync:L{layer}"]
            row["d2h_sync_us"] = d2h["duration_us"].sum() if not d2h.empty else 0.0

            records.append(row)

    return pd.DataFrame(records)


def get_dma_per_layer(
    memcpy: pd.DataFrame,
    steps: pd.DataFrame,
    nvtx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attribute H2D DMA transfers to (step, layer) by looking at which
    runkv:attention_compute range they fall under (they run on load_stream
    which overlaps with the compute of the *previous* layer).

    Returns DataFrame with columns: step, layer, dma_total_us, dma_bytes
    """
    records = []
    for _, s_row in steps.iterrows():
        step = s_row["step"]
        s0, s1 = s_row["start_ns"], s_row["end_ns"]

        # H2D transfers in this step window
        dma_win = memcpy[(memcpy["start_ns"] >= s0) & (memcpy["end_ns"] <= s1)]
        if dma_win.empty:
            continue

        # Attention compute windows → layer boundaries
        attn_win = nvtx[
            (nvtx["start_ns"] >= s0)
            & (nvtx["end_ns"] <= s1)
            & nvtx["text"].str.contains(r":attention_compute:", regex=True, na=False)
        ].copy()
        attn_win["layer"] = attn_win["text"].apply(_extract_layer_idx)

        for _, a_row in attn_win.iterrows():
            layer = a_row["layer"]
            if layer is None:
                continue
            # DMA that overlaps with this attention range
            lo, hi = a_row["start_ns"], a_row["end_ns"]
            dma_layer = dma_win[(dma_win["start_ns"] < hi) & (dma_win["end_ns"] > lo)]
            if not dma_layer.empty:
                records.append(
                    {
                        "step": step,
                        "layer": layer,
                        "dma_total_us": dma_layer["duration_us"].sum(),
                        "dma_bytes": dma_layer["bytes"].sum(),
                    }
                )

    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["step", "layer", "dma_total_us", "dma_bytes"])
    )


def get_cuda_layer_timing(
    kernel_df: pd.DataFrame,
    memcpy_df: pd.DataFrame,
    nvtx: pd.DataFrame,
    steps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-layer, per-step GPU-accurate compute and IO breakdown using CUDA event data.

    Compute window for layer N  = [attn_N.start_ns, attn_N+1.start_ns)
      Covers: attention_N kernels + FFN_N kernels
      Source: runkv:attention_compute:*:L{N} NVTX range start times

    IO window for layer N       = runkv:mseg_dma:L{N}:* NVTX range
      Covers: H2D CUDA memcpy transferring KV blocks for layer N
      Source: CUPTI_ACTIVITY_KIND_MEMCPY copyKind=1 within that window

    Returns DataFrame with columns:
      step, layer,
      gpu_compute_us   — sum of CUDA kernel GPU durations in compute window
      gpu_attn_us      — sum of CUDA kernel GPU durations in attention-only window
      gpu_io_us        — sum of H2D CUDA memcpy GPU durations in mseg_dma window
      gpu_io_bytes     — bytes transferred (H2D) in mseg_dma window
    """
    # Pre-filter: only H2D memcpy (copyKind=1)
    h2d_df = memcpy_df[memcpy_df["copyKind"] == 1] if memcpy_df is not None else None

    records = []

    for _, s_row in steps.iterrows():
        step = s_row["step"]
        s0, s1 = s_row["start_ns"], s_row["end_ns"]

        # All NVTX events in this step
        win_nvtx = nvtx[(nvtx["start_ns"] >= s0) & (nvtx["end_ns"] <= s1)]

        # --- attention compute NVTX ranges → layer compute boundaries ---
        attn_ranges = win_nvtx[
            win_nvtx["text"].str.contains(r":attention_compute:", regex=True, na=False)
        ].copy()
        attn_ranges["layer"] = attn_ranges["text"].apply(_extract_layer_idx)
        attn_ranges = attn_ranges.dropna(subset=["layer"])
        attn_ranges["layer"] = attn_ranges["layer"].astype(int)
        attn_ranges = attn_ranges.sort_values("start_ns").reset_index(drop=True)

        if attn_ranges.empty:
            continue

        # Build per-layer compute windows: [attn_N.start, attn_N+1.start)
        # Last layer: [attn_N.start, step_end)
        attn_starts = attn_ranges.set_index("layer")["start_ns"].to_dict()
        attn_ends = attn_ranges.set_index("layer")["end_ns"].to_dict()

        layer_list = sorted(attn_starts.keys())
        compute_windows: dict[int, tuple[int, int]] = {}
        for i, layer in enumerate(layer_list):
            win_start = attn_starts[layer]
            win_end = attn_starts[layer_list[i + 1]] if i + 1 < len(layer_list) else s1
            compute_windows[layer] = (win_start, win_end)

        # --- mseg_dma NVTX ranges → IO windows ---
        dma_nvtx = win_nvtx[
            win_nvtx["text"].str.match(r"runkv:mseg_dma:L\d+:", na=False)
        ].copy()
        dma_nvtx["layer"] = dma_nvtx["text"].apply(_extract_layer_idx)
        dma_nvtx = dma_nvtx.dropna(subset=["layer"])
        dma_nvtx["layer"] = dma_nvtx["layer"].astype(int)

        # --- CUDA events in this step ---
        if kernel_df is not None:
            step_kernels = kernel_df[
                (kernel_df["start_ns"] >= s0) & (kernel_df["end_ns"] <= s1)
            ]
        else:
            step_kernels = None

        if h2d_df is not None:
            step_h2d = h2d_df[(h2d_df["start_ns"] >= s0) & (h2d_df["end_ns"] <= s1)]
        else:
            step_h2d = None

        # --- Per layer ---
        for layer in layer_list:
            row: dict = {"step": step, "layer": layer}

            # GPU compute: CUDA kernels in [attn_N.start, attn_N+1.start)
            c0, c1 = compute_windows[layer]
            if step_kernels is not None and not step_kernels.empty:
                k_win = step_kernels[
                    (step_kernels["start_ns"] >= c0) & (step_kernels["end_ns"] <= c1)
                ]
                row["gpu_compute_us"] = k_win["duration_us"].sum()
            else:
                row["gpu_compute_us"] = 0.0

            # GPU attention only: CUDA kernels strictly within attention range
            a0, a1 = attn_starts[layer], attn_ends[layer]
            if step_kernels is not None and not step_kernels.empty:
                k_attn = step_kernels[
                    (step_kernels["start_ns"] >= a0) & (step_kernels["end_ns"] <= a1)
                ]
                row["gpu_attn_us"] = k_attn["duration_us"].sum()
            else:
                row["gpu_attn_us"] = 0.0

            # GPU IO: H2D memcpy within mseg_dma window for this layer
            dma_layer_rows = dma_nvtx[dma_nvtx["layer"] == layer]
            gpu_io_us = 0.0
            gpu_io_bytes = 0
            if step_h2d is not None and not step_h2d.empty and not dma_layer_rows.empty:
                for _, d_row in dma_layer_rows.iterrows():
                    d0, d1 = d_row["start_ns"], d_row["end_ns"]
                    h2d_win = step_h2d[
                        (step_h2d["start_ns"] >= d0) & (step_h2d["end_ns"] <= d1)
                    ]
                    gpu_io_us += h2d_win["duration_us"].sum()
                    gpu_io_bytes += h2d_win["bytes"].sum()
            row["gpu_io_us"] = gpu_io_us
            row["gpu_io_bytes"] = gpu_io_bytes

            records.append(row)

    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(
            columns=[
                "step",
                "layer",
                "gpu_compute_us",
                "gpu_attn_us",
                "gpu_io_us",
                "gpu_io_bytes",
            ]
        )
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(
    label: str,
    steps: pd.DataFrame,
    prehook: pd.DataFrame,
    cuda_layer: pd.DataFrame | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    print(f"\n[Step latency] (n={len(steps)})")
    print(
        f"  mean={steps['duration_us'].mean():.1f} us  "
        f"median={steps['duration_us'].median():.1f} us  "
        f"p95={steps['duration_us'].quantile(0.95):.1f} us  "
        f"std={steps['duration_us'].std():.1f} us"
    )

    if prehook.empty:
        return

    per_layer = (
        prehook.groupby("layer")
        .mean(numeric_only=True)
        .drop(columns=["step"], errors="ignore")
    )
    total_cols = [c for c in per_layer.columns if c.endswith("_us")]

    print(
        f"\n[Per-layer NVTX averages, µs]  (aggregated over {prehook['step'].nunique()} steps)"
    )
    col_w = 16
    header = f"{'layer':>5} " + "".join(
        f"{c.replace('_us', ''):>{col_w}}" for c in total_cols
    )
    print(header)
    print("-" * len(header))
    for layer, row in per_layer.iterrows():
        vals = "".join(f"{row[c]:>{col_w}.2f}" for c in total_cols)
        print(f"{int(layer):>5} {vals}")

    print("\n[Summed NVTX across all layers per step, µs]")
    layer_sum = prehook.groupby("step")[total_cols].sum()
    print(
        layer_sum.describe(percentiles=[0.5, 0.95])
        .loc[["mean", "50%", "95%", "std"]]
        .to_string()
    )

    # --- GPU-accurate IO vs Compute ---
    if cuda_layer is not None and not cuda_layer.empty:
        gpu_cols = ["gpu_compute_us", "gpu_attn_us", "gpu_io_us"]
        gpu_cols = [c for c in gpu_cols if c in cuda_layer.columns]
        gpu_per_layer = cuda_layer.groupby("layer")[gpu_cols].mean()

        print(
            f"\n[Per-layer GPU-accurate IO & Compute, µs]  "
            f"(CUDA CUPTI events, {cuda_layer['step'].nunique()} steps)"
        )
        print("  gpu_compute = attn + FFN kernels in [attn_N.start, attn_N+1.start)")
        print("  gpu_attn    = kernels strictly within attention_N NVTX range")
        print("  gpu_io      = H2D CUDA memcpy within mseg_dma_N NVTX range")

        col_w2 = 18
        header2 = f"{'layer':>5} " + "".join(
            f"{c.replace('_us', '').replace('gpu_', ''):>{col_w2}}" for c in gpu_cols
        )
        # Add derived columns: ffn = compute - attn, overlap = io - wait_ready
        if "gpu_compute_us" in gpu_cols and "gpu_attn_us" in gpu_cols:
            header2 += f"{'ffn':>{col_w2}}"
        if "gpu_io_us" in gpu_cols and "gpu_io_bytes" in cuda_layer.columns:
            gpu_io_bytes_layer = cuda_layer.groupby("layer")["gpu_io_bytes"].mean()
            header2 += f"{'io_bytes(MB)':>{col_w2}}"
        print(header2)
        print("-" * len(header2))

        for layer in gpu_per_layer.index:
            row = gpu_per_layer.loc[layer]
            vals = "".join(f"{row[c]:>{col_w2}.2f}" for c in gpu_cols)
            if "gpu_compute_us" in gpu_cols and "gpu_attn_us" in gpu_cols:
                ffn = row["gpu_compute_us"] - row["gpu_attn_us"]
                vals += f"{ffn:>{col_w2}.2f}"
            if "gpu_io_us" in gpu_cols and "gpu_io_bytes" in cuda_layer.columns:
                mb = gpu_io_bytes_layer.get(layer, 0) / 1e6
                vals += f"{mb:>{col_w2}.2f}"
            print(f"{int(layer):>5} {vals}")

        # Step-level sums
        print("\n[Summed GPU across all layers per step, µs]")
        gpu_step_sum = cuda_layer.groupby("step")[gpu_cols].sum()
        print(
            gpu_step_sum.describe(percentiles=[0.5, 0.95])
            .loc[["mean", "50%", "95%", "std"]]
            .to_string()
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tightllm_sqlite", help="nsys SQLite export for TightLLM run")
    ap.add_argument("runkv_sqlite", help="nsys SQLite export for RunKV run")
    ap.add_argument(
        "--out",
        default=None,
        help="Directory to write CSV files (default: no CSV output)",
    )
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="Also extract per-layer DMA H2D timing from CUDA memcpy events",
    )
    ap.add_argument(
        "--skip-warmup",
        type=int,
        default=2,
        help="Skip first N steps (warmup). Default: 2",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for label, path, planner in [
        ("TightLLM", args.tightllm_sqlite, "tightllm"),
        ("RunKV", args.runkv_sqlite, "runkv"),
    ]:
        print(f"\nLoading {label} from {path} ...")
        con = _connect(path)
        nvtx = _nvtx_df(con)
        print(f"  NVTX events: {len(nvtx):,}")

        steps = get_step_intervals(nvtx, planner)
        print(f"  Steps found: {len(steps)}")

        # drop warmup
        steps = steps.iloc[args.skip_warmup :].reset_index(drop=True)
        steps["step"] = range(len(steps))
        print(f"  Steps after warmup skip ({args.skip_warmup}): {len(steps)}")

        prehook = get_prehook_data(nvtx, steps, planner)

        dma_df = None
        cuda_layer_df = None
        if args.cuda:
            memcpy = _memcpy_df(con)
            kernel = _kernel_df(con)
            if memcpy is not None:
                dma_df = get_dma_per_layer(memcpy[memcpy["copyKind"] == 1], steps, nvtx)
                print(f"  DMA transfer groups: {len(dma_df)}")
                cuda_layer_df = get_cuda_layer_timing(kernel, memcpy, nvtx, steps)
                n_layers_found = (
                    cuda_layer_df["layer"].nunique() if not cuda_layer_df.empty else 0
                )
                print(
                    f"  CUDA layer timing: {n_layers_found} layers × {cuda_layer_df['step'].nunique() if not cuda_layer_df.empty else 0} steps"
                )
            else:
                print("  WARNING: CUDA memcpy table not found, skipping DMA analysis")

        results[label] = {
            "steps": steps,
            "prehook": prehook,
            "dma": dma_df,
            "cuda_layer": cuda_layer_df,
            "planner": planner,
        }

        if out_dir:
            steps.to_csv(out_dir / f"{planner}_steps.csv", index=False)
            prehook.to_csv(out_dir / f"{planner}_prehook.csv", index=False)
            if dma_df is not None:
                dma_df.to_csv(out_dir / f"{planner}_dma.csv", index=False)
            if cuda_layer_df is not None and not cuda_layer_df.empty:
                cuda_layer_df.to_csv(out_dir / f"{planner}_cuda_layer.csv", index=False)
            print(f"  CSVs written to {out_dir}/")

        con.close()

    # ---------- Summary ----------
    for label, data in results.items():
        print_summary(label, data["steps"], data["prehook"], data.get("cuda_layer"))

    # ---------- Head-to-head diff ----------
    tl = results["TightLLM"]
    rk = results["RunKV"]

    print(f"\n{'=' * 60}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 60}")

    tl_mean = tl["steps"]["duration_us"].mean()
    rk_mean = rk["steps"]["duration_us"].mean()
    print(
        f"\nStep latency: TightLLM={tl_mean:.1f} µs  RunKV={rk_mean:.1f} µs  "
        f"diff={rk_mean - tl_mean:+.1f} µs  ({(rk_mean / tl_mean - 1) * 100:+.1f}%)"
    )

    # Per-column per-layer diff
    us_cols = [c for c in tl["prehook"].columns if c.endswith("_us")]
    if us_cols:
        tl_layeravg = tl["prehook"].groupby("layer")[us_cols].mean()
        rk_layeravg = rk["prehook"].groupby("layer")[us_cols].mean()

        # Align on common layers
        common_layers = tl_layeravg.index.intersection(rk_layeravg.index)
        diff = rk_layeravg.loc[common_layers] - tl_layeravg.loc[common_layers]

        print(
            "\n[Per-layer diff: RunKV − TightLLM (µs, averaged per layer over steps)]"
        )
        print("  Positive = RunKV is SLOWER in that component")
        col_w = 18
        header = f"{'layer':>5} " + "".join(
            f"{c.replace('_us', ''):>{col_w}}" for c in us_cols
        )
        print(header)
        print("-" * len(header))
        for layer in common_layers:
            row = diff.loc[layer]
            vals = "".join(f"{row[c]:>{col_w}.2f}" for c in us_cols)
            print(f"{int(layer):>5} {vals}")

        # Summary: total extra overhead per step
        total_diff_per_step = diff[us_cols].sum().sum() * (
            len(common_layers) / len(tl_layeravg)
        )
        print(
            f"\n  Est. total extra prehook overhead / step (RunKV − TightLLM): "
            f"{total_diff_per_step:.1f} µs"
        )

    # DMA comparison
    if tl["dma"] is not None and rk["dma"] is not None:
        tl_dma = tl["dma"].groupby("layer")["dma_total_us"].mean()
        rk_dma = rk["dma"].groupby("layer")["dma_total_us"].mean()
        common = tl_dma.index.intersection(rk_dma.index)
        if not common.empty:
            print("\n[Per-layer H2D DMA duration diff: RunKV − TightLLM (µs)]")
            for layer in common:
                print(
                    f"  L{layer:2d}: {rk_dma[layer] - tl_dma[layer]:+.1f} µs "
                    f"(TightLLM={tl_dma[layer]:.1f}  RunKV={rk_dma[layer]:.1f})"
                )

    # GPU IO vs Compute comparison
    tl_cl = tl.get("cuda_layer")
    rk_cl = rk.get("cuda_layer")
    if tl_cl is not None and rk_cl is not None and not tl_cl.empty and not rk_cl.empty:
        gpu_cmp_cols = ["gpu_compute_us", "gpu_attn_us", "gpu_io_us"]
        gpu_cmp_cols = [
            c for c in gpu_cmp_cols if c in tl_cl.columns and c in rk_cl.columns
        ]

        tl_gpu = tl_cl.groupby("layer")[gpu_cmp_cols].mean()
        rk_gpu = rk_cl.groupby("layer")[gpu_cmp_cols].mean()
        common_gpu = tl_gpu.index.intersection(rk_gpu.index)
        diff_gpu = rk_gpu.loc[common_gpu] - tl_gpu.loc[common_gpu]

        print("\n[Per-layer GPU IO & Compute diff: RunKV − TightLLM (µs)]")
        print("  Positive = RunKV has MORE GPU time in that bucket")
        col_w = 18
        hdr = f"{'layer':>5} " + "".join(
            f"{c.replace('gpu_', '').replace('_us', ''):>{col_w}}" for c in gpu_cmp_cols
        )
        if "gpu_compute_us" in gpu_cmp_cols and "gpu_attn_us" in gpu_cmp_cols:
            hdr += f"{'ffn_diff':>{col_w}}"
        print(hdr)
        print("-" * len(hdr))
        for layer in common_gpu:
            row = diff_gpu.loc[layer]
            vals = "".join(f"{row[c]:>{col_w}.2f}" for c in gpu_cmp_cols)
            if "gpu_compute_us" in gpu_cmp_cols and "gpu_attn_us" in gpu_cmp_cols:
                ffn_diff = row["gpu_compute_us"] - row["gpu_attn_us"]
                vals += f"{ffn_diff:>{col_w}.2f}"
            print(f"{int(layer):>5} {vals}")

        # Step-level total GPU work diff
        tl_gpu_step = tl_cl.groupby("step")[gpu_cmp_cols].sum().mean()
        rk_gpu_step = rk_cl.groupby("step")[gpu_cmp_cols].sum().mean()
        print("\n  Per-step total GPU work diff (RunKV − TightLLM, µs):")
        for c in gpu_cmp_cols:
            print(
                f"    {c.replace('gpu_', '').replace('_us', ''):15s}: {rk_gpu_step[c] - tl_gpu_step[c]:+.1f} µs "
                f"(TightLLM={tl_gpu_step[c]:.1f}  RunKV={rk_gpu_step[c]:.1f})"
            )

    print()


if __name__ == "__main__":
    main()
