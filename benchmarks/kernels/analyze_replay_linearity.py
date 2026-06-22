"""
Analyze results from profile_replay_position_sensitivity.py.

Usage:
    python benchmarks/kernels/analyze_replay_linearity.py \
        --output-dir /tmp/replay_linearity

Reads raw.csv and local_fits.csv from --output-dir.
Prints:
  1. Per-(module, ctx_len, anchor_block) linearity summary
  2. Overall pass/fail verdict (R² ≥ threshold and CV ≤ threshold)
  3. Slope table: ms-per-replay-block for each config
Optionally saves a matplotlib figure if --plot is set.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass mirrors (must match profile_replay_position_sensitivity.py)
# ---------------------------------------------------------------------------

@dataclass
class RawRow:
    module: str
    ctx_len: int
    scheduled_len: int
    anchor_block: int
    anchor_token: int
    suffix_len: int
    replay_blocks: int
    baseline_ms: float
    baseline_std_ms: float
    replay_ms: float
    replay_std_ms: float
    delta_ms: float


@dataclass
class FitRow:
    module: str
    ctx_len: int
    scheduled_len: int
    anchor_block: int
    window_start_blocks: int
    window_end_blocks: int
    n_points: int
    slope_ms_per_block: float
    intercept_ms: float
    r2: float
    max_abs_residual_ms: float
    cv_marginal_delta: float


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _cast(val: str, typ):
    if typ is int:
        return int(val)
    if typ is float:
        return float(val)
    return val


def load_raw_csv(path: Path) -> list[RawRow]:
    rows = []
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(RawRow(
                module=rec["module"],
                ctx_len=int(rec["ctx_len"]),
                scheduled_len=int(rec["scheduled_len"]),
                anchor_block=int(rec["anchor_block"]),
                anchor_token=int(rec["anchor_token"]),
                suffix_len=int(rec["suffix_len"]),
                replay_blocks=int(rec["replay_blocks"]),
                baseline_ms=float(rec["baseline_ms"]),
                baseline_std_ms=float(rec["baseline_std_ms"]),
                replay_ms=float(rec["replay_ms"]),
                replay_std_ms=float(rec["replay_std_ms"]),
                delta_ms=float(rec["delta_ms"]),
            ))
    return rows


def load_fit_csv(path: Path) -> list[FitRow]:
    rows = []
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(FitRow(
                module=rec["module"],
                ctx_len=int(rec["ctx_len"]),
                scheduled_len=int(rec["scheduled_len"]),
                anchor_block=int(rec["anchor_block"]),
                window_start_blocks=int(rec["window_start_blocks"]),
                window_end_blocks=int(rec["window_end_blocks"]),
                n_points=int(rec["n_points"]),
                slope_ms_per_block=float(rec["slope_ms_per_block"]),
                intercept_ms=float(rec["intercept_ms"]),
                r2=float(rec["r2"]),
                max_abs_residual_ms=float(rec["max_abs_residual_ms"]),
                cv_marginal_delta=float(rec["cv_marginal_delta"]),
            ))
    return rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def groupby(rows, key):
    from itertools import groupby as _gb
    rows = sorted(rows, key=key)
    return _gb(rows, key=key)


def analyze_fits(fits: list[FitRow], r2_threshold: float, cv_threshold: float):
    """
    For each (module, ctx_len, anchor_block), aggregate all windows and report:
      - median R²  (want ≥ r2_threshold)
      - max CV     (want ≤ cv_threshold)
      - median slope  ms/block
      - worst-case max_abs_residual_ms
    """
    import statistics

    key = lambda r: (r.module, r.ctx_len, r.anchor_block)
    print(f"\n{'='*100}")
    print(f"  Local linearity summary   (R² ≥ {r2_threshold:.2f}  AND  CV ≤ {cv_threshold:.2f} → PASS)")
    print(f"{'='*100}")
    hdr = (f"{'module':<15} {'ctx':>6} {'anchor':>7}  "
           f"{'med_R²':>8} {'min_R²':>8} {'med_slope(ms/blk)':>18} "
           f"{'max_CV':>8} {'max_resid_ms':>13}  {'verdict':>7}")
    print(hdr)
    print("-" * 100)

    verdicts = []
    for (mod, ctx, anc), grp in groupby(fits, key):
        grp = list(grp)
        r2s    = [r.r2                  for r in grp]
        cvs    = [r.cv_marginal_delta   for r in grp]
        slopes = [r.slope_ms_per_block  for r in grp]
        resids = [r.max_abs_residual_ms for r in grp]

        med_r2    = statistics.median(r2s)
        min_r2    = min(r2s)
        max_cv    = max(cvs)
        med_slope = statistics.median(slopes)
        max_resid = max(resids)

        ok = (min_r2 >= r2_threshold) and (max_cv <= cv_threshold)
        verdict = "PASS" if ok else "FAIL"
        verdicts.append(ok)

        print(f"{mod:<15} {ctx:>6} {anc:>7}  "
              f"{med_r2:>8.4f} {min_r2:>8.4f} {med_slope:>18.5f} "
              f"{max_cv:>8.4f} {max_resid:>13.4f}  {verdict:>7}")

    pass_pct = 100 * sum(verdicts) / max(1, len(verdicts))
    print(f"\nOverall: {sum(verdicts)}/{len(verdicts)} configs passed ({pass_pct:.0f}%)")
    return verdicts


def slope_table(fits: list[FitRow]):
    """Print median slope (ms per replay block) organized as a table."""
    import statistics

    key = lambda r: (r.module, r.ctx_len, r.anchor_block)
    entries = {}
    for (mod, ctx, anc), grp in groupby(fits, key):
        slopes = [r.slope_ms_per_block for r in grp]
        entries[(mod, ctx, anc)] = statistics.median(slopes)

    modules = sorted({k[0] for k in entries})
    for mod in modules:
        print(f"\n  [{mod}]  slope (ms / replay-block)")
        ctx_vals = sorted({k[1] for k in entries if k[0] == mod})
        anc_vals = sorted({k[2] for k in entries if k[0] == mod})
        # header
        print(f"  {'anchor \\ ctx':>14} " + "".join(f"{c:>10}" for c in ctx_vals))
        for anc in anc_vals:
            row_str = f"  {'blk='+str(anc):>14} "
            for ctx in ctx_vals:
                v = entries.get((mod, ctx, anc))
                row_str += f"{v:>10.5f}" if v is not None else f"{'n/a':>10}"
            print(row_str)


def plot_delta_curves(
    raw: list[RawRow],
    fits: list[FitRow],
    out_dir: Path,
):
    """One subplot per (ctx_len, anchor_block): scatter delta_ms vs replay_blocks,
    overlay the median local-fit line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not found — skipping plots.")
        return

    import statistics
    from itertools import groupby as _gb

    modules = sorted({r.module for r in raw})
    for mod in modules:
        mod_raw  = [r for r in raw  if r.module == mod and r.replay_blocks > 0]
        mod_fits = [r for r in fits if r.module == mod]

        ctx_vals = sorted({r.ctx_len      for r in mod_raw})
        anc_vals = sorted({r.anchor_block for r in mod_raw})

        fig, axes = plt.subplots(
            len(anc_vals), len(ctx_vals),
            figsize=(4 * len(ctx_vals), 3.5 * len(anc_vals)),
            squeeze=False,
        )
        fig.suptitle(f"delta_ms vs replay_blocks  [{mod}]", fontsize=13)

        for ci, ctx in enumerate(ctx_vals):
            for ai, anc in enumerate(anc_vals):
                ax = axes[ai][ci]
                pts = [r for r in mod_raw if r.ctx_len == ctx and r.anchor_block == anc]
                if not pts:
                    ax.set_visible(False)
                    continue
                pts.sort(key=lambda r: r.replay_blocks)
                xs = np.array([r.replay_blocks for r in pts])
                ys = np.array([r.delta_ms      for r in pts])
                ax.scatter(xs, ys, s=10, alpha=0.7, label="measured")

                # median fit line
                fpts = [f for f in mod_fits if f.ctx_len == ctx and f.anchor_block == anc]
                if fpts:
                    slope = statistics.median(f.slope_ms_per_block for f in fpts)
                    intercept = statistics.median(f.intercept_ms for f in fpts)
                    x_fit = np.linspace(xs.min(), xs.max(), 200)
                    ax.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5,
                            label=f"slope={slope:.4f}")

                ax.set_title(f"ctx={ctx} anchor_blk={anc}", fontsize=9)
                ax.set_xlabel("replay_blocks")
                ax.set_ylabel("delta_ms")
                ax.legend(fontsize=7)

        plt.tight_layout()
        fig_path = out_dir / f"delta_curves_{mod}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"Saved plot → {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir  = Path(args.output_dir)
    raw_path = out_dir / "raw.csv"
    fit_path = out_dir / "local_fits.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run the benchmark first.")
    if not fit_path.exists():
        raise FileNotFoundError(f"{fit_path} not found. Run the benchmark first.")

    raw  = load_raw_csv(raw_path)
    fits = load_fit_csv(fit_path)

    print(f"\nLoaded {len(raw)} raw rows, {len(fits)} fit windows")
    print(f"Modules  : {sorted({r.module for r in raw})}")
    print(f"ctx_lens : {sorted({r.ctx_len for r in raw})}")
    print(f"anchors  : {sorted({r.anchor_block for r in raw})}")

    analyze_fits(fits, r2_threshold=args.r2_threshold, cv_threshold=args.cv_threshold)

    print(f"\n{'='*60}")
    print("  Slope table (median ms / replay-block)")
    print(f"{'='*60}")
    slope_table(fits)

    if args.plot:
        plot_delta_curves(raw, fits, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze replay-linearity benchmark results."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing raw.csv and local_fits.csv.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.95,
        help="Minimum acceptable R² for a config to PASS (default: 0.95).",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.20,
        help="Maximum acceptable CV (std/mean of marginal deltas) to PASS (default: 0.20).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate delta_curves_<module>.png plots in --output-dir.",
    )
    args = parser.parse_args()
    main(args)
