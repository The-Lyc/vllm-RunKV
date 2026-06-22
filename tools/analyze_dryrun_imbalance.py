#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze per-step imbalance mean/stdev from RunKV dryrun flat.jsonl files.

For each decode step the flat.jsonl contains one record per layer.  This script
computes, within each step, the mean and standard-deviation of imbalance_ms
across all layers, then aggregates per IO-pressure stage.  For constant-resource
dryrun experiments, use ``--only-modal-step-shape`` to retain the dominant
step shape in each run.

Typical usage
-------------
Single run:

    python tools/analyze_dryrun_imbalance.py \\
        --input exp_results/staged_offline_pilot/io_multistage_step/\\
                a800_opt30b_<TAG>/runkv_dryrun/r0/*.flat.jsonl

Compare 1k vs 4k context (two --input groups, each with a label):

    python tools/analyze_dryrun_imbalance.py \\
        --input exp_results/.../runkv_dryrun/r0/*.flat.jsonl --label 1k \\
        --input exp_results/.../runkv_dryrun/r0/*.flat.jsonl --label 4k \\
        --output-dir exp_results/analysis/dryrun_imbalance/<TAG>

Output
------
* Console table: per-stage mean±stdev summary for every input group.
* Per-step CSV: step, stage, mean_imbalance_ms, stdev_imbalance_ms.
* Optional per-context mean/stdev box plots (requires matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(p))
    return paths


def _load_flat_jsonl(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Per-step stats
# ─────────────────────────────────────────────────────────────────────────────

def _compute_per_step_stats(
    records: list[dict],
    skip_warmup_steps: int = 1,
    drop_boundary_imbalances: bool = False,
) -> list[dict]:
    """Return one row per step with mean/stdev of imbalance across layers."""
    # collect (layer_idx, imbalance_ms) per step
    step_data: dict[int, dict] = {}
    for r in records:
        imb = r.get("imbalance_ms")
        if imb is None:
            continue
        step = r["step"]
        if step not in step_data:
            step_data[step] = {
                "step": step,
                "resource_stage": r.get("resource_stage", "?"),
                "resource_stage_id": r.get("resource_stage_id", -1),
                "resource_requested_target": r.get("resource_requested_target"),
                "resource_target": r.get("resource_target"),
                "resource_target_unit": r.get("resource_target_unit", ""),
                "num_reqs": r.get("num_reqs"),
                "total_scheduled_tokens": r.get("total_scheduled_tokens"),
                "imbalances": [],
            }
        step_data[step]["imbalances"].append((r.get("layer_idx", -1), float(imb)))

    rows = []
    for step in sorted(step_data):
        if step < skip_warmup_steps:
            continue
        d = step_data[step]
        vals = [value for _, value in sorted(d["imbalances"])]
        if drop_boundary_imbalances:
            vals = vals[1:-1]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        stdev = statistics.stdev(vals) if n >= 2 else 0.0
        rows.append(
            {
                "step": step,
                "n_layers": n,
                "resource_stage": d["resource_stage"],
                "resource_stage_id": d["resource_stage_id"],
                "resource_requested_target": d["resource_requested_target"],
                "resource_target": d["resource_target"],
                "resource_target_unit": d["resource_target_unit"],
                "num_reqs": d["num_reqs"],
                "total_scheduled_tokens": d["total_scheduled_tokens"],
                "mean_imbalance_ms": mean,
                "stdev_imbalance_ms": stdev,
                "abs_mean_imbalance_ms": abs(mean),
            }
        )
    return rows


def _stage_summary(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-step rows into per-stage summary stats."""
    stage_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        stage_rows[r["resource_stage"]].append(r)

    summary: dict[str, dict] = {}
    for stage in sorted(stage_rows, key=lambda s: stage_rows[s][0]["resource_stage_id"]):
        srows = stage_rows[stage]
        means = [r["mean_imbalance_ms"] for r in srows]
        stdevs = [r["stdev_imbalance_ms"] for r in srows]
        abs_means = [r["abs_mean_imbalance_ms"] for r in srows]
        target = srows[0].get("resource_requested_target")
        unit = srows[0].get("resource_target_unit", "")
        summary[stage] = {
            "stage": stage,
            "n_steps": len(srows),
            "resource_target": f"{target} {unit}".strip() if target is not None else "?",
            # mean of per-step means
            "mean_of_means": sum(means) / len(means),
            "stdev_of_means": statistics.stdev(means) if len(means) >= 2 else 0.0,
            # mean of per-step stdevs
            "mean_of_stdevs": sum(stdevs) / len(stdevs),
            "stdev_of_stdevs": statistics.stdev(stdevs) if len(stdevs) >= 2 else 0.0,
            # absolute-value summary
            "mean_of_abs_means": sum(abs_means) / len(abs_means),
        }
    return summary


def _only_modal_step_shape(rows: list[dict]) -> tuple[list[dict], tuple[object, object] | None]:
    """Keep the dominant workload shape in the run."""
    shapes = Counter(
        (r["num_reqs"], r["total_scheduled_tokens"])
        for r in rows
        if r["num_reqs"] is not None and r["total_scheduled_tokens"] is not None
    )
    if not shapes:
        return rows, None

    shape, _ = shapes.most_common(1)[0]
    return [
        r
        for r in rows
        if (r["num_reqs"], r["total_scheduled_tokens"]) == shape
    ], shape


def _only_step_shape(rows: list[dict], shape: tuple[int, int]) -> list[dict]:
    """Keep an explicitly selected workload shape in the run."""
    return [
        r
        for r in rows
        if (r["num_reqs"], r["total_scheduled_tokens"]) == shape
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{decimals}f}"


def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = _col_widths(headers, rows)
    sep = "  ".join("-" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def _print_stage_summary(label: str, summary: dict[str, dict]) -> None:
    print(f"\n── {label} ─────────────────────────────────────────────────────────")
    headers = [
        "stage",
        "n_steps",
        "io_target",
        "mean(mean_imb)",
        "stdev(mean_imb)",
        "mean(stdev_imb)",
        "stdev(stdev_imb)",
        "|mean| avg",
    ]
    rows = []
    for s in summary.values():
        rows.append([
            s["stage"],
            str(s["n_steps"]),
            str(s["resource_target"]),
            _fmt(s["mean_of_means"]),
            _fmt(s["stdev_of_means"]),
            _fmt(s["mean_of_stdevs"]),
            _fmt(s["stdev_of_stdevs"]),
            _fmt(s["mean_of_abs_means"]),
        ])
    _print_table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "step",
        "n_layers",
        "resource_stage",
        "resource_requested_target",
        "resource_target_unit",
        "num_reqs",
        "total_scheduled_tokens",
        "mean_imbalance_ms",
        "stdev_imbalance_ms",
        "abs_mean_imbalance_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({"label": label, **r})
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_STAGE_COLORS = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]


def _plot_per_step(
    all_rows: list[tuple[str, list[dict]]],
    out_dir: Path,
) -> None:
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    ax_mean, ax_stdev = axes

    for idx, (label, rows) in enumerate(all_rows):
        color = _COLORS[idx % len(_COLORS)]
        steps = [r["step"] for r in rows]
        means = [r["mean_imbalance_ms"] for r in rows]
        stdevs = [r["stdev_imbalance_ms"] for r in rows]

        ax_mean.plot(steps, means, color=color, linewidth=1.0, label=label)
        ax_stdev.plot(steps, stdevs, color=color, linewidth=1.0, label=label)

        # shade stage regions on first group
        if idx == 0:
            _shade_stages(ax_mean, rows)
            _shade_stages(ax_stdev, rows)

    ax_mean.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_mean.set_ylabel("mean imbalance_ms (across layers)")
    ax_mean.set_title("Per-step mean imbalance (negative = IO has slack; positive = GPU stalls)")
    ax_mean.legend(fontsize=8)

    ax_stdev.set_ylabel("stdev imbalance_ms (across layers)")
    ax_stdev.set_xlabel("decode step")
    ax_stdev.set_title("Per-step stdev of imbalance (higher = larger spread across layers)")
    ax_stdev.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "dryrun_imbalance_per_step.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot → {out_path}")


def _shade_stages(ax, rows: list[dict]) -> None:
    """Draw light background bands for each stage."""
    from itertools import groupby

    for stage_id, group in groupby(rows, key=lambda r: r["resource_stage_id"]):
        grp = list(group)
        x0 = grp[0]["step"] - 0.5
        x1 = grp[-1]["step"] + 0.5
        color = _STAGE_COLORS[stage_id % len(_STAGE_COLORS)]
        ax.axvspan(x0, x1, alpha=0.15, color=color, label=f"_{grp[0]['resource_stage']}")

    # add stage text labels once
    seen: set = set()
    for r in rows:
        stage = r["resource_stage"]
        if stage not in seen:
            seen.add(stage)
            ax.text(
                r["step"],
                ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] != 0 else 1,
                stage,
                fontsize=7,
                color="grey",
            )


def _plot_stage_box(
    all_rows: list[tuple[str, list[dict]]],
    out_dir: Path,
) -> None:
    """Box plot of per-step stdev grouped by stage, one subplot per input label."""
    if not HAS_MPL:
        return

    n = len(all_rows)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax_idx, (label, rows) in enumerate(all_rows):
        ax = axes[0][ax_idx]
        stage_stdevs: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            stage_stdevs[r["resource_stage"]].append(r["stdev_imbalance_ms"])

        stages = sorted(stage_stdevs, key=lambda s: rows[[r["resource_stage"] for r in rows].index(s)]["resource_stage_id"])
        data = [stage_stdevs[s] for s in stages]
        bp = ax.boxplot(data, labels=stages, patch_artist=True)
        for patch, color in zip(bp["boxes"], _STAGE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{label}\nstdev(imbalance_ms) by stage")
        ax.set_ylabel("stdev imbalance_ms (ms)")
        ax.set_xlabel("IO stage")

    fig.tight_layout()
    out_path = out_dir / "dryrun_imbalance_stdev_box.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot → {out_path}")


def _plot_context_box(
    all_rows: list[tuple[str, list[dict]]],
    out_dir: Path,
) -> None:
    """Box plots comparing per-step imbalance summaries across contexts."""
    if not HAS_MPL:
        return

    labels = [label for label, _ in all_rows]
    mean_data = [[r["mean_imbalance_ms"] for r in rows] for _, rows in all_rows]
    stdev_data = [[r["stdev_imbalance_ms"] for r in rows] for _, rows in all_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, ylabel in (
        (
            axes[0],
            mean_data,
            "Per-step mean imbalance by context",
            "mean imbalance_ms (across layers)",
        ),
        (
            axes[1],
            stdev_data,
            "Per-step stdev of imbalance by context",
            "stdev imbalance_ms (across layers)",
        ),
    ):
        boxes = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
        for patch, color in zip(boxes["boxes"], _COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_xlabel("context")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "dryrun_imbalance_context_box.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

class _MultiAction(argparse.Action):
    """Collect (patterns, label) pairs from repeated --input / --label args."""

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(values)
        setattr(namespace, self.dest, items)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        dest="inputs",
        metavar="GLOB_OR_PATH",
        action=_MultiAction,
        nargs="+",
        required=True,
        help=(
            "One or more flat.jsonl paths / globs for one experiment group. "
            "Repeat --input (and optionally --label) for additional groups."
        ),
    )
    p.add_argument(
        "--label",
        dest="labels",
        metavar="LABEL",
        action="append",
        default=[],
        help="Display label for the preceding --input group (e.g. '1k', '4k').",
    )
    p.add_argument(
        "--output-dir",
        default="exp_results/analysis/dryrun_imbalance",
        help="Directory for CSV and plot outputs.",
    )
    p.add_argument(
        "--skip-warmup-steps",
        type=int,
        default=1,
        help="Number of initial steps to discard (default: 1).",
    )
    p.add_argument(
        "--only-modal-step-shape",
        action="store_true",
        help=(
            "Only retain the most common (num_reqs, total_scheduled_tokens) "
            "shape in each input group."
        ),
    )
    p.add_argument(
        "--step-shape",
        action="append",
        default=[],
        metavar="LABEL=NUM_REQS,TOKENS",
        help=(
            "Override the retained step shape for one input label, for example "
            "'2k-32x128=32,32'. Overrides --only-modal-step-shape for that label."
        ),
    )
    p.add_argument(
        "--drop-boundary-imbalances",
        action="store_true",
        help=(
            "Exclude the first and last layer-ordered imbalance sample from "
            "each step before calculating mean/stdev."
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation even if matplotlib is available.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pad labels
    labels: list[str] = list(args.labels)
    while len(labels) < len(args.inputs):
        labels.append(f"run{len(labels) + 1}")

    shape_overrides: dict[str, tuple[int, int]] = {}
    for spec in args.step_shape:
        try:
            label, values = spec.rsplit("=", 1)
            num_reqs, tokens = values.split(",", 1)
            shape_overrides[label] = (int(num_reqs), int(tokens))
        except ValueError:
            raise SystemExit(
                f"[ERROR] invalid --step-shape {spec!r}; expected LABEL=NUM_REQS,TOKENS"
            )

    all_rows: list[tuple[str, list[dict]]] = []

    print(f"\n{'=' * 72}")
    print("  Dryrun Imbalance Analysis")
    print(f"{'=' * 72}")

    for label, patterns in zip(labels, args.inputs):
        paths = _expand(patterns)
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[WARN] {label}: files not found: {missing}", file=sys.stderr)
            paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"[ERROR] {label}: no valid input files", file=sys.stderr)
            continue

        print(f"\n── Loading {label} ({len(paths)} file(s)) ──")
        for p in paths:
            print(f"   {p}")

        records = _load_flat_jsonl(paths)
        rows = _compute_per_step_stats(
            records,
            skip_warmup_steps=args.skip_warmup_steps,
            drop_boundary_imbalances=args.drop_boundary_imbalances,
        )
        if label in shape_overrides:
            unfiltered_count = len(rows)
            shape = shape_overrides[label]
            rows = _only_step_shape(rows, shape)
            print(
                "  selected step shape: "
                f"num_reqs={shape[0]}, total_scheduled_tokens={shape[1]} "
                f"({len(rows)}/{unfiltered_count} steps retained)"
            )
        elif args.only_modal_step_shape:
            unfiltered_count = len(rows)
            rows, shape = _only_modal_step_shape(rows)
            if shape is None:
                print("  [WARN] step-shape fields unavailable; retaining all steps")
            else:
                print(
                    "  fixed step shape: "
                    f"num_reqs={shape[0]}, total_scheduled_tokens={shape[1]} "
                    f"({len(rows)}/{unfiltered_count} steps retained)"
                )
        all_rows.append((label, rows))

        summary = _stage_summary(rows)
        _print_stage_summary(label, summary)

        # Per-step CSV
        _write_csv(rows, out_dir / f"per_step_{label}.csv", label=label)

    if not all_rows:
        print("[ERROR] No data loaded.", file=sys.stderr)
        sys.exit(1)

    # Combined CSV
    combined_path = out_dir / "per_step_combined.csv"
    all_flat: list[dict] = []
    for label, rows in all_rows:
        for r in rows:
            all_flat.append({"label": label, **r})
    fieldnames = [
        "label", "step", "n_layers", "resource_stage",
        "resource_requested_target", "resource_target_unit",
        "num_reqs", "total_scheduled_tokens",
        "mean_imbalance_ms", "stdev_imbalance_ms", "abs_mean_imbalance_ms",
    ]
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_flat)
    print(f"  combined CSV → {combined_path}")

    if not args.no_plot:
        if HAS_MPL:
            _plot_per_step(all_rows, out_dir)
            _plot_stage_box(all_rows, out_dir)
            _plot_context_box(all_rows, out_dir)
        else:
            print("[WARN] matplotlib not available; skipping plots.")

    print(f"\n{'=' * 72}")
    print(f"  Output dir: {out_dir}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
