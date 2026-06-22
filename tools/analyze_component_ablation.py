#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize no-throttle RunKV single-disable component ablations."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_per_layer_timing import (  # noqa: E402
    extract_inference_loop_seconds,
    extract_kernel_timing_from_sqlite,
    load_nvtx,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASELINES: dict[str, dict[str, str]] = {
    "1k": {
        "manifest": "exp_results/manifests/runkv_20260525_0143.json",
        "summary": "exp_results/analysis/per_layer/20260525_0143-3080-2.7b-1k-v1.1/per_layer_summary.txt",
        "sqlite": "exp_results/sqlite/runkv_20260525_0143.sqlite",
    },
    "2k": {
        "manifest": "exp_results/manifests/runkv_20260525_0225.json",
        "summary": "exp_results/analysis/per_layer/20260525_0225-3080-2.7b-2k-v1.1/per_layer_summary.txt",
        "sqlite": "exp_results/sqlite/runkv_20260525_0225.sqlite",
    },
    "4k": {
        "manifest": "exp_results/manifests/runkv_20260525_1634.json",
        "summary": "exp_results/analysis/per_layer/20260525_1634-3080-2.7b-4k-v1.1/per_layer_summary.txt",
        "sqlite": "exp_results/sqlite/runkv_20260525_1634.sqlite",
    },
    "8k": {
        "manifest": "exp_results/manifests/runkv_20260526_0136.json",
        "summary": "exp_results/analysis/per_layer/20260526_0136-3080-2.7b-8k-v1.1/per_layer_summary.txt",
        "sqlite": "exp_results/sqlite/runkv_20260526_0136.sqlite",
    },
}
VARIANTS = ["full", "no_async_plan", "no_segment_dma", "no_state_machine"]
MATCH_KEYS = [
    "model",
    "prefix_blocks",
    "num_prompts",
    "prompt_words",
    "max_tokens",
    "gpu_memory_fraction",
    "gpu_memory_utilization",
    "num_device_buffers",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "planner",
]
FIELDS = [
    "workload",
    "variant",
    "config_group",
    "num_prompts",
    "prompt_words",
    "max_tokens",
    "decode_tokens",
    "gpu_memory_fraction",
    "gpu_memory_utilization",
    "baseline_config_match",
    "valid_comparison",
    "comparison_note",
    "decode_duration_s",
    "decode_tokens_per_s",
    "delta_duration_s",
    "delta_decode_tokens_per_s",
    "delta_decode_tokens_per_s_pct",
    "mean_abs_imbalance_ms",
    "p95_abs_imbalance_ms",
    "mean_replay_tokens",
    "mean_replay_ratio",
    "avg_gpu_bubble_ms_per_layer",
    "prehook_total_ms",
    "prehook_build_plan_ms",
    "prehook_sync_plan_ms",
    "schedule_io_ms",
    "h2d_dma_enqueue_ms",
    "manifest",
]
LONG_METRICS = {
    "decode_duration_s": ("Decode duration", "s"),
    "decode_tokens_per_s": ("Decode throughput", "tokens/s"),
    "mean_abs_imbalance_ms": ("Mean absolute imbalance", "ms"),
    "p95_abs_imbalance_ms": ("P95 absolute imbalance", "ms"),
    "mean_replay_tokens": ("Mean replay tokens", "tokens"),
    "mean_replay_ratio": ("Mean replay ratio", "ratio"),
    "avg_gpu_bubble_ms_per_layer": ("Average GPU bubble per layer", "ms"),
    "prehook_total_ms": ("Prehook total", "ms"),
    "prehook_build_plan_ms": ("Prehook build-plan total", "ms"),
    "prehook_sync_plan_ms": ("Synchronous full-plan build total", "ms"),
    "schedule_io_ms": ("Schedule IO total", "ms"),
    "h2d_dma_enqueue_ms": ("H2D DMA enqueue total", "ms"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "exp_results/ablation/no_throttle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exp_results/analysis/component_ablation/no_throttle",
    )
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _add_config_columns(row: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key in (
        "num_prompts",
        "prompt_words",
        "max_tokens",
        "gpu_memory_fraction",
        "gpu_memory_utilization",
    ):
        row[key] = manifest.get(key)
    row["decode_tokens"] = int(manifest["num_prompts"]) * int(manifest["max_tokens"])
    row["config_group"] = (
        f"{manifest['prompt_words']}-"
        f"{manifest['num_prompts']}x{manifest['max_tokens']}-"
        f"f{manifest['gpu_memory_fraction']}"
    )


def _flat_metrics(path_pattern: str | None) -> dict[str, float | None]:
    paths = sorted(glob.glob(path_pattern or ""))
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    imbalance = [abs(float(r["imbalance_ms"])) for r in records if r.get("imbalance_ms") is not None]
    replay_tokens = [float(r["replay_token_count"]) for r in records if r.get("replay_token_count") is not None]
    replay_ratio = [float(r["replay_ratio"]) for r in records if r.get("replay_ratio") is not None]
    if imbalance:
        ordered = sorted(imbalance)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    else:
        p95 = None
    return {
        "mean_abs_imbalance_ms": statistics.fmean(imbalance) if imbalance else None,
        "p95_abs_imbalance_ms": p95,
        "mean_replay_tokens": statistics.fmean(replay_tokens) if replay_tokens else None,
        "mean_replay_ratio": statistics.fmean(replay_ratio) if replay_ratio else None,
    }


def _prehook_metrics(run_dir: Path) -> dict[str, float | None]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "prehook_timing").glob("prehook_timing_*.jsonl")):
        with path.open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    keys = {
        "prehook_total_ms": "total_ms",
        "prehook_build_plan_ms": "build_plan_ms",
        "prehook_sync_plan_ms": "bp_sync_plan_ms",
        "schedule_io_ms": "schedule_io_ms",
    }
    result: dict[str, float | None] = {
        out: sum(float(r.get(src, 0.0)) for r in records) if records else None
        for out, src in keys.items()
    }
    result["h2d_dma_enqueue_ms"] = (
        sum(
            float(r.get("sio_mseg_dma_ms", 0.0))
            + float(r.get("sio_gather_dma_ms", 0.0))
            for r in records
        )
        if records
        else None
    )
    return result


def _sqlite_metrics(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"decode_duration_s": None, "avg_gpu_bubble_ms_per_layer": None}
    duration = extract_inference_loop_seconds(load_nvtx(str(path)))
    layer_timing = extract_kernel_timing_from_sqlite(str(path))
    bubbles = [
        value
        for metrics in layer_timing.values()
        for value in metrics.get("gpu_bubble", [])
    ]
    return {
        "decode_duration_s": duration,
        "avg_gpu_bubble_ms_per_layer": (
            statistics.fmean(bubbles) if bubbles else None
        ),
    }


def _baseline_row(workload: str) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = BASELINES[workload]
    manifest_path = ROOT / paths["manifest"]
    manifest = _load_json(manifest_path)
    summary = (ROOT / paths["summary"]).read_text()
    timing_match = re.search(r"^RunKV\s+([0-9.]+)\s+([0-9.]+)", summary, re.MULTILINE)
    abs_match = re.search(r"Absolute avg \|imbalance\|: RunKV=([0-9.]+)ms", summary)
    bubble_match = re.search(r"RunKV:\s+avg kernel_active=.*?avg bubble=([0-9.]+)ms", summary)
    if timing_match is None:
        raise RuntimeError(f"Cannot parse baseline throughput: {paths['summary']}")
    row: dict[str, Any] = {
        "workload": workload,
        "variant": "full",
        "baseline_config_match": True,
        "valid_comparison": True,
        "comparison_note": "historical full reference",
        "decode_duration_s": float(timing_match.group(1)),
        "decode_tokens_per_s": float(timing_match.group(2)),
        "mean_abs_imbalance_ms": float(abs_match.group(1)) if abs_match else None,
        "avg_gpu_bubble_ms_per_layer": (
            float(bubble_match.group(1)) if bubble_match else None
        ),
        "manifest": str(manifest_path),
    }
    _add_config_columns(row, manifest)
    row.update(_flat_metrics(manifest.get("mfu_flat_jsonl_glob")))
    if abs_match:
        row["mean_abs_imbalance_ms"] = float(abs_match.group(1))
    return row, manifest


def _latest_variant_manifest(
    input_root: Path, workload: str, variant: str, include_smoke: bool
) -> Path | None:
    candidates = []
    for path in (input_root / workload / variant).glob("**/manifest.json"):
        manifest = _load_json(path)
        if include_smoke or not manifest.get("smoke", False):
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _variant_row(
    workload: str,
    variant: str,
    path: Path,
    baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(path)
    mismatches = [
        key
        for key in MATCH_KEYS
        if str(manifest.get(key)) != str(baseline_manifest.get(key))
    ]
    sqlite_path = Path(manifest.get("sqlite", path.parent / "runkv.sqlite"))
    row: dict[str, Any] = {
        "workload": workload,
        "variant": variant,
        "baseline_config_match": not mismatches,
        "valid_comparison": not mismatches
        and manifest.get("resource_pressure_kind") == "none",
        "comparison_note": (
            "matches historical full"
            if not mismatches
            else f"historical full differs: {', '.join(mismatches)}"
        ),
        "manifest": str(path),
    }
    _add_config_columns(row, manifest)
    row.update(_sqlite_metrics(sqlite_path))
    decode_tokens = int(manifest["num_prompts"]) * int(manifest["max_tokens"])
    duration = row["decode_duration_s"]
    row["decode_tokens_per_s"] = decode_tokens / duration if duration else None
    row.update(_flat_metrics(manifest.get("mfu_flat_jsonl_glob")))
    row.update(_prehook_metrics(path.parent))
    if mismatches:
        row["validation_error"] = f"baseline mismatch: {', '.join(mismatches)}"
    return row


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# No-Throttle Component Ablation",
        "",
        "| Workload | Variant | Valid | Duration (s) | Decode tok/s | Delta tok/s | Delta % | Mean |imb| (ms) | P95 |imb| (ms) | Prehook (ms) | H2D enqueue (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {variant} | {valid} | {dur} | {tps} | {delta} | "
            "{pct} | {mean} | {p95} | {prehook} | {h2d} |".format(
                workload=row["workload"],
                variant=row["variant"],
                valid=_fmt(row.get("valid_comparison")),
                dur=_fmt(row.get("decode_duration_s")),
                tps=_fmt(row.get("decode_tokens_per_s")),
                delta=_fmt(row.get("delta_decode_tokens_per_s")),
                pct=_fmt(row.get("delta_decode_tokens_per_s_pct")),
                mean=_fmt(row.get("mean_abs_imbalance_ms")),
                p95=_fmt(row.get("p95_abs_imbalance_ms")),
                prehook=_fmt(row.get("prehook_total_ms")),
                h2d=_fmt(row.get("h2d_dma_enqueue_ms")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_long_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "workload",
        "variant",
        "config_group",
        "num_prompts",
        "prompt_words",
        "max_tokens",
        "gpu_memory_fraction",
        "gpu_memory_utilization",
        "baseline_config_match",
        "valid_comparison",
        "metric",
        "metric_label",
        "unit",
        "value",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for metric, (label, unit) in LONG_METRICS.items():
                value = row.get(metric)
                if value is None:
                    continue
                writer.writerow(
                    {
                        "workload": row["workload"],
                        "variant": row["variant"],
                        "config_group": row["config_group"],
                        "num_prompts": row["num_prompts"],
                        "prompt_words": row["prompt_words"],
                        "max_tokens": row["max_tokens"],
                        "gpu_memory_fraction": row["gpu_memory_fraction"],
                        "gpu_memory_utilization": row["gpu_memory_utilization"],
                        "baseline_config_match": row["baseline_config_match"],
                        "valid_comparison": row["valid_comparison"],
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "value": value,
                    }
                )


def _plot_workload(rows: list[dict[str, Any]], workload: str, out_dir: Path) -> None:
    if not HAS_MPL:
        return
    selected = [r for r in rows if r["workload"] == workload and r.get("decode_tokens_per_s") is not None]
    if not selected:
        return
    labels = [r["variant"] for r in selected]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [r["decode_tokens_per_s"] for r in selected])
    axes[0].set_ylabel("decode tokens/s")
    axes[0].set_title(f"{workload} throughput")
    axes[1].bar(labels, [r["decode_duration_s"] for r in selected])
    axes[1].set_ylabel("seconds")
    axes[1].set_title(f"{workload} decode duration")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{workload}_throughput_latency.png", dpi=180)
    plt.close(fig)


def _plot_mechanisms(rows: list[dict[str, Any]], workload: str, out_dir: Path) -> None:
    if not HAS_MPL:
        return
    selected = [r for r in rows if r["workload"] == workload]
    metrics = [
        ("mean_abs_imbalance_ms", "mean |imbalance| (ms)"),
        ("prehook_build_plan_ms", "prehook plan build total (ms)"),
        ("h2d_dma_enqueue_ms", "H2D enqueue total (ms)"),
        ("avg_gpu_bubble_ms_per_layer", "GPU bubble/layer (ms)"),
    ]
    populated = [
        (field, title)
        for field, title in metrics
        if any(row.get(field) is not None for row in selected)
    ]
    if not populated:
        return
    fig, axes = plt.subplots(1, len(populated), figsize=(5 * len(populated), 4))
    if len(populated) == 1:
        axes = [axes]
    labels = [row["variant"] for row in selected]
    for axis, (field, title) in zip(axes, populated):
        axis.bar(labels, [row.get(field) or 0.0 for row in selected])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"{workload} mechanism metrics")
    fig.tight_layout()
    fig.savefig(out_dir / f"{workload}_mechanisms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for workload in BASELINES:
        baseline, baseline_manifest = _baseline_row(workload)
        rows.append(baseline)
        for variant in VARIANTS[1:]:
            manifest_path = _latest_variant_manifest(
                args.input_root, workload, variant, args.include_smoke
            )
            if manifest_path is None:
                continue
            rows.append(_variant_row(workload, variant, manifest_path, baseline_manifest))
        full_tps = baseline["decode_tokens_per_s"]
        full_duration = baseline["decode_duration_s"]
        for row in rows:
            if row["workload"] != workload or row["variant"] == "full":
                continue
            if row.get("decode_tokens_per_s") is not None:
                row["delta_decode_tokens_per_s"] = row["decode_tokens_per_s"] - full_tps
                row["delta_decode_tokens_per_s_pct"] = (
                    row["delta_decode_tokens_per_s"] / full_tps * 100
                )
            if row.get("decode_duration_s") is not None:
                row["delta_duration_s"] = row["decode_duration_s"] - full_duration

    csv_path = args.output_dir / "component_ablation_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.output_dir / "component_ablation_summary.md"
    _write_markdown(rows, md_path)
    long_path = args.output_dir / "component_ablation_metrics_long.csv"
    _write_long_metrics(rows, long_path)
    for workload in BASELINES:
        _plot_workload(rows, workload, args.output_dir)
        _plot_mechanisms(rows, workload, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
