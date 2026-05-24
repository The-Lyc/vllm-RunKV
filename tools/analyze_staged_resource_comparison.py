#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze runner-controlled staged-resource RunKV vs TightLLM experiments.

This analysis intentionally does not require Nsight Systems.  It joins:

* resource_steps*.jsonl: one record per engine.step() from the resource controller
* opt_component_mfu*.jsonl / .flat.jsonl: per-step and per-layer RunKV timing records
* pressure*.csv: resource pressure worker target/actual samples

The output focuses on the staged-resource metrics used for the formal
comparison: S2 throughput deficit, S2 latency inflation, and S3 overshoot checks.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class RunInputs:
    system: str
    run_dir: Path | None
    step_logs: list[Path]
    mfu_steps: list[Path]
    mfu_flat: list[Path]
    pressure_logs: list[Path]
    sqlite_paths: list[Path] = field(default_factory=list)


def _expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(paths)


def _discover_run_inputs(system: str, run_dirs: list[str]) -> list[RunInputs]:
    runs: list[RunInputs] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        runs.append(
            RunInputs(
                system=system,
                run_dir=run_dir,
                step_logs=sorted(run_dir.glob("resource_steps*.jsonl")),
                mfu_steps=sorted(
                    path
                    for path in run_dir.glob("opt_component_mfu_*.jsonl")
                    if ".flat." not in path.name
                ),
                mfu_flat=sorted(run_dir.glob("opt_component_mfu_*.flat.jsonl")),
                pressure_logs=sorted(run_dir.glob("pressure*.csv")),
                sqlite_paths=sorted(run_dir.glob("*.sqlite")),
            )
        )
    return runs


def _manual_run_inputs(
    system: str,
    *,
    step_logs: list[str],
    mfu_steps: list[str],
    mfu_flat: list[str],
    pressure_logs: list[str],
) -> list[RunInputs]:
    if not (step_logs or mfu_steps or mfu_flat or pressure_logs):
        return []
    return [
        RunInputs(
            system=system,
            run_dir=None,
            step_logs=_expand_patterns(step_logs),
            mfu_steps=_expand_patterns(mfu_steps),
            mfu_flat=_expand_patterns(mfu_flat),
            pressure_logs=_expand_patterns(pressure_logs),
        )
    ]


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"  [warn] missing JSONL: {path}", file=sys.stderr)
            continue
        with path.open() as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _load_pressure(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"  [warn] missing pressure CSV: {path}", file=sys.stderr)
            continue
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                records.append(dict(row))
    return records


def _stats(values: list[float]) -> dict[str, float]:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return {}
    n = len(clean)

    def pct(q: float) -> float:
        if n == 1:
            return clean[0]
        idx = min(n - 1, max(0, int(round((n - 1) * q))))
        return clean[idx]

    return {
        "n": float(n),
        "mean": sum(clean) / n,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": clean[0],
        "max": clean[-1],
    }


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_sort_key(stage: str) -> tuple[int, str]:
    if stage.startswith("S") and stage[1:].isdigit():
        return int(stage[1:]), stage
    return 10_000, stage


def _controller_budget_after(update: Any) -> float | None:
    if not isinstance(update, dict):
        return None
    for key in ("budget_after", "new_budget", "budget"):
        value = _safe_float(update.get(key))
        if value is not None:
            return value
    return None


def _build_step_layer_metrics(flat_records: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in flat_records:
        step = record.get("step")
        if step is None:
            continue
        grouped[int(step)].append(record)

    metrics: dict[int, dict[str, float]] = {}
    for step, records in grouped.items():
        imbalances = [
            float(record["imbalance_ms"])
            for record in records
            if record.get("imbalance_ms") is not None
        ]
        replay_tokens = [
            float(record.get("replay_token_count") or 0.0) for record in records
        ]
        actual_tokens = [
            float(record.get("num_actual_tokens") or 0.0) for record in records
        ]
        budgets = [
            value
            for value in (_controller_budget_after(record.get("controller_update")) for record in records)
            if value is not None
        ]
        replay_total = sum(replay_tokens)
        actual_total = sum(actual_tokens)
        entry: dict[str, float] = {
            "mean_imbalance_ms": _mean(imbalances) or 0.0,
            "mean_abs_imbalance_ms": _mean([abs(value) for value in imbalances]) or 0.0,
            "replay_token_count": replay_total,
            "num_actual_tokens": actual_total,
            "replay_ratio": replay_total / actual_total if actual_total > 0 else 0.0,
        }
        if budgets:
            entry["mean_budget_after"] = _mean(budgets) or 0.0
        metrics[step] = entry
    return metrics


def _load_mfu_step_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for record in _load_jsonl(paths):
        if "layers" not in record:
            continue
        step = record.get("step")
        if step is not None:
            by_step[int(step)] = record
    return by_step


def analyze_run(inputs: RunInputs, skip_warmup_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step_records = [
        record
        for record in _load_jsonl(inputs.step_logs)
        if int(record.get("resource_pressure_step", record.get("step", 0)) or 0)
        >= skip_warmup_steps
    ]
    mfu_steps = _load_mfu_step_records(inputs.mfu_steps)
    flat_metrics = _build_step_layer_metrics(_load_jsonl(inputs.mfu_flat))
    pressure_records = _load_pressure(inputs.pressure_logs)

    run_name = inputs.run_dir.name if inputs.run_dir is not None else inputs.system
    stage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in step_records:
        stage = str(record.get("resource_stage") or f"S{int(record.get('resource_stage_id', 0)) + 1}")
        stage_groups[stage].append(record)

    rows: list[dict[str, Any]] = []
    for stage in sorted(stage_groups, key=_stage_sort_key):
        records = stage_groups[stage]
        steps = [int(record.get("resource_pressure_step", 0)) for record in records]
        step_wall_ms = [
            float(record["step_wall_s"]) * 1000.0
            for record in records
            if record.get("step_wall_s") is not None
        ]
        total_wall_s = sum(float(record.get("step_wall_s") or 0.0) for record in records)
        total_scheduled_tokens = 0.0
        step_metric_records = []
        for step in steps:
            mfu_record = mfu_steps.get(step)
            if mfu_record is not None:
                total_scheduled_tokens += float(mfu_record.get("total_scheduled_tokens") or 0.0)
            if step in flat_metrics:
                step_metric_records.append(flat_metrics[step])

        latency_stats = _stats(step_wall_ms)
        throughput = total_scheduled_tokens / total_wall_s if total_wall_s > 0 else None
        row: dict[str, Any] = {
            "system": inputs.system,
            "run": run_name,
            "run_dir": str(inputs.run_dir) if inputs.run_dir is not None else "",
            "stage": stage,
            "step_count": len(records),
            "step_start": min(steps) if steps else None,
            "step_end": max(steps) if steps else None,
            "target": records[0].get("resource_target") if records else None,
            "target_unit": records[0].get("resource_target_unit") if records else None,
            "total_wall_s": total_wall_s,
            "total_scheduled_tokens": total_scheduled_tokens,
            "scheduled_tokens_per_s": throughput,
            "step_wall_mean_ms": latency_stats.get("mean"),
            "step_wall_p50_ms": latency_stats.get("p50"),
            "step_wall_p95_ms": latency_stats.get("p95"),
            "step_wall_p99_ms": latency_stats.get("p99"),
            "mean_imbalance_ms": _mean([m["mean_imbalance_ms"] for m in step_metric_records]),
            "mean_abs_imbalance_ms": _mean([m["mean_abs_imbalance_ms"] for m in step_metric_records]),
            "mean_replay_ratio": _mean([m["replay_ratio"] for m in step_metric_records]),
            "mean_replay_token_count": _mean([m["replay_token_count"] for m in step_metric_records]),
            "mean_budget_after": _mean([
                m["mean_budget_after"]
                for m in step_metric_records
                if "mean_budget_after" in m
            ]),
        }
        rows.append(row)

    pressure_targets = [
        _safe_float(record.get("target"))
        for record in pressure_records
        if _safe_float(record.get("target")) is not None
    ]
    pressure_actuals = [
        _safe_float(record.get("actual"))
        for record in pressure_records
        if _safe_float(record.get("actual")) is not None
    ]
    run_summary = {
        "system": inputs.system,
        "run": run_name,
        "run_dir": str(inputs.run_dir) if inputs.run_dir is not None else "",
        "step_log_count": len(step_records),
        "pressure_sample_count": len(pressure_records),
        "pressure_target_mean": _mean([v for v in pressure_targets if v is not None]),
        "pressure_actual_mean": _mean([v for v in pressure_actuals if v is not None]),
    }
    return rows, run_summary


def _aggregate_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["system"]), str(row["stage"]))].append(row)

    out: list[dict[str, Any]] = []
    for (system, stage), group in sorted(grouped.items(), key=lambda item: (item[0][0], _stage_sort_key(item[0][1]))):
        out.append(
            {
                "system": system,
                "stage": stage,
                "runs": len(group),
                "target": group[0].get("target"),
                "target_unit": group[0].get("target_unit"),
                "step_count_mean": _mean([float(row["step_count"]) for row in group]),
                "scheduled_tokens_per_s_mean": _mean([
                    float(row["scheduled_tokens_per_s"])
                    for row in group
                    if row.get("scheduled_tokens_per_s") is not None
                ]),
                "step_wall_mean_ms_mean": _mean([
                    float(row["step_wall_mean_ms"])
                    for row in group
                    if row.get("step_wall_mean_ms") is not None
                ]),
                "step_wall_p95_ms_mean": _mean([
                    float(row["step_wall_p95_ms"])
                    for row in group
                    if row.get("step_wall_p95_ms") is not None
                ]),
                "mean_imbalance_ms_mean": _mean([
                    float(row["mean_imbalance_ms"])
                    for row in group
                    if row.get("mean_imbalance_ms") is not None
                ]),
                "mean_abs_imbalance_ms_mean": _mean([
                    float(row["mean_abs_imbalance_ms"])
                    for row in group
                    if row.get("mean_abs_imbalance_ms") is not None
                ]),
                "mean_replay_ratio_mean": _mean([
                    float(row["mean_replay_ratio"])
                    for row in group
                    if row.get("mean_replay_ratio") is not None
                ]),
                "mean_budget_after_mean": _mean([
                    float(row["mean_budget_after"])
                    for row in group
                    if row.get("mean_budget_after") is not None
                ]),
            }
        )
    return out


def _stage_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["system"]), str(row["stage"])): row for row in rows}


def _derive_comparison_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _stage_lookup(stage_rows)
    systems = sorted({str(row["system"]) for row in stage_rows})
    out: list[dict[str, Any]] = []
    for system in systems:
        s1 = lookup.get((system, "S1"))
        s2 = lookup.get((system, "S2"))
        s3 = lookup.get((system, "S3"))
        if s1 and s2:
            s1_tps = s1.get("scheduled_tokens_per_s_mean")
            s2_tps = s2.get("scheduled_tokens_per_s_mean")
            s1_p95 = s1.get("step_wall_p95_ms_mean")
            s2_p95 = s2.get("step_wall_p95_ms_mean")
            out.append(
                {
                    "system": system,
                    "comparison": "S2_vs_S1",
                    "throughput_drop_pct": _pct_drop(s1_tps, s2_tps),
                    "latency_p95_inflation_pct": _pct_increase(s1_p95, s2_p95),
                    "mean_abs_imbalance_delta_ms": _delta(
                        s1.get("mean_abs_imbalance_ms_mean"),
                        s2.get("mean_abs_imbalance_ms_mean"),
                    ),
                    "replay_ratio_delta": _delta(
                        s1.get("mean_replay_ratio_mean"),
                        s2.get("mean_replay_ratio_mean"),
                    ),
                }
            )
        if s1 and s3:
            out.append(
                {
                    "system": system,
                    "comparison": "S3_vs_S1",
                    "throughput_drop_pct": _pct_drop(
                        s1.get("scheduled_tokens_per_s_mean"),
                        s3.get("scheduled_tokens_per_s_mean"),
                    ),
                    "latency_p95_inflation_pct": _pct_increase(
                        s1.get("step_wall_p95_ms_mean"),
                        s3.get("step_wall_p95_ms_mean"),
                    ),
                    "mean_abs_imbalance_delta_ms": _delta(
                        s1.get("mean_abs_imbalance_ms_mean"),
                        s3.get("mean_abs_imbalance_ms_mean"),
                    ),
                    "replay_ratio_delta": _delta(
                        s1.get("mean_replay_ratio_mean"),
                        s3.get("mean_replay_ratio_mean"),
                    ),
                }
            )
    return out


def _pct_drop(base: Any, value: Any) -> float | None:
    base_f = _safe_float(base)
    value_f = _safe_float(value)
    if base_f is None or value_f is None or base_f == 0:
        return None
    return (base_f - value_f) / base_f * 100.0


def _pct_increase(base: Any, value: Any) -> float | None:
    base_f = _safe_float(base)
    value_f = _safe_float(value)
    if base_f is None or value_f is None or base_f == 0:
        return None
    return (value_f - base_f) / base_f * 100.0


def _delta(base: Any, value: Any) -> float | None:
    base_f = _safe_float(base)
    value_f = _safe_float(value)
    if base_f is None or value_f is None:
        return None
    return value_f - base_f


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any, digits: int = 3) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.{digits}f}"


def _write_summary_md(
    path: Path,
    *,
    run_summaries: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    throughput_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines: list[str] = ["# Staged Resource Comparison", ""]
    lines.append("## Runs")
    lines.append("")
    for summary in run_summaries:
        lines.append(
            f"- {summary['system']} / {summary['run']}: "
            f"steps={summary['step_log_count']}, pressure_samples={summary['pressure_sample_count']}"
        )
    lines.append("")
    lines.append("## Stage Summary")
    lines.append("")
    lines.append(
        "| system | stage | runs | target | tokens/s | step p95 ms | abs imbalance ms | replay ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in stage_rows:
        lines.append(
            "| "
            f"{row['system']} | {row['stage']} | {row['runs']} | "
            f"{_format_value(row.get('target'))} {row.get('target_unit') or ''} | "
            f"{_format_value(row.get('scheduled_tokens_per_s_mean'))} | "
            f"{_format_value(row.get('step_wall_p95_ms_mean'))} | "
            f"{_format_value(row.get('mean_abs_imbalance_ms_mean'))} | "
            f"{_format_value(row.get('mean_replay_ratio_mean'))} |"
        )
    lines.append("")
    lines.append("## S2 Loss / S3 Overshoot")
    lines.append("")
    lines.append(
        "| system | comparison | throughput drop % | p95 latency inflation % | abs imbalance delta ms | replay ratio delta |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in comparison_rows:
        lines.append(
            "| "
            f"{row['system']} | {row['comparison']} | "
            f"{_format_value(row.get('throughput_drop_pct'))} | "
            f"{_format_value(row.get('latency_p95_inflation_pct'))} | "
            f"{_format_value(row.get('mean_abs_imbalance_delta_ms'))} | "
            f"{_format_value(row.get('replay_ratio_delta'))} |"
        )
    lines.append("")
    if throughput_rows:
        lines.append("## Decode Throughput (nsys sqlite step-span)")
        lines.append("")
        lines.append(
            "| system | total tokens | marker | steps | window (s) | tokens/s |"
        )
        lines.append("|---|---:|---|---:|---:|---:|")
        for row in throughput_rows:
            lines.append(
                "| "
                f"{row['system']} | "
                f"{_format_value(row.get('total_tokens'))} | "
                f"{row.get('step_marker') or '—'} | "
                f"{_format_value(row.get('step_count'))} | "
                f"{_format_value(row.get('decode_window_s'))} | "
                f"{_format_value(row.get('decode_throughput_tokens_per_s'))} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _plot_stage_bars(out_dir: Path, stage_rows: list[dict[str, Any]]) -> None:
    if not HAS_MPL or not stage_rows:
        return
    metrics = [
        ("scheduled_tokens_per_s_mean", "Scheduled tokens/s", "throughput_by_stage.png"),
        ("step_wall_p95_ms_mean", "Step wall P95 (ms)", "step_p95_by_stage.png"),
        ("mean_abs_imbalance_ms_mean", "Mean |imbalance| (ms)", "imbalance_by_stage.png"),
        ("mean_replay_ratio_mean", "Mean replay ratio", "replay_ratio_by_stage.png"),
    ]
    systems = sorted({str(row["system"]) for row in stage_rows})
    stages = sorted({str(row["stage"]) for row in stage_rows}, key=_stage_sort_key)
    lookup = _stage_lookup(stage_rows)
    for metric, ylabel, filename in metrics:
        x_positions = list(range(len(stages)))
        width = 0.8 / max(1, len(systems))
        fig, ax = plt.subplots(figsize=(8, 4))
        for system_idx, system in enumerate(systems):
            values = [lookup.get((system, stage), {}).get(metric) for stage in stages]
            offsets = [x + (system_idx - (len(systems) - 1) / 2) * width for x in x_positions]
            ax.bar(offsets, [0.0 if value is None else float(value) for value in values], width, label=system)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stages)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


_STEP_NVTX_CANDIDATES: tuple[tuple[str, str], ...] = (
    # (label, SQL LIKE pattern) — first candidate that yields a non-empty
    # result wins. ``step_%`` is the preferred marker (emitted by
    # opt_replay_component_mfu.run_prompts_with_engine), but in practice the
    # ``nvtx_range`` helper there only fires when the ``nvtx`` python package
    # is importable AND nsys captures it. ``gpu_model_runner: forward`` is
    # emitted via torch.profiler.record_function from inside engine.step()
    # and is reliably captured by ``--trace=nvtx``, so we fall back to it
    # — one event per step, same min/max time window.
    ("step_*", "step\\_%"),
    ("gpu_model_runner: forward", "gpu_model_runner: forward"),
)


def _query_step_span_ns(sqlite_path: Path) -> tuple[int, int, int, str] | None:
    """Return ``(min_start_ns, max_end_ns, step_count, marker)`` over the best
    available step-boundary NVTX range in the nsys-exported sqlite database.

    Probes candidates in ``_STEP_NVTX_CANDIDATES`` in priority order and
    returns the first non-empty match together with the marker label that
    won, so callers can record which range was used.
    """
    if not sqlite_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'"
        )
        if cur.fetchone() is None:
            return None
        for marker, like_pattern in _STEP_NVTX_CANDIDATES:
            row = cur.execute(
                "SELECT MIN(start), MAX(end), COUNT(*) "
                "FROM NVTX_EVENTS WHERE text LIKE ? ESCAPE '\\'",
                (like_pattern,),
            ).fetchone()
            if row is None or row[0] is None or row[1] is None or (row[2] or 0) == 0:
                continue
            return int(row[0]), int(row[1]), int(row[2]), marker
    finally:
        conn.close()
    return None


def _decode_throughput_for_run(
    run: RunInputs,
    *,
    total_tokens: int | None,
    sqlite_override: Path | None = None,
) -> dict[str, Any]:
    """Compute decode throughput for one run from its nsys sqlite + token count.

    ``total_tokens`` is the *total decode token count* for the whole run
    (typically ``num_prompts * max_tokens``). When sqlite or total_tokens is
    missing, the corresponding fields come back as ``None``.
    """
    sqlite_path: Path | None = sqlite_override
    if sqlite_path is None and run.sqlite_paths:
        # Pick the most recent sqlite when multiple exist.
        sqlite_path = max(run.sqlite_paths, key=lambda p: p.stat().st_mtime)
    record: dict[str, Any] = {
        "system": run.system,
        "run_dir": str(run.run_dir) if run.run_dir is not None else None,
        "sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
        "total_tokens": total_tokens,
        "step_marker": None,
        "step_count": None,
        "first_step_start_ns": None,
        "last_step_end_ns": None,
        "decode_window_s": None,
        "decode_throughput_tokens_per_s": None,
    }
    if sqlite_path is None:
        return record
    span = _query_step_span_ns(sqlite_path)
    if span is None:
        return record
    start_ns, end_ns, step_count, marker = span
    window_s = (end_ns - start_ns) / 1e9
    record["step_marker"] = marker
    record["first_step_start_ns"] = start_ns
    record["last_step_end_ns"] = end_ns
    record["step_count"] = step_count
    record["decode_window_s"] = window_s
    if total_tokens is not None and window_s > 0:
        record["decode_throughput_tokens_per_s"] = total_tokens / window_s
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze staged-resource RunKV vs TightLLM runs without requiring nsys."
    )
    parser.add_argument("--runkv-run-dir", nargs="*", default=[])
    parser.add_argument("--tightllm-run-dir", nargs="*", default=[])
    parser.add_argument(
        "--dryrun-run-dir",
        nargs="*",
        default=[],
        help=(
            "Run directories produced by RunKV with DRY_RUN=1 (vanilla "
            "baseline; planner emits no replay/skip decisions). Stage and "
            "comparison rows for these runs are emitted under the system "
            "label 'runkv-dryrun'."
        ),
    )
    parser.add_argument("--runkv-step-log", nargs="*", default=[])
    parser.add_argument("--tightllm-step-log", nargs="*", default=[])
    parser.add_argument("--dryrun-step-log", nargs="*", default=[])
    parser.add_argument("--runkv-mfu", nargs="*", default=[])
    parser.add_argument("--tightllm-mfu", nargs="*", default=[])
    parser.add_argument("--dryrun-mfu", nargs="*", default=[])
    parser.add_argument("--runkv-mfu-flat", nargs="*", default=[])
    parser.add_argument("--tightllm-mfu-flat", nargs="*", default=[])
    parser.add_argument("--dryrun-mfu-flat", nargs="*", default=[])
    parser.add_argument("--runkv-pressure", nargs="*", default=[])
    parser.add_argument("--tightllm-pressure", nargs="*", default=[])
    parser.add_argument("--dryrun-pressure", nargs="*", default=[])
    parser.add_argument("--output-dir", default="exp_results/analysis/staged_resource")
    parser.add_argument("--skip-warmup-steps", type=int, default=1)
    # Decode-throughput inputs. Sqlite paths are optional — when omitted, the
    # script auto-picks the newest *.sqlite under each run_dir. Token totals
    # come from the inference config; pass either ``--total-decode-tokens``
    # directly or the (num_prompts, max_tokens) pair so the window from
    # step_0 start → last step end can be turned into tokens/s.
    parser.add_argument("--runkv-sqlite", nargs="*", default=[])
    parser.add_argument("--tightllm-sqlite", nargs="*", default=[])
    parser.add_argument("--dryrun-sqlite", nargs="*", default=[])
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of inference prompts; combined with --max-tokens to derive total decode tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-prompt decode token count.",
    )
    parser.add_argument(
        "--total-decode-tokens",
        type=int,
        default=None,
        help="Total decode tokens for the run; overrides --num-prompts * --max-tokens when set.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunInputs] = []
    runs.extend(_discover_run_inputs("runkv-feedback", args.runkv_run_dir))
    runs.extend(_discover_run_inputs("tightllm-replay", args.tightllm_run_dir))
    runs.extend(_discover_run_inputs("runkv-dryrun", args.dryrun_run_dir))
    runs.extend(
        _manual_run_inputs(
            "runkv-feedback",
            step_logs=args.runkv_step_log,
            mfu_steps=args.runkv_mfu,
            mfu_flat=args.runkv_mfu_flat,
            pressure_logs=args.runkv_pressure,
        )
    )
    runs.extend(
        _manual_run_inputs(
            "tightllm-replay",
            step_logs=args.tightllm_step_log,
            mfu_steps=args.tightllm_mfu,
            mfu_flat=args.tightllm_mfu_flat,
            pressure_logs=args.tightllm_pressure,
        )
    )
    runs.extend(
        _manual_run_inputs(
            "runkv-dryrun",
            step_logs=args.dryrun_step_log,
            mfu_steps=args.dryrun_mfu,
            mfu_flat=args.dryrun_mfu_flat,
            pressure_logs=args.dryrun_pressure,
        )
    )
    if not runs:
        raise SystemExit(
            "No run inputs provided. Use --runkv-run-dir / --tightllm-run-dir / "
            "--dryrun-run-dir or explicit file arguments."
        )

    all_stage_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run in runs:
        print(f"Analyzing {run.system}: {run.run_dir or '<manual>'}")
        stage_rows, run_summary = analyze_run(run, args.skip_warmup_steps)
        all_stage_rows.extend(stage_rows)
        run_summaries.append(run_summary)

    stage_rows = _aggregate_stage_rows(all_stage_rows)
    comparison_rows = _derive_comparison_rows(stage_rows)

    # ── Decode throughput from nsys sqlite + token-count CLI ────────────────
    if args.total_decode_tokens is not None:
        total_tokens = args.total_decode_tokens
    elif args.num_prompts is not None and args.max_tokens is not None:
        total_tokens = args.num_prompts * args.max_tokens
    else:
        total_tokens = None

    sqlite_overrides_runkv = _expand_patterns(args.runkv_sqlite)
    sqlite_overrides_tightllm = _expand_patterns(args.tightllm_sqlite)
    sqlite_overrides_dryrun = _expand_patterns(args.dryrun_sqlite)
    throughput_rows: list[dict[str, Any]] = []
    runkv_idx = tightllm_idx = dryrun_idx = 0
    for run in runs:
        if run.system == "runkv-feedback" and runkv_idx < len(sqlite_overrides_runkv):
            override = sqlite_overrides_runkv[runkv_idx]
            runkv_idx += 1
        elif run.system == "tightllm-replay" and tightllm_idx < len(sqlite_overrides_tightllm):
            override = sqlite_overrides_tightllm[tightllm_idx]
            tightllm_idx += 1
        elif run.system == "runkv-dryrun" and dryrun_idx < len(sqlite_overrides_dryrun):
            override = sqlite_overrides_dryrun[dryrun_idx]
            dryrun_idx += 1
        else:
            override = None
        throughput_rows.append(
            _decode_throughput_for_run(
                run, total_tokens=total_tokens, sqlite_override=override
            )
        )

    _write_csv(out_dir / "run_stage_summary.csv", all_stage_rows)
    _write_csv(out_dir / "stage_summary.csv", stage_rows)
    _write_csv(out_dir / "comparison_summary.csv", comparison_rows)
    _write_csv(out_dir / "decode_throughput.csv", throughput_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": run_summaries,
                "stage_summary": stage_rows,
                "comparison_summary": comparison_rows,
                "decode_throughput": throughput_rows,
            },
            indent=2,
        )
        + "\n"
    )
    _write_summary_md(
        out_dir / "analysis_summary.md",
        run_summaries=run_summaries,
        stage_rows=stage_rows,
        comparison_rows=comparison_rows,
        throughput_rows=throughput_rows,
    )
    _plot_stage_bars(out_dir, stage_rows)

    print(f"\nAnalysis written to: {out_dir}")
    print(f"  {out_dir / 'analysis_summary.md'}")
    print(f"  {out_dir / 'stage_summary.csv'}")
    print(f"  {out_dir / 'comparison_summary.csv'}")
    print(f"  {out_dir / 'decode_throughput.csv'}")


if __name__ == "__main__":
    main()
