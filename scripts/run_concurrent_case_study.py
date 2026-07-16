#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run RunKV+RunKV and TightLLM+TightLLM overlap case studies."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/offline_inference/opt_replay_component_mfu.py"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/concurrent_case_study"
DEFAULT_PROFILE = ROOT / "exp_results/tightllm_profiles/ubuntu/opt-2.7b-8k.json"


def _sanitize(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in value
    ).strip("_.")


def _latest_step(path: Path) -> int:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return -1
    for line in reversed(lines):
        try:
            return int(line)
        except ValueError:
            continue
    return -1


def _command(
    args: argparse.Namespace,
    *,
    planner: str,
    run_tag: str,
    run_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--model",
        args.model,
        "--planner",
        "feedback" if planner == "runkv" else "tightllm",
        "--prefix-blocks",
        str(args.prefix_blocks),
        "--num-prompts",
        str(args.num_prompts),
        "--prompt-words",
        str(args.prompt_words),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--num-device-buffers",
        str(args.num_device_buffers),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-staging-blocks",
        str(args.max_staging_blocks),
        "--cpu-memory-gb",
        str(args.cpu_memory_gb),
        "--cpu-memory-fraction",
        str(args.cpu_memory_fraction),
        "--output-dir",
        str(run_dir),
        "--run-tag",
        run_tag,
        "--step-progress-path",
        str(run_dir / "step_progress.log"),
        "--step-metrics-path",
        str(run_dir / "step_metrics.jsonl"),
    ]
    if planner == "runkv":
        cmd.append("--use-state-machine")
    else:
        cmd.extend(
            ["--tightllm-profile-path", str(args.tightllm_profile_path)]
        )
    return cmd


def _launch(
    args: argparse.Namespace,
    *,
    planner: str,
    task_name: str,
    run_tag: str,
    run_dir: Path,
) -> tuple[subprocess.Popen[bytes], TextIO]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _command(
        args,
        planner=planner,
        run_tag=run_tag,
        run_dir=run_dir,
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUNKV_PREHOOK_TIMING": "1",
            "RUNKV_PREHOOK_TIMING_DIR": str(run_dir / "prehook_timing"),
        }
    )
    log_file = (run_dir / "run.log").open("w", buffering=1)
    log_file.write(f"command: {shlex.join(cmd)}\n")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    print(f"  launched {task_name}: pid={process.pid}, log={run_dir / 'run.log'}")
    return process, log_file


def _stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run_pair(
    args: argparse.Namespace,
    *,
    planner: str,
    case_dir: Path,
    case_tag: str,
) -> dict[str, object]:
    group = f"{planner}_{planner}"
    group_dir = case_dir / group
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[TextIO] = []
    print(f"\n[{group}]")
    try:
        task_a, log_a = _launch(
            args,
            planner=planner,
            task_name="task_a",
            run_tag=f"{case_tag}_{group}_a",
            run_dir=group_dir / "task_a",
        )
        processes.append(task_a)
        logs.append(log_a)

        progress_path = group_dir / "task_a/step_progress.log"
        deadline = time.monotonic() + args.trigger_timeout_s
        observed_step = -1
        trigger = "timeout"
        while time.monotonic() < deadline:
            observed_step = _latest_step(progress_path)
            if observed_step >= args.trigger_step:
                trigger = "step_reached"
                break
            if task_a.poll() is not None:
                trigger = f"task_a_exited_{task_a.returncode}"
                break
            time.sleep(args.poll_interval_s)

        print(f"  task_b trigger: {trigger}, observed_step={observed_step}")
        task_a_alive = task_a.poll() is None
        task_b, log_b = _launch(
            args,
            planner=planner,
            task_name="task_b",
            run_tag=f"{case_tag}_{group}_b",
            run_dir=group_dir / "task_b",
        )
        processes.append(task_b)
        logs.append(log_b)

        task_b_progress = group_dir / "task_b/step_progress.log"
        inference_overlap = False
        overlap_deadline = time.monotonic() + args.trigger_timeout_s
        while time.monotonic() < overlap_deadline:
            if task_a.poll() is not None or task_b.poll() is not None:
                break
            if _latest_step(task_b_progress) >= 0:
                inference_overlap = True
                break
            time.sleep(args.poll_interval_s)

        task_a_rc = task_a.wait()
        task_b_rc = task_b.wait()
        metrics = _summarize_group(args, group_dir)
        return {
            "group": group,
            "trigger": trigger,
            "observed_step": observed_step,
            "task_a_alive_when_task_b_launched": task_a_alive,
            "inference_overlap_observed": inference_overlap,
            "task_a_pid": task_a.pid,
            "task_b_pid": task_b.pid,
            "task_a_return_code": task_a_rc,
            "task_b_return_code": task_b_rc,
            "output_dir": str(group_dir),
            "metrics": metrics,
        }
    except KeyboardInterrupt:
        _stop(processes)
        raise
    except Exception as exc:
        _stop(processes)
        return {"group": group, "error": str(exc), "output_dir": str(group_dir)}
    finally:
        for log_file in logs:
            log_file.close()


def _group_worker_command(
    args: argparse.Namespace,
    *,
    planner: str,
    case_dir: Path,
    case_tag: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--group-worker",
        planner,
        "--case-dir",
        str(case_dir),
        "--case-tag",
        case_tag,
        "--model",
        args.model,
        "--tightllm-profile-path",
        str(args.tightllm_profile_path),
        "--trigger-step",
        str(args.trigger_step),
        "--trigger-timeout-s",
        str(args.trigger_timeout_s),
        "--poll-interval-s",
        str(args.poll_interval_s),
        "--prefix-blocks",
        str(args.prefix_blocks),
        "--num-prompts",
        str(args.num_prompts),
        "--prompt-words",
        str(args.prompt_words),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--num-device-buffers",
        str(args.num_device_buffers),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-staging-blocks",
        str(args.max_staging_blocks),
        "--cpu-memory-gb",
        str(args.cpu_memory_gb),
        "--cpu-memory-fraction",
        str(args.cpu_memory_fraction),
    ]


def _nsys_profile_command(
    args: argparse.Namespace,
    *,
    planner: str,
    case_dir: Path,
    case_tag: str,
) -> tuple[list[str], Path]:
    group = f"{planner}_{planner}"
    report_stem = case_dir / group / f"{group}_combined"
    worker_cmd = _group_worker_command(
        args,
        planner=planner,
        case_dir=case_dir,
        case_tag=case_tag,
    )
    cmd = [
        args.nsys_cmd,
        "profile",
        "--trace=cuda,nvtx,osrt",
        f"--sample={args.nsys_sample}",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--wait=all",
        "--force-overwrite=true",
        "--output",
        str(report_stem),
        *shlex.split(args.nsys_extra_args),
        *worker_cmd,
    ]
    return cmd, Path(f"{report_stem}.nsys-rep")


def _run_logged_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with log_path.open("w", buffering=1) as log_file:
        log_file.write(f"command: {shlex.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def _export_nsys_sqlite(
    args: argparse.Namespace,
    *,
    report_path: Path,
    group_dir: Path,
) -> tuple[Path | None, int | None]:
    if args.skip_nsys_sqlite_export or not report_path.is_file():
        return None, None
    sqlite_path = report_path.with_suffix(".sqlite")
    cmd = [
        args.nsys_cmd,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(report_path),
    ]
    rc = _run_logged_command(cmd, group_dir / "nsys_export.log")
    return (sqlite_path if sqlite_path.is_file() else None), rc


def _run_profiled_pair(
    args: argparse.Namespace,
    *,
    planner: str,
    case_dir: Path,
    case_tag: str,
) -> dict[str, object]:
    group = f"{planner}_{planner}"
    group_dir = case_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    result_path = group_dir / "group_result.json"
    cmd, report_path = _nsys_profile_command(
        args,
        planner=planner,
        case_dir=case_dir,
        case_tag=case_tag,
    )
    print(f"\n[{group} combined nsys]")
    print(f"  report: {report_path}")
    nsys_rc = _run_logged_command(cmd, group_dir / "nsys.log")

    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result = {
            "group": group,
            "error": f"combined nsys/group worker failed: {exc}",
            "output_dir": str(group_dir),
        }
    if not isinstance(result, dict):
        result = {
            "group": group,
            "error": "group_result.json is not a JSON object",
            "output_dir": str(group_dir),
        }

    sqlite_path, export_rc = _export_nsys_sqlite(
        args,
        report_path=report_path,
        group_dir=group_dir,
    )
    result["nsys"] = {
        "profile_return_code": nsys_rc,
        "report": str(report_path) if report_path.is_file() else None,
        "sqlite": str(sqlite_path) if sqlite_path is not None else None,
        "sqlite_export_return_code": export_rc,
        "log": str(group_dir / "nsys.log"),
        "export_log": (
            str(group_dir / "nsys_export.log")
            if not args.skip_nsys_sqlite_export
            else None
        ),
        "contains_tasks": ["task_a", "task_b"],
    }
    return result


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as file:
        for line in file:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }

    def percentile(q: float) -> float:
        index = min(len(clean) - 1, round((len(clean) - 1) * q))
        return clean[index]

    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": clean[0],
        "max": clean[-1],
    }


def _latest_artifact(run_dir: Path, pattern: str) -> Path | None:
    paths = list(run_dir.glob(pattern))
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _summarize_prehook(run_dir: Path) -> dict[str, Any] | None:
    path = _latest_artifact(run_dir / "prehook_timing", "prehook_timing_*.jsonl")
    records = _read_jsonl(path)
    if not records:
        return None
    fields = (
        "total_ms",
        "sync_wait_ms",
        "imbalance_ms",
        "build_plan_ms",
        "build_meta_ms",
        "schedule_io_ms",
    )
    return {
        "artifact": str(path),
        **{
            field: _stats(
                [float(record[field]) for record in records if record.get(field) is not None]
            )
            for field in fields
        },
    }


def _summarize_task(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    mfu_path = _latest_artifact(run_dir, "opt_component_mfu_*.jsonl")
    if mfu_path is not None and ".flat." in mfu_path.name:
        candidates = [
            path
            for path in run_dir.glob("opt_component_mfu_*.jsonl")
            if ".flat." not in path.name
        ]
        mfu_path = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    mfu_records = _read_jsonl(mfu_path)

    step_info: dict[int, dict[str, float]] = {}
    imbalances: list[float] = []
    budgets: list[float] = []
    replay_tokens = 0.0
    actual_tokens = 0.0
    cpu_fill_tokens = 0.0
    gpu_reuse_tokens = 0.0
    for record in mfu_records:
        step = int(record.get("step", len(step_info)))
        scheduled = float(record.get("total_scheduled_tokens") or 0.0)
        layers = record.get("layers")
        if not isinstance(layers, list):
            layers = []
        compute_ends = [
            float(layer["compute_end_ms_from_anchor"])
            for layer in layers
            if isinstance(layer, dict)
            and layer.get("compute_end_ms_from_anchor") is not None
        ]
        step_info[step] = {
            "scheduled_tokens": scheduled,
            "num_reqs": float(record.get("num_reqs") or 0.0),
            "cuda_span_s": max(compute_ends, default=0.0) / 1000.0,
        }
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if layer.get("imbalance_ms") is not None:
                imbalances.append(float(layer["imbalance_ms"]))
            replay_tokens += float(layer.get("replay_token_count") or 0.0)
            actual_tokens += float(layer.get("num_actual_tokens") or 0.0)
            cpu_fill_tokens += float(layer.get("cpu_fill_token_count") or 0.0)
            gpu_reuse_tokens += float(layer.get("gpu_reuse_token_count") or 0.0)
            update = layer.get("controller_update")
            if isinstance(update, dict) and update.get("budget_after") is not None:
                budgets.append(float(update["budget_after"]))

    metrics_path = run_dir / "step_metrics.jsonl"
    metric_records = _read_jsonl(metrics_path)
    samples: list[dict[str, float]] = []
    timing_source = "step_metrics_wall"
    for record in metric_records:
        step = int(record.get("step", len(samples)))
        start_s = float(record.get("step_start_monotonic_s") or 0.0)
        end_s = float(record.get("step_end_monotonic_s") or start_s)
        info = step_info.get(step, {})
        samples.append(
            {
                "step": float(step),
                "start_s": start_s,
                "end_s": end_s,
                "wall_s": max(0.0, end_s - start_s),
                "scheduled_tokens": float(info.get("scheduled_tokens", 0.0)),
                "num_reqs": float(info.get("num_reqs", 0.0)),
            }
        )

    if not samples:
        timing_source = "mfu_cuda_span_fallback"
        cursor_s = 0.0
        for step, info in sorted(step_info.items()):
            duration_s = float(info["cuda_span_s"])
            samples.append(
                {
                    "step": float(step),
                    "start_s": cursor_s,
                    "end_s": cursor_s + duration_s,
                    "wall_s": duration_s,
                    "scheduled_tokens": float(info["scheduled_tokens"]),
                    "num_reqs": float(info["num_reqs"]),
                }
            )
            cursor_s += duration_s

    duration_s = (
        samples[-1]["end_s"] - samples[0]["start_s"] if samples else 0.0
    )
    step_wall_ms = [sample["wall_s"] * 1000.0 for sample in samples]
    total_scheduled_tokens = sum(sample["scheduled_tokens"] for sample in samples)
    decode_tokens = args.num_prompts * args.max_tokens
    finished_requests = sum(
        int(record.get("finished_count") or 0) for record in metric_records
    )

    phase_metrics: dict[str, dict[str, float | int | None]] = {}
    for phase, phase_samples in (
        (
            "prefill",
            [sample for sample in samples if sample["scheduled_tokens"] > sample["num_reqs"]],
        ),
        (
            "decode",
            [sample for sample in samples if sample["scheduled_tokens"] <= sample["num_reqs"]],
        ),
    ):
        phase_wall_s = sum(sample["wall_s"] for sample in phase_samples)
        phase_tokens = sum(sample["scheduled_tokens"] for sample in phase_samples)
        phase_metrics[phase] = {
            "step_count": len(phase_samples),
            "wall_s": phase_wall_s,
            "scheduled_tokens": phase_tokens,
            "scheduled_tokens_per_s": (
                phase_tokens / phase_wall_s if phase_wall_s > 0 else None
            ),
        }

    return (
        {
            "run_dir": str(run_dir),
            "timing_source": timing_source,
            "artifacts": {
                "run_log": str(run_dir / "run.log"),
                "step_progress": str(run_dir / "step_progress.log"),
                "step_metrics": str(metrics_path) if metrics_path.exists() else None,
                "mfu_jsonl": str(mfu_path) if mfu_path is not None else None,
            },
            "step_count": len(samples),
            "finished_requests_observed": (
                finished_requests if metric_records else None
            ),
            "inference_duration_s": duration_s,
            "step_wall_ms": _stats(step_wall_ms),
            "throughput": {
                "total_scheduled_tokens": total_scheduled_tokens,
                "scheduled_tokens_per_s": (
                    total_scheduled_tokens / duration_s if duration_s > 0 else None
                ),
                "workload_decode_tokens": decode_tokens,
                "decode_tokens_per_s": (
                    decode_tokens / duration_s if duration_s > 0 else None
                ),
                "requests": args.num_prompts,
                "requests_per_s": (
                    args.num_prompts / duration_s if duration_s > 0 else None
                ),
            },
            "phase_metrics": phase_metrics,
            "replay": {
                "replay_tokens_across_layers": replay_tokens,
                "actual_tokens_across_layers": actual_tokens,
                "replay_ratio": (
                    replay_tokens / actual_tokens if actual_tokens > 0 else None
                ),
                "cpu_fill_tokens_across_layers": cpu_fill_tokens,
                "gpu_reuse_tokens_across_layers": gpu_reuse_tokens,
                "imbalance_ms": _stats(imbalances),
                "absolute_imbalance_ms": _stats([abs(value) for value in imbalances]),
                "controller_budget_after": _stats(budgets),
            },
            "prehook": _summarize_prehook(run_dir),
        },
        samples,
    )


def _interval_scheduled_tokens(
    samples: list[dict[str, float]], start_s: float, end_s: float
) -> float:
    total = 0.0
    for sample in samples:
        duration_s = sample["wall_s"]
        if duration_s <= 0:
            continue
        covered_s = max(
            0.0,
            min(sample["end_s"], end_s) - max(sample["start_s"], start_s),
        )
        total += sample["scheduled_tokens"] * covered_s / duration_s
    return total


def _summarize_group(args: argparse.Namespace, group_dir: Path) -> dict[str, Any]:
    task_a, samples_a = _summarize_task(args, group_dir / "task_a")
    task_b, samples_b = _summarize_task(args, group_dir / "task_b")
    task_rates = [
        float(task["throughput"]["scheduled_tokens_per_s"] or 0.0)
        for task in (task_a, task_b)
    ]
    fairness = None
    if any(task_rates):
        fairness = sum(task_rates) ** 2 / (2.0 * sum(rate**2 for rate in task_rates))

    result: dict[str, Any] = {
        "task_a": task_a,
        "task_b": task_b,
        "per_task_scheduled_tokens_per_s_sum": sum(task_rates),
        "scheduled_throughput_jain_fairness": fairness,
        "aggregate": None,
        "overlap_window": None,
    }
    exact_timing = (
        task_a["timing_source"] == "step_metrics_wall"
        and task_b["timing_source"] == "step_metrics_wall"
        and samples_a
        and samples_b
    )
    if not exact_timing:
        return result

    group_start_s = min(samples_a[0]["start_s"], samples_b[0]["start_s"])
    group_end_s = max(samples_a[-1]["end_s"], samples_b[-1]["end_s"])
    makespan_s = group_end_s - group_start_s
    total_scheduled = sum(
        float(task["throughput"]["total_scheduled_tokens"])
        for task in (task_a, task_b)
    )
    total_decode = 2 * args.num_prompts * args.max_tokens
    result["aggregate"] = {
        "inference_makespan_s": makespan_s,
        "total_scheduled_tokens": total_scheduled,
        "scheduled_tokens_per_s": (
            total_scheduled / makespan_s if makespan_s > 0 else None
        ),
        "workload_decode_tokens": total_decode,
        "decode_tokens_per_s": total_decode / makespan_s if makespan_s > 0 else None,
        "requests": 2 * args.num_prompts,
        "requests_per_s": (
            2 * args.num_prompts / makespan_s if makespan_s > 0 else None
        ),
    }

    overlap_start_s = max(samples_a[0]["start_s"], samples_b[0]["start_s"])
    overlap_end_s = min(samples_a[-1]["end_s"], samples_b[-1]["end_s"])
    overlap_s = max(0.0, overlap_end_s - overlap_start_s)
    a_overlap_tokens = _interval_scheduled_tokens(
        samples_a, overlap_start_s, overlap_end_s
    )
    b_overlap_tokens = _interval_scheduled_tokens(
        samples_b, overlap_start_s, overlap_end_s
    )
    result["overlap_window"] = {
        "duration_s": overlap_s,
        "task_a_scheduled_tokens": a_overlap_tokens,
        "task_b_scheduled_tokens": b_overlap_tokens,
        "combined_scheduled_tokens": a_overlap_tokens + b_overlap_tokens,
        "combined_scheduled_tokens_per_s": (
            (a_overlap_tokens + b_overlap_tokens) / overlap_s
            if overlap_s > 0
            else None
        ),
        "task_a_solo_before_task_b_inference_s": max(
            0.0, samples_b[0]["start_s"] - samples_a[0]["start_s"]
        ),
    }
    return result


def _comparison(results: list[dict[str, object]]) -> dict[str, Any] | None:
    by_group = {str(result.get("group")): result for result in results}
    runkv = by_group.get("runkv_runkv")
    tightllm = by_group.get("tightllm_tightllm")
    if runkv is None or tightllm is None:
        return None

    def group_rate(result: dict[str, object]) -> tuple[float | None, str]:
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            return None, "unavailable"
        aggregate = metrics.get("aggregate")
        if isinstance(aggregate, dict) and aggregate.get("scheduled_tokens_per_s"):
            return float(aggregate["scheduled_tokens_per_s"]), "group_makespan"
        rate = metrics.get("per_task_scheduled_tokens_per_s_sum")
        return (float(rate), "sum_of_per_task_rates") if rate is not None else (None, "unavailable")

    runkv_rate, runkv_source = group_rate(runkv)
    tightllm_rate, tightllm_source = group_rate(tightllm)

    def metric_dict(result: dict[str, object]) -> dict[str, Any]:
        value = result.get("metrics")
        return value if isinstance(value, dict) else {}

    def summed_task_metric(result: dict[str, object], key: str) -> float | None:
        metrics = metric_dict(result)
        values: list[float] = []
        for task_name in ("task_a", "task_b"):
            task = metrics.get(task_name)
            throughput = task.get("throughput") if isinstance(task, dict) else None
            if isinstance(throughput, dict) and throughput.get(key) is not None:
                values.append(float(throughput[key]))
        return sum(values) if values else None

    def aggregate_metric(result: dict[str, object], key: str) -> float | None:
        aggregate = metric_dict(result).get("aggregate")
        if isinstance(aggregate, dict) and aggregate.get(key) is not None:
            return float(aggregate[key])
        return None

    def mean_task_p95(result: dict[str, object]) -> float | None:
        metrics = metric_dict(result)
        values: list[float] = []
        for task_name in ("task_a", "task_b"):
            task = metrics.get(task_name)
            stats = task.get("step_wall_ms") if isinstance(task, dict) else None
            if isinstance(stats, dict) and stats.get("p95") is not None:
                values.append(float(stats["p95"]))
        return sum(values) / len(values) if values else None

    runkv_decode_rate = aggregate_metric(runkv, "decode_tokens_per_s")
    tightllm_decode_rate = aggregate_metric(tightllm, "decode_tokens_per_s")
    if runkv_decode_rate is None:
        runkv_decode_rate = summed_task_metric(runkv, "decode_tokens_per_s")
    if tightllm_decode_rate is None:
        tightllm_decode_rate = summed_task_metric(tightllm, "decode_tokens_per_s")
    runkv_p95 = mean_task_p95(runkv)
    tightllm_p95 = mean_task_p95(tightllm)
    runkv_makespan = aggregate_metric(runkv, "inference_makespan_s")
    tightllm_makespan = aggregate_metric(tightllm, "inference_makespan_s")

    runkv_overlap = metric_dict(runkv).get("overlap_window")
    tightllm_overlap = metric_dict(tightllm).get("overlap_window")
    runkv_overlap_rate = (
        float(runkv_overlap["combined_scheduled_tokens_per_s"])
        if isinstance(runkv_overlap, dict)
        and runkv_overlap.get("combined_scheduled_tokens_per_s") is not None
        else None
    )
    tightllm_overlap_rate = (
        float(tightllm_overlap["combined_scheduled_tokens_per_s"])
        if isinstance(tightllm_overlap, dict)
        and tightllm_overlap.get("combined_scheduled_tokens_per_s") is not None
        else None
    )

    scheduled_speedup = (
        runkv_rate / tightllm_rate
        if runkv_rate is not None and tightllm_rate not in (None, 0.0)
        else None
    )
    return {
        "scheduled_throughput": {
            "metric": "scheduled_tokens_per_s",
            "runkv_runkv": runkv_rate,
            "tightllm_tightllm": tightllm_rate,
            "runkv_source": runkv_source,
            "tightllm_source": tightllm_source,
            "runkv_over_tightllm_speedup": scheduled_speedup,
            "runkv_vs_tightllm_change_pct": (
                (scheduled_speedup - 1.0) * 100.0
                if scheduled_speedup is not None
                else None
            ),
        },
        "decode_throughput": {
            "metric": "decode_tokens_per_s",
            "runkv_runkv": runkv_decode_rate,
            "tightllm_tightllm": tightllm_decode_rate,
            "runkv_over_tightllm_speedup": (
                runkv_decode_rate / tightllm_decode_rate
                if runkv_decode_rate is not None
                and tightllm_decode_rate not in (None, 0.0)
                else None
            ),
        },
        "step_wall_p95_ms": {
            "runkv_runkv_task_mean": runkv_p95,
            "tightllm_tightllm_task_mean": tightllm_p95,
            "runkv_vs_tightllm_reduction_pct": (
                (tightllm_p95 - runkv_p95) / tightllm_p95 * 100.0
                if runkv_p95 is not None and tightllm_p95 not in (None, 0.0)
                else None
            ),
        },
        "inference_makespan_s": {
            "runkv_runkv": runkv_makespan,
            "tightllm_tightllm": tightllm_makespan,
            "runkv_vs_tightllm_reduction_pct": (
                (tightllm_makespan - runkv_makespan) / tightllm_makespan * 100.0
                if runkv_makespan is not None
                and tightllm_makespan not in (None, 0.0)
                else None
            ),
        },
        "overlap_scheduled_throughput": {
            "runkv_runkv": runkv_overlap_rate,
            "tightllm_tightllm": tightllm_overlap_rate,
            "runkv_over_tightllm_speedup": (
                runkv_overlap_rate / tightllm_overlap_rate
                if runkv_overlap_rate is not None
                and tightllm_overlap_rate not in (None, 0.0)
                else None
            ),
        },
    }


def _nested_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _detailed_comparison(results: list[dict[str, object]]) -> dict[str, Any]:
    by_group = {str(result.get("group")): result for result in results}
    task_columns = (
        ("runkv_task_a", "runkv_runkv", "task_a"),
        ("runkv_task_b", "runkv_runkv", "task_b"),
        ("tightllm_task_a", "tightllm_tightllm", "task_a"),
        ("tightllm_task_b", "tightllm_tightllm", "task_b"),
    )

    def task_row(metric: str, unit: str, *path: str) -> dict[str, Any]:
        row: dict[str, Any] = {"metric": metric, "unit": unit}
        for column, group, task_name in task_columns:
            result = by_group.get(group, {})
            metrics = result.get("metrics") if isinstance(result, dict) else None
            task = metrics.get(task_name) if isinstance(metrics, dict) else None
            row[column] = _nested_value(task, tuple(path))
        return row

    performance_specs = (
        ("step_count", "steps", ("step_count",)),
        ("inference_duration", "s", ("inference_duration_s",)),
        (
            "total_scheduled_tokens",
            "tokens",
            ("throughput", "total_scheduled_tokens"),
        ),
        (
            "scheduled_throughput",
            "tokens/s",
            ("throughput", "scheduled_tokens_per_s"),
        ),
        (
            "decode_throughput",
            "tokens/s",
            ("throughput", "decode_tokens_per_s"),
        ),
        ("request_throughput", "requests/s", ("throughput", "requests_per_s")),
        ("step_wall_mean", "ms", ("step_wall_ms", "mean")),
        ("step_wall_p50", "ms", ("step_wall_ms", "p50")),
        ("step_wall_p95", "ms", ("step_wall_ms", "p95")),
        ("step_wall_p99", "ms", ("step_wall_ms", "p99")),
    )
    phase_specs = tuple(
        (
            f"{phase}_{metric}",
            unit,
            ("phase_metrics", phase, source_key),
        )
        for phase in ("prefill", "decode")
        for metric, unit, source_key in (
            ("step_count", "steps", "step_count"),
            ("wall", "s", "wall_s"),
            ("scheduled_tokens", "tokens", "scheduled_tokens"),
            ("scheduled_throughput", "tokens/s", "scheduled_tokens_per_s"),
        )
    )
    replay_specs = (
        ("replay_ratio", "ratio", ("replay", "replay_ratio")),
        (
            "replay_tokens_across_layers",
            "tokens",
            ("replay", "replay_tokens_across_layers"),
        ),
        (
            "cpu_fill_tokens_across_layers",
            "tokens",
            ("replay", "cpu_fill_tokens_across_layers"),
        ),
        (
            "gpu_reuse_tokens_across_layers",
            "tokens",
            ("replay", "gpu_reuse_tokens_across_layers"),
        ),
        ("imbalance_mean", "ms", ("replay", "imbalance_ms", "mean")),
        ("imbalance_p95", "ms", ("replay", "imbalance_ms", "p95")),
        (
            "absolute_imbalance_mean",
            "ms",
            ("replay", "absolute_imbalance_ms", "mean"),
        ),
        (
            "absolute_imbalance_p95",
            "ms",
            ("replay", "absolute_imbalance_ms", "p95"),
        ),
        (
            "controller_budget_mean",
            "blocks",
            ("replay", "controller_budget_after", "mean"),
        ),
        (
            "controller_budget_p95",
            "blocks",
            ("replay", "controller_budget_after", "p95"),
        ),
    )
    prehook_specs = tuple(
        (
            f"{field}_{stat}",
            "ms",
            ("prehook", field, stat),
        )
        for field in (
            "total_ms",
            "sync_wait_ms",
            "imbalance_ms",
            "build_plan_ms",
            "schedule_io_ms",
        )
        for stat in ("mean", "p95")
    )

    def rows(specs: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [task_row(metric, unit, *path) for metric, unit, path in specs]

    group_rows: list[dict[str, Any]] = []
    for metric, unit, path in (
        (
            "sum_of_per_task_scheduled_throughput",
            "tokens/s",
            ("per_task_scheduled_tokens_per_s_sum",),
        ),
        (
            "scheduled_throughput_jain_fairness",
            "ratio",
            ("scheduled_throughput_jain_fairness",),
        ),
        ("inference_makespan", "s", ("aggregate", "inference_makespan_s")),
        (
            "aggregate_scheduled_throughput",
            "tokens/s",
            ("aggregate", "scheduled_tokens_per_s"),
        ),
        (
            "aggregate_decode_throughput",
            "tokens/s",
            ("aggregate", "decode_tokens_per_s"),
        ),
        ("overlap_duration", "s", ("overlap_window", "duration_s")),
        (
            "overlap_scheduled_throughput",
            "tokens/s",
            ("overlap_window", "combined_scheduled_tokens_per_s"),
        ),
        (
            "task_a_solo_before_task_b_inference",
            "s",
            ("overlap_window", "task_a_solo_before_task_b_inference_s"),
        ),
    ):
        row = {"metric": metric, "unit": unit}
        for column, group in (
            ("runkv_runkv", "runkv_runkv"),
            ("tightllm_tightllm", "tightllm_tightllm"),
        ):
            result = by_group.get(group, {})
            metrics = result.get("metrics") if isinstance(result, dict) else None
            row[column] = _nested_value(metrics, path)
        group_rows.append(row)

    return {
        "task_performance": rows(performance_specs),
        "phase_performance": rows(phase_specs),
        "group_concurrency": group_rows,
        "replay_and_imbalance": rows(replay_specs),
        "prehook_timing": rows(prehook_specs),
    }


def _process_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in result.items() if key != "metrics"}
        for result in results
    ]


def _artifact_table(results: list[dict[str, object]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        group = str(result.get("group"))
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for task_name in ("task_a", "task_b"):
            task = metrics.get(task_name)
            if not isinstance(task, dict):
                continue
            artifacts = task.get("artifacts")
            prehook = task.get("prehook")
            rows.append(
                {
                    "group": group,
                    "task": task_name,
                    "timing_source": task.get("timing_source"),
                    **(artifacts if isinstance(artifacts, dict) else {}),
                    "prehook_timing": (
                        prehook.get("artifact") if isinstance(prehook, dict) else None
                    ),
                }
            )
    return rows


def _write_summary(
    path: Path,
    *,
    case_tag: str,
    status: str,
    results: list[dict[str, object]],
    configuration: dict[str, Any],
) -> None:
    summary = {
        "schema_version": 2,
        "case_tag": case_tag,
        "status": status,
        "headline_comparison": _comparison(results),
        "configuration": configuration,
        "detailed_comparison": _detailed_comparison(results),
        "process_results": _process_results(results),
        "artifacts": _artifact_table(results),
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n")
    temporary_path.replace(path)


def _handle_termination(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "tasks_per_group": 2,
        "trigger_step": args.trigger_step,
        "workload_per_task": {
            "batch_size": args.num_prompts,
            "prompt_words": args.prompt_words,
            "decode_tokens_per_request": args.max_tokens,
            "total_decode_tokens": args.num_prompts * args.max_tokens,
        },
        "memory_per_task": {
            "cpu_memory_gb": args.cpu_memory_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_staging_blocks": args.max_staging_blocks,
        },
        "nsys": {
            "enabled": args.enable_nsys,
            "scope": "one combined report per group",
            "trace": "cuda,nvtx,osrt",
            "sample": args.nsys_sample,
            "sqlite_export": not args.skip_nsys_sqlite_export,
        },
    }


def _summarize_existing(args: argparse.Namespace, case_dir: Path) -> int:
    if not case_dir.is_absolute():
        case_dir = ROOT / case_dir
    summary_path = case_dir / "case_summary.json"
    try:
        existing = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {summary_path}: {exc}", file=sys.stderr)
        return 2

    if isinstance(existing, list):
        results = existing
        status = "complete"
        case_tag = case_dir.name
        configuration = _configuration(args)
    else:
        results = existing.get("groups")
        if results is None:
            results = existing.get("process_results", [])
        status = str(existing.get("status", "complete"))
        case_tag = str(existing.get("case_tag", case_dir.name))
        configuration = existing.get("configuration")
        if not isinstance(configuration, dict):
            configuration = _configuration(args)
    if not isinstance(results, list):
        print(f"ERROR: invalid process results in {summary_path}", file=sys.stderr)
        return 2

    for result in results:
        if not isinstance(result, dict) or not result.get("group"):
            continue
        group_dir = Path(str(result.get("output_dir") or case_dir / result["group"]))
        result["metrics"] = _summarize_group(args, group_dir)

    _write_summary(
        summary_path,
        case_tag=case_tag,
        status=status,
        results=results,
        configuration=configuration,
    )
    print(f"Updated case-study metrics: {summary_path}")
    comparison = _comparison(results)
    scheduled = comparison.get("scheduled_throughput") if comparison else None
    if isinstance(scheduled, dict) and all(
        scheduled.get(key) is not None
        for key in (
            "runkv_runkv",
            "tightllm_tightllm",
            "runkv_over_tightllm_speedup",
        )
    ):
        print(
            "  scheduled-token throughput: "
            f"RunKV+RunKV={scheduled['runkv_runkv']:.3f}, "
            f"TightLLM+TightLLM={scheduled['tightllm_tightllm']:.3f}, "
            f"speedup={scheduled['runkv_over_tightllm_speedup']:.3f}x"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-worker",
        choices=["runkv", "tightllm"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-tag", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="/data/models/opt-2.7b-8k")
    parser.add_argument("--tightllm-profile-path", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trigger-step", type=int, default=20)
    parser.add_argument("--trigger-timeout-s", type=float, default=900.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.2)
    parser.add_argument("--prefix-blocks", default="10000")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-words", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--num-device-buffers", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-staging-blocks", type=int, default=1152)
    parser.add_argument("--cpu-memory-gb", type=float, default=25.0)
    parser.add_argument("--cpu-memory-fraction", type=float, default=0.8)
    parser.add_argument(
        "--enable-nsys",
        dest="enable_nsys",
        action="store_true",
        default=True,
        help="Write one combined nsys report containing both tasks in each group.",
    )
    parser.add_argument(
        "--disable-nsys",
        dest="enable_nsys",
        action="store_false",
        help="Run groups directly without Nsight Systems.",
    )
    parser.add_argument(
        "--nsys-cmd",
        default=os.environ.get("NSYS_CMD", "nsys"),
    )
    parser.add_argument("--nsys-sample", default="none")
    parser.add_argument(
        "--nsys-extra-args",
        default="",
        help="Additional arguments inserted into each group-level nsys profile command.",
    )
    parser.add_argument(
        "--skip-nsys-sqlite-export",
        action="store_true",
        help="Keep the combined .nsys-rep without exporting a combined SQLite file.",
    )
    parser.add_argument(
        "--summarize-existing",
        type=Path,
        help="Rebuild metrics in an existing case-study case_summary.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.summarize_existing is not None:
        return _summarize_existing(args, args.summarize_existing.expanduser())

    args.tightllm_profile_path = args.tightllm_profile_path.expanduser()
    if not args.tightllm_profile_path.is_absolute():
        args.tightllm_profile_path = ROOT / args.tightllm_profile_path
    if not args.tightllm_profile_path.is_file():
        print(
            f"ERROR: TightLLM profile not found: {args.tightllm_profile_path}",
            file=sys.stderr,
        )
        return 2

    if args.group_worker is not None:
        if args.case_dir is None or not args.case_tag:
            print("ERROR: group worker requires --case-dir and --case-tag", file=sys.stderr)
            return 2
        case_dir = args.case_dir.expanduser()
        if not case_dir.is_absolute():
            case_dir = ROOT / case_dir
        previous_sigterm_handler = signal.signal(
            signal.SIGTERM, _handle_termination
        )
        try:
            result = _run_pair(
                args,
                planner=args.group_worker,
                case_dir=case_dir,
                case_tag=args.case_tag,
            )
        except KeyboardInterrupt:
            print("Group worker interrupted; active tasks were terminated.", file=sys.stderr)
            return 130
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        result_path = case_dir / f"{args.group_worker}_{args.group_worker}" / "group_result.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, indent=2) + "\n")
        temporary_path.replace(result_path)
        return 0

    if args.enable_nsys and shutil.which(args.nsys_cmd) is None:
        print(f"ERROR: nsys command not found: {args.nsys_cmd}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_tag = (
        f"{timestamp}_{_sanitize(Path(args.model).name)}_step{args.trigger_step}_"
        f"bs{args.num_prompts}_p{args.prompt_words}_d{args.max_tokens}"
    )
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    case_dir = output_root / case_tag

    print(f"Case study: {case_tag}")
    print(f"  output: {case_dir}")
    print(
        "  gpu_memory_utilization total for two processes: "
        f"{2 * args.gpu_memory_utilization:g}"
    )
    if args.dry_run:
        for planner in ("runkv", "tightllm"):
            group = f"{planner}_{planner}"
            if args.enable_nsys:
                cmd, _ = _nsys_profile_command(
                    args,
                    planner=planner,
                    case_dir=case_dir,
                    case_tag=case_tag,
                )
            else:
                cmd = _command(
                    args,
                    planner=planner,
                    run_tag=f"{case_tag}_{group}_a",
                    run_dir=case_dir / group / "task_a",
                )
            print(f"  {group}: {shlex.join(cmd)}")
        return 0

    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "case_summary.json"
    results: list[dict[str, object]] = []
    configuration = _configuration(args)
    _write_summary(
        summary_path,
        case_tag=case_tag,
        status="running",
        results=results,
        configuration=configuration,
    )
    print(f"  summary: {summary_path} (updated after each group)")
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_termination)
    try:
        for planner in ("runkv", "tightllm"):
            if args.enable_nsys:
                result = _run_profiled_pair(
                    args,
                    planner=planner,
                    case_dir=case_dir,
                    case_tag=case_tag,
                )
            else:
                result = _run_pair(
                    args,
                    planner=planner,
                    case_dir=case_dir,
                    case_tag=case_tag,
                )
            results.append(result)
            _write_summary(
                summary_path,
                case_tag=case_tag,
                status="running",
                results=results,
                configuration=configuration,
            )
    except KeyboardInterrupt:
        _write_summary(
            summary_path,
            case_tag=case_tag,
            status="interrupted",
            results=results,
            configuration=configuration,
        )
        print("\nInterrupted; active tasks were terminated.", file=sys.stderr)
        print(f"Partial summary: {summary_path}", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    _write_summary(
        summary_path,
        case_tag=case_tag,
        status="complete",
        results=results,
        configuration=configuration,
    )
    failed = False
    print("\nCase-study summary")
    for result in results:
        return_codes = (
            result.get("task_a_return_code"),
            result.get("task_b_return_code"),
        )
        passed = "error" not in result and return_codes == (0, 0)
        passed = passed and result.get("trigger") == "step_reached"
        passed = passed and result.get("inference_overlap_observed") is True
        nsys_result = result.get("nsys")
        if args.enable_nsys:
            passed = passed and isinstance(nsys_result, dict)
            passed = passed and nsys_result.get("profile_return_code") == 0
            passed = passed and nsys_result.get("report") is not None
        failed |= not passed
        print(
            f"  {'PASS' if passed else 'FAIL'} {result['group']}: "
            f"rc={return_codes}, trigger={result.get('trigger')}, "
            f"inference_overlap={result.get('inference_overlap_observed')}"
        )
        if isinstance(nsys_result, dict):
            print(f"    nsys report: {nsys_result.get('report')}")
            print(f"    nsys sqlite: {nsys_result.get('sqlite')}")
    print(f"  summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
