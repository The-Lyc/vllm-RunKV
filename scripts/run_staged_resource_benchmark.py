#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run staged-resource RunKV vs TightLLM benchmark pipeline.

This pipeline is the resource-controller counterpart of
scripts/run_benchmark_pipeline.py.  It runs RunKV feedback and TightLLM with the
same workload and a runner-controlled resource pressure schedule, discovers the
JSONL/CSV/nsys artifacts, exports nsys reports to sqlite, and calls both the
staged-resource analysis script and the per-layer timing analysis script.

The default IO pressure mode uses real PCIe contention: after the configured
step, the runner launches independent CUDA worker processes that continuously
copy pinned CPU memory to the GPU.  If the no-contention PCIe bandwidth is
24 GB/s, targets 12, 8, and 6 GB/s correspond to 1, 2, and 3 extra contending
processes, respectively.

Default mode mirrors scripts/run_benchmark_pipeline.py: nsys/NVTX/CUDA profiler
and RunKV prehook timing are enabled so timeline-level metrics are available.
Use --disable-nsys/--disable-nvtx/--disable-profile/--disable-prehook-timing for
lightweight smoke runs.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

RUNKV_SCRIPT = ROOT / "examples/offline_inference/run_opt_feedback_observation.py"
TIGHTLLM_SCRIPT = ROOT / "examples/offline_inference/run_tightllm_observation.py"
STAGE_ANALYSIS_SCRIPT = ROOT / "tools/analyze_staged_resource_comparison.py"
PER_LAYER_ANALYSIS_SCRIPT = ROOT / "tools/analyze_per_layer_timing.py"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/staged_offline_pilot"
MANIFEST_ROOT = ROOT / "exp_results/manifests/staged_resource"
ANALYSIS_ROOT = ROOT / "exp_results/analysis/staged_resource"
PER_LAYER_ANALYSIS_ROOT = ROOT / "exp_results/analysis/staged_resource_per_layer"
DEFAULT_TIGHTLLM_PROFILE_ROOT = ROOT / "exp_results/tightllm_profiles"


def _require_plotting_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import matplotlib"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Analysis plots are enabled, but matplotlib is not installed in "
            f"{sys.executable}. Install the repository bench dependencies "
            "(`python -m pip install -e '.[bench]'`) before running this "
            "pipeline, or pass --skip-analysis to run without plots."
        )


def _sanitize_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return token.strip("_.") or "run"


def _default_pattern_name(kind: str, clock: str, pattern: str) -> str:
    if kind == "io" and pattern == "0:24,16:12,48:24" and clock == "step":
        return "io_pcie12_step"
    if kind == "sm" and pattern == "0:100,16:50,48:100" and clock == "step":
        return "sm_process50_step"
    if kind == "io" and pattern == "0:0,16:15,48:0" and clock == "step":
        return "io_severe_step"
    if kind == "sm" and pattern == "0:0,16:50,48:0" and clock == "step":
        return "sm_severe_step"
    return _sanitize_token(f"{kind}_{clock}_{pattern}")


def _model_tag(model: str) -> str:
    return _sanitize_token(Path(model).name or model)


def _resolve_tightllm_profile_path(args: argparse.Namespace) -> str:
    if args.tightllm_profile_path:
        return args.tightllm_profile_path
    if args.hardware_platform:
        return str(
            Path(args.tightllm_profile_root)
            / _sanitize_token(args.hardware_platform.lower())
            / f"{_model_tag(args.model)}.json"
        )
    return "tightllm_profile.json"


def _validate_tightllm_profile_path(args: argparse.Namespace, path: str) -> None:
    if Path(path).exists():
        return
    command = (
        f"{sys.executable} -m vllm.v1.profiling.tightllm_offline_profiler "
        f"--model {shlex.quote(args.model)} --output {shlex.quote(path)} "
        "--seq-lengths 128 256 512 1024 2048 4096 8192 16384"
    )
    raise SystemExit(
        f"ERROR: TightLLM profile not found: {path}\n"
        f"Generate the profile on hardware platform {args.hardware_platform or 'current'}:\n"
        f"  {command}"
    )


def _resolve_glob(pattern: str) -> list[str]:
    return sorted(_glob.glob(pattern))


def _find_latest(paths: list[str]) -> str:
    if not paths:
        raise FileNotFoundError("No files found matching the pattern")
    return max(paths, key=os.path.getmtime)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _run_step(
    step_name: str,
    cmd: list[str],
    env: dict[str, str],
    *,
    manifest_path: Path | None,
    log_path: Path | None,
) -> int:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(shlex.quote(part) for part in cmd)}")
    if log_path is not None:
        print(f"  LOG:  {log_path}")
    print(f"{sep}\n")

    run_env = os.environ.copy()
    run_env.update(env)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        run_env["MANIFEST_FILE"] = str(manifest_path)

    if log_path is None:
        result = subprocess.run(cmd, env=run_env)
        return result.returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        print(f"\n[ERROR] {step_name} failed with code {return_code}")
    else:
        print(f"\n[DONE] {step_name}")
    return return_code


def _find_run_artifacts(run_dir: Path, run_tag: str, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path) or {}
    mfu_glob = manifest.get("mfu_jsonl_glob") or str(run_dir / f"opt_component_mfu_*_{run_tag}.jsonl")
    mfu_flat_glob = manifest.get("mfu_flat_jsonl_glob") or str(
        run_dir / f"opt_component_mfu_*_{run_tag}.flat.jsonl"
    )
    mfu_files = [path for path in _resolve_glob(mfu_glob) if ".flat." not in Path(path).name]
    mfu_flat_files = _resolve_glob(mfu_flat_glob)
    return {
        "manifest": str(manifest_path),
        "mfu_jsonl": _find_latest(mfu_files) if mfu_files else None,
        "mfu_flat_jsonl": _find_latest(mfu_flat_files) if mfu_flat_files else None,
        "pressure_csv": str(run_dir / "pressure.csv"),
        "resource_steps_jsonl": str(run_dir / "resource_steps.jsonl"),
        "run_log": str(run_dir / "run.log"),
        "nsys_report": manifest.get("nsys_report"),
        "nsys_sqlite": None,
    }


def _find_existing_run_artifacts(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_manifest(manifest_path) or {}
    run_tag = str(manifest.get("run_tag") or "*")
    mfu_glob = manifest.get("mfu_jsonl_glob") or str(run_dir / f"opt_component_mfu_*_{run_tag}.jsonl")
    mfu_flat_glob = manifest.get("mfu_flat_jsonl_glob") or str(
        run_dir / f"opt_component_mfu_*_{run_tag}.flat.jsonl"
    )
    mfu_files = [path for path in _resolve_glob(mfu_glob) if ".flat." not in Path(path).name]
    mfu_flat_files = _resolve_glob(mfu_flat_glob)
    nsys_reports = _resolve_glob(str(run_dir / "*.nsys-rep"))
    sqlite_files = _resolve_glob(str(run_dir / "*.sqlite"))
    return {
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "mfu_jsonl": _find_latest(mfu_files) if mfu_files else None,
        "mfu_flat_jsonl": _find_latest(mfu_flat_files) if mfu_flat_files else None,
        "pressure_csv": str(run_dir / "pressure.csv"),
        "resource_steps_jsonl": str(run_dir / "resource_steps.jsonl"),
        "run_log": str(run_dir / "run.log"),
        "nsys_report": manifest.get("nsys_report") or (_find_latest(nsys_reports) if nsys_reports else None),
        "nsys_sqlite": _find_latest(sqlite_files) if sqlite_files else None,
    }


def _export_nsys_sqlite(
    *,
    args: argparse.Namespace,
    system: str,
    run_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if not args.enable_nsys or args.skip_sqlite_export:
        return
    if artifacts.get("nsys_sqlite") and Path(str(artifacts["nsys_sqlite"])).exists():
        return

    report = artifacts.get("nsys_report")
    if not report:
        print(f"[WARN] {system}: nsys enabled but manifest has no nsys_report")
        return
    report_path = Path(report)
    if not report_path.exists():
        print(f"[WARN] {system}: nsys report not found: {report_path}")
        return

    nsys_cmd = os.environ.get("NSYS_CMD", "nsys")
    if shutil.which(nsys_cmd) is None:
        print(f"[WARN] {system}: cannot export sqlite; {nsys_cmd!r} is not on PATH")
        return

    sqlite_path = run_dir / f"{report_path.stem}.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    cmd = [
        nsys_cmd,
        "export",
        "--type",
        "sqlite",
        "-o",
        str(sqlite_path),
        str(report_path),
    ]
    rc = _run_step(
        f"{system} export nsys sqlite",
        cmd,
        {},
        manifest_path=None,
        log_path=run_dir / "nsys_export.log",
    )
    if rc != 0:
        raise SystemExit(rc)
    artifacts["nsys_sqlite"] = str(sqlite_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged-resource RunKV vs TightLLM benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ctrl = parser.add_argument_group("Pipeline control")
    ctrl.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M"))
    ctrl.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ctrl.add_argument("--analysis-output-dir", default="")
    ctrl.add_argument("--repeats", type=int, default=1)
    ctrl.add_argument("--skip-runkv", action="store_true")
    ctrl.add_argument("--skip-tightllm", action="store_true")
    ctrl.add_argument("--skip-analysis", action="store_true")

    test = parser.add_argument_group("Test parameters")
    test.add_argument("--model", default="/data/models/opt-1.3b-8k")
    test.add_argument("--prefix-blocks", default="10000")
    test.add_argument("--num-prompts", default="64")
    test.add_argument("--prompt-words", default="2000")
    test.add_argument("--max-tokens", default="256")
    test.add_argument("--gpu-memory-utilization", default="0.8")
    test.add_argument("--gpu-memory-fraction", default="0.9")
    test.add_argument("--num-device-buffers", default="3")
    test.add_argument(
        "--max-num-seqs",
        default="",
        help=(
            "Maximum concurrently active requests; empty preserves the runner "
            "default of --num-prompts."
        ),
    )
    test.add_argument(
        "--max-staging-blocks",
        default="",
        help=(
            "Explicit KV blocks per GPU staging buffer; empty uses "
            "--gpu-memory-fraction auto sizing."
        ),
    )
    test.add_argument(
        "--cpu-memory-gb",
        "--cpu-kv-memory-gb",
        dest="cpu_memory_gb",
        default=str(6e10 / (1024**3)),
        help=(
            "Total CPU cache-store budget in GiB; with dynamic replay this "
            "covers both full-layer KV and hidden-state stores. 0 derives "
            "it from --cpu-memory-fraction; --cpu-kv-memory-gb remains an alias."
        ),
    )
    test.add_argument(
        "--cpu-memory-fraction",
        default="0.6",
        help="Clamp total CPU cache-store budget to this available-RAM fraction.",
    )
    test.add_argument(
        "--hardware-platform",
        default="",
        help=(
            "Profile key for the current GPU/node, e.g. rtx4090 or a800. "
            "When set, auto-selects a model-specific TightLLM profile unless "
            "--tightllm-profile-path is given."
        ),
    )
    test.add_argument(
        "--tightllm-profile-root",
        default=str(DEFAULT_TIGHTLLM_PROFILE_ROOT),
        help="Root for platform/model TightLLM profiles.",
    )
    test.add_argument(
        "--tightllm-profile-path",
        default="",
        help=(
            "Explicit TightLLM profile JSON override. Without this option, "
            "--hardware-platform resolves "
            "<profile-root>/<platform>/<model-tag>.json; without either, "
            "uses legacy tightllm_profile.json."
        ),
    )
    test.add_argument(
        "--tightllm-feedback-correction",
        action="store_true",
        help="Run TightLLM with additive feedback correction ablation.",
    )
    test.add_argument(
        "--runkv-replay-allocation-policy",
        choices=("spread", "concentrate"),
        default="spread",
        help=(
            "Per-request replay budget redistribution policy for the RunKV "
            "feedback planner. 'spread' is the legacy behaviour; "
            "'concentrate' keeps replay concentrated on few requests."
        ),
    )
    test.add_argument(
        "--tightllm-replay-allocation-policy",
        choices=("concentrate", "spread"),
        default="concentrate",
        help=(
            "Per-request budget distribution for the TightLLM planner. "
            "'concentrate' is the native greedy baseline; 'spread' "
            "equalises the replay ratio across requests (experimental "
            "contrast knob)."
        ),
    )

    pressure = parser.add_argument_group("Resource pressure")
    pressure.add_argument("--resource-pressure-kind", choices=["io", "sm"], default="io")
    pressure.add_argument("--resource-pressure-clock", choices=["step", "time"], default="step")
    pressure.add_argument(
        "--resource-pressure-pattern",
        default="0:24,16:12,48:24",
        help=(
            "Comma-separated start:target schedule. In process mode, positive "
            "IO targets are desired benchmark-visible GB/s, and positive SM "
            "targets are desired benchmark-visible compute percent. Target 0 "
            "means no process contention. With a 24 GB/s IO baseline, 12/8/6 "
            "GB/s starts 1/2/3 contending processes; for SM, 50/33.3/25 "
            "percent starts 1/2/3 contending processes."
        ),
    )
    pressure.add_argument("--resource-pattern-name", default="")
    pressure.add_argument("--resource-pressure-device", default="cuda:0")
    pressure.add_argument("--resource-pressure-buffer-mb", default="256")
    pressure.add_argument(
        "--resource-pressure-direction",
        choices=["h2d", "d2h", "bidirectional"],
        default="h2d",
    )
    pressure.add_argument("--resource-pressure-matrix-size", default="4096")
    pressure.add_argument(
        "--resource-pressure-dtype", choices=["float16", "bfloat16"], default="float16"
    )
    pressure.add_argument("--resource-pressure-window-s", default="0.25")
    pressure.add_argument("--resource-pressure-period-ms", default="100.0")
    pressure.add_argument("--resource-pressure-max-fraction", default="0.5")
    pressure.add_argument("--resource-pressure-io-calibration-s", default="0.5")
    pressure.add_argument(
        "--resource-pressure-io-max-gbps",
        default="",
        help=(
            "No-contention PCIe bandwidth in GB/s. Process mode defaults to "
            "24.0 when this is omitted; other modes keep their legacy "
            "calibration behavior."
        ),
    )
    pressure.add_argument(
        "--resource-pressure-mode",
        choices=["thread", "inline", "throttle", "process"],
        default="process",
        help=(
            "thread: legacy background-worker pressure. "
            "inline: pressure injected synchronously from RunKV pre_hook on "
            "each layer (IO before prefetch on a dedicated stream, SM before "
            "FA on compute stream). Inline produces identical interference "
            "for RunKV / TightLLM and does not race with host syncs. "
            "throttle: IO-only; cap the KV / cpu-fill load streams to the "
            "stage's target GB/s via a stream-side torch.cuda._sleep paired "
            "with each copy. With throttle, --resource-pressure-pattern "
            "targets are interpreted as GB/s (not a contention fraction), "
            "and bytes-saved by RunKV translate linearly to wall-time saved. "
            "process: launch independent CUDA worker processes at "
            "each stage so the benchmark physically shares PCIe bandwidth or "
            "SM time with them. In process mode, IO targets are desired "
            "benchmark-visible GB/s and SM targets are desired benchmark-"
            "visible compute percent. See docs/design/io_bandwidth_throttle_"
            "design.md for the older throttle mode."
        ),
    )
    pressure.add_argument(
        "--resource-pressure-inline-layer-period-ms",
        default="5.0",
        help="Per-layer wall-time estimate used to size each inline burst.",
    )

    profile = parser.add_argument_group("Diagnostic profiling")
    profile.add_argument("--enable-nsys", dest="enable_nsys", action="store_true", default=True)
    profile.add_argument("--disable-nsys", dest="enable_nsys", action="store_false")
    profile.add_argument("--enable-nvtx", dest="enable_nvtx", action="store_true", default=True)
    profile.add_argument("--disable-nvtx", dest="enable_nvtx", action="store_false")
    profile.add_argument("--enable-profile", dest="enable_profile", action="store_true", default=True)
    profile.add_argument("--disable-profile", dest="enable_profile", action="store_false")
    profile.add_argument(
        "--enable-prehook-timing",
        dest="enable_prehook_timing",
        action="store_true",
        default=True,
    )
    profile.add_argument(
        "--disable-prehook-timing",
        dest="enable_prehook_timing",
        action="store_false",
    )
    profile.add_argument(
        "--skip-sqlite-export",
        action="store_true",
        help="When --enable-nsys is set, keep only .nsys-rep and skip sqlite export.",
    )
    profile.add_argument("--nsys-sample", default="cpu")
    profile.add_argument(
        "--nsys-extra-args",
        default="--capture-range=cudaProfilerApi --capture-range-end=stop",
    )

    paths = parser.add_argument_group("Analysis path overrides")
    paths.add_argument("--runkv-run-dir", nargs="*", default=[])
    paths.add_argument("--tightllm-run-dir", nargs="*", default=[])
    paths.add_argument("--skip-warmup-steps", type=int, default=1)
    paths.add_argument("--skip-stage-analysis", action="store_true")
    paths.add_argument("--skip-per-layer-analysis", action="store_true")
    paths.add_argument("--compute-stream", type=int, default=7)
    paths.add_argument("--dma-tol-ms", type=float, default=2.0)
    return parser


def _common_env(args: argparse.Namespace, run_tag: str, output_dir: Path) -> dict[str, str]:
    env = {
        "RUN_TAG": run_tag,
        "MODEL": args.model,
        "PREFIX_BLOCKS": args.prefix_blocks,
        "NUM_PROMPTS": args.num_prompts,
        "PROMPT_WORDS": args.prompt_words,
        "MAX_TOKENS": args.max_tokens,
        "GPU_MEMORY_UTILIZATION": args.gpu_memory_utilization,
        "GPU_MEMORY_FRACTION": args.gpu_memory_fraction,
        "NUM_DEVICE_BUFFERS": args.num_device_buffers,
        "MAX_NUM_SEQS": args.max_num_seqs,
        "MAX_STAGING_BLOCKS": args.max_staging_blocks,
        "CPU_MEMORY_GB": args.cpu_memory_gb,
        "CPU_MEMORY_FRACTION": args.cpu_memory_fraction,
        "HARDWARE_PLATFORM": args.hardware_platform,
        "OUTPUT_DIR": str(output_dir),
        "ENABLE_OPT_COMPONENT_MFU_PROFILING": "1",
        "ENABLE_NSYS": "1" if args.enable_nsys else "0",
        "ENABLE_NVTX": "1" if args.enable_nvtx else "0",
        "ENABLE_PROFILE": "1" if args.enable_profile else "0",
        "RUNKV_PREHOOK_TIMING": "1" if args.enable_prehook_timing else "0",
        "NSYS_SAMPLE": args.nsys_sample,
        "NSYS_EXTRA_ARGS": args.nsys_extra_args,
    }
    if args.enable_prehook_timing:
        prehook_dir = output_dir / "prehook_timing"
        prehook_dir.mkdir(parents=True, exist_ok=True)
        env["RUNKV_PREHOOK_TIMING_DIR"] = str(prehook_dir)
    return env


def _resource_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    resource_args = [
        "--resource-pressure-kind",
        args.resource_pressure_kind,
        "--resource-pressure-clock",
        args.resource_pressure_clock,
        "--resource-pressure-pattern",
        args.resource_pressure_pattern,
        "--resource-pressure-device",
        args.resource_pressure_device,
        "--resource-pressure-log-path",
        str(output_dir / "pressure.csv"),
        "--resource-pressure-step-log-path",
        str(output_dir / "resource_steps.jsonl"),
        "--resource-pressure-window-s",
        args.resource_pressure_window_s,
        "--resource-pressure-period-ms",
        args.resource_pressure_period_ms,
        "--resource-pressure-max-fraction",
        args.resource_pressure_max_fraction,
        "--resource-pressure-io-calibration-s",
        args.resource_pressure_io_calibration_s,
    ]
    if args.resource_pressure_io_max_gbps:
        resource_args.extend(
            [
                "--resource-pressure-io-max-gbps",
                args.resource_pressure_io_max_gbps,
            ]
        )
    resource_args.extend(
        [
            "--resource-pressure-mode",
            args.resource_pressure_mode,
            "--resource-pressure-inline-layer-period-ms",
            args.resource_pressure_inline_layer_period_ms,
        ]
    )
    if args.resource_pressure_kind == "io":
        resource_args.extend(
            [
                "--resource-pressure-direction",
                args.resource_pressure_direction,
                "--resource-pressure-buffer-mb",
                args.resource_pressure_buffer_mb,
            ]
        )
    else:
        resource_args.extend(
            [
                "--resource-pressure-matrix-size",
                args.resource_pressure_matrix_size,
                "--resource-pressure-dtype",
                args.resource_pressure_dtype,
            ]
        )
    return resource_args


def _run_system(
    *,
    system: str,
    script: Path,
    args: argparse.Namespace,
    pattern_name: str,
    repeat_idx: int,
    system_dir_name: str,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    run_tag = f"{pattern_name}_{system_dir_name}_r{repeat_idx}_{args.run_tag}"
    run_dir = (
        Path(args.output_root)
        / pattern_name
        / _sanitize_token(args.run_tag)
        / system_dir_name
        / f"r{repeat_idx}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    env = _common_env(args, run_tag, run_dir)
    env.update(extra_env)
    cmd = [sys.executable, str(script), *_resource_args(args, run_dir)]
    rc = _run_step(
        f"{system} staged resource run r{repeat_idx}",
        cmd,
        env,
        manifest_path=manifest_path,
        log_path=run_dir / "run.log",
    )
    if rc != 0:
        raise SystemExit(rc)
    artifacts = _find_run_artifacts(run_dir, run_tag, manifest_path)
    _export_nsys_sqlite(
        args=args,
        system=system,
        run_dir=run_dir,
        artifacts=artifacts,
    )
    return {
        "system": system,
        "repeat": repeat_idx,
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "artifacts": artifacts,
    }


def _run_stage_analysis(
    *,
    args: argparse.Namespace,
    pattern_name: str,
    runkv_dirs: list[str],
    tightllm_dirs: list[str],
) -> Path | None:
    if args.skip_analysis or args.skip_stage_analysis:
        print("[SKIP] staged-resource analysis")
        return None
    output_dir = (
        Path(args.analysis_output_dir)
        if args.analysis_output_dir
        else ANALYSIS_ROOT / pattern_name / args.run_tag
    )
    cmd = [
        sys.executable,
        str(STAGE_ANALYSIS_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--skip-warmup-steps",
        str(args.skip_warmup_steps),
        "--num-prompts",
        str(args.num_prompts),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if runkv_dirs:
        cmd.extend(["--runkv-run-dir", *runkv_dirs])
    if tightllm_dirs:
        cmd.extend(["--tightllm-run-dir", *tightllm_dirs])
    rc = _run_step("Staged resource analysis", cmd, {}, manifest_path=None, log_path=None)
    if rc != 0:
        raise SystemExit(rc)
    return output_dir


def _result_from_existing_run_dir(
    *,
    args: argparse.Namespace,
    system: str,
    run_dir: str,
) -> dict[str, Any]:
    path = Path(run_dir)
    artifacts = _find_existing_run_artifacts(path)
    _export_nsys_sqlite(
        args=args,
        system=system,
        run_dir=path,
        artifacts=artifacts,
    )
    return {
        "system": system,
        "repeat": None,
        "run_tag": None,
        "run_dir": str(path),
        "artifacts": artifacts,
    }


def _require_artifact(result: dict[str, Any], key: str) -> str:
    value = result.get("artifacts", {}).get(key)
    if not value:
        raise FileNotFoundError(
            f"Missing {key} for {result.get('system')} run: {result.get('run_dir')}"
        )
    return str(value)


def _run_per_layer_analysis(
    *,
    args: argparse.Namespace,
    pattern_name: str,
    runkv_results: list[dict[str, Any]],
    tightllm_results: list[dict[str, Any]],
) -> list[str]:
    if args.skip_analysis or args.skip_per_layer_analysis:
        print("[SKIP] per-layer timing analysis")
        return []
    if not args.enable_nsys:
        print("[SKIP] per-layer timing analysis because nsys is disabled")
        return []

    pair_count = min(len(runkv_results), len(tightllm_results))
    if pair_count == 0:
        print("[SKIP] per-layer timing analysis; missing RunKV or TightLLM runs")
        return []
    if len(runkv_results) != len(tightllm_results):
        print(
            "[WARN] per-layer analysis pairs only the first "
            f"{pair_count} RunKV/TightLLM runs"
        )

    output_dirs: list[str] = []
    for pair_idx, (runkv_result, tightllm_result) in enumerate(
        zip(runkv_results, tightllm_results)
    ):
        output_dir = PER_LAYER_ANALYSIS_ROOT / pattern_name / args.run_tag / f"r{pair_idx}"
        cmd = [
            sys.executable,
            str(PER_LAYER_ANALYSIS_SCRIPT),
            "--runkv-mfu",
            _require_artifact(runkv_result, "mfu_flat_jsonl"),
            "--tightllm-mfu",
            _require_artifact(tightllm_result, "mfu_flat_jsonl"),
            "--runkv-sqlite",
            _require_artifact(runkv_result, "nsys_sqlite"),
            "--tightllm-sqlite",
            _require_artifact(tightllm_result, "nsys_sqlite"),
            "--output-dir",
            str(output_dir),
            "--skip-warmup-steps",
            str(args.skip_warmup_steps),
            "--compute-stream",
            str(args.compute_stream),
            "--dma-tol-ms",
            str(args.dma_tol_ms),
            "--num-prompts",
            str(args.num_prompts),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if not args.runkv_run_dir and not args.tightllm_run_dir:
            cmd.append("--fixed-output-length")
        rc = _run_step(
            f"Per-layer timing analysis r{pair_idx}",
            cmd,
            {},
            manifest_path=None,
            log_path=None,
        )
        if rc != 0:
            raise SystemExit(rc)
        output_dirs.append(str(output_dir))
    return output_dirs


def _print_settings(args: argparse.Namespace, pattern_name: str) -> None:
    print("Running OPT staged-resource benchmark")
    print(f"  model:                       {args.model}")
    print(f"  prefix_blocks:               {args.prefix_blocks}")
    print(f"  num_prompts:                 {args.num_prompts}")
    print(f"  prompt_words:                {args.prompt_words}")
    print(f"  max_tokens:                  {args.max_tokens}")
    print(f"  gpu_memory_fraction:         {args.gpu_memory_fraction}")
    print(f"  gpu_memory_utilization:      {args.gpu_memory_utilization}")
    print(f"  num_device_buffers:          {args.num_device_buffers}")
    if args.max_num_seqs:
        print(f"  max_num_seqs:                {args.max_num_seqs}")
    if args.max_staging_blocks:
        print(f"  max_staging_blocks:          {args.max_staging_blocks}")
    print(f"  cpu_memory_gb:               {args.cpu_memory_gb}")
    print(f"  cpu_memory_fraction:         {args.cpu_memory_fraction}")
    if args.hardware_platform:
        print(f"  hardware_platform:           {args.hardware_platform}")
    print(f"  resource_pressure_kind:      {args.resource_pressure_kind}")
    print(f"  resource_pressure_mode:      {args.resource_pressure_mode}")
    print(f"  resource_pressure_clock:     {args.resource_pressure_clock}")
    print(f"  resource_pressure_pattern:   {args.resource_pressure_pattern}")
    print(f"  resource_pressure_device:    {args.resource_pressure_device}")
    print(f"  resource_pressure_max_fraction: {args.resource_pressure_max_fraction}")
    print(f"  resource_pressure_io_calibration_s: {args.resource_pressure_io_calibration_s}")
    if args.resource_pressure_io_max_gbps:
        print(f"  resource_pressure_io_max_gbps: {args.resource_pressure_io_max_gbps}")
    print(f"  pattern_name:                {pattern_name}")
    print(f"  run_tag:                     {args.run_tag}")
    print(f"  output_root:                 {args.output_root}")
    print(f"  repeats:                     {args.repeats}")
    print(f"  enable_nsys:                 {int(args.enable_nsys)}")
    print(f"  enable_nvtx:                 {int(args.enable_nvtx)}")
    print(f"  enable_profile:              {int(args.enable_profile)}")
    print(f"  skip_runkv:                  {int(args.skip_runkv)}")
    print(f"  skip_tightllm:               {int(args.skip_tightllm)}")
    print(f"  skip_analysis:               {int(args.skip_analysis)}")
    print()


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.resource_pressure_mode == "process"
        and args.resource_pressure_kind == "sm"
        and args.resource_pressure_pattern == "0:24,16:12,48:24"
    ):
        args.resource_pressure_pattern = "0:100,16:50,48:100"
    if (
        args.resource_pressure_mode == "process"
        and args.resource_pressure_kind == "io"
        and not args.resource_pressure_io_max_gbps
    ):
        args.resource_pressure_io_max_gbps = "24.0"
    if not args.skip_analysis and (
        not args.skip_stage_analysis or not args.skip_per_layer_analysis
    ):
        _require_plotting_dependency()
    args.tightllm_profile_path = _resolve_tightllm_profile_path(args)
    if not args.skip_tightllm and not args.tightllm_run_dir:
        _validate_tightllm_profile_path(args, args.tightllm_profile_path)
    pattern_name = args.resource_pattern_name or _default_pattern_name(
        args.resource_pressure_kind,
        args.resource_pressure_clock,
        args.resource_pressure_pattern,
    )
    _print_settings(args, pattern_name)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    if args.enable_nsys and shutil.which(os.environ.get("NSYS_CMD", "nsys")) is None:
        print("[WARN] --enable-nsys set, but nsys is not on PATH and NSYS_CMD is not resolvable")

    pipeline_manifest: dict[str, Any] = {
        "run_tag": args.run_tag,
        "pattern_name": pattern_name,
        "resource_pressure": {
            "kind": args.resource_pressure_kind,
            "clock": args.resource_pressure_clock,
            "pattern": args.resource_pressure_pattern,
        },
        "settings": vars(args),
        "runs": [],
    }

    runkv_dirs = list(args.runkv_run_dir)
    tightllm_dirs = list(args.tightllm_run_dir)
    runkv_results: list[dict[str, Any]] = []
    tightllm_results: list[dict[str, Any]] = []

    for run_dir in args.runkv_run_dir:
        runkv_results.append(
            _result_from_existing_run_dir(
                args=args,
                system="runkv-feedback",
                run_dir=run_dir,
            )
        )
    for run_dir in args.tightllm_run_dir:
        tightllm_results.append(
            _result_from_existing_run_dir(
                args=args,
                system="tightllm-replay",
                run_dir=run_dir,
            )
        )
    if not args.skip_runkv and not args.runkv_run_dir:
        for repeat_idx in range(args.repeats):
            result = _run_system(
                system="runkv-feedback",
                script=RUNKV_SCRIPT,
                args=args,
                pattern_name=pattern_name,
                repeat_idx=repeat_idx,
                system_dir_name="runkv_feedback",
                extra_env={
                    "PLANNER": "feedback",
                    "DRY_RUN": "0",
                    "USE_STATE_MACHINE": "1",
                    "REPLAY_ALLOCATION_POLICY": args.runkv_replay_allocation_policy,
                },
            )
            pipeline_manifest["runs"].append(result)
            runkv_results.append(result)
            runkv_dirs.append(result["run_dir"])
    elif args.skip_runkv:
        print("[SKIP] RunKV runs")

    if not args.skip_tightllm and not args.tightllm_run_dir:
        for repeat_idx in range(args.repeats):
            tightllm_extra_env = {
                "TIGHTLLM_PROFILE_PATH": args.tightllm_profile_path,
                "TIGHTLLM_FEEDBACK_CORRECTION": "1" if args.tightllm_feedback_correction else "0",
                "TIGHTLLM_REPLAY_ALLOCATION_POLICY": (
                    args.tightllm_replay_allocation_policy
                ),
            }
            result = _run_system(
                system=(
                    "tightllm-feedback" if args.tightllm_feedback_correction else "tightllm-replay"
                ),
                script=TIGHTLLM_SCRIPT,
                args=args,
                pattern_name=pattern_name,
                repeat_idx=repeat_idx,
                system_dir_name=(
                    "tightllm_feedback" if args.tightllm_feedback_correction else "tightllm_replay"
                ),
                extra_env=tightllm_extra_env,
            )
            pipeline_manifest["runs"].append(result)
            tightllm_results.append(result)
            tightllm_dirs.append(result["run_dir"])
    elif args.skip_tightllm:
        print("[SKIP] TightLLM runs")

    manifest_path = MANIFEST_ROOT / f"{pattern_name}_{args.run_tag}.json"
    manifest_path.write_text(json.dumps(pipeline_manifest, indent=2) + "\n")
    print(f"\nPipeline manifest written to: {manifest_path}")

    stage_analysis_dir = _run_stage_analysis(
        args=args,
        pattern_name=pattern_name,
        runkv_dirs=runkv_dirs,
        tightllm_dirs=tightllm_dirs,
    )
    per_layer_analysis_dirs = _run_per_layer_analysis(
        args=args,
        pattern_name=pattern_name,
        runkv_results=runkv_results,
        tightllm_results=tightllm_results,
    )
    pipeline_manifest["analysis"] = {
        "stage_analysis_dir": str(stage_analysis_dir) if stage_analysis_dir else None,
        "per_layer_analysis_dirs": per_layer_analysis_dirs,
    }
    manifest_path.write_text(json.dumps(pipeline_manifest, indent=2) + "\n")

    print(f"\n{'=' * 72}")
    print("  Staged resource pipeline complete")
    print(f"  pattern: {pattern_name}")
    print(f"  run_tag: {args.run_tag}")
    print(f"  manifest: {manifest_path}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
