#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one RunKV case with Flux and capture both in one Nsight report.

The selected case comes from the same JSON format as ``run_benchmark_batch.py``.
An outer ``nsys profile`` follows the complete process tree. Its worker starts
Flux from the diffusion virtual environment, runs RunKV with the current vLLM
Python, and always stops Flux afterward. RunKV's inner nsys launch is disabled;
its CUDA profiler start/stop calls instead delimit the combined capture.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_benchmark_batch import _load_settings, _run_tag

ROOT = Path(__file__).resolve().parents[1]
RUNKV_SCRIPT = ROOT / "examples/offline_inference/run_opt_feedback_observation.py"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/runkv_flux_nsys"
FLUX_ROOT = Path("/home/lyc/inference/val")
FLUX_VENV_ACTIVATE = Path("diffusion/.venv/bin/activate")
FLUX_MANAGER = Path("diffusion/manage_flux_contender.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--test-index",
        type=int,
        default=1,
        help="1-based index of the batch test to run (default: 1).",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Explicit output tag; the default is generated from the selected case.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--nsys-cmd",
        default=os.environ.get("NSYS_CMD", "nsys"),
    )
    parser.add_argument("--nsys-sample", default="cpu")
    parser.add_argument(
        "--nsys-extra-args",
        default="",
        help="Additional arguments inserted into the outer nsys profile command.",
    )
    parser.add_argument("--dry-run", action="store_true")

    worker = parser.add_argument_group("internal worker")
    worker.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def _select_setting(
    config_path: Path,
    test_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, settings = _load_settings(config_path)
    if not 1 <= test_index <= len(settings):
        raise ValueError(
            f"--test-index must be between 1 and {len(settings)}; got {test_index}"
        )
    return config, settings[test_index - 1]


def _resolve_output_root(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _flux_shell_command(action: str) -> str:
    manager_args = [f"./{FLUX_MANAGER}", action]
    if action == "start":
        manager_args.extend(["--gpu", "0"])
    return " ".join(
        [
            "source",
            shlex.quote(str(FLUX_VENV_ACTIVATE)),
            "&&",
            *(shlex.quote(arg) for arg in manager_args),
        ]
    )


def _run_flux_command(action: str) -> int:
    shell_command = _flux_shell_command(action)
    print(f"\n[FLUX] {action}")
    print(f"  cwd: {FLUX_ROOT}")
    print(f"  cmd: {shell_command}")
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", shell_command],
            cwd=FLUX_ROOT,
        )
    except OSError as exc:
        print(f"[ERROR] Flux {action} failed: {exc}", file=sys.stderr)
        return 1
    if result.returncode:
        print(
            f"[ERROR] Flux {action} failed with code {result.returncode}",
            file=sys.stderr,
        )
    return result.returncode


def _runkv_env(
    setting: dict[str, Any],
    *,
    run_tag: str,
    case_dir: Path,
) -> dict[str, str]:
    runkv_dir = case_dir / "runkv"
    env = os.environ.copy()
    env.update(
        {
            "RUN_TAG": run_tag,
            "MODEL": str(setting["model"]),
            "PREFIX_BLOCKS": str(setting.get("prefix_blocks", 10000)),
            "NUM_PROMPTS": str(setting["batch_size"]),
            "PROMPT_WORDS": str(setting["prompt_length"]),
            "MAX_TOKENS": str(setting["decode_length"]),
            "CPU_MEMORY_GB": str(setting["cpu_memory_gb"]),
            "CPU_MEMORY_FRACTION": str(setting["cpu_memory_fraction"]),
            "GPU_MEMORY_UTILIZATION": str(setting["gpu_memory_utilization"]),
            "GPU_MEMORY_FRACTION": str(setting["gpu_memory_fraction"]),
            "NUM_DEVICE_BUFFERS": str(setting.get("num_device_buffers", 3)),
            "MAX_NUM_SEQS": str(setting.get("max_num_seqs") or ""),
            "MAX_STAGING_BLOCKS": str(setting.get("max_staging_blocks") or ""),
            "HARDWARE_PLATFORM": str(setting.get("hardware_platform") or ""),
            "DRY_RUN": "0",
            "USE_STATE_MACHINE": "1",
            "H2D_COPY_MODE": str(setting["runkv_h2d_copy_mode"]),
            "REPLAY_ALLOCATION_POLICY": str(
                setting["runkv_replay_allocation_policy"]
            ),
            "ENABLE_NVTX": "1",
            "ENABLE_PROFILE": "1",
            "ENABLE_OPT_COMPONENT_MFU_PROFILING": "1",
            "RUNKV_PREHOOK_TIMING": "1",
            "RUNKV_PREHOOK_TIMING_DIR": str(runkv_dir / "prehook_timing"),
            # The single outer nsys process captures both RunKV and Flux.
            "ENABLE_NSYS": "0",
            "OUTPUT_DIR": str(runkv_dir),
            "MANIFEST_FILE": str(case_dir / "runkv_manifest.json"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _run_worker(
    setting: dict[str, Any],
    *,
    run_tag: str,
    case_dir: Path,
) -> int:
    case_dir.mkdir(parents=True, exist_ok=True)
    start_rc = _run_flux_command("start")
    if start_rc:
        print("[ERROR] RunKV was not launched because Flux failed to start")
        return start_rc

    runkv_rc = 1
    try:
        cmd = [sys.executable, str(RUNKV_SCRIPT)]
        print("\n[RUNKV]")
        print(f"  python: {sys.executable}")
        print(f"  cmd: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_runkv_env(setting, run_tag=run_tag, case_dir=case_dir),
        )
        runkv_rc = result.returncode
    finally:
        stop_rc = _run_flux_command("stop")

    if runkv_rc:
        if stop_rc:
            print(
                f"[ERROR] Flux also failed to stop with code {stop_rc}",
                file=sys.stderr,
            )
        return runkv_rc
    return stop_rc


def _worker_command(
    args: argparse.Namespace,
    *,
    config_path: Path,
    run_tag: str,
    case_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        str(config_path),
        "--test-index",
        str(args.test_index),
        "--run-tag",
        run_tag,
        "--case-dir",
        str(case_dir),
        "--worker",
    ]


def _nsys_command(
    args: argparse.Namespace,
    *,
    worker_cmd: list[str],
    report_stem: Path,
) -> list[str]:
    return [
        args.nsys_cmd,
        "profile",
        "--trace=cuda,nvtx,osrt",
        f"--sample={args.nsys_sample}",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--wait=all",
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--output",
        str(report_stem),
        *shlex.split(args.nsys_extra_args),
        *worker_cmd,
    ]


def _run_logged(cmd: list[str], log_path: Path) -> int:
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


def _write_combined_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    setting: dict[str, Any],
    run_tag: str,
    report_path: Path,
    log_path: Path,
    profile_return_code: int,
) -> None:
    manifest = {
        "run_tag": run_tag,
        "test_index": args.test_index,
        "test_name": setting.get("name", f"case{args.test_index:03d}"),
        "model": setting["model"],
        "profile_return_code": profile_return_code,
        "nsys_report": str(report_path) if report_path.is_file() else None,
        "nsys_log": str(log_path),
        "contains_processes": ["RunKV", "Flux"],
        "capture_range": "RunKV cudaProfilerStart/Stop",
        "runkv_manifest": str(path.parent / "runkv_manifest.json"),
        "flux_root": str(FLUX_ROOT),
        "vllm_python": sys.executable,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        _, setting = _select_setting(config_path, args.test_index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.worker:
        if args.case_dir is None or not args.run_tag:
            print(
                "ERROR: internal worker requires --case-dir and --run-tag",
                file=sys.stderr,
            )
            return 2
        return _run_worker(
            setting,
            run_tag=args.run_tag,
            case_dir=args.case_dir.resolve(),
        )

    if shutil.which(args.nsys_cmd) is None:
        print(f"ERROR: nsys command not found: {args.nsys_cmd}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag or _run_tag(setting, args.test_index, timestamp)
    case_dir = _resolve_output_root(args.output_root) / run_tag
    report_stem = case_dir / "runkv_flux_combined"
    report_path = report_stem.with_suffix(".nsys-rep")
    log_path = case_dir / "nsys.log"
    worker_cmd = _worker_command(
        args,
        config_path=config_path,
        run_tag=run_tag,
        case_dir=case_dir,
    )
    cmd = _nsys_command(args, worker_cmd=worker_cmd, report_stem=report_stem)

    print(f"RunKV + Flux combined profile: {run_tag}")
    print(f"  case: {setting.get('name', f'case{args.test_index:03d}')}")
    print(f"  vLLM python: {sys.executable}")
    print(f"  output: {case_dir}")
    print(f"  report: {report_path}")
    print(f"  cmd: {shlex.join(cmd)}")
    if args.dry_run:
        print("\nDry run complete; Flux, RunKV, and nsys were not launched.")
        return 0

    profile_rc = _run_logged(cmd, log_path)
    _write_combined_manifest(
        case_dir / "combined_manifest.json",
        args=args,
        setting=setting,
        run_tag=run_tag,
        report_path=report_path,
        log_path=log_path,
        profile_return_code=profile_rc,
    )
    if profile_rc:
        print(f"\n[ERROR] combined nsys run failed with code {profile_rc}")
        return profile_rc
    if not report_path.is_file():
        print(f"\n[ERROR] nsys completed without creating {report_path}")
        return 1

    print("\nCombined profile complete")
    print(f"  report: {report_path}")
    print(f"  log: {log_path}")
    print(f"  manifest: {case_dir / 'combined_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
