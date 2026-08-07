#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one Llama RunKV vs TightLLM benchmark and collect profiling artifacts.

This is the Llama counterpart of ``scripts/run_benchmark_pipeline.py``.
It runs one RunKV planner (``feedback`` or ``static``) and the model-aware
TightLLM ILP planner, optionally exports both nsys reports to sqlite, and runs
the existing paired per-layer analyzer.
"""

from __future__ import annotations

import argparse
import glob
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
RUNKV_RUNNER = ROOT / "examples/offline_inference/run_llama_feedback_observation.py"
TIGHTLLM_RUNNER = ROOT / "examples/offline_inference/run_tightllm_observation.py"
ANALYZER = ROOT / "tools/analyze_per_layer_timing.py"
DEFAULT_MODEL = Path("/data/models/Llama-2-7b-hf-8k")
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/llama_benchmark"
DEFAULT_ANALYSIS_ROOT = ROOT / "exp_results/analysis/llama_per_layer"
DEFAULT_TIGHTLLM_PROFILE_ROOT = ROOT / "exp_results/tightllm_profiles"
RUNKV_COMPONENT_PREFIX = "llama_runkv_component"
TIGHTLLM_COMPONENT_PREFIX = "llama_tightllm_component"


def _sanitize(value: Any) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)
    )
    return token.strip("_.") or "run"


def _compact_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _sanitize(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def _default_run_tag(args: argparse.Namespace) -> str:
    return "_".join(
        [
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            _sanitize(Path(args.model).name or args.model),
            args.planner,
            f"pb{_sanitize(args.prefix_blocks)}",
            f"cpu{_compact_number(args.cpu_memory_gb)}",
            f"gu{_compact_number(args.gpu_memory_utilization)}",
            f"gf{_compact_number(args.gpu_memory_fraction)}",
            f"bs{args.num_prompts}",
            f"p{args.prompt_words}",
            f"d{args.max_tokens}",
        ]
    )


def _boolean_option(
    group: argparse._ArgumentGroup,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    group.add_argument(
        f"--enable-{name}",
        dest=destination,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--disable-{name}",
        dest=destination,
        action="store_false",
        help=f"Disable {help_text[0].lower()}{help_text[1:]}",
    )
    group.set_defaults(**{destination: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Llama RunKV benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    control = parser.add_argument_group("Pipeline control")
    control.add_argument("--run-tag", default="")
    control.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    control.add_argument("--analysis-output-dir", default="")
    control.add_argument("--skip-runkv", action="store_true")
    control.add_argument("--skip-tightllm", action="store_true")
    control.add_argument("--skip-sqlite-export", action="store_true")
    control.add_argument("--skip-analysis", action="store_true")
    control.add_argument("--dry-run", action="store_true")
    control.add_argument(
        "--flux-contender",
        action="store_true",
        help="Start and stop the repository's Flux contender around the Llama run.",
    )

    test = parser.add_argument_group("Llama RunKV workload")
    test.add_argument("--model", default=str(DEFAULT_MODEL))
    test.add_argument("--planner", choices=("feedback", "static"), default="feedback")
    test.add_argument("--planner-dry-run", action="store_true")
    test.add_argument("--prefix-blocks", default="128")
    test.add_argument(
        "--prompt-word",
        default="the",
        help=(
            "Word repeated to build prompts; 'the' is one token for this "
            "Llama tokenizer."
        ),
    )
    test.add_argument("--num-prompts", type=int, default=8)
    test.add_argument("--prompt-words", type=int, default=1000)
    test.add_argument("--max-tokens", type=int, default=64)
    test.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    test.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    test.add_argument("--num-device-buffers", type=int, default=3)
    test.add_argument("--max-num-seqs", type=int, default=None)
    test.add_argument("--max-staging-blocks", type=int, default=None)
    test.add_argument("--cpu-memory-gb", type=float, default=47.9)
    test.add_argument("--cpu-memory-fraction", type=float, default=0.8)
    test.add_argument(
        "--h2d-copy-mode", choices=("segment", "gather"), default="segment"
    )
    test.add_argument(
        "--replay-allocation-policy",
        choices=("spread", "concentrate"),
        default="concentrate",
    )
    test.add_argument(
        "--tightllm-h2d-copy-mode",
        choices=("segment", "gather"),
        default="segment",
    )
    test.add_argument(
        "--tightllm-replay-allocation-policy",
        choices=("concentrate", "spread"),
        default="concentrate",
    )
    test.add_argument("--tightllm-feedback-correction", action="store_true")
    test.add_argument("--hardware-platform", default="")
    test.add_argument(
        "--tightllm-profile-root",
        default=str(DEFAULT_TIGHTLLM_PROFILE_ROOT),
    )
    test.add_argument("--tightllm-profile-path", default="")
    test.add_argument(
        "--use-state-machine",
        dest="use_state_machine",
        action="store_true",
    )
    test.add_argument(
        "--no-state-machine",
        dest="use_state_machine",
        action="store_false",
    )
    test.set_defaults(use_state_machine=True)
    test.add_argument(
        "--async-plan-build",
        dest="async_plan_build",
        action="store_true",
    )
    test.add_argument(
        "--no-async-plan-build",
        dest="async_plan_build",
        action="store_false",
    )
    test.set_defaults(async_plan_build=True)

    profiling = parser.add_argument_group("Profiling")
    _boolean_option(
        profiling,
        "nsys",
        default=True,
        help_text="Capture an Nsight Systems report.",
    )
    _boolean_option(
        profiling,
        "nvtx",
        default=True,
        help_text="Emit NVTX ranges.",
    )
    _boolean_option(
        profiling,
        "profile",
        default=True,
        help_text="Use the CUDA profiler capture range.",
    )
    _boolean_option(
        profiling,
        "component-timing",
        default=True,
        help_text="Write per-step and per-layer RunKV timing JSONL.",
    )
    _boolean_option(
        profiling,
        "prehook-timing",
        default=True,
        help_text="Write RunKV pre-hook timing records.",
    )
    profiling.add_argument("--nsys-sample", default="cpu")
    profiling.add_argument(
        "--nsys-extra-args",
        default="--capture-range=cudaProfilerApi --capture-range-end=stop",
    )

    paths = parser.add_argument_group("Existing artifact overrides")
    paths.add_argument("--runkv-manifest", default="")
    paths.add_argument("--runkv-nsys-rep", default="")
    paths.add_argument("--runkv-component-glob", default="")
    paths.add_argument("--runkv-sqlite", default="")
    paths.add_argument("--tightllm-manifest", default="")
    paths.add_argument("--tightllm-nsys-rep", default="")
    paths.add_argument("--tightllm-component-glob", default="")
    paths.add_argument("--tightllm-sqlite", default="")
    paths.add_argument("--skip-warmup-steps", type=int, default=1)
    paths.add_argument("--compute-stream", type=int, default=7)
    return parser


def _resolve_tightllm_profile_path(args: argparse.Namespace) -> Path:
    if args.tightllm_profile_path:
        path = Path(args.tightllm_profile_path).expanduser()
    elif args.hardware_platform:
        path = (
            Path(args.tightllm_profile_root).expanduser()
            / _sanitize(args.hardware_platform.lower())
            / f"{Path(args.model).name}.json"
        )
    else:
        path = Path("tightllm_profile.json")
    return path if path.is_absolute() else ROOT / path


def _validate_tightllm_profile(path: Path, model: str) -> None:
    if not path.is_file():
        raise SystemExit(
            f"ERROR: Llama TightLLM profile not found: {path}\n"
            "Generate it with:\n"
            "  .venv/bin/python -m "
            "vllm.v1.profiling.tightllm_offline_profiler "
            f"--model {model} --output {path} "
            "--seq-lengths 128 256 512 1024 2048 4096 8192"
        )
    try:
        profile = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: invalid TightLLM profile {path}: {exc}") from exc
    if profile.get("model_type") != "llama":
        raise SystemExit(
            f"ERROR: TightLLM profile {path} is not a model-aware Llama "
            "profile. Regenerate it with the current offline profiler."
        )


def _validate(args: argparse.Namespace) -> None:
    for name in ("num_prompts", "prompt_words", "max_tokens", "num_device_buffers"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"ERROR: --{name.replace('_', '-')} must be > 0")
    for name in ("max_num_seqs", "max_staging_blocks"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"ERROR: --{name.replace('_', '-')} must be > 0")
    if args.cpu_memory_gb < 0:
        raise SystemExit("ERROR: --cpu-memory-gb must be >= 0")
    if not 0 < args.cpu_memory_fraction <= 1:
        raise SystemExit("ERROR: --cpu-memory-fraction must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("ERROR: --gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise SystemExit("ERROR: --gpu-memory-fraction must be in (0, 1]")
    if not args.prompt_word.strip():
        raise SystemExit("ERROR: --prompt-word cannot be empty")
    try:
        prefix_blocks = int(args.prefix_blocks)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "ERROR: --prefix-blocks must be one non-negative integer; "
            "the Llama benchmark pipeline does not support 'baseline' or "
            "comma-separated sweeps"
        ) from exc
    if prefix_blocks < 0:
        raise SystemExit("ERROR: --prefix-blocks must be >= 0")
    if args.planner_dry_run and args.planner != "feedback":
        raise SystemExit(
            "ERROR: --planner-dry-run is only valid with --planner feedback"
        )
    if not args.skip_analysis and not args.component_timing:
        raise SystemExit(
            "ERROR: analysis requires --enable-component-timing; "
            "pass --skip-analysis or enable component timing"
        )
    if not args.skip_tightllm and not args.dry_run:
        _validate_tightllm_profile(
            _resolve_tightllm_profile_path(args),
            args.model,
        )

    config_path = Path(args.model) / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"ERROR: cannot read model config {config_path}: {exc}"
            ) from exc
        model_type = config.get("model_type")
        if model_type != "llama":
            raise SystemExit(
                f"ERROR: Llama pipeline requires model_type='llama'; got {model_type!r}"
            )


def _run_step(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {shlex.join(command)}")
    print(f"{'=' * 72}\n")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(command, cwd=ROOT, env=run_env)
    if result.returncode:
        raise SystemExit(result.returncode)


def _run_observation(
    step_name: str,
    command: list[str],
    env: dict[str, str],
    *,
    flux_contender: bool,
) -> None:
    if not flux_contender:
        _run_step(step_name, command, env=env)
        return

    # Reuse the exact Flux lifecycle already used by the OPT pipeline so the
    # two model suites exercise the same contender process.
    from run_benchmark_pipeline import _run_flux_command

    start_rc = _run_flux_command("start")
    if start_rc:
        raise SystemExit(start_rc)
    try:
        _run_step(f"{step_name} with Flux contender", command, env=env)
    finally:
        stop_rc = _run_flux_command("stop")
    if stop_rc:
        raise SystemExit(stop_rc)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Run manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid run manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run manifest must contain an object: {path}")
    return value


def _latest(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No artifacts match: {pattern}")
    return max(matches, key=os.path.getmtime)


def _resolve_component_flat(
    override: str,
    manifest: dict[str, Any],
    run_dir: Path,
    run_tag: str,
    artifact_prefix: str,
) -> str:
    pattern = override or manifest.get("mfu_flat_jsonl_glob")
    if not pattern:
        pattern = str(run_dir / f"{artifact_prefix}_*_{run_tag}.flat.jsonl")
    return _latest(str(pattern))


def _resolve_nsys_report(
    override: str,
    manifest: dict[str, Any],
) -> str:
    report = override or manifest.get("nsys_report")
    if not report:
        raise FileNotFoundError("Run manifest does not contain an nsys report")
    if not Path(report).is_file():
        raise FileNotFoundError(f"nsys report not found: {report}")
    return str(report)


def _collect_artifacts(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    manifest_path: Path,
    component_override: str,
    nsys_override: str,
    sqlite_override: str,
    artifact_prefix: str,
    system_label: str,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    artifacts: dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "component_jsonl": None,
        "component_flat_jsonl": None,
        "nsys_report": None,
        "nsys_sqlite": None,
        "analysis_dir": None,
    }
    if args.component_timing:
        artifacts["component_flat_jsonl"] = _resolve_component_flat(
            component_override,
            manifest,
            run_dir,
            args.run_tag,
            artifact_prefix,
        )
        jsonl_pattern = manifest.get("mfu_jsonl_glob")
        if jsonl_pattern:
            candidates = [
                path
                for path in glob.glob(str(jsonl_pattern))
                if ".flat." not in path
            ]
            if candidates:
                artifacts["component_jsonl"] = max(
                    candidates,
                    key=os.path.getmtime,
                )

    sqlite_path = (
        Path(sqlite_override).expanduser()
        if sqlite_override
        else run_dir / f"{_sanitize(system_label)}_{args.run_tag}.sqlite"
    )
    if args.nsys:
        report = _resolve_nsys_report(nsys_override, manifest)
        artifacts["nsys_report"] = report
        if not args.skip_sqlite_export:
            nsys_cmd = os.environ.get("NSYS_CMD", "nsys")
            if shutil.which(nsys_cmd) is None:
                raise SystemExit(
                    f"ERROR: cannot export nsys report; {nsys_cmd!r} is not on PATH"
                )
            if sqlite_path.exists():
                sqlite_path.unlink()
            _run_step(
                f"Export {system_label} nsys report to sqlite",
                [
                    nsys_cmd,
                    "export",
                    "--type",
                    "sqlite",
                    "-o",
                    str(sqlite_path),
                    report,
                ],
            )
            artifacts["nsys_sqlite"] = str(sqlite_path)
        elif sqlite_path.is_file():
            artifacts["nsys_sqlite"] = str(sqlite_path)
        else:
            print(f"[SKIP] {system_label} nsys sqlite export")
    return artifacts


def _write_pipeline_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    systems: dict[str, dict[str, Any]],
    analysis_dir: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_tag": args.run_tag,
        "model_family": "llama",
        "model": args.model,
        "planner": args.planner,
        "settings": vars(args),
        "tightllm_profile": str(_resolve_tightllm_profile_path(args)),
        "systems": systems,
        # Preserve the original single-system fields for existing consumers.
        "run_manifest": systems.get("runkv", {}).get("manifest"),
        "artifacts": systems.get("runkv", {}),
        "analysis_dir": str(analysis_dir) if analysis_dir else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    _validate(args)
    args.run_tag = _sanitize(args.run_tag) if args.run_tag else _default_run_tag(args)

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    case_dir = output_root / args.run_tag
    runkv_dir = case_dir / f"runkv_{args.planner}"
    tightllm_dir = case_dir / "tightllm_replay"
    runkv_manifest_path = (
        Path(args.runkv_manifest).expanduser()
        if args.runkv_manifest
        else runkv_dir / "manifest.json"
    )
    tightllm_manifest_path = (
        Path(args.tightllm_manifest).expanduser()
        if args.tightllm_manifest
        else tightllm_dir / "manifest.json"
    )
    analysis_dir = (
        Path(args.analysis_output_dir).expanduser()
        if args.analysis_output_dir
        else DEFAULT_ANALYSIS_ROOT / args.run_tag
    )
    pipeline_manifest_path = case_dir / "pipeline_manifest.json"

    tightllm_profile_path = _resolve_tightllm_profile_path(args)
    common_env = {
        "RUN_TAG": args.run_tag,
        "MODEL": args.model,
        "MODEL_FAMILY": "llama",
        "PREFIX_BLOCKS": str(args.prefix_blocks),
        "PROMPT_WORD": args.prompt_word,
        "NUM_PROMPTS": str(args.num_prompts),
        "PROMPT_WORDS": str(args.prompt_words),
        "MAX_TOKENS": str(args.max_tokens),
        "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "GPU_MEMORY_FRACTION": str(args.gpu_memory_fraction),
        "NUM_DEVICE_BUFFERS": str(args.num_device_buffers),
        "MAX_NUM_SEQS": "" if args.max_num_seqs is None else str(args.max_num_seqs),
        "MAX_STAGING_BLOCKS": (
            "" if args.max_staging_blocks is None else str(args.max_staging_blocks)
        ),
        "CPU_MEMORY_GB": str(args.cpu_memory_gb),
        "CPU_MEMORY_FRACTION": str(args.cpu_memory_fraction),
        "ENABLE_NSYS": "1" if args.nsys else "0",
        "ENABLE_NVTX": "1" if args.nvtx else "0",
        "ENABLE_PROFILE": "1" if args.profile else "0",
        "ENABLE_COMPONENT_TIMING_PROFILING": (
            "1" if args.component_timing else "0"
        ),
        "ENABLE_OPT_COMPONENT_MFU_PROFILING": (
            "1" if args.component_timing else "0"
        ),
        "RUNKV_PREHOOK_TIMING": "1" if args.prehook_timing else "0",
        "NSYS_SAMPLE": args.nsys_sample,
        "NSYS_EXTRA_ARGS": args.nsys_extra_args,
    }
    runkv_env = {
        **common_env,
        "OUTPUT_DIR": str(runkv_dir),
        "NSYS_OUTPUT_DIR": str(runkv_dir),
        "PLANNER": args.planner,
        "DRY_RUN": "1" if args.planner_dry_run else "0",
        "USE_STATE_MACHINE": "1" if args.use_state_machine else "0",
        "ASYNC_PLAN_BUILD": "1" if args.async_plan_build else "0",
        "H2D_COPY_MODE": args.h2d_copy_mode,
        "REPLAY_ALLOCATION_POLICY": args.replay_allocation_policy,
        "COMPONENT_ARTIFACT_PREFIX": RUNKV_COMPONENT_PREFIX,
        "RUNKV_PREHOOK_TIMING_DIR": str(runkv_dir / "prehook_timing"),
        "MANIFEST_FILE": str(runkv_manifest_path),
    }
    tightllm_env = {
        **common_env,
        "OUTPUT_DIR": str(tightllm_dir),
        "NSYS_OUTPUT_DIR": str(tightllm_dir),
        "TIGHTLLM_PROFILE_PATH": str(tightllm_profile_path),
        "TIGHTLLM_FEEDBACK_CORRECTION": (
            "1" if args.tightllm_feedback_correction else "0"
        ),
        "TIGHTLLM_REPLAY_ALLOCATION_POLICY": (
            args.tightllm_replay_allocation_policy
        ),
        "H2D_COPY_MODE": args.tightllm_h2d_copy_mode,
        "COMPONENT_ARTIFACT_PREFIX": TIGHTLLM_COMPONENT_PREFIX,
        "RUNKV_PREHOOK_TIMING_DIR": str(tightllm_dir / "prehook_timing"),
        "MANIFEST_FILE": str(tightllm_manifest_path),
    }
    runkv_command = [sys.executable, str(RUNKV_RUNNER)]
    tightllm_command = [sys.executable, str(TIGHTLLM_RUNNER)]

    print("Llama RunKV vs TightLLM benchmark pipeline")
    print(f"  model:        {args.model}")
    print(f"  planner:      {args.planner}")
    print(f"  prompt_word:  {args.prompt_word!r}")
    print(
        "  workload:     "
        f"bs={args.num_prompts}, p={args.prompt_words}, d={args.max_tokens}"
    )
    print(f"  run_tag:      {args.run_tag}")
    print(f"  output:       {case_dir}")
    print(f"  profile:      {tightllm_profile_path}")

    if args.dry_run:
        for label, command, env, skipped in (
            ("RunKV", runkv_command, runkv_env, args.skip_runkv),
            ("TightLLM", tightllm_command, tightllm_env, args.skip_tightllm),
        ):
            if skipped:
                print(f"\n  {label}: skipped")
                continue
            print(f"\n  {label} command: {shlex.join(command)}")
            for key in sorted(env):
                print(f"  {label} env {key}={shlex.quote(env[key])}")
        if not args.skip_tightllm and not tightllm_profile_path.is_file():
            print(
                "\n  TightLLM profile is not present yet; generate it before "
                "a non-dry run."
            )
        print("\nDry run complete; no benchmark was launched.")
        return 0

    systems: dict[str, dict[str, Any]] = {}
    if not args.skip_runkv:
        runkv_dir.mkdir(parents=True, exist_ok=True)
        _run_observation(
            "Llama RunKV observation",
            runkv_command,
            runkv_env,
            flux_contender=args.flux_contender,
        )
    else:
        print("[SKIP] Llama RunKV observation")
    if not args.skip_tightllm:
        tightllm_dir.mkdir(parents=True, exist_ok=True)
        _run_observation(
            "Llama TightLLM observation",
            tightllm_command,
            tightllm_env,
            flux_contender=args.flux_contender,
        )
    else:
        print("[SKIP] Llama TightLLM observation")

    if not args.skip_runkv or args.runkv_manifest:
        systems["runkv"] = _collect_artifacts(
            args=args,
            run_dir=runkv_dir,
            manifest_path=runkv_manifest_path,
            component_override=args.runkv_component_glob,
            nsys_override=args.runkv_nsys_rep,
            sqlite_override=args.runkv_sqlite,
            artifact_prefix=RUNKV_COMPONENT_PREFIX,
            system_label="llama_runkv",
        )
    if not args.skip_tightllm or args.tightllm_manifest:
        systems["tightllm"] = _collect_artifacts(
            args=args,
            run_dir=tightllm_dir,
            manifest_path=tightllm_manifest_path,
            component_override=args.tightllm_component_glob,
            nsys_override=args.tightllm_nsys_rep,
            sqlite_override=args.tightllm_sqlite,
            artifact_prefix=TIGHTLLM_COMPONENT_PREFIX,
            system_label="llama_tightllm",
        )
    if not args.nsys:
        print("[SKIP] nsys sqlite export because nsys capture is disabled")

    completed_analysis_dir: Path | None = None
    if not args.skip_analysis:
        matplotlib_check = subprocess.run(
            [sys.executable, "-c", "import matplotlib"],
            capture_output=True,
            text=True,
        )
        if matplotlib_check.returncode:
            raise SystemExit(
                "ERROR: per-layer analysis requires matplotlib; install bench "
                "dependencies or pass --skip-analysis"
            )
        analysis_command = [
            sys.executable,
            str(ANALYZER),
            "--output-dir",
            str(analysis_dir),
            "--skip-warmup-steps",
            str(args.skip_warmup_steps),
            "--compute-stream",
            str(args.compute_stream),
            "--num-prompts",
            str(args.num_prompts),
            "--max-tokens",
            str(args.max_tokens),
            "--fixed-output-length",
        ]
        for system, mfu_option, sqlite_option in (
            ("runkv", "--runkv-mfu", "--runkv-sqlite"),
            ("tightllm", "--tightllm-mfu", "--tightllm-sqlite"),
        ):
            artifacts = systems.get(system)
            if not artifacts:
                continue
            analysis_command.extend(
                [mfu_option, str(artifacts["component_flat_jsonl"])]
            )
            if artifacts["nsys_sqlite"]:
                analysis_command.extend(
                    [sqlite_option, str(artifacts["nsys_sqlite"])]
                )
        if not systems:
            raise SystemExit("ERROR: no RunKV or TightLLM artifacts to analyze")
        _run_step("Analyze Llama RunKV vs TightLLM timing", analysis_command)
        completed_analysis_dir = analysis_dir
        for artifacts in systems.values():
            artifacts["analysis_dir"] = str(analysis_dir)
    else:
        print("[SKIP] per-layer analysis")

    _write_pipeline_manifest(
        pipeline_manifest_path,
        args=args,
        systems=systems,
        analysis_dir=completed_analysis_dir,
    )
    print(f"\nPipeline complete: {pipeline_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
