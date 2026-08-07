#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run complete RunKV vs TightLLM benchmark pipeline.

Steps:
  1. Run RunKV feedback observation test  (with nsys profiling)
  2. Run TightLLM ILP planner observation test  (with nsys profiling)
  3. Export nsys .nsys-rep → .sqlite for both
  4. Run per-layer timing analysis script

Basic usage (full pipeline, auto-generated run tag)::

    python scripts/run_benchmark_pipeline.py

Only re-run analysis (skip tests and sqlite export)::

    python scripts/run_benchmark_pipeline.py \\
        --skip-runkv --skip-tightllm --skip-sqlite \\
        --runkv-mfu-glob "exp_results/opt_feedback_observation/opt_component_mfu_*_20260424*.flat.jsonl" \\
        --tightllm-mfu-glob "exp_results/tightllm_observation/opt_component_mfu_*_20260424*.flat.jsonl" \\
        --runkv-sqlite exp_results/sqlite/runkv_20260424.sqlite \\
        --tightllm-sqlite exp_results/sqlite/tightllm_20260424.sqlite

Only re-export sqlite from existing nsys reports::

    python scripts/run_benchmark_pipeline.py \\
        --skip-runkv --skip-tightllm --skip-analysis \\
        --runkv-nsys-rep exp_results/opt_feedback_observation/opt_gap_*.nsys-rep \\
        --tightllm-nsys-rep exp_results/tightllm_observation/tightllm_obs_*.nsys-rep

Set a custom run tag for the next experiment::

    python scripts/run_benchmark_pipeline.py --run-tag my_experiment_v2
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

RUNKV_SCRIPT = ROOT / "examples/offline_inference/run_opt_feedback_observation.py"
TIGHTLLM_SCRIPT = ROOT / "examples/offline_inference/run_tightllm_observation.py"
ANALYSIS_SCRIPT = ROOT / "tools/analyze_per_layer_timing.py"
RUNKV_OUTPUT_DIR = ROOT / "exp_results/opt_feedback_observation"
TIGHTLLM_OUTPUT_DIR = ROOT / "exp_results/tightllm_observation"
SQLITE_OUTPUT_DIR = ROOT / "exp_results/sqlite"
ANALYSIS_OUTPUT_DIR = ROOT / "exp_results/analysis/per_layer"
MANIFEST_DIR = ROOT / "exp_results/manifests"
DEFAULT_TIGHTLLM_PROFILE_ROOT = ROOT / "exp_results/tightllm_profiles"
FLUX_ROOT = Path("/home/lyc/inference/val")
FLUX_VENV_ACTIVATE = Path("diffusion/.venv/bin/activate")
FLUX_MANAGER = Path("diffusion/manage_flux_contender.py")


def _require_plotting_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import matplotlib"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Per-layer analysis is enabled, but matplotlib is not installed in "
            f"{sys.executable}. Install the repository bench dependencies "
            "(`python -m pip install -e '.[bench]'`) before running this "
            "pipeline, or pass --skip-analysis to run without analysis plots."
        )


def _sanitize_token(value: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    )
    return token.strip("_.") or "run"


def _model_tag(model: str) -> str:
    return _sanitize_token(Path(model).name or model)


def _compact_tag_value(value: str) -> str:
    """Make numeric-looking values short and filename friendly."""
    try:
        number = float(value)
    except ValueError:
        return _sanitize_token(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def _default_run_tag(args: argparse.Namespace) -> str:
    """Build a collision-resistant tag that identifies the benchmark setting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        timestamp,
        _model_tag(args.model),
        f"cpu{_compact_tag_value(args.cpu_memory_gb)}",
        f"gu{_compact_tag_value(args.gpu_memory_utilization)}",
        f"gf{_compact_tag_value(args.gpu_memory_fraction)}",
        f"rkcopy-{args.runkv_h2d_copy_mode}",
        f"tlcopy-{args.tightllm_h2d_copy_mode}",
        f"bs{_compact_tag_value(args.num_prompts)}",
        f"p{_compact_tag_value(args.prompt_words)}",
        f"d{_compact_tag_value(args.max_tokens)}",
    ]
    # Keep legacy tags byte-identical for the default policy so existing
    # tooling that matches directory names is unaffected.
    if args.runkv_replay_allocation_policy != "spread":
        parts.insert(6, f"rkalloc-{args.runkv_replay_allocation_policy}")
    if args.tightllm_replay_allocation_policy != "concentrate":
        tl_idx = parts.index(f"tlcopy-{args.tightllm_h2d_copy_mode}") + 1
        parts.insert(tl_idx, f"tlalloc-{args.tightllm_replay_allocation_policy}")
    return "_".join(parts)


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
    """Resolve a glob pattern relative to ROOT, returning sorted matches."""
    paths = _glob.glob(pattern)
    return sorted(paths)


def _find_latest(paths: list[str]) -> str:
    """Return the most recently modified path from a list."""
    if not paths:
        raise FileNotFoundError("No files found matching the pattern")
    return max(paths, key=os.path.getmtime)


def _run_step(
    step_name: str,
    cmd: list[str],
    env: dict[str, str],
    manifest_path: Optional[str],
) -> int:
    """Run a subprocess step, printing progress info.

    Returns the exit code.  If *manifest_path* is set, the env is extended
    with ``MANIFEST_FILE`` so the child process writes a JSON manifest.
    """
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{sep}\n")

    run_env = os.environ.copy()
    run_env.update(env)
    if manifest_path:
        run_env["MANIFEST_FILE"] = manifest_path

    result = subprocess.run(cmd, env=run_env)
    if result.returncode != 0:
        print(f"\n[ERROR] {step_name} failed with code {result.returncode}")
    else:
        print(f"\n[DONE] {step_name}")
    return result.returncode


def _run_flux_command(action: str) -> int:
    """Run the Flux manager in its own virtual environment."""
    manager_args = [f"./{FLUX_MANAGER}", action]
    if action == "start":
        manager_args.extend(["--gpu", "0"])
    shell_command = " ".join(
        [
            "source",
            shlex.quote(str(FLUX_VENV_ACTIVATE)),
            "&&",
            *(shlex.quote(arg) for arg in manager_args),
        ]
    )

    print(f"\n[FLUX] {action}")
    print(f"  cwd: {FLUX_ROOT}")
    print(f"  cmd: {shell_command}")
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", shell_command],
            cwd=FLUX_ROOT,
        )
    except OSError as exc:
        print(f"[ERROR] Flux {action} failed: {exc}")
        return 1
    if result.returncode != 0:
        print(f"[ERROR] Flux {action} failed with code {result.returncode}")
    else:
        print(f"[DONE] Flux {action}")
    return result.returncode


def _run_test_step(
    step_name: str,
    cmd: list[str],
    env: dict[str, str],
    manifest_path: Optional[str],
    *,
    with_flux_contender: bool,
) -> int:
    """Run one benchmark, optionally with a fresh Flux service around it."""
    if not with_flux_contender:
        return _run_step(step_name, cmd, env, manifest_path)

    start_rc = _run_flux_command("start")
    if start_rc != 0:
        print(f"[ERROR] {step_name} was not launched because Flux failed to start")
        return start_rc

    test_rc = 1
    try:
        test_rc = _run_step(step_name, cmd, env, manifest_path)
    finally:
        stop_rc = _run_flux_command("stop")

    if test_rc != 0:
        if stop_rc != 0:
            print(
                f"[ERROR] Flux also failed to stop with code {stop_rc} "
                f"after {step_name} failed"
            )
        return test_rc
    return stop_rc


def _load_manifest(path: str) -> Optional[dict]:
    """Load a JSON manifest file if it exists."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _find_nsys_rep(manifest_path: str, glob_pattern: str) -> str:
    """Find nsys-rep file from manifest first, then fall back to glob."""
    m = _load_manifest(manifest_path)
    if m and m.get("nsys_report") and Path(m["nsys_report"]).exists():
        return m["nsys_report"]
    matches = _resolve_glob(glob_pattern)
    return _find_latest(matches)


def _find_mfu_flat(manifest_path: str, glob_pattern: str) -> str:
    """Find .flat.jsonl file from manifest (glob) or direct glob fallback."""
    m = _load_manifest(manifest_path)
    if m:
        flat_glob = m.get("mfu_flat_jsonl_glob")
        if flat_glob:
            matches = _resolve_glob(flat_glob)
            if matches:
                return _find_latest(matches)
        jsonl_glob = m.get("mfu_jsonl_glob")
        if jsonl_glob:
            matches = _resolve_glob(jsonl_glob)
            if matches:
                return _find_latest(matches)
    matches = _resolve_glob(glob_pattern)
    return _find_latest(matches)


def _check_nsys_available() -> bool:
    """Check if nsys is on PATH."""
    return shutil.which("nsys") is not None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RunKV vs TightLLM benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Pipeline control ──────────────────────────────────────────────────
    ctrl = p.add_argument_group("Pipeline control")
    ctrl.add_argument(
        "--run-tag",
        default="",
        help=(
            "Tag for output file names. The default includes timestamp, model, "
            "memory, batch, prompt, and decode settings."
        ),
    )
    ctrl.add_argument(
        "--skip-runkv", action="store_true", help="Skip RunKV test step"
    )
    ctrl.add_argument(
        "--skip-tightllm", action="store_true", help="Skip TightLLM test step"
    )
    ctrl.add_argument(
        "--skip-sqlite", action="store_true", help="Skip nsys→sqlite export step"
    )
    ctrl.add_argument(
        "--skip-analysis", action="store_true", help="Skip per-layer analysis step"
    )
    ctrl.add_argument(
        "--flux-contender",
        action="store_true",
        help=(
            "Start the Flux contender service before each RunKV/TightLLM test "
            "and stop it immediately afterward."
        ),
    )

    # ── Test parameters ───────────────────────────────────────────────────
    test = p.add_argument_group("Test parameters")
    test.add_argument("--model", default="/data/models/opt-6.7b-8k")
    test.add_argument("--prefix-blocks", default="10000")
    test.add_argument("--num-prompts", default="32")
    test.add_argument("--prompt-words", default="2000")
    test.add_argument("--max-tokens", default="32")
    test.add_argument("--gpu-memory-utilization", default="0.9")
    test.add_argument("--gpu-memory-fraction", default="0.6")
    test.add_argument("--num-device-buffers", default="3")
    test.add_argument(
        "--runkv-h2d-copy-mode",
        choices=("segment", "gather"),
        default="segment",
        help="KV H2D copy implementation for the RunKV benchmark step.",
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
        "--tightllm-h2d-copy-mode",
        choices=("segment", "gather"),
        default="segment",
        help="KV H2D copy implementation for the TightLLM benchmark step.",
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
        default=str(5e10 / (1024**3)),
        help=(
            "Total CPU cache-store budget in GiB; with dynamic replay this "
            "covers both full-layer KV and hidden-state stores. 0 derives "
            "it from --cpu-memory-fraction; --cpu-kv-memory-gb remains an alias."
        ),
    )
    test.add_argument(
        "--cpu-memory-fraction",
        default="0.8",
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

    # ── Path overrides (for re-running specific steps) ────────────────────
    paths = p.add_argument_group("Path overrides (skip auto-discovery)")
    paths.add_argument("--runkv-nsys-rep", default="")
    paths.add_argument("--tightllm-nsys-rep", default="")
    paths.add_argument("--runkv-mfu-glob", default="")
    paths.add_argument("--tightllm-mfu-glob", default="")
    paths.add_argument("--runkv-sqlite", default="")
    paths.add_argument("--tightllm-sqlite", default="")
    paths.add_argument("--analysis-output-dir", default="")

    # ── Analysis parameters ────────────────────────────────────────────────
    ana = p.add_argument_group("Analysis parameters")
    ana.add_argument("--skip-warmup-steps", type=int, default=1)
    ana.add_argument("--compute-stream", type=int, default=7)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.skip_analysis:
        _require_plotting_dependency()
    tightllm_profile_path = _resolve_tightllm_profile_path(args)
    if not args.skip_tightllm:
        _validate_tightllm_profile_path(args, tightllm_profile_path)

    if not _check_nsys_available():
        print(
            "[WARN] nsys not found on PATH — nsys profiling may fail unless "
            "NSYS_CMD is set."
        )

    run_tag = (
        _sanitize_token(args.run_tag) if args.run_tag else _default_run_tag(args)
    )
    manifest_dir = MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runkv_manifest = str(manifest_dir / f"runkv_{run_tag}.json")
    tightllm_manifest = str(manifest_dir / f"tightllm_{run_tag}.json")

    # ── Common environment variables ──────────────────────────────────────
    common_env: dict[str, str] = {
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
        "ENABLE_NVTX": "1",
        "ENABLE_PROFILE": "1",
        "ENABLE_OPT_COMPONENT_MFU_PROFILING": "1",
        "RUNKV_PREHOOK_TIMING": "1",
        "ENABLE_NSYS": "1",
        "NSYS_SAMPLE": "cpu",
        "NSYS_EXTRA_ARGS": "--capture-range=cudaProfilerApi --capture-range-end=stop",
    }

    # ── Step 1: RunKV ─────────────────────────────────────────────────────
    if not args.skip_runkv:
        runkv_env = dict(common_env)
        runkv_env["OUTPUT_DIR"] = str(RUNKV_OUTPUT_DIR)
        runkv_env["DRY_RUN"] = "0"
        runkv_env["USE_STATE_MACHINE"] = "1"
        runkv_env["H2D_COPY_MODE"] = args.runkv_h2d_copy_mode
        runkv_env["REPLAY_ALLOCATION_POLICY"] = args.runkv_replay_allocation_policy
        rc = _run_test_step(
            "RunKV feedback observation",
            [sys.executable, str(RUNKV_SCRIPT)],
            runkv_env,
            runkv_manifest,
            with_flux_contender=args.flux_contender,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("[SKIP] RunKV test")

    # ── Step 2: TightLLM ──────────────────────────────────────────────────
    if not args.skip_tightllm:
        tightllm_env = dict(common_env)
        tightllm_env["OUTPUT_DIR"] = str(TIGHTLLM_OUTPUT_DIR)
        tightllm_env["TIGHTLLM_PROFILE_PATH"] = tightllm_profile_path
        tightllm_env["H2D_COPY_MODE"] = args.tightllm_h2d_copy_mode
        tightllm_env["TIGHTLLM_REPLAY_ALLOCATION_POLICY"] = (
            args.tightllm_replay_allocation_policy
        )
        rc = _run_test_step(
            "TightLLM ILP planner observation",
            [sys.executable, str(TIGHTLLM_SCRIPT)],
            tightllm_env,
            tightllm_manifest,
            with_flux_contender=args.flux_contender,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("[SKIP] TightLLM test")

    # ── Step 3: nsys export → sqlite ──────────────────────────────────────
    sqlite_dir = SQLITE_OUTPUT_DIR
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    runkv_sqlite = args.runkv_sqlite or str(sqlite_dir / f"runkv_{run_tag}.sqlite")
    tightllm_sqlite = args.tightllm_sqlite or str(
        sqlite_dir / f"tightllm_{run_tag}.sqlite"
    )

    if not args.skip_sqlite:
        # Find nsys-rep files
        if args.runkv_nsys_rep:
            runkv_rep = args.runkv_nsys_rep
        else:
            runkv_rep = _find_nsys_rep(
                runkv_manifest,
                str(RUNKV_OUTPUT_DIR / f"opt_gap_*_{run_tag}.nsys-rep"),
            )
        if args.tightllm_nsys_rep:
            tightllm_rep = args.tightllm_nsys_rep
        else:
            tightllm_rep = _find_nsys_rep(
                tightllm_manifest,
                str(TIGHTLLM_OUTPUT_DIR / f"tightllm_obs_*_{run_tag}.nsys-rep"),
            )

        print(f"\n  RunKV    nsys-rep: {runkv_rep}")
        print(f"  TightLLM nsys-rep: {tightllm_rep}")

        nsys_cmd = os.environ.get("NSYS_CMD", "nsys")
        for label, rep, output_path in [
            ("RunKV", runkv_rep, runkv_sqlite),
            ("TightLLM", tightllm_rep, tightllm_sqlite),
        ]:
            if os.path.exists(output_path):
                print(f"  [{label}] Removing existing sqlite: {output_path}")
                os.remove(output_path)
            rc = _run_step(
                f"nsys export → sqlite ({label})",
                [
                    nsys_cmd,
                    "export",
                    "--type",
                    "sqlite",
                    "-o",
                    output_path,
                    rep,
                ],
                {},
                None,
            )
            if rc != 0:
                sys.exit(rc)
    else:
        print("[SKIP] nsys → sqlite export")
        # Still resolve sqlite paths for the analysis step
        if args.runkv_sqlite:
            runkv_sqlite = args.runkv_sqlite
        if args.tightllm_sqlite:
            tightllm_sqlite = args.tightllm_sqlite

    # ── Step 4: Per-layer analysis ─────────────────────────────────────────
    if not args.skip_analysis:
        # Find mfu flat.jsonl files
        if args.runkv_mfu_glob:
            runkv_mfu = _find_latest(_resolve_glob(args.runkv_mfu_glob))
        else:
            runkv_mfu = _find_mfu_flat(
                runkv_manifest,
                str(RUNKV_OUTPUT_DIR / f"opt_component_mfu_*_{run_tag}.flat.jsonl"),
            )
        if args.tightllm_mfu_glob:
            tightllm_mfu = _find_latest(
                _resolve_glob(args.tightllm_mfu_glob)
            )
        else:
            tightllm_mfu = _find_mfu_flat(
                tightllm_manifest,
                str(
                    TIGHTLLM_OUTPUT_DIR
                    / f"opt_component_mfu_*_{run_tag}.flat.jsonl"
                ),
            )

        print(f"\n  RunKV    mfu flat: {runkv_mfu}")
        print(f"  TightLLM mfu flat: {tightllm_mfu}")
        print(f"  RunKV    sqlite:   {runkv_sqlite}")
        print(f"  TightLLM sqlite:   {tightllm_sqlite}")

        output_root = Path(args.analysis_output_dir or ANALYSIS_OUTPUT_DIR)
        output_dir = str(output_root / run_tag)

        analysis_cmd = [
            sys.executable,
            str(ANALYSIS_SCRIPT),
            "--runkv-mfu",
            runkv_mfu,
            "--tightllm-mfu",
            tightllm_mfu,
            "--runkv-sqlite",
            runkv_sqlite,
            "--tightllm-sqlite",
            tightllm_sqlite,
            "--output-dir",
            output_dir,
            "--skip-warmup-steps",
            str(args.skip_warmup_steps),
            "--compute-stream",
            str(args.compute_stream),
            "--num-prompts",
            str(args.num_prompts),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if not args.skip_runkv and not args.skip_tightllm:
            analysis_cmd.append("--fixed-output-length")
        rc = _run_step("Per-layer timing analysis", analysis_cmd, {}, None)
        if rc != 0:
            sys.exit(rc)
    else:
        print("[SKIP] Per-layer analysis")

    print(f"\n{'=' * 68}")
    print(f"  Pipeline complete!")
    print(f"  Run tag: {run_tag}")
    print(f"  Manifests: {manifest_dir}")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    import shutil

    main()
