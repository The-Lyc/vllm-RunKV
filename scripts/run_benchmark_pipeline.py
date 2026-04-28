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
        default=datetime.now().strftime("%Y%m%d_%H%M"),
        help="Tag for output file names (default: YYYYMMDD_HHMM)",
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

    # ── Test parameters ───────────────────────────────────────────────────
    test = p.add_argument_group("Test parameters")
    test.add_argument("--model", default="/home/lyc/hf_models/opt-2.7b-8k")
    test.add_argument("--prefix-blocks", default="1000")
    test.add_argument("--num-prompts", default="32")
    test.add_argument("--prompt-words", default="4000")
    test.add_argument("--max-tokens", default="32")
    test.add_argument("--gpu-memory-fraction", default="0.9")
    test.add_argument("--num-device-buffers", default="3")
    test.add_argument(
        "--tightllm-profile-path", default="tightllm_profile.json"
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

    if not _check_nsys_available():
        print(
            "[WARN] nsys not found on PATH — nsys profiling may fail unless "
            "NSYS_CMD is set."
        )

    run_tag = args.run_tag
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
        "GPU_MEMORY_FRACTION": args.gpu_memory_fraction,
        "NUM_DEVICE_BUFFERS": args.num_device_buffers,
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
        rc = _run_step(
            "RunKV feedback observation",
            [sys.executable, str(RUNKV_SCRIPT)],
            runkv_env,
            runkv_manifest,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("[SKIP] RunKV test")

    # ── Step 2: TightLLM ──────────────────────────────────────────────────
    if not args.skip_tightllm:
        tightllm_env = dict(common_env)
        tightllm_env["OUTPUT_DIR"] = str(TIGHTLLM_OUTPUT_DIR)
        tightllm_env["TIGHTLLM_PROFILE_PATH"] = args.tightllm_profile_path
        rc = _run_step(
            "TightLLM ILP planner observation",
            [sys.executable, str(TIGHTLLM_SCRIPT)],
            tightllm_env,
            tightllm_manifest,
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

        output_dir = args.analysis_output_dir or str(
            ANALYSIS_OUTPUT_DIR / run_tag
        )

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
        ]
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
