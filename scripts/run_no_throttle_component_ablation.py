#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run no-pressure RunKV single-disable component ablations.

The full-on reference is intentionally not executed here; it is taken from the
historical manifests recorded in BASELINES. Each new run retains the workload
settings while disabling exactly one component.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/offline_inference/run_opt_feedback_observation.py"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_results/ablation/no_throttle"

BASELINES: dict[str, dict[str, str]] = {
    "1k": {
        "manifest": "exp_results/manifests/runkv_20260525_0143.json",
        "analysis": "exp_results/analysis/per_layer/20260525_0143-3080-2.7b-1k-v1.1",
        "prompt_words": "1000",
        "num_prompts": "32",
        "max_tokens": "128",
        "gpu_memory_fraction": "0.7",
    },
    "2k": {
        "manifest": "exp_results/manifests/runkv_20260525_0225.json",
        "analysis": "exp_results/analysis/per_layer/20260525_0225-3080-2.7b-2k-v1.1",
        "prompt_words": "2000",
        "num_prompts": "32",
        "max_tokens": "128",
        "gpu_memory_fraction": "0.7",
    },
    "4k": {
        "manifest": "exp_results/manifests/runkv_20260525_1634.json",
        "analysis": "exp_results/analysis/per_layer/20260525_1634-3080-2.7b-4k-v1.1",
        "prompt_words": "4000",
        "num_prompts": "16",
        "max_tokens": "32",
        "gpu_memory_fraction": "0.75",
    },
    "8k": {
        "manifest": "exp_results/manifests/runkv_20260526_0136.json",
        "analysis": "exp_results/analysis/per_layer/20260526_0136-3080-2.7b-8k-v1.1",
        "prompt_words": "8000",
        "num_prompts": "16",
        "max_tokens": "32",
        "gpu_memory_fraction": "0.7",
    },
}

VARIANTS: dict[str, dict[str, str | bool]] = {
    "no_async_plan": {
        "ASYNC_PLAN_BUILD": "0",
        "H2D_COPY_MODE": "segment",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "async_spec_plan_build",
    },
    "no_segment_dma": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "gather",
        "USE_STATE_MACHINE": "1",
        "disabled_component": "segment_dma",
    },
    "no_state_machine": {
        "ASYNC_PLAN_BUILD": "1",
        "H2D_COPY_MODE": "segment",
        "USE_STATE_MACHINE": "0",
        "disabled_component": "state_machine",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--gpu-memory-fraction",
        default=None,
        help="Override GPU staging-buffer fraction for selected workloads.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Override request batch size for selected workloads.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without executing.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Within --run-tag, skip runs with a complete manifest and expected artifacts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one short request and two output tokens to validate switch paths.",
    )
    parser.add_argument("--disable-nsys", action="store_true")
    parser.add_argument("--disable-profile", action="store_true")
    parser.add_argument("--skip-sqlite", action="store_true")
    parser.add_argument("--nsys-cmd", default=os.environ.get("NSYS_CMD", "nsys"))
    return parser.parse_args()


def _run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    print(f"  CMD: {' '.join(shlex.quote(part) for part in cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
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
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _export_sqlite(args: argparse.Namespace, manifest: dict[str, Any], out: Path) -> None:
    report = manifest.get("nsys_report")
    if not report:
        raise RuntimeError("Nsight report missing from manifest; cannot export sqlite.")
    cmd = [
        args.nsys_cmd,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "-o",
        str(out),
        str(report),
    ]
    print(f"  SQLITE: {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _is_complete(
    args: argparse.Namespace,
    manifest_path: Path,
    expected_config: dict[str, str],
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    for key, expected in expected_config.items():
        if str(manifest.get(key)) != expected:
            return False
    flat_glob = manifest.get("mfu_flat_jsonl_glob")
    if not flat_glob or not glob.glob(flat_glob):
        return False
    if not args.disable_nsys and not Path(manifest.get("nsys_report", "")).exists():
        return False
    if not args.skip_sqlite and not args.disable_nsys:
        sqlite = manifest.get("sqlite")
        if not sqlite or not Path(sqlite).exists():
            return False
    return True


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    model = "/home/lyc/hf_models/opt-2.7b-8k"

    for workload in args.workloads:
        baseline = BASELINES[workload]
        for variant in args.variants:
            switches = VARIANTS[variant]
            gpu_memory_fraction = (
                args.gpu_memory_fraction or baseline["gpu_memory_fraction"]
            )
            num_prompts = (
                str(args.num_prompts)
                if args.num_prompts is not None and not args.smoke
                else ("1" if args.smoke else baseline["num_prompts"])
            )
            suffix = "_smoke" if args.smoke else ""
            run_tag = f"{args.run_tag}_{workload}_{variant}{suffix}"
            run_dir = output_root / workload / variant / args.run_tag
            manifest_path = run_dir / "manifest.json"
            sqlite_path = run_dir / "runkv.sqlite"
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": model,
                    "PREFIX_BLOCKS": "10000",
                    "NUM_PROMPTS": num_prompts,
                    "PROMPT_WORDS": "64" if args.smoke else baseline["prompt_words"],
                    "MAX_TOKENS": "2" if args.smoke else baseline["max_tokens"],
                    "GPU_MEMORY_FRACTION": gpu_memory_fraction,
                    "GPU_MEMORY_UTILIZATION": "0.9",
                    "NUM_DEVICE_BUFFERS": "3",
                    "CPU_MEMORY_GB": "93.13225746154785",
                    "CPU_MEMORY_FRACTION": "0.6",
                    "PLANNER": "feedback",
                    "DRY_RUN": "0",
                    "ASYNC_PLAN_BUILD": str(switches["ASYNC_PLAN_BUILD"]),
                    "H2D_COPY_MODE": str(switches["H2D_COPY_MODE"]),
                    "USE_STATE_MACHINE": str(switches["USE_STATE_MACHINE"]),
                    "ENABLE_NVTX": "1",
                    "ENABLE_PROFILE": "0" if args.disable_profile else "1",
                    "ENABLE_OPT_COMPONENT_MFU_PROFILING": "1",
                    "RUNKV_PREHOOK_TIMING": "1",
                    "RUNKV_PREHOOK_TIMING_DIR": str(run_dir / "prehook_timing"),
                    "ENABLE_NSYS": "0" if args.disable_nsys else "1",
                    "NSYS_SAMPLE": "cpu",
                    "NSYS_EXTRA_ARGS": (
                        "--capture-range=cudaProfilerApi --capture-range-end=stop"
                    ),
                    "NSYS_OUTPUT_DIR": str(run_dir),
                    "OUTPUT_DIR": str(run_dir),
                    "RUN_TAG": run_tag,
                    "MANIFEST_FILE": str(manifest_path),
                }
            )
            cmd = [sys.executable, str(RUNNER)]
            print(f"\n[{workload}/{variant}] output={run_dir}")
            expected_config = {
                "num_prompts": num_prompts,
                "prompt_words": "64" if args.smoke else baseline["prompt_words"],
                "max_tokens": "2" if args.smoke else baseline["max_tokens"],
                "gpu_memory_fraction": gpu_memory_fraction,
                "gpu_memory_utilization": "0.9",
                "layer_recompute_async_plan_build": str(
                    switches["ASYNC_PLAN_BUILD"] == "1"
                ),
                "h2d_copy_mode": str(switches["H2D_COPY_MODE"]),
                "layer_recompute_use_state_machine": str(
                    switches["USE_STATE_MACHINE"] == "1"
                ),
            }
            if args.resume and _is_complete(args, manifest_path, expected_config):
                print("  SKIP: completed run already exists for this tag")
                continue
            if args.dry_run:
                print(f"  CMD: {' '.join(shlex.quote(part) for part in cmd)}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            _run_logged(cmd, env, run_dir / "run.log")
            manifest = _load_json(manifest_path)
            manifest.update(
                {
                    "ablation": "no_throttle_single_disable",
                    "workload": workload,
                    "variant": variant,
                    "disabled_component": switches["disabled_component"],
                    "resource_pressure_kind": "none",
                    "smoke": args.smoke,
                    "baseline_manifest": str((ROOT / baseline["manifest"]).resolve()),
                    "baseline_analysis": str((ROOT / baseline["analysis"]).resolve()),
                }
            )
            if not args.skip_sqlite and not args.disable_nsys:
                _export_sqlite(args, manifest, sqlite_path)
                manifest["sqlite"] = str(sqlite_path.resolve())
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"  DONE: {manifest_path}")


if __name__ == "__main__":
    main()
