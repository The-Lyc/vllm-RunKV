#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run benchmark pipeline settings sequentially from a JSON file."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts/run_benchmark_pipeline.py"
DEFAULT_LOG_ROOT = ROOT / "exp_results/logs/benchmark_batch"
H2D_COPY_MODES = ("segment", "gather")
REPLAY_ALLOCATION_POLICIES = ("spread", "concentrate")
TIGHTLLM_ALLOCATION_POLICIES = ("concentrate", "spread")

REQUIRED_FIELDS = (
    "model",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "gpu_memory_utilization",
    "gpu_memory_fraction",
    "tightllm_profile",
    "batch_size",
    "prompt_length",
    "decode_length",
)


def _sanitize(value: Any) -> str:
    text = str(value).replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _tag_number(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p")


def _load_settings(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text())
    defaults = config.get("defaults", {})
    tests = config.get("tests", [])
    if not isinstance(defaults, dict) or not isinstance(tests, list) or not tests:
        raise ValueError("config requires an object 'defaults' and a non-empty 'tests'")

    settings = []
    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise ValueError(f"tests[{index}] must be an object")
        setting = {**defaults, **test}
        missing = [key for key in REQUIRED_FIELDS if key not in setting]
        if missing:
            raise ValueError(f"tests[{index}] is missing: {', '.join(missing)}")
        for key in ("runkv_h2d_copy_mode", "tightllm_h2d_copy_mode"):
            mode = setting.get(key, "segment")
            if mode not in H2D_COPY_MODES:
                choices = ", ".join(H2D_COPY_MODES)
                raise ValueError(
                    f"tests[{index}].{key} must be one of: {choices}; got {mode!r}"
                )
            setting[key] = mode
        policy = setting.get("runkv_replay_allocation_policy", "spread")
        if policy not in REPLAY_ALLOCATION_POLICIES:
            choices = ", ".join(REPLAY_ALLOCATION_POLICIES)
            raise ValueError(
                f"tests[{index}].runkv_replay_allocation_policy must be one "
                f"of: {choices}; got {policy!r}"
            )
        setting["runkv_replay_allocation_policy"] = policy
        tl_policy = setting.get("tightllm_replay_allocation_policy", "concentrate")
        if tl_policy not in TIGHTLLM_ALLOCATION_POLICIES:
            choices = ", ".join(TIGHTLLM_ALLOCATION_POLICIES)
            raise ValueError(
                f"tests[{index}].tightllm_replay_allocation_policy must be "
                f"one of: {choices}; got {tl_policy!r}"
            )
        setting["tightllm_replay_allocation_policy"] = tl_policy
        settings.append(setting)
    return config, settings


def _run_tag(setting: dict[str, Any], index: int, timestamp: str) -> str:
    model = Path(str(setting["model"])).name
    name = setting.get("name", f"case{index:03d}")
    parts = [
        timestamp,
        f"j{index:03d}",
        _sanitize(name),
        _sanitize(model),
        f"cpu{_tag_number(setting['cpu_memory_gb'])}",
        f"gu{_tag_number(setting['gpu_memory_utilization'])}",
        f"gf{_tag_number(setting['gpu_memory_fraction'])}",
        f"rkcopy-{_sanitize(setting['runkv_h2d_copy_mode'])}",
        f"tlcopy-{_sanitize(setting['tightllm_h2d_copy_mode'])}",
        f"bs{_sanitize(setting['batch_size'])}",
        f"p{_sanitize(setting['prompt_length'])}",
        f"d{_sanitize(setting['decode_length'])}",
    ]
    # Keep legacy tags byte-identical for the default policy.
    if setting["runkv_replay_allocation_policy"] != "spread":
        parts.insert(
            8, f"rkalloc-{_sanitize(setting['runkv_replay_allocation_policy'])}"
        )
    if setting["tightllm_replay_allocation_policy"] != "concentrate":
        tl_idx = (
            parts.index(f"tlcopy-{_sanitize(setting['tightllm_h2d_copy_mode'])}") + 1
        )
        parts.insert(
            tl_idx,
            f"tlalloc-{_sanitize(setting['tightllm_replay_allocation_policy'])}",
        )
    return "_".join(parts)


def _command(setting: dict[str, Any], run_tag: str) -> list[str]:
    profile = Path(str(setting["tightllm_profile"])).expanduser()
    if not profile.is_absolute():
        profile = ROOT / profile
    if not profile.is_file():
        raise ValueError(f"TightLLM profile not found: {profile}")

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--run-tag",
        run_tag,
        "--model",
        str(setting["model"]),
        "--prefix-blocks",
        str(setting.get("prefix_blocks", 10000)),
        "--num-prompts",
        str(setting["batch_size"]),
        "--prompt-words",
        str(setting["prompt_length"]),
        "--max-tokens",
        str(setting["decode_length"]),
        "--cpu-memory-gb",
        str(setting["cpu_memory_gb"]),
        "--cpu-memory-fraction",
        str(setting["cpu_memory_fraction"]),
        "--gpu-memory-utilization",
        str(setting["gpu_memory_utilization"]),
        "--gpu-memory-fraction",
        str(setting["gpu_memory_fraction"]),
        "--num-device-buffers",
        str(setting.get("num_device_buffers", 3)),
        "--runkv-h2d-copy-mode",
        str(setting["runkv_h2d_copy_mode"]),
        "--runkv-replay-allocation-policy",
        str(setting["runkv_replay_allocation_policy"]),
        "--tightllm-h2d-copy-mode",
        str(setting["tightllm_h2d_copy_mode"]),
        "--tightllm-replay-allocation-policy",
        str(setting["tightllm_replay_allocation_policy"]),
        "--tightllm-profile-path",
        str(profile),
    ]
    for key, option in (
        ("max_num_seqs", "--max-num-seqs"),
        ("max_staging_blocks", "--max-staging-blocks"),
        ("hardware_platform", "--hardware-platform"),
    ):
        if setting.get(key) not in (None, ""):
            cmd.extend([option, str(setting[key])])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        config, settings = _load_settings(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    log_root = Path(config.get("log_root", DEFAULT_LOG_ROOT)).expanduser()
    if not log_root.is_absolute():
        log_root = ROOT / log_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    succeeded: list[tuple[str, Path]] = []
    failed: list[tuple[str, str, Path]] = []

    for index, setting in enumerate(settings, start=1):
        name = str(setting.get("name", f"case{index:03d}"))
        fallback_dir = f"{timestamp}_j{index:03d}_{_sanitize(name)}"
        log_path = log_root / fallback_dir / "pipeline.log"
        try:
            run_tag = _run_tag(setting, index, timestamp)
            cmd = _command(setting, run_tag)
            log_path = log_root / run_tag / "pipeline.log"
            print(f"\n[{index}/{len(settings)}] {name}")
            print(f"  log: {log_path}")
            print(f"  cmd: {shlex.join(cmd)}")
            if args.dry_run:
                continue

            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode != 0:
                reason = f"exit code {result.returncode}"
                failed.append((name, reason, log_path))
                print(f"  FAILED ({reason}); continuing")
            else:
                succeeded.append((name, log_path))
                print("  DONE")
        except (OSError, ValueError) as exc:
            failed.append((name, str(exc), log_path))
            print(f"  FAILED ({exc}); continuing")

    if args.dry_run:
        print("\nDry run complete; no tests were launched.")
        return 0

    print("\nBatch summary")
    print(f"  succeeded: {len(succeeded)}")
    for name, log_path in succeeded:
        print(f"    PASS {name}: {log_path}")
    print(f"  failed: {len(failed)}")
    for name, reason, log_path in failed:
        print(f"    FAIL {name} ({reason}): {log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
