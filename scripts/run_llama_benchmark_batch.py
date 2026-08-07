#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Llama RunKV benchmark settings sequentially from a JSON file."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts/run_llama_benchmark_pipeline.py"
DEFAULT_LOG_ROOT = ROOT / "exp_results/logs/llama_benchmark_batch"

ALIASES = {
    "batch_size": "num_prompts",
    "prompt_length": "prompt_words",
    "decode_length": "max_tokens",
    "runkv_h2d_copy_mode": "h2d_copy_mode",
    "runkv_replay_allocation_policy": "replay_allocation_policy",
    "tightllm_profile": "tightllm_profile_path",
}
WORKLOAD_FIELDS = {"num_prompts", "prompt_words", "max_tokens"}
REQUIRED_FIELDS = (
    "model",
    "prompt_word",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "gpu_memory_utilization",
    "gpu_memory_fraction",
    "num_prompts",
    "prompt_words",
    "max_tokens",
)
VALUE_OPTIONS = {
    "output_root": "--output-root",
    "analysis_output_dir": "--analysis-output-dir",
    "model": "--model",
    "planner": "--planner",
    "prefix_blocks": "--prefix-blocks",
    "prompt_word": "--prompt-word",
    "num_prompts": "--num-prompts",
    "prompt_words": "--prompt-words",
    "max_tokens": "--max-tokens",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "gpu_memory_fraction": "--gpu-memory-fraction",
    "num_device_buffers": "--num-device-buffers",
    "max_num_seqs": "--max-num-seqs",
    "max_staging_blocks": "--max-staging-blocks",
    "cpu_memory_gb": "--cpu-memory-gb",
    "cpu_memory_fraction": "--cpu-memory-fraction",
    "h2d_copy_mode": "--h2d-copy-mode",
    "replay_allocation_policy": "--replay-allocation-policy",
    "tightllm_h2d_copy_mode": "--tightllm-h2d-copy-mode",
    "tightllm_replay_allocation_policy": (
        "--tightllm-replay-allocation-policy"
    ),
    "hardware_platform": "--hardware-platform",
    "tightllm_profile_root": "--tightllm-profile-root",
    "tightllm_profile_path": "--tightllm-profile-path",
    "nsys_sample": "--nsys-sample",
    "nsys_extra_args": "--nsys-extra-args",
    "skip_warmup_steps": "--skip-warmup-steps",
    "compute_stream": "--compute-stream",
}
TRUE_FLAGS = {
    "planner_dry_run": "--planner-dry-run",
    "skip_runkv": "--skip-runkv",
    "skip_tightllm": "--skip-tightllm",
    "skip_sqlite_export": "--skip-sqlite-export",
    "skip_analysis": "--skip-analysis",
    "flux_contender": "--flux-contender",
    "tightllm_feedback_correction": "--tightllm-feedback-correction",
}
BOOLEAN_OPTIONS = {
    "use_state_machine": ("--use-state-machine", "--no-state-machine"),
    "async_plan_build": ("--async-plan-build", "--no-async-plan-build"),
    "enable_nsys": ("--enable-nsys", "--disable-nsys"),
    "enable_nvtx": ("--enable-nvtx", "--disable-nvtx"),
    "enable_profile": ("--enable-profile", "--disable-profile"),
    "enable_component_timing": (
        "--enable-component-timing",
        "--disable-component-timing",
    ),
    "enable_prehook_timing": (
        "--enable-prehook-timing",
        "--disable-prehook-timing",
    ),
}
CHOICES = {
    "planner": {"feedback", "static"},
    "h2d_copy_mode": {"segment", "gather"},
    "replay_allocation_policy": {"spread", "concentrate"},
    "tightllm_h2d_copy_mode": {"segment", "gather"},
    "tightllm_replay_allocation_policy": {"spread", "concentrate"},
}
META_FIELDS = {"name"}
SUPPORTED_FIELDS = (
    set(VALUE_OPTIONS)
    | set(TRUE_FLAGS)
    | set(BOOLEAN_OPTIONS)
    | META_FIELDS
)


def _sanitize(value: Any) -> str:
    text = str(value).replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _normalize(raw: dict[str, Any], location: str) -> dict[str, Any]:
    value = dict(raw)
    for alias, canonical in ALIASES.items():
        if alias not in value:
            continue
        if canonical in value:
            raise ValueError(
                f"{location} cannot set both {alias!r} and {canonical!r}"
            )
        value[canonical] = value.pop(alias)
    return value


def _validate_model(setting: dict[str, Any], index: int) -> None:
    config_path = Path(str(setting["model"])).expanduser() / "config.json"
    if not config_path.is_file():
        return
    try:
        model_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"tests[{index}] cannot read {config_path}: {exc}") from exc
    if model_config.get("model_type") != "llama":
        raise ValueError(
            f"tests[{index}].model must be a Llama model; "
            f"got model_type={model_config.get('model_type')!r}"
        )


def _validate_setting(setting: dict[str, Any], index: int) -> None:
    missing = [key for key in REQUIRED_FIELDS if setting.get(key) in (None, "")]
    if missing:
        raise ValueError(f"tests[{index}] is missing: {', '.join(missing)}")
    unknown = sorted(set(setting) - SUPPORTED_FIELDS)
    if unknown:
        raise ValueError(f"tests[{index}] has unknown fields: {', '.join(unknown)}")
    for key, choices in CHOICES.items():
        if key in setting and setting[key] not in choices:
            raise ValueError(
                f"tests[{index}].{key} must be one of "
                f"{', '.join(sorted(choices))}; got {setting[key]!r}"
            )
    for key in set(TRUE_FLAGS) | set(BOOLEAN_OPTIONS):
        if key in setting and not isinstance(setting[key], bool):
            raise ValueError(f"tests[{index}].{key} must be a boolean")
    if (
        setting.get("planner_dry_run")
        and setting.get("planner", "feedback") != "feedback"
    ):
        raise ValueError(
            f"tests[{index}].planner_dry_run requires planner='feedback'"
        )
    if (
        setting.get("skip_analysis", False) is False
        and setting.get("enable_component_timing", True) is False
    ):
        raise ValueError(
            f"tests[{index}] must enable component timing or set skip_analysis=true"
        )
    if (
        not setting.get("skip_tightllm", False)
        and not setting.get("tightllm_profile_path")
        and not setting.get("hardware_platform")
    ):
        raise ValueError(
            f"tests[{index}] must set tightllm_profile_path or "
            "hardware_platform when TightLLM is enabled"
        )
    _validate_model(setting, index)


def _load_settings(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    defaults = config.get("defaults", {})
    tests = config.get("tests", [])
    if not isinstance(defaults, dict) or not isinstance(tests, list) or not tests:
        raise ValueError("config requires an object 'defaults' and a non-empty 'tests'")
    defaults = _normalize(defaults, "defaults")
    misplaced = sorted(WORKLOAD_FIELDS & set(defaults))
    if misplaced:
        raise ValueError(
            "workload fields must be set in every test, not defaults: "
            + ", ".join(misplaced)
        )

    settings: list[dict[str, Any]] = []
    for index, raw_test in enumerate(tests):
        if not isinstance(raw_test, dict):
            raise ValueError(f"tests[{index}] must be an object")
        test = _normalize(raw_test, f"tests[{index}]")
        missing_workload = sorted(WORKLOAD_FIELDS - set(test))
        if missing_workload:
            raise ValueError(
                f"tests[{index}] is missing workload fields: "
                + ", ".join(missing_workload)
            )
        setting = {**defaults, **test}
        _validate_setting(setting, index)
        settings.append(setting)
    return config, settings


def _run_tag(setting: dict[str, Any], index: int, timestamp: str) -> str:
    return "_".join(
        [
            timestamp,
            f"j{index:03d}",
            _sanitize(setting.get("name", f"case{index:03d}")),
            _sanitize(Path(str(setting["model"])).name),
            _sanitize(setting.get("planner", "feedback")),
            f"pb{_sanitize(setting.get('prefix_blocks', 128))}",
            f"bs{_sanitize(setting['num_prompts'])}",
            f"p{_sanitize(setting['prompt_words'])}",
            f"d{_sanitize(setting['max_tokens'])}",
        ]
    )


def _command(
    setting: dict[str, Any],
    run_tag: str,
    pipeline_args: Sequence[str] = (),
) -> list[str]:
    command = [sys.executable, str(PIPELINE), "--run-tag", run_tag]
    for key, option in VALUE_OPTIONS.items():
        value = setting.get(key)
        if value not in (None, ""):
            value_text = str(value)
            if value_text.startswith("-"):
                command.append(f"{option}={value_text}")
            else:
                command.extend([option, value_text])
    for key, option in TRUE_FLAGS.items():
        if setting.get(key, False):
            command.append(option)
    for key, (true_option, false_option) in BOOLEAN_OPTIONS.items():
        if key in setting:
            command.append(true_option if setting[key] else false_option)
    command.extend(pipeline_args)
    return command


def main(
    *,
    pipeline_args: Sequence[str] = (),
    description: str = __doc__,
) -> int:
    parser = argparse.ArgumentParser(description=description)
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
        run_tag = _run_tag(setting, index, timestamp)
        log_path = log_root / run_tag / "pipeline.log"
        try:
            command = _command(setting, run_tag, pipeline_args)
            print(f"\n[{index}/{len(settings)}] {name}")
            print(f"  log: {log_path}")
            print(f"  cmd: {shlex.join(command)}")
            if args.dry_run:
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log_file:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=os.environ.copy(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                reason = f"exit code {result.returncode}"
                failed.append((name, reason, log_path))
                print(f"  FAILED ({reason}); continuing")
            else:
                succeeded.append((name, log_path))
                print("  DONE")
        except OSError as exc:
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
