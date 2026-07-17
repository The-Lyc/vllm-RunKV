#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run staged-resource benchmark settings sequentially from a JSON file.

The configuration uses a ``defaults`` object plus a non-empty ``tests`` list.
Each test overrides values from ``defaults``.  See
configs/staged_resource_benchmark_batch.example.json for a complete example.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts/run_staged_resource_benchmark.py"
DEFAULT_LOG_ROOT = ROOT / "exp_results/logs/staged_resource_benchmark_batch"

# Keep the workload field names compatible with benchmark_batch.example.json.
ALIASES = {
    "batch_size": "num_prompts",
    "prompt_length": "prompt_words",
    "decode_length": "max_tokens",
    "tightllm_profile": "tightllm_profile_path",
}

PER_TEST_WORKLOAD_FIELDS = {
    "num_prompts",
    "prompt_words",
    "max_tokens",
}

REQUIRED_FIELDS = (
    "model",
    "cpu_memory_gb",
    "cpu_memory_fraction",
    "gpu_memory_utilization",
    "gpu_memory_fraction",
    "num_prompts",
    "prompt_words",
    "max_tokens",
    "resource_pressure_kind",
    "resource_pressure_pattern",
)

VALUE_OPTIONS = {
    "output_root": "--output-root",
    "analysis_output_dir": "--analysis-output-dir",
    "repeats": "--repeats",
    "model": "--model",
    "prefix_blocks": "--prefix-blocks",
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
    "hardware_platform": "--hardware-platform",
    "tightllm_profile_root": "--tightllm-profile-root",
    "tightllm_profile_path": "--tightllm-profile-path",
    "resource_pressure_kind": "--resource-pressure-kind",
    "resource_pressure_clock": "--resource-pressure-clock",
    "resource_pressure_pattern": "--resource-pressure-pattern",
    "resource_pattern_name": "--resource-pattern-name",
    "resource_pressure_device": "--resource-pressure-device",
    "resource_pressure_buffer_mb": "--resource-pressure-buffer-mb",
    "resource_pressure_direction": "--resource-pressure-direction",
    "resource_pressure_matrix_size": "--resource-pressure-matrix-size",
    "resource_pressure_dtype": "--resource-pressure-dtype",
    "resource_pressure_window_s": "--resource-pressure-window-s",
    "resource_pressure_period_ms": "--resource-pressure-period-ms",
    "resource_pressure_max_fraction": "--resource-pressure-max-fraction",
    "resource_pressure_io_calibration_s": "--resource-pressure-io-calibration-s",
    "resource_pressure_io_max_gbps": "--resource-pressure-io-max-gbps",
    "resource_pressure_mode": "--resource-pressure-mode",
    "resource_pressure_inline_layer_period_ms": (
        "--resource-pressure-inline-layer-period-ms"
    ),
    "nsys_sample": "--nsys-sample",
    "nsys_extra_args": "--nsys-extra-args",
    "skip_warmup_steps": "--skip-warmup-steps",
    "compute_stream": "--compute-stream",
    "dma_tol_ms": "--dma-tol-ms",
}

TRUE_FLAGS = {
    "skip_runkv": "--skip-runkv",
    "skip_tightllm": "--skip-tightllm",
    "skip_analysis": "--skip-analysis",
    "tightllm_feedback_correction": "--tightllm-feedback-correction",
    "skip_sqlite_export": "--skip-sqlite-export",
    "skip_stage_analysis": "--skip-stage-analysis",
    "skip_per_layer_analysis": "--skip-per-layer-analysis",
}

BOOLEAN_OPTIONS = {
    "enable_nsys": ("--enable-nsys", "--disable-nsys"),
    "enable_nvtx": ("--enable-nvtx", "--disable-nvtx"),
    "enable_profile": ("--enable-profile", "--disable-profile"),
    "enable_prehook_timing": (
        "--enable-prehook-timing",
        "--disable-prehook-timing",
    ),
}

LIST_OPTIONS = {
    "runkv_run_dir": "--runkv-run-dir",
    "tightllm_run_dir": "--tightllm-run-dir",
}

CHOICES = {
    "resource_pressure_kind": {"io", "sm"},
    "resource_pressure_clock": {"step", "time"},
    "resource_pressure_direction": {"h2d", "d2h", "bidirectional"},
    "resource_pressure_dtype": {"float16", "bfloat16"},
    "resource_pressure_mode": {"thread", "inline", "throttle", "process"},
}

META_FIELDS = {"name"}
SUPPORTED_FIELDS = (
    set(VALUE_OPTIONS)
    | set(TRUE_FLAGS)
    | set(BOOLEAN_OPTIONS)
    | set(LIST_OPTIONS)
    | META_FIELDS
)


def _sanitize(value: Any) -> str:
    text = str(value).replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _normalize_setting(setting: dict[str, Any], location: str) -> dict[str, Any]:
    normalized = dict(setting)
    for alias, canonical in ALIASES.items():
        if alias not in normalized:
            continue
        if canonical in normalized:
            raise ValueError(
                f"{location} cannot set both {alias!r} and {canonical!r}"
            )
        normalized[canonical] = normalized.pop(alias)
    return normalized


def _validate_setting(setting: dict[str, Any], index: int) -> None:
    missing = [key for key in REQUIRED_FIELDS if setting.get(key) in (None, "")]
    if missing:
        raise ValueError(f"tests[{index}] is missing: {', '.join(missing)}")

    unknown = sorted(set(setting) - SUPPORTED_FIELDS)
    if unknown:
        raise ValueError(f"tests[{index}] has unknown fields: {', '.join(unknown)}")

    for key, choices in CHOICES.items():
        if key in setting and setting[key] not in choices:
            choices_text = ", ".join(sorted(choices))
            raise ValueError(
                f"tests[{index}].{key} must be one of: {choices_text}; "
                f"got {setting[key]!r}"
            )

    for key in set(TRUE_FLAGS) | set(BOOLEAN_OPTIONS):
        if key in setting and not isinstance(setting[key], bool):
            raise ValueError(f"tests[{index}].{key} must be a boolean")

    for key in LIST_OPTIONS:
        if key in setting and not isinstance(setting[key], list):
            raise ValueError(f"tests[{index}].{key} must be a list")


def _load_settings(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    defaults = config.get("defaults", {})
    tests = config.get("tests", [])
    if not isinstance(defaults, dict) or not isinstance(tests, list) or not tests:
        raise ValueError("config requires an object 'defaults' and a non-empty 'tests'")

    normalized_defaults = _normalize_setting(defaults, "defaults")
    misplaced = sorted(PER_TEST_WORKLOAD_FIELDS & set(normalized_defaults))
    if misplaced:
        aliases = {
            canonical: alias
            for alias, canonical in ALIASES.items()
            if canonical in PER_TEST_WORKLOAD_FIELDS
        }
        names = ", ".join(aliases.get(key, key) for key in misplaced)
        raise ValueError(
            f"workload fields must be set in every test, not defaults: {names}"
        )

    settings = []
    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise ValueError(f"tests[{index}] must be an object")
        normalized_test = _normalize_setting(test, f"tests[{index}]")
        missing_workload = sorted(PER_TEST_WORKLOAD_FIELDS - set(normalized_test))
        if missing_workload:
            aliases = {
                canonical: alias
                for alias, canonical in ALIASES.items()
                if canonical in PER_TEST_WORKLOAD_FIELDS
            }
            names = ", ".join(aliases.get(key, key) for key in missing_workload)
            raise ValueError(f"tests[{index}] is missing workload fields: {names}")
        setting = {**normalized_defaults, **normalized_test}
        _validate_setting(setting, index)
        settings.append(setting)

    pressure_kinds = {setting["resource_pressure_kind"] for setting in settings}
    if len(pressure_kinds) != 1:
        kinds = ", ".join(sorted(pressure_kinds))
        raise ValueError(
            "one batch config can contain only IO tests or only SM tests; "
            f"found: {kinds}"
        )
    return config, settings


def _run_tag(setting: dict[str, Any], index: int, timestamp: str) -> str:
    model = Path(str(setting["model"])).name
    name = setting.get("name", f"case{index:03d}")
    pattern = setting.get(
        "resource_pattern_name",
        "-".join(
            [
                str(setting["resource_pressure_kind"]),
                str(setting.get("resource_pressure_mode", "process")),
            ]
        ),
    )
    return "_".join(
        [
            timestamp,
            f"j{index:03d}",
            _sanitize(name),
            _sanitize(model),
            _sanitize(pattern),
            f"bs{_sanitize(setting['num_prompts'])}",
            f"p{_sanitize(setting['prompt_words'])}",
            f"d{_sanitize(setting['max_tokens'])}",
        ]
    )


def _resolve_profile(setting: dict[str, Any]) -> None:
    value = setting.get("tightllm_profile_path")
    if value in (None, ""):
        return
    profile = Path(str(value)).expanduser()
    if not profile.is_absolute():
        profile = ROOT / profile
    has_existing_runs = bool(setting.get("tightllm_run_dir"))
    if not setting.get("skip_tightllm", False) and not has_existing_runs:
        if not profile.is_file():
            raise ValueError(f"TightLLM profile not found: {profile}")
    setting["tightllm_profile_path"] = str(profile)


def _command(setting: dict[str, Any], run_tag: str) -> list[str]:
    setting = dict(setting)
    _resolve_profile(setting)
    cmd = [sys.executable, str(PIPELINE), "--run-tag", run_tag]

    for key, option in VALUE_OPTIONS.items():
        value = setting.get(key)
        if value not in (None, ""):
            value_text = str(value)
            if value_text.startswith("-"):
                cmd.append(f"{option}={value_text}")
            else:
                cmd.extend([option, value_text])
    for key, option in TRUE_FLAGS.items():
        if setting.get(key, False):
            cmd.append(option)
    for key, (true_option, false_option) in BOOLEAN_OPTIONS.items():
        if key in setting:
            cmd.append(true_option if setting[key] else false_option)
    for key, option in LIST_OPTIONS.items():
        values = setting.get(key)
        if values:
            cmd.append(option)
            cmd.extend(str(value) for value in values)
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
        log_path = log_root / fallback_dir / "staged_pipeline.log"
        try:
            run_tag = _run_tag(setting, index, timestamp)
            cmd = _command(setting, run_tag)
            log_path = log_root / run_tag / "staged_pipeline.log"
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
                    env=os.environ.copy(),
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
