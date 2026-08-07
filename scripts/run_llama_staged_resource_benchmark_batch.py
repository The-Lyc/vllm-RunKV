#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Llama 2 7B staged-resource settings sequentially from JSON.

The JSON schema and ``--dry-run`` behavior match
``run_staged_resource_benchmark_batch.py``.  In addition, ``prompt_word`` is a
configurable value option.  Every setting is normalized to the local Llama 2
7B model and uses the model-aware TightLLM profile when TightLLM is enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_staged_resource_benchmark_batch as batch  # noqa: E402

LLAMA_MODEL = "/data/models/Llama-2-7b-hf-8k"
LLAMA_PIPELINE = ROOT / "scripts/run_llama_staged_resource_benchmark.py"
LLAMA_LOG_ROOT = ROOT / "exp_results/logs/staged_resource_benchmark_llama2_7b"


def _install_llama_adapter() -> None:
    batch.PIPELINE = LLAMA_PIPELINE
    batch.DEFAULT_LOG_ROOT = LLAMA_LOG_ROOT
    batch.VALUE_OPTIONS = {
        **batch.VALUE_OPTIONS,
        "prompt_word": "--prompt-word",
    }
    batch.SUPPORTED_FIELDS = (
        set(batch.VALUE_OPTIONS)
        | set(batch.TRUE_FLAGS)
        | set(batch.BOOLEAN_OPTIONS)
        | set(batch.LIST_OPTIONS)
        | batch.META_FIELDS
    )

    def resolve_profile(setting: dict) -> None:
        value = setting.get("tightllm_profile_path")
        if value in (None, ""):
            return
        profile = Path(str(value)).expanduser()
        if not profile.is_absolute():
            profile = ROOT / profile
        # Keep --dry-run useful before the hardware-specific profile exists.
        # The staged pipeline validates the file on a real run.
        setting["tightllm_profile_path"] = str(profile)

    batch._resolve_profile = resolve_profile

    original_load_settings = batch._load_settings

    def load_settings(config_path: Path):
        config, settings = original_load_settings(config_path)
        for setting in settings:
            setting["model"] = LLAMA_MODEL
            setting.setdefault("prompt_word", "the")
        return config, settings

    batch._load_settings = load_settings


def main() -> int:
    _install_llama_adapter()
    return batch.main()


if __name__ == "__main__":
    sys.exit(main())
