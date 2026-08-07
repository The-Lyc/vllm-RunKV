#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the staged-resource benchmark for Llama 2 7B.

This is a thin, model-specific adapter around
``run_staged_resource_benchmark.py``.  It keeps the staged runner's CLI and
artifacts while selecting the local Llama 2 model and Llama-aware RunKV and
TightLLM observation runners.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_staged_resource_benchmark as staged  # noqa: E402

LLAMA_MODEL = "/data/models/Llama-2-7b-hf-8k"
DEFAULT_PROMPT_WORD = "the"
LLAMA_OUTPUT_ROOT = ROOT / "exp_results/staged_llama2_7b"
LLAMA_MANIFEST_ROOT = ROOT / "exp_results/manifests/staged_resource_llama2_7b"
LLAMA_ANALYSIS_ROOT = ROOT / "exp_results/analysis/staged_resource_llama2_7b"
LLAMA_PER_LAYER_ANALYSIS_ROOT = (
    ROOT / "exp_results/analysis/staged_resource_per_layer_llama2_7b"
)
LLAMA_RUNNER = ROOT / "examples/offline_inference/run_llama_feedback_observation.py"
LLAMA_TIGHTLLM_RUNNER = (
    ROOT / "examples/offline_inference/run_tightllm_observation.py"
)


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _install_llama_adapter() -> None:
    staged.DEFAULT_OUTPUT_ROOT = LLAMA_OUTPUT_ROOT
    staged.MANIFEST_ROOT = LLAMA_MANIFEST_ROOT
    staged.ANALYSIS_ROOT = LLAMA_ANALYSIS_ROOT
    staged.PER_LAYER_ANALYSIS_ROOT = LLAMA_PER_LAYER_ANALYSIS_ROOT
    staged.RUNKV_SCRIPT = LLAMA_RUNNER
    staged.TIGHTLLM_SCRIPT = LLAMA_TIGHTLLM_RUNNER
    staged.BENCHMARK_LABEL = "Llama 2 7B"

    original_build_parser = staged.build_parser

    def build_parser():
        parser = original_build_parser()
        parser.description = (
            "Run the paired Llama 2 7B RunKV/TightLLM staged-resource pipeline"
        )
        parser.epilog = __doc__
        if not any(
            "--prompt-word" in action.option_strings
            for action in parser._actions
        ):
            parser.add_argument(
                "--prompt-word",
                default=DEFAULT_PROMPT_WORD,
                help=(
                    "Single tokenizer-friendly word repeated to build each "
                    "prompt (Llama default: the)."
                ),
            )
        return parser

    staged.build_parser = build_parser

    original_common_env = staged._common_env

    def common_env(args, run_tag: str, output_dir: Path) -> dict[str, str]:
        env = original_common_env(args, run_tag, output_dir)
        env["PROMPT_WORD"] = getattr(args, "prompt_word", DEFAULT_PROMPT_WORD)
        env["MODEL_FAMILY"] = "llama"
        return env

    staged._common_env = common_env

    def validate_tightllm_profile(args, path: str) -> None:
        profile_path = Path(path)
        if profile_path.is_file():
            try:
                profile = json.loads(profile_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"ERROR: invalid TightLLM profile {profile_path}: {exc}"
                ) from exc
            if profile.get("model_type") != "llama":
                raise SystemExit(
                    f"ERROR: TightLLM profile {profile_path} is not a "
                    "model-aware Llama profile. Regenerate it with the "
                    "current offline profiler."
                )
            return

        command = [
            sys.executable,
            "-m",
            "vllm.v1.profiling.tightllm_offline_profiler",
            "--model",
            args.model,
            "--output",
            path,
            "--seq-lengths",
            "128",
            "256",
            "512",
            "1024",
            "2048",
            "4096",
            "8192",
            "--device",
            "cuda:0",
        ]
        raise SystemExit(
            f"ERROR: Llama TightLLM profile not found: {profile_path}\n"
            "Generate it on the benchmark GPU without resource contention:\n"
            f"  {shlex.join(command)}"
        )

    staged._validate_tightllm_profile_path = validate_tightllm_profile


def main(argv: Sequence[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)

    if not _has_option(forwarded, "--output-root"):
        forwarded.extend(["--output-root", str(LLAMA_OUTPUT_ROOT)])
    if not _has_option(forwarded, "--prompt-word"):
        forwarded.extend(["--prompt-word", DEFAULT_PROMPT_WORD])

    # Append the enforced model last so repeated argparse values cannot select
    # a non-Llama model.
    forwarded.extend(
        [
            "--model",
            LLAMA_MODEL,
        ]
    )

    _install_llama_adapter()
    sys.argv = [sys.argv[0], *forwarded]
    staged.main()


if __name__ == "__main__":
    main()
