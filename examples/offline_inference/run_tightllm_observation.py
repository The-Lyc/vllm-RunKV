#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TightLLM ILP planner observation runner.

Wraps opt_replay_component_mfu.py with --planner=tightllm and optionally
wraps the whole run with nsys so that TightLLM-specific NVTX ranges
(tightllm:begin_step, tightllm:ilp_solve, tightllm:observe_feedback:L*)
appear in the Nsight Systems trace alongside the existing runkv ranges
(runkv:h2d_sync, runkv:d2h_copy, runkv:attention_compute, etc.).

Usage (basic — no nsys):
    python examples/offline_inference/run_tightllm_observation.py

Usage (with nsys trace):
    ENABLE_NSYS=1 python examples/offline_inference/run_tightllm_observation.py

Environment variables (all optional, with sensible defaults):
    MODEL                  model path
    TIGHTLLM_PROFILE_PATH  offline profile JSON (required)
    TIGHTLLM_FEEDBACK_CORRECTION  set to "1" to enable hybrid ILP+feedback
    PREFIX_BLOCKS          io prefix blocks (default 1000)
    NUM_PROMPTS            number of prompts (default 64)
    PROMPT_WORDS           words per prompt (default 8000)
    MAX_TOKENS             max output tokens (default 32)
    GPU_MEMORY_FRACTION    KV offload GPU fraction (default 0.9)
    NUM_DEVICE_BUFFERS     DMA device buffers (default 3)
    ENABLE_NVTX            "1" (default) to emit NVTX scopes
    ENABLE_LAYERWISE_NVTX  "1" to also emit per-module NVTX
    ENABLE_NSYS            "1" to wrap with nsys profile
    NSYS_CMD               path to nsys (default "nsys")
    NSYS_OUTPUT_DIR        nsys output directory
    NSYS_EXTRA_ARGS        extra nsys arguments (space-separated)
    RUN_TAG                tag for output file names
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _sanitize_token(value: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    )
    return sanitized.strip("_.") or "run"


def _model_tag(model: str) -> str:
    return _sanitize_token(Path(model).name or model)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    model = os.environ.get("MODEL", "/home/lyc/hf_models/opt-2.7b-8k")
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "/home/lyc/inference/vllm/exp_results/tightllm_observation",
    )
    tightllm_profile_path = os.environ.get(
        "TIGHTLLM_PROFILE_PATH", "tightllm_profile.json"
    )
    tightllm_feedback_correction = (
        os.environ.get("TIGHTLLM_FEEDBACK_CORRECTION", "0") == "1"
    )
    prefix_blocks = os.environ.get("PREFIX_BLOCKS", "1000")
    num_prompts = os.environ.get("NUM_PROMPTS", "32")
    prompt_words = os.environ.get("PROMPT_WORDS", "4000")
    max_tokens = os.environ.get("MAX_TOKENS", "32")
    gpu_memory_fraction = os.environ.get("GPU_MEMORY_FRACTION", "0.9")
    num_device_buffers = os.environ.get("NUM_DEVICE_BUFFERS", "3")
    enable_nvtx = os.environ.get("ENABLE_NVTX", "1") == "1"
    enable_layerwise_nvtx = os.environ.get("ENABLE_LAYERWISE_NVTX", "0") == "1"
    enable_opt_component_mfu = (
        os.environ.get("ENABLE_OPT_COMPONENT_MFU_PROFILING", "1") == "1"
    )
    enable_profile = os.environ.get("ENABLE_PROFILE", "0") == "1"
    enable_nsys = os.environ.get("ENABLE_NSYS", "0") == "1"
    nsys_cmd = os.environ.get("NSYS_CMD", "nsys")
    run_tag = os.environ.get(
        "RUN_TAG",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    nsys_output_dir = Path(os.environ.get("NSYS_OUTPUT_DIR", output_dir)).expanduser()
    nsys_extra_args = shlex.split(os.environ.get("NSYS_EXTRA_ARGS", ""))
    layerwise_tag = "layerwise" if enable_layerwise_nvtx else "coarse"
    nsys_stem = os.environ.get(
        "NSYS_OUTPUT_STEM",
        str(
            nsys_output_dir
            / (
                "tightllm_obs"
                f"_{_model_tag(model)}"
                f"_pb{_sanitize_token(prefix_blocks)}"
                f"_{layerwise_tag}"
                f"_{run_tag}"
            )
        ),
    )

    # Validate profile path
    if not tightllm_profile_path or not Path(tightllm_profile_path).exists():
        print(f"ERROR: TightLLM profile not found: {tightllm_profile_path}")
        print("Run the offline profiler first:")
        print("  python -m vllm.v1.profiling.tightllm_offline_profiler \\")
        print(f"      --model {model} --output tightllm_profile.json")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(root_dir)
        if not env.get("PYTHONPATH")
        else f"{root_dir}:{env['PYTHONPATH']}"
    )
    if enable_nvtx:
        env.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")

    cmd = [
        python_bin,
        str(root_dir / "examples/offline_inference/opt_replay_component_mfu.py"),
        "--model",
        model,
        "--prefix-blocks",
        prefix_blocks,
        "--num-prompts",
        num_prompts,
        "--prompt-words",
        prompt_words,
        "--max-tokens",
        max_tokens,
        "--gpu-memory-fraction",
        gpu_memory_fraction,
        "--num-device-buffers",
        num_device_buffers,
        "--planner",
        "tightllm",
        "--tightllm-profile-path",
        tightllm_profile_path,
        "--output-dir",
        output_dir,
        "--run-tag",
        run_tag,
    ]
    if tightllm_feedback_correction:
        cmd.append("--tightllm-feedback-correction")
    if not enable_opt_component_mfu:
        cmd.append("--disable-opt-component-mfu-profiling")
    if not enable_nvtx:
        cmd.append("--disable-nvtx-scopes")
    if enable_layerwise_nvtx:
        cmd.append("--enable-layerwise-nvtx-tracing")
    if enable_profile:
        cmd.append("--profile")

    print("Running TightLLM ILP planner observation")
    print(f"  model:              {model}")
    print("  planner:            tightllm")
    print(f"  profile:            {tightllm_profile_path}")
    print(f"  feedback_correction:{int(tightllm_feedback_correction)}")
    print(f"  opt_component_mfu:  {int(enable_opt_component_mfu)}")
    print(f"  nvtx_scopes:        {int(enable_nvtx)}")
    print(f"  layerwise_nvtx:     {int(enable_layerwise_nvtx)}")
    print(f"  cuda_profiler:      {int(enable_profile)}")
    print(f"  prefix_blocks:      {prefix_blocks}")
    print(f"  num_prompts:        {num_prompts}")
    print(f"  prompt_words:       {prompt_words}")
    print(f"  max_tokens:         {max_tokens}")
    print(f"  run_tag:            {run_tag}")
    print(f"  output_dir:         {output_dir}")
    print(f"  enable_nsys:        {int(enable_nsys)}")
    if enable_nsys:
        print(f"  nsys_output:        {nsys_stem}")
    print()
    print("Expected NVTX ranges in nsys trace:")
    print("  tightllm:begin_step         — per-step ILP budget computation")
    print("  tightllm:ilp_solve           — exhaustive enumeration solver")
    print("  tightllm:observe_feedback:L* — per-layer imbalance recording")
    print("  runkv:h2d_sync:L*            — DMA H2D transfer per layer")
    print("  runkv:d2h_copy:*             — DMA D2H offload")
    print("  runkv:prehook:imbalance:L*   — imbalance measurement")
    print("  runkv:prehook:build_plan:L*  — plan construction")
    print("  runkv:attention_compute:*    — attention kernel")
    print()

    final_cmd = cmd
    if enable_nsys:
        nsys_output_dir.mkdir(parents=True, exist_ok=True)
        nsys_sample = os.environ.get("NSYS_SAMPLE", "none")
        final_cmd = [
            nsys_cmd,
            "profile",
            "--trace=cuda,nvtx,osrt",
            f"--sample={nsys_sample}",
            "-o",
            nsys_stem,
            *nsys_extra_args,
            *cmd,
        ]

    subprocess.run(final_cmd, env=env, check=True)

    print()
    if enable_nsys:
        print("Nsight Systems report:")
        print(f"  {nsys_stem}.nsys-rep")
        print()
        print("Open with:  nsys-ui " + nsys_stem + ".nsys-rep")
    print("Done.")


if __name__ == "__main__":
    main()
