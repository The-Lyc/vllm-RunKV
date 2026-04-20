#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        "/home/lyc/inference/vllm/exp_results/opt_feedback_observation",
    )
    prefix_blocks = os.environ.get("PREFIX_BLOCKS", "1000")
    num_prompts = os.environ.get("NUM_PROMPTS", "32")
    prompt_words = os.environ.get("PROMPT_WORDS", "4000")
    max_tokens = os.environ.get("MAX_TOKENS", "32")
    gpu_memory_fraction = os.environ.get("GPU_MEMORY_FRACTION", "0.9")
    num_device_buffers = os.environ.get("NUM_DEVICE_BUFFERS", "3")
    planner = os.environ.get("PLANNER", "feedback")
    dry_run = os.environ.get("DRY_RUN", "1") == "1"
    tightllm_profile_path = os.environ.get("TIGHTLLM_PROFILE_PATH", "")
    tightllm_feedback_correction = (
        os.environ.get("TIGHTLLM_FEEDBACK_CORRECTION", "0") == "1"
    )
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
                "opt_gap"
                f"_{_model_tag(model)}"
                f"_pb{_sanitize_token(prefix_blocks)}"
                f"_{planner}"
                f"_{layerwise_tag}"
                f"_{run_tag}"
            )
        ),
    )

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
        planner,
        "--output-dir",
        output_dir,
        "--run-tag",
        run_tag,
    ]
    if dry_run:
        cmd.append("--planner-dry-run")
    if tightllm_profile_path:
        cmd.extend(["--tightllm-profile-path", tightllm_profile_path])
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

    print("Running OPT feedback observation")
    print(f"  model: {model}")
    print(f"  planner: {planner}")
    print(f"  planner_dry_run: {int(dry_run)}")
    if planner == "tightllm":
        print(f"  tightllm_profile: {tightllm_profile_path}")
        print(f"  tightllm_feedback_correction: {int(tightllm_feedback_correction)}")
    print(f"  opt_component_mfu: {int(enable_opt_component_mfu)}")
    print(f"  nvtx_scopes: {int(enable_nvtx)}")
    print(f"  layerwise_nvtx: {int(enable_layerwise_nvtx)}")
    print(f"  cuda_profiler_capture: {int(enable_profile)}")
    print(f"  prefix_blocks: {prefix_blocks}")
    print(f"  run_tag: {run_tag}")
    print(f"  output_dir: {output_dir}")
    print(f"  suggested_nsys_stem: {nsys_stem}")
    print(f"  enable_nsys: {int(enable_nsys)}")
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
    print("Trace files:")
    print(f"  {output_dir}/opt_component_mfu_*.jsonl")
    print(f"  {output_dir}/opt_component_mfu_*.flat.jsonl")
    if enable_nsys:
        print("Nsight Systems:")
        print(f"  {nsys_stem}.nsys-rep")
        print(f"  {nsys_stem}.qdstrm")
    print("Main JSONL: one line per step")
    print("Flat JSONL: one line per (step, layer)")


if __name__ == "__main__":
    main()
