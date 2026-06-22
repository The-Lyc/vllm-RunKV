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
    passthrough_args = sys.argv[1:]
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    model = os.environ.get("MODEL", "/home/lyc/hf_models/opt-2.7b-8k")
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "/home/lyc/inference/vllm/exp_results/opt_feedback_observation",
    )
    prefix_blocks = os.environ.get("PREFIX_BLOCKS", "10000")
    num_prompts = os.environ.get("NUM_PROMPTS", "32")
    prompt_words = os.environ.get("PROMPT_WORDS", "1000")
    max_tokens = os.environ.get("MAX_TOKENS", "128")
    gpu_memory_fraction = os.environ.get("GPU_MEMORY_FRACTION", "0.7")
    num_device_buffers = os.environ.get("NUM_DEVICE_BUFFERS", "3")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "")
    max_staging_blocks = os.environ.get("MAX_STAGING_BLOCKS", "")
    cpu_memory_gb = os.environ.get(
        "CPU_MEMORY_GB", os.environ.get("CPU_KV_MEMORY_GB", "")
    )
    cpu_memory_fraction = os.environ.get("CPU_MEMORY_FRACTION", "")
    planner = os.environ.get("PLANNER", "feedback")
    dry_run = os.environ.get("DRY_RUN", "1") == "1"
    use_state_machine = os.environ.get("USE_STATE_MACHINE", "0") == "1"
    async_plan_build = os.environ.get("ASYNC_PLAN_BUILD", "1") == "1"
    h2d_copy_mode = os.environ.get("H2D_COPY_MODE", "segment")
    if h2d_copy_mode not in ("segment", "gather"):
        raise ValueError(f"Unsupported H2D_COPY_MODE: {h2d_copy_mode!r}")
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
    for option, value in (
        ("--gpu-memory-utilization", gpu_memory_utilization),
        ("--max-num-seqs", max_num_seqs),
        ("--max-staging-blocks", max_staging_blocks),
        ("--cpu-memory-gb", cpu_memory_gb),
        ("--cpu-memory-fraction", cpu_memory_fraction),
    ):
        if value:
            cmd.extend([option, value])
    if dry_run:
        cmd.append("--planner-dry-run")
    if use_state_machine:
        cmd.append("--use-state-machine")
    if not async_plan_build:
        cmd.append("--no-async-plan-build")
    cmd.extend(["--h2d-copy-mode", h2d_copy_mode])
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
    if passthrough_args:
        cmd.extend(passthrough_args)

    print("Running OPT feedback observation")
    print(f"  model: {model}")
    print(f"  planner: {planner}")
    print(f"  planner_dry_run: {int(dry_run)}")
    print(f"  use_state_machine: {int(use_state_machine)}")
    print(f"  async_plan_build: {int(async_plan_build)}")
    print(f"  h2d_copy_mode: {h2d_copy_mode}")
    if planner == "tightllm":
        print(f"  tightllm_profile: {tightllm_profile_path}")
        print(f"  tightllm_feedback_correction: {int(tightllm_feedback_correction)}")
    print(f"  opt_component_mfu: {int(enable_opt_component_mfu)}")
    print(f"  nvtx_scopes: {int(enable_nvtx)}")
    print(f"  layerwise_nvtx: {int(enable_layerwise_nvtx)}")
    print(f"  cuda_profiler_capture: {int(enable_profile)}")
    print(f"  prefix_blocks: {prefix_blocks}")
    print(f"  gpu_memory_fraction: {gpu_memory_fraction}")
    print(f"  num_device_buffers:  {num_device_buffers}")
    if gpu_memory_utilization:
        print(f"  gpu_memory_utilization: {gpu_memory_utilization}")
    if max_num_seqs:
        print(f"  max_num_seqs:        {max_num_seqs}")
    if max_staging_blocks:
        print(f"  max_staging_blocks:  {max_staging_blocks}")
    if cpu_memory_gb:
        print(f"  cpu_memory_gb:       {cpu_memory_gb}")
    if cpu_memory_fraction:
        print(f"  cpu_memory_fraction: {cpu_memory_fraction}")
    print(f"  run_tag: {run_tag}")
    print(f"  output_dir: {output_dir}")
    print(f"  suggested_nsys_stem: {nsys_stem}")
    print(f"  enable_nsys: {int(enable_nsys)}")
    if passthrough_args:
        print(f"  passthrough_args: {' '.join(passthrough_args)}")
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

    # Write manifest if requested (for automation scripts)
    manifest_file = os.environ.get("MANIFEST_FILE", "")
    if manifest_file:
        import json as _json
        _manifest = {
            "run_tag": run_tag,
            "output_dir": str(Path(output_dir).resolve()),
            "prefix_blocks": prefix_blocks,
            "num_prompts": num_prompts,
            "prompt_words": prompt_words,
            "max_tokens": max_tokens,
            "fixed_output_length": True,
            "gpu_memory_fraction": gpu_memory_fraction,
            "num_device_buffers": num_device_buffers,
            "gpu_memory_utilization": gpu_memory_utilization or None,
            "max_num_seqs": max_num_seqs or None,
            "max_staging_blocks": max_staging_blocks or None,
            "cpu_memory_gb": cpu_memory_gb or None,
            "cpu_memory_fraction": cpu_memory_fraction or None,
            "planner": planner,
            "layer_recompute_async_plan_build": async_plan_build,
            "h2d_copy_mode": h2d_copy_mode,
            "layer_recompute_use_state_machine": use_state_machine,
            "model": model,
            "nsys_report": str(Path(nsys_stem + ".nsys-rep").resolve())
            if enable_nsys
            else None,
            "mfu_jsonl_glob": str(
                Path(output_dir) / f"opt_component_mfu_*_{run_tag}.jsonl"
            ),
            "mfu_flat_jsonl_glob": str(
                Path(output_dir) / f"opt_component_mfu_*_{run_tag}.flat.jsonl"
            ),
            "passthrough_args": passthrough_args,
        }
        Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_file).write_text(_json.dumps(_manifest, indent=2) + "\n")
        print(f"\nManifest written to: {manifest_file}")


if __name__ == "__main__":
    main()
