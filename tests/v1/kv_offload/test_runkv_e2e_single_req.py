#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Smoke Test for RunKV KV Cache Offloading.

This script is intentionally minimal:
- Enable RunKV *before* engine/model initialization.
- Assert RunKV is actually active in the model runner.
- Run a single short generation to ensure the end-to-end path works.

Usage:
    python test_runkv_e2e.py [--model MODEL] [--prompt PROMPT] [--max-tokens N]

Requirements:
    - CUDA available
    - vLLM installed with RunKV support

Example:
    python test_runkv_e2e.py --model "~/hf_models/Qwen3-0.6B"
    --prompt "Hello" --max-tokens 16
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA is not available. RunKV requires a CUDA-enabled GPU.")
            return False
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {total_memory_gb:.1f} GB")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False


def _build_engine(
    *,
    model_name: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_offload_config,
):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    # Now we can pass kv_offload_config directly to EngineArgs
    engine_args = EngineArgs(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        kv_offload_config=kv_offload_config,  # Direct API support!
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    executor_class = Executor.get_class(vllm_config)
    return LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
        usage_context=UsageContext.ENGINE_CONTEXT,
        multiprocess_mode=False,
    )


def _assert_runkv_active(engine) -> None:
    vllm_config = getattr(engine, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError("Cannot access engine.vllm_config to verify RunKV state.")

    cfg = getattr(vllm_config, "kv_offload_config", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        raise RuntimeError("RunKV not enabled in vllm_config.kv_offload_config.")

    executor = getattr(engine, "model_executor", None)
    if executor is None or not hasattr(executor, "driver_worker"):
        raise RuntimeError("Cannot access engine.model_executor.driver_worker.")

    driver_worker = executor.driver_worker
    worker = getattr(driver_worker, "worker", None)
    if worker is None:
        raise RuntimeError(
            "Cannot access driver_worker.worker (unexpected executor mode)."
        )

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("Cannot access worker.model_runner to verify RunKV state.")

    if not getattr(model_runner, "use_runkv", False):
        raise RuntimeError("model_runner.use_runkv is False (RunKV not active).")
    if not getattr(getattr(model_runner, "kv_offload_config", None), "enabled", False):
        raise RuntimeError("model_runner.kv_offload_config.enabled is False.")
    if not getattr(model_runner, "kv_buffers", None):
        raise RuntimeError("RunKV is active but model_runner.kv_buffers is empty.")


def _run_one_request(*, engine, prompt: str, max_tokens: int, max_steps: int) -> str:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.utils import random_uuid

    request_id = f"runkv-smoke-{random_uuid()}"
    # We only need the final output for this smoke test. Using FINAL_ONLY avoids
    # dealing with per-step cumulative vs delta semantics.
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    engine.add_request(request_id=request_id, prompt=prompt, params=params)

    for _ in range(max_steps):
        step_outputs = engine.step()
        for out in step_outputs:
            if getattr(out, "request_id", None) != request_id:
                continue
            if out.finished:
                text = out.outputs[0].text if out.outputs else ""
                if not text:
                    raise RuntimeError("Request finished but produced empty text.")
                return text

    raise TimeoutError(
        f"Timed out after {max_steps} engine steps waiting for request to finish."
    )


def _shutdown_engine(engine) -> None:
    with contextlib.suppress(Exception):
        engine.engine_core.shutdown()
    # Best-effort cleanup for uniproc runs (also resets vLLM's global group
    # coordinators so the process can re-initialize a second engine).
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory(shutdown_ray=False)


def test_without_runkv(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_steps: int,
):
    """Run a baseline (RunKV disabled) and return its generated text."""
    from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

    engine = _build_engine(
        model_name=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_offload_config=RunKVOffloadConfig(enabled=False),
    )
    try:
        text = _run_one_request(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            max_steps=max_steps,
        )
        return True, text
    finally:
        _shutdown_engine(engine)


def main():
    parser = argparse.ArgumentParser(description="RunKV Smoke Test")
    parser.add_argument(
        "--model",
        type=str,
        default="~/hf_models/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name (default: ~/hf_models/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=100.0,
        help="CPU memory limit for RunKV KV cache offloading in GB. "
        "Set <= 0 to derive from available system memory and "
        "--cpu-memory-fraction. Default: 100.0",
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.7,
        help="When --cpu-memory-gb <= 0, cap the RunKV CPU KV cache backing store "
        "to (available_system_memory * fraction). Default: 0.7",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run a baseline (RunKV disabled) and compare outputs",
    )
    parser.add_argument(
        "--skip-cuda-check", action="store_true", help="Skip CUDA availability check"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2048,
        help="Maximum LLMEngine.step() iterations to wait for completion",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Model max length for the engine",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization for vLLM (leave headroom for staging)",
    )
    parser.add_argument(
        "--num-device-buffers",
        type=int,
        default=3,
        help="RunKV GPU ring buffer count",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.85,
        help="Fraction of GPU budget used for RunKV staging buffers",
    )
    parser.add_argument(
        "--max-staging-blocks",
        type=int,
        default=0,
        help="If >0, overrides computed staging blocks per buffer",
    )

    args = parser.parse_args()

    # Expand local paths like "~/..." so Transformers doesn't treat them as HF repo IDs.
    args.model = os.path.expandvars(os.path.expanduser(args.model))

    # If the model argument looks like a local path, normalize it.
    # (HF hub repo IDs must not contain '~', and local paths commonly do.)
    model_path = Path(args.model)
    if args.model.startswith((".", os.sep)) or "~" in args.model or model_path.exists():
        args.model = str(model_path.expanduser().resolve(strict=False))
        model_path = Path(args.model)
        if model_path.is_dir() and not (model_path / "config.json").exists():
            print(
                f"Model path looks local but missing config.json: {args.model}\n"
                "Pass a valid local HF directory (must contain config.json), "
                "or pass a HuggingFace repo id like 'namespace/repo_name'."
            )
            sys.exit(1)

    print("=" * 60)
    print("RunKV KV Cache Offloading - Smoke Test")
    print("=" * 60)

    # Check CUDA
    if not args.skip_cuda_check and not check_cuda():
        sys.exit(1)

    from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

    if not (0.0 < args.cpu_memory_fraction <= 1.0):
        raise ValueError("--cpu-memory-fraction must be in (0, 1]")

    cpu_limit_bytes = None
    if args.cpu_memory_gb > 0:
        cpu_limit_bytes = int(args.cpu_memory_gb * 1024**3)

    kv_offload_config = RunKVOffloadConfig(
        enabled=True,
        num_device_buffers=args.num_device_buffers,
        gpu_memory_fraction=args.gpu_memory_fraction,
        enable_async_prefetch=True,
        enable_async_offload=True,
        cpu_memory_limit=cpu_limit_bytes,
        cpu_memory_fraction=args.cpu_memory_fraction,
        max_staging_blocks=(args.max_staging_blocks or None),
    )

    baseline_text = None
    if args.compare_baseline:
        ok, baseline_text = test_without_runkv(
            model_name=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_steps=args.max_steps,
        )
        if not ok:
            print("Baseline test failed")
            sys.exit(1)

    engine = _build_engine(
        model_name=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_offload_config=kv_offload_config,
    )

    try:
        _assert_runkv_active(engine)
        start = time.time()
        text = _run_one_request(
            engine=engine,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
        )
        dt = time.time() - start
        print(f"RunKV smoke OK in {dt:.2f}s")
        print(f"Prompt: {args.prompt}")
        print(f"Output: {text}")

        if baseline_text is not None and baseline_text != text:
            print("Baseline output differs from RunKV output.")
            print(f"Baseline: {baseline_text}")
            sys.exit(2)
    finally:
        _shutdown_engine(engine)

    sys.exit(0)


if __name__ == "__main__":
    main()
