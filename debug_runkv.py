# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Single-thread RunKV debugging script.

This mirrors debug_vllm.py but enables RunKV (layer-wise KV cache offload).

Notes:
- Prefer running with a local model path (no network needed).
- RunKV currently stages KV per Attention module (HtoD pre-hook, DtoH post-hook).
"""

import argparse
import os
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

# Important: environment variables must be set before importing vLLM.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
# Keep v1 single-process for easier debugging / introspection.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Set the Hugging Face cache directory to a user-writable location.
hf_cache_dir = str(Path("~/hf_cache").expanduser())
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("vLLM RunKV debugging script has launched")
print("=" * 60)


def _pick_default_model() -> str:
    # Prefer local Qwen models.
    local_models = [
        "~/hf_models/Qwen2.5-0.5B-Instruct",
        "~/hf_models/Qwen2.5-1.5B-Instruct",
        "~/hf_models/Qwen3-0.6B",
        "~/hf_models/Qwen2-0.5B-Instruct",
        "~/hf_models/Qwen2-1.5B-Instruct",
    ]
    for local_path in local_models:
        expanded_path = Path(local_path).expanduser()
        if expanded_path.exists() and (expanded_path / "config.json").exists():
            print(f"\n✓ Found local Qwen model: {expanded_path}")
            return str(expanded_path)

    # Fallback to a small hub model (requires network).
    print("\nNo local Qwen model found; falling back to Hugging Face Hub.")
    print("If you have no network access, pass --model /path/to/local/model.")
    return "Qwen/Qwen2.5-0.5B-Instruct"


def _assert_runkv_active(llm: LLM) -> None:
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise RuntimeError("Cannot access llm.llm_engine to verify RunKV state.")

    cfg = getattr(getattr(engine, "vllm_config", None), "kv_offload_config", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        raise RuntimeError("RunKV not enabled in engine.vllm_config.kv_offload_config.")

    # In single-process mode, LLMEngine exposes model_executor for v0 compatibility.
    executor = getattr(engine, "model_executor", None)
    if executor is None or not hasattr(executor, "driver_worker"):
        raise RuntimeError(
            "Cannot access engine.model_executor.driver_worker. "
            "Are you running with v1 multiprocessing enabled?"
        )

    worker = getattr(getattr(executor, "driver_worker", None), "worker", None)
    if worker is None:
        raise RuntimeError("Cannot access driver_worker.worker.")

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("Cannot access worker.model_runner.")

    if not getattr(model_runner, "use_runkv", False):
        raise RuntimeError("model_runner.use_runkv is False (RunKV not active).")
    if not getattr(getattr(model_runner, "kv_offload_config", None), "enabled", False):
        raise RuntimeError("model_runner.kv_offload_config.enabled is False.")
    if not getattr(model_runner, "kv_buffers", None):
        raise RuntimeError("RunKV is active but model_runner.kv_buffers is empty.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunKV debug script (LLM API).")
    parser.add_argument("--model", type=str, default=None, help="Model path or HF id.")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.set_defaults(enforce_eager=True)
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow CUDA graphs (less convenient for debugging).",
    )

    # RunKV knobs
    parser.add_argument("--num-device-buffers", type=int, default=3)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--max-staging-blocks", type=int, default=0)
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=0.0,
        help="If >0, sets kv_offload_config.cpu_memory_limit.",
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.7,
        help="Used when --cpu-memory-gb <= 0.",
    )
    parser.add_argument(
        "--disable-async-prefetch",
        action="store_true",
        help="Disable async HtoD staging (debug).",
    )
    parser.add_argument(
        "--disable-async-offload",
        action="store_true",
        help="Disable async DtoH flush (debug).",
    )

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Can be passed multiple times. Default: 2 short prompts.",
    )
    args = parser.parse_args()

    model_name = args.model or _pick_default_model()
    model_name = os.path.expandvars(os.path.expanduser(model_name))

    cpu_memory_limit = None
    if args.cpu_memory_gb and args.cpu_memory_gb > 0:
        cpu_memory_limit = int(args.cpu_memory_gb * 1024**3)

    kv_offload_config = {
        "enabled": True,
        "num_device_buffers": args.num_device_buffers,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "enable_async_prefetch": (not args.disable_async_prefetch),
        "enable_async_offload": (not args.disable_async_offload),
        "cpu_memory_limit": cpu_memory_limit,
        "cpu_memory_fraction": args.cpu_memory_fraction,
        "max_staging_blocks": (args.max_staging_blocks or None),
    }

    prompts = args.prompt or ["Hello, my name is", "The future of AI is"]

    print(f"\nModel: {model_name}")
    print(
        f"max_model_len={args.max_model_len}"
        f"gpu_memory_utilization={args.gpu_memory_utilization} "
        f"max_num_seqs={args.max_num_seqs}"
    )
    print("RunKV config:", kv_offload_config)

    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            kv_offload_config=kv_offload_config,
        )
    except Exception as e:
        print(f"\nModel load failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    try:
        _assert_runkv_active(llm)
        print("\n✓ RunKV is active (verified via model_runner.use_runkv)")
    except Exception as e:
        print(f"\nRunKV verification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nStarting generation with {len(prompts)} prompts...")
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)

    print("\n" + "=" * 50)
    print("Generated results:")
    print("=" * 50)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text if output.outputs else ""
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Generated text: {generated_text}")
        print("-" * 50)

    print("\n" + "=" * 60)
    print("RunKV debug complete")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    main()
