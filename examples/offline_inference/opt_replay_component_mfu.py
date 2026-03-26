#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real OPT inference with RunKV replay enabled and summarize "
            "attention/FFN MFU versus replay ratio."
        )
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument(
        "--prefix-blocks",
        default="baseline,4,8,16",
        help=(
            "Comma-separated replay settings. Use 'baseline' for RunKV without "
            "layer recompute, or an integer block count for "
            "layer_recompute_io_prefix_blocks."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8)
    parser.add_argument("--num-device-buffers", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--planner",
        choices=["static", "feedback"],
        default="feedback",
        help="Replay planner to use when layer recompute is enabled.",
    )
    parser.add_argument(
        "--planner-dry-run",
        action="store_true",
        help=(
            "Keep the feedback planner in observe-only mode. The runtime still "
            "emits imbalance observations into the profiling trace."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/opt_component_mfu",
        help="Directory for raw JSONL step traces.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional identifier appended to output filenames so multiple runs "
            "do not overwrite each other."
        ),
    )
    parser.add_argument(
        "--disable-opt-component-mfu-profiling",
        action="store_true",
        help=(
            "Disable per-component CUDA-event MFU timing while still emitting "
            "the step/layer JSONL logs. Useful when only collecting Nsight "
            "Systems traces."
        ),
    )
    parser.add_argument(
        "--disable-nvtx-scopes",
        action="store_true",
        help="Disable RunKV NVTX scope emission for this profiling script.",
    )
    parser.add_argument(
        "--enable-layerwise-nvtx-tracing",
        action="store_true",
        help="Also emit per-module NVTX ranges in addition to RunKV phase ranges.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable CUDA profiler start/stop for nsys --capture-range=cudaProfilerApi."
        ),
    )
    return parser.parse_args()


@contextlib.contextmanager
def nvtx_range(name: str, color: str = "blue"):
    if os.environ.get("VLLM_NVTX_SCOPES_FOR_PROFILING", "0") == "1":
        try:
            import nvtx

            with nvtx.annotate(name, color=color):
                yield
        except ImportError:
            try:
                import torch.cuda.nvtx as torch_nvtx

                with torch_nvtx.range(name):
                    yield
            except Exception:
                yield
    else:
        yield


def cuda_profiler_start() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.cudart().cudaProfilerStop()


def build_prompts(num_prompts: int, prompt_words: int) -> list[str]:
    repeated = " ".join(["replay"] * prompt_words)
    return [
        f"Request {idx}: summarize the pattern and continue briefly. {repeated}"
        for idx in range(num_prompts)
    ]


def parse_prefix_settings(raw: str) -> list[str | int]:
    settings: list[str | int] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        if value.lower() == "baseline":
            settings.append("baseline")
        else:
            settings.append(int(value))
    if not settings:
        raise ValueError("No valid --prefix-blocks settings were provided.")
    return settings


def make_kv_offload_config(
    setting: str | int,
    *,
    gpu_memory_fraction: float,
    num_device_buffers: int,
    planner: str,
    planner_dry_run: bool,
) -> dict:
    config = {
        "enabled": True,
        "num_device_buffers": num_device_buffers,
        "gpu_memory_fraction": gpu_memory_fraction,
        "enable_async_prefetch": True,
        "enable_async_offload": True,
        "cpu_memory_limit": 5e10,
    }
    if setting == "baseline":
        config["enable_layer_recompute"] = False
    else:
        config["enable_layer_recompute"] = True
        config["layer_recompute_mode"] = "prev_layer_output_dynamic"
        config["layer_recompute_io_prefix_blocks"] = [int(setting)]
        config["layer_recompute_planner"] = planner
        config["layer_recompute_planner_dry_run"] = planner_dry_run
    return config


def build_engine(
    *,
    model: str,
    gpu_memory_utilization: float,
    kv_offload_config: dict,
    enable_layerwise_nvtx_tracing: bool,
    profiler_config: dict | None,
    enable_opt_component_mfu_profiling: bool,
    opt_component_mfu_output_path: str | None,
    opt_component_mfu_peak_tflops: float | None,
    max_num_seqs: int,
):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    engine_args = EngineArgs(
        model=model,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_cascade_attn=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        enable_layerwise_nvtx_tracing=enable_layerwise_nvtx_tracing,
        profiler_config=profiler_config,
        kv_offload_config=kv_offload_config,
        enable_opt_component_mfu_profiling=enable_opt_component_mfu_profiling,
        opt_component_mfu_output_path=opt_component_mfu_output_path,
        opt_component_mfu_peak_tflops=opt_component_mfu_peak_tflops,
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


def run_prompts_with_engine(
    engine,
    prompts: list[str],
    *,
    max_tokens: int,
    enable_profiling: bool,
) -> None:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    pending_requests: set[str] = set()

    with nvtx_range("add_requests", color="green"):
        for idx, prompt in enumerate(prompts):
            request_id = f"req_{idx}"
            engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
            )
            pending_requests.add(request_id)

    if enable_profiling:
        cuda_profiler_start()

    step = 0
    try:
        with nvtx_range("inference_loop", color="blue"):
            while pending_requests:
                with nvtx_range(f"step_{step}", color="yellow"):
                    step_outputs = engine.step()

                for out in step_outputs:
                    request_id = getattr(out, "request_id", None)
                    if request_id is not None and getattr(out, "finished", False):
                        pending_requests.discard(request_id)

                step += 1
    finally:
        if enable_profiling:
            cuda_profiler_stop()


def summarize_run(trace_path: Path) -> dict[str, float | int | None]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return {"steps": 0, "avg_imbalance_ms": None}

    all_imbalances: list[float] = []
    for row in rows:
        for layer in row.get("layers", []):
            imb = layer.get("imbalance_ms")
            if imb is not None:
                all_imbalances.append(imb)
    return {
        "steps": len(rows),
        "avg_imbalance_ms": mean(all_imbalances) if all_imbalances else None,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_summary_table(
    results: list[tuple[str, dict[str, float | int | None]]],
) -> None:
    print("\nOPT Replay Imbalance Summary")
    print(f"{'setting':<12} {'steps':>6} {'avg_imbalance_ms':>18}")
    print("-" * 40)
    for setting, summary in results:
        print(
            f"{setting:<12} "
            f"{summary['steps']:>6} "
            f"{format_float(summary['avg_imbalance_ms']):>18}"
        )


def main() -> None:
    args = parse_args()
    if not args.disable_nvtx_scopes:
        os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    mfu_profiler_enabled = not args.disable_opt_component_mfu_profiling

    summaries: list[tuple[str, dict[str, float | int | None]]] = []
    for setting in parse_prefix_settings(args.prefix_blocks):
        label = str(setting)
        trace_path = (
            output_dir / f"opt_component_mfu_{label}_{run_tag}.jsonl"
            if mfu_profiler_enabled
            else None
        )
        prompts = build_prompts(args.num_prompts, args.prompt_words)

        with nvtx_range("build_engine", color="purple"):
            engine = build_engine(
                model=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
                profiler_config=(
                    {"profiler": "cuda"} if not args.disable_nvtx_scopes else None
                ),
                kv_offload_config=make_kv_offload_config(
                    setting,
                    gpu_memory_fraction=args.gpu_memory_fraction,
                    num_device_buffers=args.num_device_buffers,
                    planner=args.planner,
                    planner_dry_run=args.planner_dry_run,
                ),
                enable_opt_component_mfu_profiling=mfu_profiler_enabled,
                opt_component_mfu_output_path=(
                    str(trace_path) if trace_path is not None else None
                ),
                opt_component_mfu_peak_tflops=args.peak_tflops,
                max_num_seqs=max(args.num_prompts, 1),
            )

        run_prompts_with_engine(
            engine,
            prompts,
            max_tokens=args.max_tokens,
            enable_profiling=args.profile,
        )

        if trace_path is not None:
            summaries.append((label, summarize_run(trace_path)))
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary_table(summaries)
    print(f"\nRaw step traces written to {output_dir}")
    print(
        "Inspect per-layer data in each JSONL row under: "
        "layers[*].{compute_end_ms_from_anchor,load_ready_ms_from_anchor,"
        "imbalance_ms,controller_update,replay_ratio}"
    )
    print("Flat step-layer JSONL is written alongside the main file as *.flat.jsonl")


if __name__ == "__main__":
    main()
