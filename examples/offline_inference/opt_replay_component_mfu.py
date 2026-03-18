#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean

import torch

from vllm import LLM, SamplingParams


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
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--num-device-buffers", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/tmp/opt_component_mfu",
        help="Directory for raw JSONL step traces.",
    )
    return parser.parse_args()


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
) -> dict:
    config = {
        "enabled": True,
        "num_device_buffers": num_device_buffers,
        "gpu_memory_fraction": gpu_memory_fraction,
        "enable_async_prefetch": True,
        "enable_async_offload": True,
    }
    if setting == "baseline":
        config["enable_layer_recompute"] = False
    else:
        config["enable_layer_recompute"] = True
        config["layer_recompute_mode"] = "prev_layer_output_dynamic"
        config["layer_recompute_io_prefix_blocks"] = [int(setting)]
    return config


def summarize_run(trace_path: Path) -> dict[str, dict[str, float | int | None]]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return {
            "attention": empty_summary(),
            "ffn": empty_summary(),
        }

    return {
        "attention": summarize_component(rows, "attention"),
        "ffn": summarize_component(rows, "ffn"),
    }


def empty_summary() -> dict[str, float | int | None]:
    return {
        "steps": 0,
        "avg_replay_ratio": None,
        "avg_mfu": None,
        "avg_tflops": None,
        "avg_time_ms": None,
    }


def summarize_component(
    rows: list[dict],
    component: str,
) -> dict[str, float | int | None]:
    payloads = [row[component] for row in rows if row[component]["calls"] > 0]
    if not payloads:
        return empty_summary()

    return {
        "steps": len(payloads),
        "avg_replay_ratio": mean(
            payload["weighted_replay_ratio"] for payload in payloads
        ),
        "avg_mfu": mean(
            payload["mfu"] for payload in payloads if payload["mfu"] is not None
        )
        if any(payload["mfu"] is not None for payload in payloads)
        else None,
        "avg_tflops": mean(
            payload["tflops"] for payload in payloads if payload["tflops"] is not None
        )
        if any(payload["tflops"] is not None for payload in payloads)
        else None,
        "avg_time_ms": mean(payload["time_ms"] for payload in payloads),
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_summary_table(
    results: list[tuple[str, dict[str, dict[str, float | int | None]]]],
) -> None:
    print("\nOPT Replay MFU Summary")
    print(
        f"{'setting':<12} {'steps':>6} {'attn_ratio':>12} {'attn_mfu':>10} "
        f"{'attn_ms':>10} {'ffn_ratio':>12} {'ffn_mfu':>10} {'ffn_ms':>10}"
    )
    print("-" * 92)
    for setting, summary in results:
        attention = summary["attention"]
        ffn = summary["ffn"]
        steps = max(int(attention["steps"] or 0), int(ffn["steps"] or 0))
        print(
            f"{setting:<12} "
            f"{steps:>6} "
            f"{format_float(attention['avg_replay_ratio']):>12} "
            f"{format_float(attention['avg_mfu']):>10} "
            f"{format_float(attention['avg_time_ms']):>10} "
            f"{format_float(ffn['avg_replay_ratio']):>12} "
            f"{format_float(ffn['avg_mfu']):>10} "
            f"{format_float(ffn['avg_time_ms']):>10}"
        )


def main() -> None:
    args = parse_args()
    prompts = build_prompts(args.num_prompts, args.prompt_words)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, dict[str, dict[str, float | int | None]]]] = []
    for setting in parse_prefix_settings(args.prefix_blocks):
        label = str(setting)
        trace_path = output_dir / f"opt_component_mfu_{label}.jsonl"
        trace_path.unlink(missing_ok=True)

        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_cascade_attn=True,
            kv_offload_config=make_kv_offload_config(
                setting,
                gpu_memory_fraction=args.gpu_memory_fraction,
                num_device_buffers=args.num_device_buffers,
            ),
            enable_opt_component_mfu_profiling=True,
            opt_component_mfu_output_path=str(trace_path),
            opt_component_mfu_peak_tflops=args.peak_tflops,
        )
        llm.generate(prompts, sampling_params)

        summaries.append((label, summarize_run(trace_path)))
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary_table(summaries)
    print(f"\nRaw step traces written to {output_dir}")


if __name__ == "__main__":
    main()
