#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
A/B comparison: runkv feedback (online) vs TightLLM ILP (offline).

Both planners share the same underlying recompute/DMA infrastructure.
The only difference is how the replay block budget is determined:
  - feedback:  online Newton/secant controller driven by runtime imbalance
  - tightllm:  offline ILP solver using profiled MFU + bandwidth data

Each arm runs in a **separate subprocess** so the CUDA context is fully
released between runs — this avoids the GPU-memory-leak issue that occurs
when destroying and recreating a vLLM engine in the same process.

NVTX / nsys support:
  The script enables VLLM_NVTX_SCOPES_FOR_PROFILING by default so all
  runkv NVTX ranges (h2d_sync, d2h_copy, prehook:imbalance, prehook:plan,
  attention_compute, etc.) appear in Nsight Systems traces.  Use
  --enable-nsys to wrap each arm with ``nsys profile``.

Imbalance statistics:
  After each arm finishes, the script collects imbalance_ms observations
  from the replay plan provider and prints summary statistics (mean,
  stdev, p95, etc.) alongside the throughput comparison.

Usage:
    # Step 1: offline profile (once per model + GPU)
    python -m vllm.v1.profiling.tightllm_offline_profiler \
        --model /path/to/opt-2.7b-8k \
        --output tightllm_profile.json

    # Step 2: A/B comparison
    python examples/offline_inference/run_planner_ab_comparison.py \
        --model /path/to/opt-2.7b-8k \
        --tightllm-profile-path tightllm_profile.json \
        --num-prompts 32 --prompt-words 4000 --max-tokens 64

    # Step 2b: with nsys trace collection
    python examples/offline_inference/run_planner_ab_comparison.py \
        --model /path/to/opt-2.7b-8k \
        --tightllm-profile-path tightllm_profile.json \
        --enable-nsys --nsys-output-dir /tmp/nsys_ab

Environment variables (override CLI defaults):
    MODEL, PREFIX_BLOCKS, NUM_PROMPTS, PROMPT_WORDS, MAX_TOKENS,
    GPU_MEMORY_FRACTION, NUM_DEVICE_BUFFERS, TIGHTLLM_PROFILE_PATH
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures (pickle-safe, no torch/vllm at module level)
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    label: str
    total_time_s: float
    num_prompts: int
    total_output_tokens: int
    outputs: dict[str, str]  # request_id -> generated text
    imbalance_stats: dict | None = None  # from planner.get_imbalance_stats()

    @property
    def throughput_tok_per_s(self) -> float:
        return (
            self.total_output_tokens / self.total_time_s if self.total_time_s > 0 else 0
        )

    @property
    def throughput_req_per_s(self) -> float:
        return self.num_prompts / self.total_time_s if self.total_time_s > 0 else 0


def build_prompts(num_prompts: int, prompt_words: int) -> list[str]:
    repeated = " ".join(["replay"] * prompt_words)
    return [
        f"Request {idx}: summarize the pattern and continue briefly. {repeated}"
        for idx in range(num_prompts)
    ]


def make_kv_offload_config(
    *,
    prefix_blocks: int,
    gpu_memory_fraction: float,
    num_device_buffers: int,
    planner: str,
    planner_dry_run: bool = False,
    tightllm_profile_path: str | None = None,
    tightllm_feedback_correction: bool = False,
) -> dict:
    config: dict = {
        "enabled": True,
        "num_device_buffers": num_device_buffers,
        "gpu_memory_fraction": gpu_memory_fraction,
        "enable_async_prefetch": True,
        "enable_async_offload": True,
        "cpu_memory_limit": int(5e10),
        "enable_layer_recompute": True,
        "layer_recompute_mode": "prev_layer_output_dynamic",
        "layer_recompute_io_prefix_blocks": [prefix_blocks],
        "layer_recompute_planner": planner,
        "layer_recompute_planner_dry_run": planner_dry_run,
    }
    if planner == "tightllm":
        if not tightllm_profile_path:
            raise ValueError("tightllm planner requires tightllm_profile_path")
        config["tightllm_profile_path"] = tightllm_profile_path
        config["tightllm_enable_feedback_correction"] = tightllm_feedback_correction
    return config


# ---------------------------------------------------------------------------
# Subprocess worker — runs one arm in a fresh process / CUDA context
# ---------------------------------------------------------------------------


def _run_arm_in_subprocess(
    model: str,
    prompts: list[str],
    max_tokens: int,
    gpu_memory_utilization: float,
    kv_offload_config: dict,
    label: str,
    result_path: str,
) -> None:
    """Entry point executed inside a spawned subprocess.

    All vLLM / torch imports happen here so the parent process stays clean.
    Results are written to *result_path* as JSON.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Enable NVTX scopes so runkv ranges appear in nsys traces
    os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")

    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    planner_name = kv_offload_config.get("layer_recompute_planner", "unknown")
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  planner = {planner_name}")
    print(f"  (subprocess pid={os.getpid()})")
    print(f"{'=' * 60}")

    engine_args = EngineArgs(
        model=model,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_cascade_attn=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max(len(prompts), 1),
        kv_offload_config=kv_offload_config,
        profiler_config={"profiler": "cuda"},
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    executor_class = Executor.get_class(vllm_config)
    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
        usage_context=UsageContext.ENGINE_CONTEXT,
        multiprocess_mode=False,
    )

    pending: set[str] = set()
    for idx, prompt in enumerate(prompts):
        rid = f"req_{idx}"
        engine.add_request(
            request_id=rid,
            prompt=prompt,
            params=SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
        )
        pending.add(rid)

    # ---- Optional: NVTX range wrapping the entire inference loop ----
    _nvtx_push = _nvtx_pop = None
    try:
        import torch.cuda.nvtx as torch_nvtx

        _nvtx_push = torch_nvtx.range_push
        _nvtx_pop = torch_nvtx.range_pop
    except Exception:
        pass

    outputs: dict[str, str] = {}
    total_output_tokens = 0

    if _nvtx_push:
        _nvtx_push(f"ab:{planner_name}:inference_loop")

    t0 = time.perf_counter()
    step_idx = 0
    while pending:
        if _nvtx_push:
            _nvtx_push(f"ab:{planner_name}:step_{step_idx}")
        step_outputs = engine.step()
        if _nvtx_pop:
            _nvtx_pop()
        for out in step_outputs:
            rid = getattr(out, "request_id", None)
            if rid and getattr(out, "finished", False):
                pending.discard(rid)
                text = ""
                if out.outputs:
                    text = out.outputs[0].text
                    total_output_tokens += len(out.outputs[0].token_ids)
                outputs[rid] = text
        step_idx += 1
    elapsed = time.perf_counter() - t0

    if _nvtx_pop:
        _nvtx_pop()  # close inference_loop range

    print(f"  [{label}] Completed {len(prompts)} requests in {elapsed:.2f}s")
    print(f"  [{label}] Output tokens: {total_output_tokens}")
    tps = total_output_tokens / elapsed if elapsed > 0 else 0
    print(f"  [{label}] Throughput: {tps:.1f} tok/s")

    # ---- Collect imbalance statistics from the replay plan provider ----
    imbalance_stats: dict | None = None
    try:
        model_runner = engine.model_executor.driver_worker.worker.model_runner
        provider = getattr(model_runner, "replay_plan_provider", None)
        if provider is not None and hasattr(provider, "get_imbalance_stats"):
            imbalance_stats = provider.get_imbalance_stats()
            print(f"  [{label}] Imbalance stats:")
            for k, v in imbalance_stats.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    except Exception as e:
        print(f"  [{label}] Warning: could not collect imbalance stats: {e}")

    # Write results to file — the parent process will read this.
    result = {
        "label": label,
        "total_time_s": elapsed,
        "num_prompts": len(prompts),
        "total_output_tokens": total_output_tokens,
        "outputs": outputs,
        "imbalance_stats": imbalance_stats,
    }
    Path(result_path).write_text(json.dumps(result))


def run_arm(
    *,
    model: str,
    prompts: list[str],
    max_tokens: int,
    gpu_memory_utilization: float,
    kv_offload_config: dict,
    label: str,
) -> RunStats:
    """Launch a subprocess to run one experimental arm, wait, return results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_run_arm_in_subprocess,
        args=(
            model,
            prompts,
            max_tokens,
            gpu_memory_utilization,
            kv_offload_config,
            label,
            result_path,
        ),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        print(f"ERROR: {label} subprocess exited with code {proc.exitcode}")
        # Return empty stats so the comparison can still proceed
        return RunStats(
            label=label,
            total_time_s=0,
            num_prompts=len(prompts),
            total_output_tokens=0,
            outputs={},
        )

    result = json.loads(Path(result_path).read_text())
    Path(result_path).unlink(missing_ok=True)
    return RunStats(
        label=result["label"],
        total_time_s=result["total_time_s"],
        num_prompts=result["num_prompts"],
        total_output_tokens=result["total_output_tokens"],
        outputs=result["outputs"],
        imbalance_stats=result.get("imbalance_stats"),
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _print_phase_stats(s: dict, label: str, phase: str, indent: str = "    ") -> None:
    """Print phase-separated imbalance stats (decode or prefill)."""
    count_key = f"{phase}_count"
    count = s.get(count_key, 0)
    if count == 0:
        print(f"{indent}[{phase}] no observations")
        return
    print(f"{indent}[{phase}] observations: {count}")
    print(f"{indent}[{phase}] mean:         {s[f'{phase}_mean_ms']:+.4f} ms")
    print(f"{indent}[{phase}] stdev:        {s[f'{phase}_stdev_ms']:.4f} ms")
    print(f"{indent}[{phase}] |imb| mean:   {s[f'{phase}_abs_mean_ms']:.4f} ms")
    print(f"{indent}[{phase}] |imb| max:    {s[f'{phase}_abs_max_ms']:.4f} ms")
    print(f"{indent}[{phase}] median:       {s[f'{phase}_median_ms']:+.4f} ms")
    print(f"{indent}[{phase}] p95 (|val|):  {s[f'{phase}_p95_ms']:.4f} ms")
    print(f"{indent}[{phase}] pos ratio:    {s[f'{phase}_positive_ratio']:.1%}")


def _print_imbalance_stats(label: str, s: dict | None, indent: str = "  ") -> None:
    """Print full imbalance stats for one arm (overall + phase breakdown)."""
    if s is None or s.get("count", 0) == 0:
        print(f"{indent}{label}: no imbalance data")
        return
    print(f"{indent}{label}:")
    inner = indent + "  "
    print(f"{inner}observations:    {s['count']}")
    print(f"{inner}mean:            {s['mean_ms']:+.4f} ms")
    print(f"{inner}stdev:           {s['stdev_ms']:.4f} ms")
    print(f"{inner}|imbalance| mean:{s['abs_mean_ms']:.4f} ms")
    print(f"{inner}|imbalance| max: {s['abs_max_ms']:.4f} ms")
    print(f"{inner}median:          {s['median_ms']:+.4f} ms")
    print(f"{inner}p95 (|val|):     {s['p95_ms']:.4f} ms")
    print(
        f"{inner}positive ratio:  {s['positive_ratio']:.1%}  (>0 = transfer bottleneck)"
    )
    print(f"{inner}budget mean:     {s['budget_mean']:.1f} blocks")
    print(f"{inner}budget stdev:    {s['budget_stdev']:.1f} blocks")
    # Phase breakdowns
    _print_phase_stats(s, label, "decode", indent=inner)
    _print_phase_stats(s, label, "prefill", indent=inner)


def compare_outputs(stats_a: RunStats, stats_b: RunStats) -> int:
    """Compare output text between two runs. Returns number of mismatches."""
    mismatches = 0
    all_keys = sorted(set(stats_a.outputs) | set(stats_b.outputs))
    for key in all_keys:
        text_a = stats_a.outputs.get(key, "<missing>")
        text_b = stats_b.outputs.get(key, "<missing>")
        if text_a != text_b:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH {key}:")
                print(f"    {stats_a.label}: {text_a[:120]}...")
                print(f"    {stats_b.label}: {text_b[:120]}...")
    return mismatches


def print_comparison(stats_a: RunStats, stats_b: RunStats) -> None:
    print(f"\n{'=' * 70}")
    print("  A/B Performance Comparison")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {stats_a.label:>18} {stats_b.label:>18}")
    print(f"{'-' * 70}")

    rows = [
        ("Total time (s)", stats_a.total_time_s, stats_b.total_time_s, False),
        (
            "Output tokens",
            stats_a.total_output_tokens,
            stats_b.total_output_tokens,
            True,
        ),
        (
            "Throughput (tok/s)",
            stats_a.throughput_tok_per_s,
            stats_b.throughput_tok_per_s,
            True,
        ),
        (
            "Throughput (req/s)",
            stats_a.throughput_req_per_s,
            stats_b.throughput_req_per_s,
            True,
        ),
    ]
    for name, va, vb, higher_is_better in rows:
        diff = ""
        if va > 0:
            pct = ((vb - va) / va) * 100
            sign = "+" if pct >= 0 else ""
            diff = f"{sign}{pct:.1f}%"
        print(f"{name:<30} {va:>18.2f} {vb:>18.2f}  {diff}")

    print("\nOutput correctness:")
    n_mismatch = compare_outputs(stats_a, stats_b)
    total = max(len(stats_a.outputs), len(stats_b.outputs))
    if n_mismatch == 0:
        print(f"  All {total} outputs MATCH")
    else:
        print(f"  {n_mismatch}/{total} outputs MISMATCH")
        print(
            "  (Some divergence is expected when planners choose different replay budgets)"
        )

    # ---- Imbalance statistics side-by-side ----
    print(f"\n{'=' * 70}")
    print("  Imbalance Statistics (compute vs DMA)")
    print(f"{'=' * 70}")
    for stats in (stats_a, stats_b):
        _print_imbalance_stats(stats.label, stats.imbalance_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B comparison: feedback (online) vs TightLLM ILP (offline)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "/home/lyc/hf_models/opt-2.7b-8k"),
    )
    parser.add_argument(
        "--tightllm-profile-path",
        default=os.environ.get("TIGHTLLM_PROFILE_PATH", ""),
        help="Path to offline profile JSON (required).",
    )
    parser.add_argument(
        "--prefix-blocks",
        type=int,
        default=int(os.environ.get("PREFIX_BLOCKS", "1000")),
    )
    parser.add_argument(
        "--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "32"))
    )
    parser.add_argument(
        "--prompt-words", type=int, default=int(os.environ.get("PROMPT_WORDS", "4000"))
    )
    parser.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "64"))
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=float(os.environ.get("GPU_MEMORY_FRACTION", "0.9")),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument(
        "--num-device-buffers",
        type=int,
        default=int(os.environ.get("NUM_DEVICE_BUFFERS", "3")),
    )
    parser.add_argument(
        "--tightllm-feedback-correction",
        action="store_true",
        help="Enable hybrid mode: ILP + online feedback correction.",
    )
    parser.add_argument(
        "--skip-feedback",
        action="store_true",
        help="Skip the feedback planner run (only run TightLLM).",
    )
    parser.add_argument(
        "--skip-tightllm",
        action="store_true",
        help="Skip the TightLLM planner run (only run feedback).",
    )
    parser.add_argument(
        "--disable-nvtx",
        action="store_true",
        help="Disable NVTX scope emission (default: enabled).",
    )
    args = parser.parse_args()

    # NVTX environment — propagated to subprocesses via os.environ
    if not args.disable_nvtx:
        os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"
    else:
        os.environ.pop("VLLM_NVTX_SCOPES_FOR_PROFILING", None)

    if not args.skip_tightllm and not args.tightllm_profile_path:
        print("ERROR: --tightllm-profile-path is required for TightLLM planner.")
        print("Run the offline profiler first:")
        print("  python -m vllm.v1.profiling.tightllm_offline_profiler \\")
        print(f"      --model {args.model} --output tightllm_profile.json")
        sys.exit(1)

    prompts = build_prompts(args.num_prompts, args.prompt_words)

    print(f"Model:          {args.model}")
    print(f"Prompts:        {args.num_prompts} x ~{args.prompt_words} words")
    print(f"Max tokens:     {args.max_tokens}")
    print(f"Prefix blocks:  {args.prefix_blocks}")
    print(f"NVTX scopes:    {'enabled' if not args.disable_nvtx else 'disabled'}")
    print("Isolation:      each arm in a separate subprocess (spawn)")
    if not args.disable_nvtx:
        print("\nTip: wrap with nsys to collect traces:")
        print("  nsys profile --trace=cuda,nvtx,osrt -o ab_trace \\")
        print(f"    python {' '.join(sys.argv)}")

    results: list[RunStats] = []

    # --- Group A: feedback (online) ---
    if not args.skip_feedback:
        feedback_config = make_kv_offload_config(
            prefix_blocks=args.prefix_blocks,
            gpu_memory_fraction=args.gpu_memory_fraction,
            num_device_buffers=args.num_device_buffers,
            planner="feedback",
            planner_dry_run=False,
        )
        feedback_stats = run_arm(
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_offload_config=feedback_config,
            label="Feedback (online)",
        )
        results.append(feedback_stats)

    # --- Group B: tightllm ILP (offline) ---
    if not args.skip_tightllm:
        tightllm_config = make_kv_offload_config(
            prefix_blocks=args.prefix_blocks,
            gpu_memory_fraction=args.gpu_memory_fraction,
            num_device_buffers=args.num_device_buffers,
            planner="tightllm",
            tightllm_profile_path=args.tightllm_profile_path,
            tightllm_feedback_correction=args.tightllm_feedback_correction,
        )
        tightllm_stats = run_arm(
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_offload_config=tightllm_config,
            label="TightLLM ILP (offline)",
        )
        results.append(tightllm_stats)

    # --- Comparison ---
    if len(results) == 2:
        print_comparison(results[0], results[1])
    elif len(results) == 1:
        r = results[0]
        print(f"\nSingle-arm result: {r.label}")
        print(
            f"  Time: {r.total_time_s:.2f}s  Tokens: {r.total_output_tokens}  "
            f"Throughput: {r.throughput_tok_per_s:.1f} tok/s"
        )
        if r.imbalance_stats and r.imbalance_stats.get("count", 0) > 0:
            print(f"\n  Imbalance stats ({r.imbalance_stats['provider']}):")
            _print_imbalance_stats(r.label, r.imbalance_stats, indent="  ")

    print("\nDone.")


if __name__ == "__main__":
    main()
