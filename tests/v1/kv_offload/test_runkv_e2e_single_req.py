#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Single-Request E2E Test for RunKV KV Cache Offloading.

This script mirrors the output style and reporting format of
`test_runkv_e2e_concurrent.py`, while running exactly one request.

Usage:
    python test_runkv_e2e_single_req.py --model MODEL --prompt-set short
    python test_runkv_e2e_single_req.py --model MODEL --prompt-set medium \
        --prompt-index 2
    python test_runkv_e2e_single_req.py --model MODEL --prompt-set very-long \
        --random-prompt

Requirements:
    - CUDA available
    - vLLM installed with RunKV support
"""

import argparse
import contextlib
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from tests.v1.kv_offload.prompt_dataset import (
        MEDIUM_PROMPTS,
        PROMPT_POOLS,
        SHORT_PROMPTS,
        VERY_LONG_PROMPTS,
    )
except ModuleNotFoundError:
    from prompt_dataset import (
        PROMPT_POOLS,
    )


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _should_use_color() -> bool:
    force = os.environ.get("VLLM_RUNKV_COLOR")
    if force == "0":
        return False
    if force == "1":
        return True
    return os.environ.get("NO_COLOR") is None


_USE_COLOR = _should_use_color()


def _c(text: str, *styles: str) -> str:
    if not _USE_COLOR or not styles:
        return text
    return f"{''.join(styles)}{text}{_Ansi.RESET}"


def _status_color(text: str) -> str:
    if "MATCH" in text and "MISMATCH" not in text:
        return _c(text, _Ansi.GREEN, _Ansi.BOLD)
    if "MISMATCH" in text:
        return _c(text, _Ansi.RED, _Ansi.BOLD)
    return _c(text, _Ansi.YELLOW, _Ansi.BOLD)


@contextlib.contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """
    Context manager for NVTX range marking.
    Only active when VLLM_NVTX_SCOPES_FOR_PROFILING=1 is set.
    """
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


def cuda_profiler_start():
    """Start CUDA profiler (for nsys --capture-range=cudaProfilerApi)."""
    try:
        import torch.cuda

        torch.cuda.cudart().cudaProfilerStart()
        print(_c("[PROFILE] CUDA profiler started", _Ansi.CYAN))
    except Exception as e:
        print(_c(f"[PROFILE] Could not start CUDA profiler: {e}", _Ansi.RED))


def cuda_profiler_stop():
    """Stop CUDA profiler."""
    try:
        import torch.cuda

        torch.cuda.cudart().cudaProfilerStop()
        print(_c("[PROFILE] CUDA profiler stopped", _Ansi.CYAN))
    except Exception as e:
        print(_c(f"[PROFILE] Could not stop CUDA profiler: {e}", _Ansi.RED))


@dataclass
class TokenTrace:
    """Per-token decode trace for diagnosing KV cache issues."""

    per_req_idx: int
    global_idx: int
    token_id: int


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: str
    prompt: str
    output_text: str
    num_tokens: int
    latency_ms: float
    finished: bool
    error: str | None = None
    token_ids: list[int] | None = None
    token_traces: list[TokenTrace] | None = None


@dataclass
class BenchmarkStats:
    """Statistics from a benchmark run."""

    total_requests: int
    completed_requests: int
    failed_requests: int
    total_tokens: int
    total_time_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    throughput_req_per_s: float
    throughput_tok_per_s: float


def check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                _c(
                    "CUDA is not available. RunKV requires a CUDA-enabled GPU.",
                    _Ansi.RED,
                )
            )
            return False
        print(_c(f"CUDA: {torch.cuda.get_device_name(0)}", _Ansi.CYAN))
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(_c(f"GPU memory: {total_memory_gb:.1f} GB", _Ansi.CYAN))
        return True
    except ImportError:
        print(_c("PyTorch not installed", _Ansi.RED))
        return False


def _build_engine(
    *,
    model_name: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_offload_config,
    max_num_seqs: int = 256,
    enable_layerwise_nvtx_tracing: bool = False,
):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    # vLLM requires: max_num_batched_tokens >= max_num_seqs.
    # Keep this invariant explicitly to avoid SchedulerConfig validation errors
    # when users set a large --max-num-seqs.
    max_num_batched_tokens = max(max_model_len, max_num_seqs)

    engine_args = EngineArgs(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_offload_config=kv_offload_config,
        enable_layerwise_nvtx_tracing=enable_layerwise_nvtx_tracing,
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
    """Verify that RunKV is properly enabled in the engine."""
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


def _shutdown_engine(engine) -> None:
    """Gracefully shutdown the engine."""
    with contextlib.suppress(Exception):
        engine.engine_core.shutdown()
    del engine
    with contextlib.suppress(Exception):
        import gc

        import torch

        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory(shutdown_ray=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_concurrent_requests(
    *,
    engine,
    requests: list[tuple[str, str, int]],
    max_steps: int,
    verbose: bool = False,
    enable_profiling: bool = False,
) -> tuple[list[RequestResult], float]:
    """Run requests through the engine (single-request variant uses one request)."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    results: dict[str, RequestResult] = {}
    pending_requests: set[str] = set()
    start_times: dict[str, float] = {}
    request_params: dict[str, tuple[str, int]] = {}

    accumulated_text: dict[str, str] = {}
    accumulated_token_ids: dict[str, list[int]] = {}
    accumulated_traces: dict[str, list[TokenTrace]] = {}
    global_decode_counter = 0

    overall_start = time.time()

    with nvtx_range("add_requests", color="green"):
        for request_id, prompt, max_tokens in requests:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=0.1,
                top_k=1,
                output_kind=RequestOutputKind.DELTA,
            )
            engine.add_request(request_id=request_id, prompt=prompt, params=params)
            pending_requests.add(request_id)
            start_times[request_id] = time.time()
            request_params[request_id] = (prompt, max_tokens)
            accumulated_text[request_id] = ""
            accumulated_token_ids[request_id] = []
            accumulated_traces[request_id] = []

    if verbose:
        print(_c(f"Added {len(requests)} requests to engine", _Ansi.BLUE))

    if enable_profiling:
        cuda_profiler_start()

    step = 0
    try:
        with nvtx_range("inference_loop", color="blue"):
            while pending_requests and step < max_steps:
                with nvtx_range(f"step_{step}", color="yellow"):
                    step_outputs = engine.step()

                for out in step_outputs:
                    req_id = getattr(out, "request_id", None)
                    if req_id is None or req_id not in pending_requests:
                        continue

                    if out.outputs:
                        delta_token_ids = list(out.outputs[0].token_ids)
                        delta_text = out.outputs[0].text
                        accumulated_text[req_id] += delta_text
                        for tid in delta_token_ids:
                            per_req_idx = len(accumulated_token_ids[req_id])
                            accumulated_token_ids[req_id].append(tid)
                            accumulated_traces[req_id].append(
                                TokenTrace(
                                    per_req_idx=per_req_idx,
                                    global_idx=global_decode_counter,
                                    token_id=tid,
                                )
                            )
                            global_decode_counter += 1

                    if out.finished:
                        end_time = time.time()
                        latency_ms = (end_time - start_times[req_id]) * 1000
                        prompt, _ = request_params[req_id]

                        results[req_id] = RequestResult(
                            request_id=req_id,
                            prompt=prompt,
                            output_text=accumulated_text[req_id],
                            num_tokens=len(accumulated_token_ids[req_id]),
                            latency_ms=latency_ms,
                            finished=True,
                            token_ids=accumulated_token_ids[req_id],
                            token_traces=accumulated_traces[req_id],
                        )
                        pending_requests.remove(req_id)

                        if verbose:
                            print(
                                _c(
                                    f"Completed {len(results)}/{len(requests)} "
                                    "requests",
                                    _Ansi.BLUE,
                                )
                            )

                step += 1
    finally:
        if enable_profiling:
            cuda_profiler_stop()

    overall_time = time.time() - overall_start

    for req_id in pending_requests:
        prompt, _ = request_params[req_id]
        results[req_id] = RequestResult(
            request_id=req_id,
            prompt=prompt,
            output_text="",
            num_tokens=0,
            latency_ms=0,
            finished=False,
            error=f"Timed out after {max_steps} steps",
        )

    return list(results.values()), overall_time


def compute_stats(results: list[RequestResult], total_time: float) -> BenchmarkStats:
    """Compute benchmark statistics from results."""
    completed = [r for r in results if r.finished and r.error is None]
    failed = [r for r in results if not r.finished or r.error is not None]

    if not completed:
        return BenchmarkStats(
            total_requests=len(results),
            completed_requests=0,
            failed_requests=len(failed),
            total_tokens=0,
            total_time_s=total_time,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p90_latency_ms=0,
            p99_latency_ms=0,
            throughput_req_per_s=0,
            throughput_tok_per_s=0,
        )

    latencies = sorted([r.latency_ms for r in completed])
    total_tokens = sum(r.num_tokens for r in completed)

    def percentile(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    return BenchmarkStats(
        total_requests=len(results),
        completed_requests=len(completed),
        failed_requests=len(failed),
        total_tokens=total_tokens,
        total_time_s=total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        p50_latency_ms=percentile(latencies, 50),
        p90_latency_ms=percentile(latencies, 90),
        p99_latency_ms=percentile(latencies, 99),
        throughput_req_per_s=len(completed) / total_time if total_time > 0 else 0,
        throughput_tok_per_s=total_tokens / total_time if total_time > 0 else 0,
    )


def print_stats(stats: BenchmarkStats, label: str = "") -> None:
    """Print benchmark statistics."""
    prefix = f"[{label}] " if label else ""
    print(_c(f"\n{prefix}Benchmark Results:", _Ansi.BOLD, _Ansi.CYAN))
    print(_c("-" * 50, _Ansi.DIM))
    print(_c(f"Total requests:      {stats.total_requests}", _Ansi.BLUE))
    print(_c(f"Completed requests:  {stats.completed_requests}", _Ansi.GREEN))
    failed_line = f"Failed requests:     {stats.failed_requests}"
    if stats.failed_requests > 0:
        print(_c(failed_line, _Ansi.RED, _Ansi.BOLD))
    else:
        print(_c(failed_line, _Ansi.GREEN))
    print(_c(f"Total tokens:        {stats.total_tokens}", _Ansi.BLUE))
    print(_c(f"Total time:          {stats.total_time_s:.2f}s", _Ansi.BLUE))
    print(_c(f"Avg latency:         {stats.avg_latency_ms:.2f}ms", _Ansi.BLUE))
    print(_c(f"P50 latency:         {stats.p50_latency_ms:.2f}ms", _Ansi.BLUE))
    print(_c(f"P90 latency:         {stats.p90_latency_ms:.2f}ms", _Ansi.BLUE))
    print(_c(f"P99 latency:         {stats.p99_latency_ms:.2f}ms", _Ansi.BLUE))
    print(_c(f"Throughput (req/s):  {stats.throughput_req_per_s:.2f}", _Ansi.MAGENTA))
    print(_c(f"Throughput (tok/s):  {stats.throughput_tok_per_s:.2f}", _Ansi.MAGENTA))
    print(_c("-" * 50, _Ansi.DIM))


def compare_outputs_multi(
    result_sets: dict[str, list[RequestResult]],
) -> None:
    """Compare outputs across multiple result sets."""
    if len(result_sets) < 2:
        return

    labels = list(result_sets.keys())
    maps: dict[str, dict[str, RequestResult]] = {
        label: {r.request_id: r for r in results}
        for label, results in result_sets.items()
    }

    all_ids: list[str] = []
    seen: set[str] = set()
    for results in result_sets.values():
        for req_res in results:
            if req_res.request_id not in seen:
                all_ids.append(req_res.request_id)
                seen.add(req_res.request_id)
    all_ids.sort()

    match_counts: dict[tuple[str, str], int] = {}
    mismatch_counts: dict[tuple[str, str], int] = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            match_counts[(la, lb)] = 0
            mismatch_counts[(la, lb)] = 0

    baseline_label = labels[0]

    print(_c("\n" + "=" * 80, _Ansi.DIM))
    print(_c(f"Output Comparison ({' / '.join(labels)})", _Ansi.BOLD, _Ansi.CYAN))
    print(_c("=" * 80, _Ansi.DIM))

    for rid in all_ids:
        outputs: dict[str, str] = {}
        for label in labels:
            maybe_res = maps[label].get(rid)
            if maybe_res is None or not maybe_res.finished:
                outputs[label] = "<not available>"
            else:
                outputs[label] = maybe_res.output_text

        pair_statuses: dict[str, str] = {}
        pair_first_diff_token: dict[str, int | None] = {}
        for label in labels[1:]:
            base_r = maps[baseline_label].get(rid)
            comp_r = maps[label].get(rid)
            base_ids = base_r.token_ids if (base_r and base_r.token_ids) else []
            comp_ids = comp_r.token_ids if (comp_r and comp_r.token_ids) else []
            if base_ids == comp_ids:
                pair_statuses[label] = "MATCH"
                pair_first_diff_token[label] = None
            else:
                min_len = min(len(base_ids), len(comp_ids))
                first_diff = min_len
                for ti in range(min_len):
                    if base_ids[ti] != comp_ids[ti]:
                        first_diff = ti
                        break
                pair_first_diff_token[label] = first_diff
                comp_traces = comp_r.token_traces if comp_r else None
                global_info = ""
                if comp_traces and first_diff < len(comp_traces):
                    g = comp_traces[first_diff].global_idx
                    global_info = f", global#{g}"
                pair_statuses[label] = f"MISMATCH@token {first_diff}{global_info}"

        for i, la in enumerate(labels):
            for lb in labels[i + 1 :]:
                if outputs[la] == outputs[lb]:
                    match_counts[(la, lb)] += 1
                else:
                    mismatch_counts[(la, lb)] += 1

        prompt = "<unknown>"
        for label in labels:
            maybe_res = maps[label].get(rid)
            if maybe_res is not None:
                prompt = maybe_res.prompt
                break

        status_parts = [f"{label}={pair_statuses[label]}" for label in labels[1:]]
        all_match = all(s == "MATCH" for s in pair_statuses.values())
        overall = "ALL MATCH" if all_match else " | ".join(status_parts)

        print(f"\n[{_status_color(overall)}] {_c(rid, _Ansi.BOLD)}")
        print(
            _c(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", _Ansi.DIM)
        )

        base_text = outputs[baseline_label]
        print(
            f"  {_c(f'{baseline_label:>12}', _Ansi.BOLD, _Ansi.CYAN)}: "
            f"{base_text[:120]}{'...' if len(base_text) > 120 else ''}"
        )

        for label in labels[1:]:
            text = outputs[label]
            tag = pair_statuses[label]
            print(
                f"  {_c(f'{label:>12}', _Ansi.BOLD, _Ansi.BLUE)}: "
                f"{text[:120]}{'...' if len(text) > 120 else ''}"
                f"  [{_status_color(tag)}]"
            )

        for label in labels[1:]:
            diff_idx = pair_first_diff_token.get(label)
            if diff_idx is None:
                continue

            base_r = maps[baseline_label].get(rid)
            comp_r = maps[label].get(rid)
            base_ids = base_r.token_ids if (base_r and base_r.token_ids) else []
            comp_ids = comp_r.token_ids if (comp_r and comp_r.token_ids) else []
            comp_traces = comp_r.token_traces if comp_r else None

            ctx_before = 3
            ctx_after = 5
            start = max(0, diff_idx - ctx_before)
            end = min(max(len(base_ids), len(comp_ids)), diff_idx + ctx_after + 1)

            print(
                _c(
                    f"  --- Token detail ({baseline_label} vs {label})"
                    f" around divergence at token {diff_idx} ---",
                    _Ansi.YELLOW,
                    _Ansi.BOLD,
                )
            )
            print(
                _c(
                    f"  {'idx':>5} {'global':>7} | "
                    f"{'baseline_tid':>13} {'compare_tid':>13} {'status':>10}",
                    _Ansi.DIM,
                )
            )
            for ti in range(start, end):
                b_tid = base_ids[ti] if ti < len(base_ids) else "<eos>"
                c_tid = comp_ids[ti] if ti < len(comp_ids) else "<eos>"

                g_str = ""
                if comp_traces and ti < len(comp_traces):
                    g_str = str(comp_traces[ti].global_idx)

                marker = " " if b_tid == c_tid else "<<"
                arrow = " >>>>" if ti == diff_idx else ""
                line = (
                    f"  {ti:>5} {g_str:>7} | {str(b_tid):>13} {str(c_tid):>13}"
                    f" {marker:>10}{arrow}"
                )
                if marker != " ":
                    print(_c(line, _Ansi.RED))
                else:
                    print(line)

    print(_c("\n" + "-" * 80, _Ansi.DIM))
    print(_c("Pairwise Summary:", _Ansi.BOLD, _Ansi.CYAN))
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            m = match_counts[(la, lb)]
            mm = mismatch_counts[(la, lb)]
            line = f"  {la} vs {lb}: {m} match, {mm} mismatch"
            if mm > 0:
                print(_c(line, _Ansi.YELLOW))
            else:
                print(_c(line, _Ansi.GREEN))
    print(_c("-" * 80, _Ansi.DIM))


def select_single_prompt(
    *,
    prompt_set: str,
    prompt_index: int,
    random_prompt: bool,
    prompt_seed: int,
) -> tuple[str, str, int, int]:
    """Select exactly one prompt from built-in pools."""
    pool = PROMPT_POOLS[prompt_set]
    pool_size = len(pool)
    if pool_size == 0:
        raise RuntimeError(f"Prompt pool '{prompt_set}' is empty.")

    if random_prompt:
        rng = random.Random(prompt_seed)
        idx = rng.randrange(pool_size)
    else:
        if prompt_index < 0 or prompt_index >= pool_size:
            raise ValueError(
                f"--prompt-index must be in [0, {pool_size - 1}] "
                f"for --prompt-set={prompt_set}"
            )
        idx = prompt_index

    return pool[idx], prompt_set, idx, pool_size


def run_benchmark(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_steps: int,
    max_num_seqs: int,
    kv_offload_config,
    label: str = "",
    verbose: bool = False,
    enable_profiling: bool = False,
    enable_layerwise_nvtx_tracing: bool = False,
) -> tuple[list[RequestResult], BenchmarkStats]:
    """Run a single-request benchmark with concurrent-style reporting."""
    print(_c(f"\n{'=' * 60}", _Ansi.DIM))
    print(_c(f"Running benchmark: {label or 'unnamed'}", _Ansi.BOLD, _Ansi.CYAN))
    if enable_profiling:
        print(
            _c(
                "[PROFILE] Profiling enabled - use nsys with"
                " --capture-range=cudaProfilerApi",
                _Ansi.YELLOW,
            )
        )
    if enable_layerwise_nvtx_tracing:
        print(_c("[PROFILE] Layer-wise NVTX tracing enabled", _Ansi.YELLOW))
    print(_c(f"{'=' * 60}", _Ansi.DIM))

    with nvtx_range("build_engine", color="purple"):
        engine = _build_engine(
            model_name=model_name,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_offload_config=kv_offload_config,
            max_num_seqs=max_num_seqs,
            enable_layerwise_nvtx_tracing=enable_layerwise_nvtx_tracing,
        )

    try:
        if kv_offload_config.enabled:
            _assert_runkv_active(engine)
            print(_c("RunKV verification: PASSED", _Ansi.GREEN, _Ansi.BOLD))

        request_id = "single-test-000000"
        requests = [(request_id, prompt, max_tokens)]
        print(_c(f"Generated {len(requests)} test requests", _Ansi.BLUE))

        print(_c("Starting concurrent request processing...", _Ansi.BLUE))
        results, total_time = run_concurrent_requests(
            engine=engine,
            requests=requests,
            max_steps=max_steps,
            verbose=verbose,
            enable_profiling=enable_profiling,
        )

        stats = compute_stats(results, total_time)
        print_stats(stats, label)

        return results, stats

    finally:
        _shutdown_engine(engine)


def main():
    parser = argparse.ArgumentParser(description="RunKV Single-Request E2E Test")
    parser.add_argument(
        "--model",
        type=str,
        default="~/hf_models/opt-1.3b",
        help="HuggingFace model name (default: ~/hf_models/opt-1.3b)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional raw prompt override. If omitted, use built-in prompt pools.",
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        choices=["short", "medium", "very-long"],
        default="short",
        help="Built-in prompt pool to use when --prompt is not set",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Prompt index in selected --prompt-set (0-based)",
    )
    parser.add_argument(
        "--random-prompt",
        action="store_true",
        help="Randomly pick one prompt from selected --prompt-set",
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=42,
        help="Seed for --random-prompt (default: 42)",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List built-in prompt pools and exit",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Maximum number of sequences to process in parallel (default: 8)",
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=30.0,
        help="CPU memory limit for RunKV in GB (default: 30.0)",
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.3,
        help="CPU memory fraction for RunKV (default: 0.3)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare RunKV outputs with baseline (RunKV disabled)",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Skip CUDA availability check",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10240,
        help="Maximum engine steps to wait for completion (default: 10240)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Model max length (default: 2048)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization for vLLM (default: 0.5)",
    )
    parser.add_argument(
        "--num-device-buffers",
        type=int,
        default=3,
        help="RunKV GPU ring buffer count (default: 3)",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.85,
        help="Fraction of GPU budget for RunKV staging buffers (default: 0.85)",
    )
    parser.add_argument(
        "--max-staging-blocks",
        type=int,
        default=0,
        help="Override computed staging blocks per buffer (0 = auto)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose progress output",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable CUDA profiling (use with nsys --capture-range=cudaProfilerApi)",
    )
    parser.add_argument(
        "--enable-layerwise-nvtx-tracing",
        action="store_true",
        help="Enable per-layer NVTX markers for detailed profiling",
    )
    parser.add_argument(
        "--enable-layer-recompute",
        action="store_true",
        help="Run an additional benchmark with layer recompute enabled",
    )
    parser.add_argument(
        "--layer-recompute-io-prefix-blocks",
        type=int,
        nargs="+",
        default=[19900],
        help="IO prefix blocks per layer for recompute (default: [19900])",
    )

    args = parser.parse_args()

    if args.list_prompts:
        print(_c("Built-in prompt pools:", _Ansi.BOLD, _Ansi.CYAN))
        for name, pool in PROMPT_POOLS.items():
            print(_c(f"  - {name}: {len(pool)} prompts", _Ansi.BLUE))
            for i, p in enumerate(pool):
                preview = p.replace("\n", " ")[:100]
                suffix = "..." if len(p) > 100 else ""
                print(_c(f"      [{i}] {preview}{suffix}", _Ansi.DIM))
        sys.exit(0)

    if args.profile or args.enable_layerwise_nvtx_tracing:
        os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")

    print(_c("=" * 60, _Ansi.DIM), flush=True)
    print(_c("RunKV Single-Request E2E Test", _Ansi.BOLD, _Ansi.CYAN), flush=True)
    print(_c("=" * 60, _Ansi.DIM), flush=True)

    args.model = os.path.expandvars(os.path.expanduser(args.model))
    model_path = Path(args.model)
    if args.model.startswith((".", os.sep)) or "~" in args.model or model_path.exists():
        args.model = str(model_path.expanduser().absolute())
        model_path = Path(args.model)
        if model_path.is_dir() and not (model_path / "config.json").exists():
            print(
                _c(
                    f"Model path looks local but missing config.json: {args.model}\n"
                    "Pass a valid local HF directory (must contain config.json), "
                    "or pass a HuggingFace repo id like 'namespace/repo_name'.",
                    _Ansi.RED,
                )
            )
            sys.exit(1)

    if not args.skip_cuda_check and not check_cuda():
        sys.exit(1)

    selected_prompt_source = "custom"
    selected_prompt_idx = -1
    selected_prompt_pool_size = 1
    if args.prompt is not None:
        selected_prompt = args.prompt
    else:
        (
            selected_prompt,
            selected_prompt_source,
            selected_prompt_idx,
            selected_prompt_pool_size,
        ) = select_single_prompt(
            prompt_set=args.prompt_set,
            prompt_index=args.prompt_index,
            random_prompt=args.random_prompt,
            prompt_seed=args.prompt_seed,
        )

    if selected_prompt_source == "custom":
        print(_c("Selected prompt: custom --prompt", _Ansi.BLUE))
    else:
        print(
            _c(
                f"Selected prompt: {selected_prompt_source}[{selected_prompt_idx}]"
                f" / {selected_prompt_pool_size}",
                _Ansi.BLUE,
            )
        )
    preview = selected_prompt.replace("\n", " ")[:120]
    print(
        _c(
            f"Prompt preview: {preview}{'...' if len(selected_prompt) > 120 else ''}",
            _Ansi.DIM,
        )
    )

    from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

    if not (0.0 < args.cpu_memory_fraction <= 1.0):
        raise ValueError("--cpu-memory-fraction must be in (0, 1]")

    cpu_limit_bytes = None
    if args.cpu_memory_gb > 0:
        cpu_limit_bytes = int(args.cpu_memory_gb * 1024**3)

    baseline_results: list[RequestResult] | None = None
    baseline_stats: BenchmarkStats | None = None
    if args.compare_baseline:
        baseline_config = RunKVOffloadConfig(enabled=False)
        baseline_results, baseline_stats = run_benchmark(
            model_name=args.model,
            prompt=selected_prompt,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_steps=args.max_steps,
            max_num_seqs=args.max_num_seqs,
            kv_offload_config=baseline_config,
            label="Baseline (RunKV disabled)",
            verbose=args.verbose,
            enable_profiling=False,
            enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
        )

    runkv_config = RunKVOffloadConfig(
        enabled=True,
        num_device_buffers=args.num_device_buffers,
        gpu_memory_fraction=args.gpu_memory_fraction,
        enable_async_prefetch=True,
        enable_async_offload=True,
        cpu_memory_limit=cpu_limit_bytes,
        cpu_memory_fraction=args.cpu_memory_fraction,
        max_staging_blocks=(args.max_staging_blocks or None),
    )

    runkv_results: list[RequestResult]
    runkv_stats: BenchmarkStats
    runkv_results, runkv_stats = run_benchmark(
        model_name=args.model,
        prompt=selected_prompt,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_steps=args.max_steps,
        max_num_seqs=args.max_num_seqs,
        kv_offload_config=runkv_config,
        label="RunKV enabled",
        verbose=args.verbose,
        enable_profiling=args.profile,
        enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
    )

    recompute_results: list[RequestResult] | None = None
    recompute_stats: BenchmarkStats | None = None
    if args.enable_layer_recompute:
        recompute_config = RunKVOffloadConfig(
            enabled=True,
            num_device_buffers=args.num_device_buffers,
            gpu_memory_fraction=args.gpu_memory_fraction,
            enable_async_prefetch=True,
            enable_async_offload=True,
            cpu_memory_limit=cpu_limit_bytes,
            cpu_memory_fraction=args.cpu_memory_fraction,
            max_staging_blocks=(args.max_staging_blocks or None),
            enable_layer_recompute=True,
            layer_recompute_io_prefix_blocks=args.layer_recompute_io_prefix_blocks,
        )
        recompute_results, recompute_stats = run_benchmark(
            model_name=args.model,
            prompt=selected_prompt,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_steps=args.max_steps,
            max_num_seqs=args.max_num_seqs,
            kv_offload_config=recompute_config,
            label="RunKV + Layer Recompute",
            verbose=args.verbose,
            enable_profiling=args.profile,
            enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
        )

    result_sets: dict[str, list[RequestResult]] = {}
    if baseline_results is not None:
        result_sets["Baseline"] = baseline_results
    result_sets["RunKV"] = runkv_results
    if recompute_results is not None:
        result_sets["Recompute"] = recompute_results

    if len(result_sets) >= 2:
        compare_outputs_multi(result_sets)

    def fmt_diff(
        baseline_val: float, compare_val: float, higher_is_better: bool = True
    ) -> str:
        if baseline_val == 0:
            return "N/A"
        diff_pct = ((compare_val - baseline_val) / baseline_val) * 100
        if not higher_is_better:
            diff_pct = -diff_pct
        sign = "+" if diff_pct >= 0 else ""
        return f"{sign}{diff_pct:.1f}%"

    def _print_perf_comparison(
        label_a: str,
        stats_a: BenchmarkStats,
        label_b: str,
        stats_b: BenchmarkStats,
    ) -> None:
        print(_c(f"\n{'=' * 80}", _Ansi.DIM))
        print(
            _c(
                f"Performance Comparison: {label_a} vs {label_b}",
                _Ansi.BOLD,
                _Ansi.CYAN,
            )
        )
        print(_c(f"{'=' * 80}", _Ansi.DIM))
        print(
            _c(f"{'Metric':<25} {label_a:>15} {label_b:>15} {'Diff':>15}", _Ansi.BOLD)
        )
        print(_c("-" * 70, _Ansi.DIM))
        for metric, attr, hib in [
            ("Throughput (req/s)", "throughput_req_per_s", True),
            ("Throughput (tok/s)", "throughput_tok_per_s", True),
            ("Avg latency (ms)", "avg_latency_ms", False),
            ("P50 latency (ms)", "p50_latency_ms", False),
            ("P90 latency (ms)", "p90_latency_ms", False),
            ("P99 latency (ms)", "p99_latency_ms", False),
        ]:
            va = getattr(stats_a, attr)
            vb = getattr(stats_b, attr)
            diff = fmt_diff(va, vb, higher_is_better=hib)
            if diff == "N/A":
                diff_colored = _c(diff, _Ansi.DIM)
            elif diff.startswith("-"):
                diff_colored = _c(diff, _Ansi.RED)
            else:
                diff_colored = _c(diff, _Ansi.GREEN)
            print(f"{metric:<25} {va:>15.2f} {vb:>15.2f} {diff_colored:>15}")

    if baseline_results is not None and baseline_stats is not None:
        _print_perf_comparison("Baseline", baseline_stats, "RunKV", runkv_stats)

    if recompute_stats is not None:
        _print_perf_comparison("RunKV", runkv_stats, "Recompute", recompute_stats)

    if baseline_stats is not None and recompute_stats is not None:
        _print_perf_comparison("Baseline", baseline_stats, "Recompute", recompute_stats)

    print(_c("\n" + "=" * 60, _Ansi.DIM))
    print(_c("Test Summary", _Ansi.BOLD, _Ansi.CYAN))
    print(_c("=" * 60, _Ansi.DIM))

    success = True

    if runkv_stats.failed_requests == 0:
        print(
            _c(
                f"✓ RunKV: All {runkv_stats.completed_requests} requests completed",
                _Ansi.GREEN,
                _Ansi.BOLD,
            )
        )
    else:
        print(
            _c(
                f"✗ RunKV: {runkv_stats.failed_requests} requests failed",
                _Ansi.RED,
                _Ansi.BOLD,
            )
        )
        success = False

    if recompute_stats is not None:
        if recompute_stats.failed_requests == 0:
            print(
                _c(
                    f"✓ Recompute: All {recompute_stats.completed_requests} "
                    "requests completed",
                    _Ansi.GREEN,
                    _Ansi.BOLD,
                )
            )
        else:
            print(
                _c(
                    f"✗ Recompute: {recompute_stats.failed_requests} requests failed",
                    _Ansi.RED,
                    _Ansi.BOLD,
                )
            )
            success = False

    if baseline_results is not None:
        print(
            _c(
                "⚠ Check output comparison above for mismatches"
                " (may be expected with sampling)",
                _Ansi.YELLOW,
            )
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
