#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
High-Concurrency E2E Test for the Feedback Planner Loop.

This script verifies end-to-end correctness of the full runkv planner loop
(layer_recompute_planner="feedback", layer_recompute_planner_dry_run=False)
under concurrent request load.  It mirrors the setup used in
examples/offline_inference/run_opt_feedback_observation.py and exercises the
same planner-driven skip/replay path that was the motivation for that script.

Scenarios tested:
  - Feedback planner (dry_run=False): planner actively adjusts IO-prefix budgets
  - Feedback planner (dry_run=True): planner observes only (no execution change)
  - Optionally compared against a plain baseline (RunKV disabled)

Usage:
    python test_feedback_planner_e2e_concurrent.py [--model MODEL]

Profiling with Nsight Systems:
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 nsys profile -o vllm_profile \\
        --trace=cuda,nvtx \\
        python test_feedback_planner_e2e_concurrent.py --num-requests 64

    # With layer-wise NVTX tracing
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 nsys profile -o vllm_detailed \\
        --trace=cuda,nvtx \\
        python test_feedback_planner_e2e_concurrent.py --num-requests 64 \\
        --enable-layerwise-nvtx-tracing

Requirements:
    - CUDA available
    - vLLM installed with runkv / dynamic-replay support

Example:
    python test_feedback_planner_e2e_concurrent.py \\
        --model ~/hf_models/opt-2.7b \\
        --num-requests 64 --prefix-blocks 1000
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import os
import random
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import regex as re

try:
    from tests.v1.kv_offload.prompt_dataset import (
        EXTRA_LONG_PROMPTS,
        MEDIUM_PROMPTS,
        SAMPLE_PROMPTS,
        VERY_LONG_PROMPTS,
    )
except ModuleNotFoundError:
    from prompt_dataset import (
        EXTRA_LONG_PROMPTS,
        MEDIUM_PROMPTS,
        SAMPLE_PROMPTS,
        VERY_LONG_PROMPTS,
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


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


def _first_diff_char_idx(a: str, b: str) -> int | None:
    if a == b:
        return None
    min_len = min(len(a), len(b))
    for idx in range(min_len):
        if a[idx] != b[idx]:
            return idx
    return min_len


def _first_diff_token_idx(a: list[int], b: list[int]) -> int | None:
    if a == b:
        return None
    min_len = min(len(a), len(b))
    for idx in range(min_len):
        if a[idx] != b[idx]:
            return idx
    return min_len


def _line_col_from_char_idx(text: str, char_idx: int) -> tuple[int, int]:
    line = text.count("\n", 0, char_idx) + 1
    last_newline = text.rfind("\n", 0, char_idx)
    col = char_idx + 1 if last_newline == -1 else char_idx - last_newline
    return line, col


def _wrap_text_for_diff(text: str, width: int) -> list[str]:
    if not text:
        return [""]

    wrapper = textwrap.TextWrapper(
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    )
    wrapped_lines: list[str] = []
    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        wrapped = wrapper.wrap(line)
        wrapped_lines.extend(wrapped if wrapped else [""])

    if text.endswith("\n"):
        wrapped_lines.append("")

    return wrapped_lines or [""]


def _highlight_text_diff(left: str, right: str) -> tuple[str, str]:
    left_parts: list[str] = []
    right_parts: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=left, b=right).get_opcodes():
        left_chunk = left[i1:i2]
        right_chunk = right[j1:j2]
        if tag == "equal":
            left_parts.append(left_chunk)
            right_parts.append(right_chunk)
            continue
        if left_chunk:
            left_parts.append(_c(left_chunk, _Ansi.RED, _Ansi.BOLD))
        if right_chunk:
            right_parts.append(_c(right_chunk, _Ansi.GREEN, _Ansi.BOLD))
    return "".join(left_parts), "".join(right_parts)


def _print_side_by_side_text_diff(
    baseline_label: str,
    baseline_text: str,
    compare_label: str,
    compare_text: str,
    mismatch_token_idx: int | None = None,
    mismatch_global_idx: int | None = None,
) -> None:
    total_width = shutil.get_terminal_size(fallback=(180, 20)).columns
    col_width = max(40, (total_width - 15) // 2)
    base_lines = _wrap_text_for_diff(baseline_text, col_width)
    comp_lines = _wrap_text_for_diff(compare_text, col_width)

    first_diff = _first_diff_char_idx(baseline_text, compare_text)
    location = ""
    if first_diff is not None:
        line, col = _line_col_from_char_idx(baseline_text, first_diff)
        location = f" first diff at char {first_diff} (line {line}, col {col})"
    token_location = ""
    if mismatch_token_idx is not None:
        token_location = f", token {mismatch_token_idx}"
        if mismatch_global_idx is not None:
            token_location += f", global#{mismatch_global_idx}"

    print(
        _c(
            "  --- Text diff "
            f"({baseline_label} vs {compare_label})"
            f"{location}{token_location} ---",
            _Ansi.YELLOW,
            _Ansi.BOLD,
        )
    )
    print(
        _c(
            f"  {'row':>4} | {baseline_label:<{col_width}} | "
            f"{compare_label:<{col_width}}",
            _Ansi.DIM,
        )
    )

    row_idx = 0
    matcher = difflib.SequenceMatcher(a=base_lines, b=comp_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_block = base_lines[i1:i2]
        right_block = comp_lines[j1:j2]
        if tag == "equal":
            for left_line, right_line in zip(left_block, right_block, strict=False):
                print(f"  {row_idx:>4} | {left_line.ljust(col_width)} | {right_line}")
                row_idx += 1
            continue

        max_rows = max(len(left_block), len(right_block))
        for offset in range(max_rows):
            left_line_opt: str | None = (
                left_block[offset] if offset < len(left_block) else None
            )
            right_line_opt: str | None = (
                right_block[offset] if offset < len(right_block) else None
            )
            if left_line_opt is None:
                left_render = _c("<no line>", _Ansi.DIM)
                right_render = _c(right_line_opt or "", _Ansi.GREEN, _Ansi.BOLD)
            elif right_line_opt is None:
                left_render = _c(left_line_opt, _Ansi.RED, _Ansi.BOLD)
                right_render = _c("<no line>", _Ansi.DIM)
            else:
                left_render, right_render = _highlight_text_diff(
                    left_line_opt,
                    right_line_opt,
                )

            left_pad = " " * max(0, col_width - _visible_len(left_render))
            print(f"  {row_idx:>4} | {left_render}{left_pad} | {right_render}")
            row_idx += 1


# ============================================================================
# Profiling Utilities
# ============================================================================
@contextlib.contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """Context manager for NVTX range marking."""
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

    per_req_idx: int  # N-th token decoded for THIS request (0-based)
    global_idx: int  # N-th token decoded across ALL requests (0-based)
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
                    "CUDA is not available. This test requires a CUDA-enabled GPU.",
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

    engine_args = EngineArgs(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        disable_cascade_attn=True,
        enable_prefix_caching=False,
        max_num_seqs=max_num_seqs,
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


def _assert_feedback_planner_active(engine, *, dry_run: bool) -> None:
    """Verify that the feedback planner is properly enabled in the engine."""
    vllm_config = getattr(engine, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError("Cannot access engine.vllm_config to verify planner state.")

    cfg = getattr(vllm_config, "kv_offload_config", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        raise RuntimeError("kv_offload_config is not enabled.")

    if not getattr(cfg, "enable_layer_recompute", False):
        raise RuntimeError("enable_layer_recompute is False in kv_offload_config.")

    planner = getattr(cfg, "layer_recompute_planner", None)
    if planner != "feedback":
        raise RuntimeError(
            f"Expected layer_recompute_planner='feedback', got '{planner}'."
        )

    actual_dry_run = getattr(cfg, "layer_recompute_planner_dry_run", None)
    if actual_dry_run != dry_run:
        raise RuntimeError(
            f"Expected layer_recompute_planner_dry_run={dry_run}, got {actual_dry_run}."
        )

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
        raise RuntimeError("Cannot access worker.model_runner to verify planner state.")

    if not getattr(model_runner, "use_runkv", False):
        raise RuntimeError("model_runner.use_runkv is False (runkv not active).")
    if not getattr(getattr(model_runner, "kv_offload_config", None), "enabled", False):
        raise RuntimeError("model_runner.kv_offload_config.enabled is False.")
    if not getattr(model_runner, "kv_buffers", None):
        raise RuntimeError(
            "Feedback planner is active but model_runner.kv_buffers is empty."
        )


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


def generate_test_requests(
    num_requests: int,
    min_tokens: int,
    max_tokens: int,
    use_long_prompts: bool = False,
    use_very_long_prompts: bool = False,
    use_extra_long_prompts: bool = False,
    long_prompt_ratio: float = 0.3,
    extra_prompt_ratio: float = 0.1,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    """Generate test requests with varying prompts and token lengths."""
    random.seed(seed)
    requests = []

    short_prompts = SAMPLE_PROMPTS
    long_prompts = []
    extra_long_prompts = []
    if use_long_prompts:
        long_prompts.extend(MEDIUM_PROMPTS)
    if use_very_long_prompts:
        long_prompts.extend(VERY_LONG_PROMPTS)
    if use_extra_long_prompts:
        extra_long_prompts.extend(EXTRA_LONG_PROMPTS)

    for i in range(num_requests):
        request_id = f"planner-test-{i:06d}"

        if extra_long_prompts and random.random() < extra_prompt_ratio:
            prompt = random.choice(extra_long_prompts)
        elif long_prompts and random.random() < long_prompt_ratio + extra_prompt_ratio:
            prompt = random.choice(long_prompts)
        else:
            prompt = random.choice(short_prompts)

        tokens = random.randint(min_tokens, max_tokens)
        requests.append((request_id, prompt, tokens))

    return requests


def run_concurrent_requests(
    *,
    engine,
    requests: list[tuple[str, str, int]],
    max_steps: int,
    verbose: bool = False,
    enable_profiling: bool = False,
) -> tuple[list[RequestResult], float]:
    """Run multiple requests concurrently through the engine."""
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
                        prompt, max_tokens = request_params[req_id]

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

                        if verbose and len(results) % 10 == 0:
                            print(
                                _c(
                                    f"Completed {len(results)}/{len(requests)} requests",
                                    _Ansi.BLUE,
                                )
                            )

                step += 1
    finally:
        if enable_profiling:
            cuda_profiler_stop()

    overall_time = time.time() - overall_start

    for req_id in pending_requests:
        prompt, max_tokens = request_params[req_id]
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
        pair_first_diff_global: dict[str, int | None] = {}
        for label in labels[1:]:
            base_r = maps[baseline_label].get(rid)
            comp_r = maps[label].get(rid)
            base_text = base_r.output_text if base_r and base_r.finished else ""
            comp_text = comp_r.output_text if comp_r and comp_r.finished else ""
            base_ids = base_r.token_ids if (base_r and base_r.token_ids) else []
            comp_ids = comp_r.token_ids if (comp_r and comp_r.token_ids) else []
            diff_token_idx = _first_diff_token_idx(base_ids, comp_ids)
            pair_first_diff_token[label] = diff_token_idx
            comp_traces = comp_r.token_traces if comp_r else None
            if (
                diff_token_idx is not None
                and comp_traces is not None
                and diff_token_idx < len(comp_traces)
            ):
                pair_first_diff_global[label] = comp_traces[diff_token_idx].global_idx
            else:
                pair_first_diff_global[label] = None

            first_diff = _first_diff_char_idx(base_text, comp_text)
            if first_diff is None:
                pair_statuses[label] = "MATCH"
            else:
                line, col = _line_col_from_char_idx(base_text, first_diff)
                token_info = ""
                if diff_token_idx is not None:
                    token_info = f", token {diff_token_idx}"
                    global_idx = pair_first_diff_global[label]
                    if global_idx is not None:
                        token_info += f", global#{global_idx}"
                pair_statuses[label] = (
                    f"MISMATCH@char {first_diff} (L{line}:C{col}{token_info})"
                )

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
            if pair_statuses.get(label) == "MATCH":
                continue

            base_r = maps[baseline_label].get(rid)
            comp_r = maps[label].get(rid)
            base_text = base_r.output_text if base_r and base_r.finished else ""
            comp_text = comp_r.output_text if comp_r and comp_r.finished else ""
            _print_side_by_side_text_diff(
                baseline_label,
                base_text,
                label,
                comp_text,
                mismatch_token_idx=pair_first_diff_token.get(label),
                mismatch_global_idx=pair_first_diff_global.get(label),
            )

    print(_c("\n" + "-" * 80, _Ansi.DIM))
    print(_c("Pairwise Summary:", _Ansi.BOLD, _Ansi.CYAN))
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            m = match_counts[(la, lb)]
            mm = mismatch_counts[(la, lb)]
            summary_line = f"  {la} vs {lb}: {m} match, {mm} mismatch"
            if mm > 0:
                print(_c(summary_line, _Ansi.YELLOW))
            else:
                print(_c(summary_line, _Ansi.GREEN))
    print(_c("-" * 80, _Ansi.DIM))


def run_benchmark(
    *,
    model_name: str,
    num_requests: int,
    min_tokens: int,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_steps: int,
    max_num_seqs: int,
    use_long_prompts: bool,
    use_very_long_prompts: bool,
    use_extra_long_prompts: bool,
    long_prompt_ratio: float,
    extra_prompt_ratio: float,
    kv_offload_config,
    dry_run: bool = False,
    label: str = "",
    verbose: bool = False,
    enable_profiling: bool = False,
    enable_layerwise_nvtx_tracing: bool = False,
) -> tuple[list[RequestResult], BenchmarkStats]:
    """Run a full benchmark with the given configuration."""
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
        # Verify feedback planner is active when enabled
        if (
            kv_offload_config.enabled
            and getattr(kv_offload_config, "layer_recompute_planner", None)
            == "feedback"
        ):
            _assert_feedback_planner_active(engine, dry_run=dry_run)
            mode = "dry-run" if dry_run else "active"
            print(
                _c(
                    f"Feedback planner verification ({mode}): PASSED",
                    _Ansi.GREEN,
                    _Ansi.BOLD,
                )
            )

        requests = generate_test_requests(
            num_requests=num_requests,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            use_long_prompts=use_long_prompts,
            use_very_long_prompts=use_very_long_prompts,
            use_extra_long_prompts=use_extra_long_prompts,
            long_prompt_ratio=long_prompt_ratio,
            extra_prompt_ratio=extra_prompt_ratio,
        )
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
    parser = argparse.ArgumentParser(description="Feedback Planner Full-Loop E2E Test")
    parser.add_argument(
        "--model",
        type=str,
        default="~/hf_models/opt-2.7b",
        help="HuggingFace model name (default: ~/hf_models/opt-2.7b)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=64,
        help="Total number of requests to process (default: 64)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=16,
        help="Minimum tokens to generate per request (default: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per request (default: 32)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences to process in parallel (default: 256)",
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=40.0,
        help="CPU memory limit for backing store in GB (default: 40.0)",
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.4,
        help="CPU memory fraction for backing store (default: 0.4)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run a baseline (RunKV disabled) for output comparison",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Skip CUDA availability check",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum engine steps to wait for completion (default: 10000)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Model max length (default: 2048, matching OPT-2.7b's max_position_embeddings)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
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
        default=0.9,
        help="Fraction of GPU budget for staging buffers (default: 0.9)",
    )
    parser.add_argument(
        "--max-staging-blocks",
        type=int,
        default=0,
        help="Override computed staging blocks per buffer (0 = auto)",
    )
    parser.add_argument(
        "--prefix-blocks",
        type=int,
        default=1000,
        help=(
            "Initial layer_recompute_io_prefix_blocks value broadcast to all layers "
            "(default: 1000, matching run_opt_feedback_observation.py)"
        ),
    )
    parser.add_argument(
        "--planner-dry-run",
        action="store_true",
        help=(
            "Keep the feedback planner in observe-only mode. "
            "The planner updates its state but does not change execution. "
            "Default: False (planner actively adjusts IO-prefix budgets)."
        ),
    )
    parser.add_argument(
        "--use-long-prompts",
        action="store_true",
        help="Include medium-length prompts (~50-100 tokens)",
    )
    parser.add_argument(
        "--use-very-long-prompts",
        action="store_true",
        help="Include very long prompts (~512 tokens)",
    )
    parser.add_argument(
        "--use-extra-long-prompts",
        action="store_true",
        default=True,
        help=(
            "Include extra long prompts (~2048 tokens) to stress the planner "
            "(default: True, matching the ~2000-word prompts in the observation script)"
        ),
    )
    parser.add_argument(
        "--long-prompt-ratio",
        type=float,
        default=0.3,
        help="Fraction of requests using longer prompts (default: 0.3)",
    )
    parser.add_argument(
        "--extra-long-prompt-ratio",
        type=float,
        default=0.5,
        help="Fraction of requests using extra long prompts (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose progress output",
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run an extended stress test with more requests and variations",
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

    args = parser.parse_args()

    if args.profile or args.enable_layerwise_nvtx_tracing:
        os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")

    print(_c("=" * 60, _Ansi.DIM), flush=True)
    print(
        _c("Feedback Planner Full-Loop E2E Test", _Ansi.BOLD, _Ansi.CYAN),
        flush=True,
    )
    print(_c("=" * 60, _Ansi.DIM), flush=True)

    # Expand and validate model path
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

    from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

    if not (0.0 < args.cpu_memory_fraction <= 1.0):
        raise ValueError("--cpu-memory-fraction must be in (0, 1]")

    cpu_limit_bytes = None
    if args.cpu_memory_gb > 0:
        cpu_limit_bytes = int(args.cpu_memory_gb * 1024**3)

    if args.stress_test:
        print(_c("\n*** STRESS TEST MODE ***", _Ansi.YELLOW, _Ansi.BOLD))
        args.num_requests = max(args.num_requests, 500)
        args.use_long_prompts = True
        args.use_very_long_prompts = True
        args.use_extra_long_prompts = True
        args.max_tokens = max(args.max_tokens, 256)
        args.long_prompt_ratio = 0.3
        args.extra_long_prompt_ratio = 0.5
        print(
            _c(
                f"Adjusted: num_requests={args.num_requests}, "
                f"max_tokens={args.max_tokens}",
                _Ansi.YELLOW,
            )
        )

    # Print active configuration
    dry_run_str = (
        "observe-only (dry_run=True)"
        if args.planner_dry_run
        else "active (dry_run=False)"
    )
    print(_c("\nPlanner config:", _Ansi.BOLD))
    print(_c("  planner:          feedback", _Ansi.BLUE))
    print(_c(f"  mode:             {dry_run_str}", _Ansi.BLUE))
    print(_c(f"  prefix_blocks:    {args.prefix_blocks}", _Ansi.BLUE))
    print(_c(f"  num_device_buf:   {args.num_device_buffers}", _Ansi.BLUE))
    print(_c(f"  gpu_mem_frac:     {args.gpu_memory_fraction}", _Ansi.BLUE))
    print()

    # Optionally run baseline
    baseline_results: list[RequestResult] | None = None
    baseline_stats: BenchmarkStats | None = None
    if args.compare_baseline:
        baseline_config = RunKVOffloadConfig(enabled=False)
        baseline_results, baseline_stats = run_benchmark(
            model_name=args.model,
            num_requests=args.num_requests,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_steps=args.max_steps,
            max_num_seqs=args.max_num_seqs,
            use_long_prompts=args.use_long_prompts,
            use_very_long_prompts=args.use_very_long_prompts,
            use_extra_long_prompts=args.use_extra_long_prompts,
            long_prompt_ratio=args.long_prompt_ratio,
            extra_prompt_ratio=args.extra_long_prompt_ratio,
            kv_offload_config=baseline_config,
            dry_run=False,
            label="Baseline",
            verbose=args.verbose,
            enable_profiling=False,
            enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
        )

    # Build feedback planner config — mirrors run_opt_feedback_observation.py
    feedback_config = RunKVOffloadConfig(
        enabled=True,
        num_device_buffers=args.num_device_buffers,
        gpu_memory_fraction=args.gpu_memory_fraction,
        enable_async_prefetch=True,
        enable_async_offload=True,
        cpu_memory_limit=cpu_limit_bytes,
        cpu_memory_fraction=args.cpu_memory_fraction,
        max_staging_blocks=(args.max_staging_blocks or None),
        enable_layer_recompute=True,
        layer_recompute_mode="prev_layer_output_dynamic",
        layer_recompute_io_prefix_blocks=[args.prefix_blocks],
        layer_recompute_planner="feedback",
        layer_recompute_planner_dry_run=args.planner_dry_run,
    )

    feedback_label = (
        "Feedback Planner (dry-run)" if args.planner_dry_run else "Feedback Planner"
    )
    feedback_results, feedback_stats = run_benchmark(
        model_name=args.model,
        num_requests=args.num_requests,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_steps=args.max_steps,
        max_num_seqs=args.max_num_seqs,
        use_long_prompts=args.use_long_prompts,
        use_very_long_prompts=args.use_very_long_prompts,
        use_extra_long_prompts=args.use_extra_long_prompts,
        long_prompt_ratio=args.long_prompt_ratio,
        extra_prompt_ratio=args.extra_long_prompt_ratio,
        kv_offload_config=feedback_config,
        dry_run=args.planner_dry_run,
        label=feedback_label,
        verbose=args.verbose,
        enable_profiling=args.profile,
        enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
    )

    # Compare outputs if we have a baseline
    result_sets: dict[str, list[RequestResult]] = {}
    if baseline_results is not None:
        result_sets["Baseline"] = baseline_results
    result_sets[feedback_label] = feedback_results

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
            _c(f"{'Metric':<25} {label_a:>20} {label_b:>20} {'Diff':>15}", _Ansi.BOLD)
        )
        print(_c("-" * 80, _Ansi.DIM))
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
            print(f"{metric:<25} {va:>20.2f} {vb:>20.2f} {diff_colored:>15}")

    if baseline_stats is not None:
        _print_perf_comparison(
            "Baseline", baseline_stats, feedback_label, feedback_stats
        )

    # Final summary
    print(_c("\n" + "=" * 60, _Ansi.DIM))
    print(_c("Test Summary", _Ansi.BOLD, _Ansi.CYAN))
    print(_c("=" * 60, _Ansi.DIM))

    success = True

    if feedback_stats.failed_requests == 0:
        print(
            _c(
                f"✓ {feedback_label}: All {feedback_stats.completed_requests}"
                " requests completed",
                _Ansi.GREEN,
                _Ansi.BOLD,
            )
        )
    else:
        print(
            _c(
                f"✗ {feedback_label}: {feedback_stats.failed_requests} requests failed",
                _Ansi.RED,
                _Ansi.BOLD,
            )
        )
        success = False

    if baseline_results is not None:
        print(
            _c(
                "⚠ Check output comparison above for mismatches"
                " (some divergence is expected when the planner actively skips layers)",
                _Ansi.YELLOW,
            )
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
