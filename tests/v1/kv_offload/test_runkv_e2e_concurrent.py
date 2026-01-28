#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
High-Concurrency E2E Test for RunKV KV Cache Offloading.

This script tests RunKV under high concurrency scenarios:
- Multiple concurrent requests processed simultaneously
- Varying prompt lengths and generation lengths
- Stress testing the offloading/prefetching pipeline
- Comparing outputs with baseline (RunKV disabled)

Usage:
    python test_runkv_e2e_concurrent.py [--model MODEL] [--num-requests N]
    [--concurrency C]

Requirements:
    - CUDA available
    - vLLM installed with RunKV support

Example:
    python test_runkv_e2e_concurrent.py --model "~/hf_models/Qwen3-0.6B" \
        --num-requests 100 --concurrency 32
"""

import argparse
import contextlib
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Sample prompts with varying lengths for realistic testing
SAMPLE_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about the ocean:",
    "What is the meaning of life?",
    "Once upon a time in a land far away,",
    "The best programming language is",
    "How do you make a perfect cup of coffee?",
    "Describe the process of photosynthesis:",
    "What are the benefits of regular exercise?",
    "In the year 2050, technology will",
    "The most important invention in history is",
    "A recipe for happiness includes",
    "The future of artificial intelligence is",
    "Climate change affects the planet by",
    "The secret to success is",
    "My favorite book is about",
    "Space exploration is important because",
    "The internet has changed society by",
    "A healthy diet should include",
]

# Medium-length prompts (~50-100 tokens)
MEDIUM_PROMPTS = [
    (
        "Please provide a detailed explanation of how neural networks learn "
        "through backpropagation, including the mathematical foundations and "
        "practical considerations:"
    ),
    (
        "Write a comprehensive essay about the history of computing, from "
        "early mechanical calculators to modern quantum computers, discussing "
        "key innovations and pioneers:"
    ),
    (
        "Explain the principles of thermodynamics and their applications in "
        "engineering, including real-world examples from power plants, "
        "refrigeration, and heat engines:"
    ),
    (
        "Describe the evolution of programming languages from assembly to "
        "modern high-level languages, including their design philosophies and "
        "use cases:"
    ),
    (
        "Discuss the impact of social media on modern communication, "
        "relationships, and mental health, with both positive and negative "
        "aspects:"
    ),
]

# Very long prompts (~512 tokens) for stress testing
VERY_LONG_PROMPTS = [
    # Prompt 1: Technical documentation style (~500 tokens)
    """You are an expert software architect reviewing a complex distributed system.
The system consists of multiple microservices communicating via message queues and
REST APIs. The main components include:

1. User Authentication Service: Handles user login, registration, and token
management. Uses JWT tokens with RSA-256 signing. Stores user credentials in
PostgreSQL with bcrypt hashing.

2. Product Catalog Service: Manages product information, categories, and
inventory. Uses Elasticsearch for full-text search and Redis for caching
frequently accessed items.

3. Order Processing Service: Handles order creation, payment processing, and
fulfillment. Integrates with external payment gateways (Stripe, PayPal) and
shipping providers (FedEx, UPS).

4. Notification Service: Sends emails, SMS, and push notifications. Uses
RabbitMQ for message queuing and supports templated messages with
internationalization.

5. Analytics Service: Collects and processes user behavior data. Uses Apache
Kafka for event streaming and ClickHouse for analytical queries.

6. API Gateway: Routes requests to appropriate services, handles rate limiting,
and manages API versioning. Implements circuit breaker patterns for fault
tolerance.

The system currently handles 10,000 requests per second during peak hours and
stores 50TB of data across all services. Recent performance issues have been
reported:
- Order processing latency increased from 200ms to 800ms
- Search queries timing out during high traffic
- Memory usage spiking on the notification service

Please analyze the architecture and provide detailed recommendations for:""",
    # Prompt 2: Scientific research context (~500 tokens)
    """Abstract: Recent advances in large language models (LLMs) have demonstrated
remarkable capabilities in natural language understanding and generation.
However, the computational requirements for training and inference remain a
significant challenge. This paper presents a comprehensive survey of
optimization techniques for LLM deployment.

Introduction:
The transformer architecture, introduced by Vaswani et al. (2017), has become
the foundation for modern language models. Models like GPT-4, Claude, and LLaMA
have shown impressive performance across diverse tasks including text
generation, code completion, and reasoning. However, these models contain
billions of parameters, requiring substantial computational resources.

Key challenges in LLM deployment include:
1. Memory bandwidth limitations during inference
2. KV cache management for long sequences
3. Batch processing efficiency under varying sequence lengths
4. Quantization effects on model quality
5. Distributed inference across multiple devices

Related Work:
Previous approaches to efficient LLM inference include:
- PagedAttention (vLLM): Manages KV cache as virtual memory pages
- FlashAttention: Optimizes attention computation through tiling
- Speculative decoding: Uses smaller models to predict multiple tokens
- Continuous batching: Dynamically adjusts batch composition
- Model parallelism: Distributes model across multiple GPUs

Our contribution focuses on KV cache offloading to CPU memory, enabling larger
batch sizes and longer sequences without GPU memory constraints.

Methodology:
We propose a novel approach called RunKV that addresses memory limitations
through intelligent cache management. The key innovations include:

Based on this context, please provide a detailed technical analysis of:""",
    # Prompt 3: Code review context (~500 tokens)
    """Code Review Request: High-Performance Key-Value Cache Implementation

File: kv_cache_manager.py

```python
import torch
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np

@dataclass
class CacheBlock:
    block_id: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    ref_count: int = 0
    last_access: float = 0.0
    is_pinned: bool = False

class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._lock = threading.RLock()
        self._blocks: Dict[int, CacheBlock] = {}
        self._free_blocks: List[int] = list(range(num_blocks))
        self._lru_order = OrderedDict()
        self._initialize_blocks()
    
    def _initialize_blocks(self):
        for block_id in range(self.num_blocks):
            key_cache = torch.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            value_cache = torch.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            self._blocks[block_id] = CacheBlock(
                block_id=block_id,
                key_cache=key_cache,
                value_cache=value_cache,
            )
    
    def allocate_block(self) -> Optional[int]:
        with self._lock:
            if self._free_blocks:
                block_id = self._free_blocks.pop()
                self._blocks[block_id].ref_count = 1
                return block_id
            return self._evict_and_allocate()
```

Please review this code and provide detailed feedback on:""",
    # Prompt 4: Long conversation context (~500 tokens)
    """Conversation History:

User: I'm building a real-time chat application and need advice on the architecture.

Assistant: I'd be happy to help! Real-time chat applications have several
architectural considerations. What scale are you targeting - hundreds,
thousands, or millions of concurrent users?

User: We're expecting around 100,000 concurrent users at peak, with about
1 million registered users total.

Assistant: That's a solid medium-scale application. For 100K concurrent users,
you'll want to consider: WebSocket connections for real-time messaging, a
message broker like Redis Pub/Sub or Kafka for distributing messages, and a
database strategy that separates hot data from cold storage.

User: We're currently using PostgreSQL for everything. Is that going to be a problem?

Assistant: PostgreSQL is excellent for user data and message history, but for
real-time message delivery, you'll want to add a caching layer. Redis is
commonly used here. For the message flow: User A sends message → Your server →
Redis Pub/Sub → Server instances → User B's WebSocket.

User: What about message persistence? We need to store all messages for
compliance reasons.

Assistant: For compliance, you'll want a robust storage strategy. Consider:
1) Write messages to Kafka first (durable log),
2) Consume from Kafka to PostgreSQL for permanent storage,
3) Use Redis only for recent messages (last 24 hours).
This gives you durability, real-time access, and audit trails.

User: We also need end-to-end encryption. How does that affect the architecture?

Assistant: E2E encryption changes things significantly. The server cannot read
message content, only route encrypted payloads. You'll need: client-side key
management, key exchange protocols (like Signal Protocol), and encrypted
payload storage. The server stores ciphertext only.

User: One more thing - we need to support file sharing up to 100MB.

Assistant: For large files with E2E encryption, I recommend:
1) Encrypt file client-side,
2) Upload encrypted blob to object storage (S3/GCS),
3) Share only the encrypted download link and decryption key via chat.
Never pass large files through WebSocket.

Based on this entire conversation, please provide a comprehensive architecture
document that includes:""",
    # Prompt 5: Multi-document analysis (~500 tokens)
    """Document Analysis Request

You are analyzing multiple related documents for a legal case. Please review
the following excerpts and provide a comprehensive analysis.

Document 1 - Contract Agreement (Excerpt):
"Section 4.2 - Performance Standards: The Service Provider agrees to maintain
system uptime of 99.9% measured on a monthly basis. Scheduled maintenance
windows (not to exceed 4 hours per month) shall be excluded from uptime
calculations. Any downtime exceeding the guaranteed uptime shall result in
service credits as specified in Schedule B.

Section 4.3 - Data Security: The Service Provider shall implement
industry-standard security measures including but not limited to: AES-256
encryption for data at rest, TLS 1.3 for data in transit, multi-factor
authentication for administrative access, and annual third-party security
audits."

Document 2 - Incident Report (Excerpt):
"On March 15, 2025, at approximately 14:32 UTC, the primary database cluster
experienced a cascading failure due to a misconfigured replication setting.
The incident resulted in complete service unavailability lasting 6 hours and
47 minutes. Root cause analysis revealed that a configuration change deployed
on March 14 introduced a race condition in the failover logic.

Impacted customers: 2,847 enterprise accounts
Data loss: None confirmed, investigation ongoing
Financial impact: Estimated $2.3M in SLA credits"

Document 3 - Email Correspondence (Excerpt):
"From: ops-team@serviceprovider.com
To: engineering-leads@serviceprovider.com
Date: March 14, 2025, 18:45 UTC
Subject: Re: Urgent - Deployment approval needed

I understand the pressure to ship this before quarter-end, but I have concerns
about the testing coverage. We only ran integration tests on the staging
environment for 2 hours. Our standard procedure requires 24-hour soak testing
for database configuration changes. Given the timeline, I'll approve with the
caveat that we monitor closely tomorrow."

Document 4 - Customer Communication:
"Dear Valued Customer, We sincerely apologize for the service disruption
experienced on March 15. We take our reliability commitments seriously and are
implementing additional safeguards including enhanced deployment procedures and
expanded monitoring coverage."

Based on these documents, please analyze:""",
]


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
    max_num_seqs: int = 256,
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
        max_num_seqs=max_num_seqs,
        kv_offload_config=kv_offload_config,
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
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory(shutdown_ray=False)


def generate_test_requests(
    num_requests: int,
    min_tokens: int,
    max_tokens: int,
    use_long_prompts: bool = False,
    use_very_long_prompts: bool = False,
    long_prompt_ratio: float = 0.3,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    """
    Generate test requests with varying prompts and token lengths.

    Args:
        num_requests: Total number of requests to generate
        min_tokens: Minimum tokens to generate per request
        max_tokens: Maximum tokens to generate per request
        use_long_prompts: Include medium-length prompts (~50-100 tokens)
        use_very_long_prompts: Include very long prompts (~512 tokens)
        long_prompt_ratio: Fraction of requests using longer prompts
        seed: Random seed for reproducibility

    Returns:
        List of (request_id, prompt, max_tokens) tuples
    """
    random.seed(seed)
    requests = []

    # Build prompt pools
    short_prompts = SAMPLE_PROMPTS
    long_prompts = []
    if use_long_prompts:
        long_prompts.extend(MEDIUM_PROMPTS)
    if use_very_long_prompts:
        long_prompts.extend(VERY_LONG_PROMPTS)

    for i in range(num_requests):
        request_id = f"concurrent-test-{i:06d}"

        # Decide whether to use long or short prompt
        if long_prompts and random.random() < long_prompt_ratio:
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
) -> tuple[list[RequestResult], float]:
    """
    Run multiple requests concurrently through the engine.

    Returns:
        (list of RequestResult, total_time_seconds)
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    results: dict[str, RequestResult] = {}
    pending_requests: set[str] = set()
    start_times: dict[str, float] = {}
    request_params: dict[
        str, tuple[str, int]
    ] = {}  # request_id -> (prompt, max_tokens)

    # Add all requests to the engine
    overall_start = time.time()

    for request_id, prompt, max_tokens in requests:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        engine.add_request(request_id=request_id, prompt=prompt, params=params)
        pending_requests.add(request_id)
        start_times[request_id] = time.time()
        request_params[request_id] = (prompt, max_tokens)

    if verbose:
        print(f"Added {len(requests)} requests to engine")

    # Process until all requests complete
    step = 0
    while pending_requests and step < max_steps:
        step_outputs = engine.step()

        for out in step_outputs:
            req_id = getattr(out, "request_id", None)
            if req_id is None or req_id not in pending_requests:
                continue

            if out.finished:
                end_time = time.time()
                latency_ms = (end_time - start_times[req_id]) * 1000
                prompt, max_tokens = request_params[req_id]

                text = out.outputs[0].text if out.outputs else ""
                num_tokens = len(out.outputs[0].token_ids) if out.outputs else 0

                results[req_id] = RequestResult(
                    request_id=req_id,
                    prompt=prompt,
                    output_text=text,
                    num_tokens=num_tokens,
                    latency_ms=latency_ms,
                    finished=True,
                )
                pending_requests.remove(req_id)

                if verbose and len(results) % 10 == 0:
                    print(f"Completed {len(results)}/{len(requests)} requests")

        step += 1

    overall_time = time.time() - overall_start

    # Mark timed-out requests
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
    print(f"\n{prefix}Benchmark Results:")
    print("-" * 50)
    print(f"Total requests:      {stats.total_requests}")
    print(f"Completed requests:  {stats.completed_requests}")
    print(f"Failed requests:     {stats.failed_requests}")
    print(f"Total tokens:        {stats.total_tokens}")
    print(f"Total time:          {stats.total_time_s:.2f}s")
    print(f"Avg latency:         {stats.avg_latency_ms:.2f}ms")
    print(f"P50 latency:         {stats.p50_latency_ms:.2f}ms")
    print(f"P90 latency:         {stats.p90_latency_ms:.2f}ms")
    print(f"P99 latency:         {stats.p99_latency_ms:.2f}ms")
    print(f"Throughput (req/s):  {stats.throughput_req_per_s:.2f}")
    print(f"Throughput (tok/s):  {stats.throughput_tok_per_s:.2f}")
    print("-" * 50)


def compare_outputs(
    runkv_results: list[RequestResult],
    baseline_results: list[RequestResult],
) -> tuple[int, int, list[str]]:
    """
    Compare RunKV outputs with baseline outputs.

    Returns:
        (matching_count, mismatched_count, list of mismatch descriptions)
    """
    baseline_map = {r.request_id: r for r in baseline_results}

    matching = 0
    mismatched = 0
    mismatches = []

    for runkv_result in runkv_results:
        baseline_result = baseline_map.get(runkv_result.request_id)
        if baseline_result is None:
            continue

        # Skip comparison if either failed
        if not runkv_result.finished or not baseline_result.finished:
            continue

        if runkv_result.output_text == baseline_result.output_text:
            matching += 1
        else:
            mismatched += 1
            mismatches.append(
                f"Request {runkv_result.request_id}:\n"
                f"  Prompt: {runkv_result.prompt[:50]}...\n"
                f"  RunKV:    {runkv_result.output_text[:100]}...\n"
                f"  Baseline: {baseline_result.output_text[:100]}..."
            )

    return matching, mismatched, mismatches


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
    long_prompt_ratio: float,
    kv_offload_config,
    label: str = "",
    verbose: bool = False,
) -> tuple[list[RequestResult], BenchmarkStats]:
    """Run a full benchmark with the given configuration."""
    print(f"\n{'=' * 60}")
    print(f"Running benchmark: {label or 'unnamed'}")
    print(f"{'=' * 60}")

    engine = _build_engine(
        model_name=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_offload_config=kv_offload_config,
        max_num_seqs=max_num_seqs,
    )

    try:
        # Verify RunKV is active if enabled
        if kv_offload_config.enabled:
            _assert_runkv_active(engine)
            print("RunKV verification: PASSED")

        # Generate test requests
        requests = generate_test_requests(
            num_requests=num_requests,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            use_long_prompts=use_long_prompts,
            use_very_long_prompts=use_very_long_prompts,
            long_prompt_ratio=long_prompt_ratio,
        )
        print(f"Generated {len(requests)} test requests")

        # Run benchmark
        print("Starting concurrent request processing...")
        results, total_time = run_concurrent_requests(
            engine=engine,
            requests=requests,
            max_steps=max_steps,
            verbose=verbose,
        )

        # Compute and print stats
        stats = compute_stats(results, total_time)
        print_stats(stats, label)

        return results, stats

    finally:
        _shutdown_engine(engine)


def main():
    parser = argparse.ArgumentParser(description="RunKV High-Concurrency E2E Test")
    parser.add_argument(
        "--model",
        type=str,
        default="~/hf_models/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name (default: ~/hf_models/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Total number of requests to process (default: 100)",
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
        default=256,
        help="Maximum tokens to generate per request (default: 256)",
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
        default=100.0,
        help="CPU memory limit for RunKV in GB (default: 100.0)",
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.7,
        help="CPU memory fraction for RunKV (default: 0.7)",
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
        default=10000,
        help="Maximum engine steps to wait for completion (default: 10000)",
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
        default=0.4,
        help="GPU memory utilization for vLLM (default: 0.4)",
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
        "--use-long-prompts",
        action="store_true",
        help="Include medium-length prompts (~50-100 tokens)",
    )
    parser.add_argument(
        "--use-very-long-prompts",
        action="store_true",
        help="Include very long prompts (~512 tokens) for stress testing",
    )
    parser.add_argument(
        "--long-prompt-ratio",
        type=float,
        default=0.3,
        help="Fraction of requests using longer prompts (default: 0.3)",
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

    args = parser.parse_args()

    # Expand and validate model path
    args.model = os.path.expandvars(os.path.expanduser(args.model))
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
    print("RunKV High-Concurrency E2E Test")
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

    # Stress test mode increases parameters
    if args.stress_test:
        print("\n*** STRESS TEST MODE ***")
        args.num_requests = max(args.num_requests, 500)
        args.use_long_prompts = True
        args.use_very_long_prompts = True
        args.max_tokens = max(args.max_tokens, 512)
        args.long_prompt_ratio = 0.5
        print(
            f"Adjusted: num_requests={args.num_requests}, max_tokens={args.max_tokens}"
        )
        print(
            "          use_very_long_prompts=True, "
            f"long_prompt_ratio={args.long_prompt_ratio}"
        )

    # Run baseline if requested
    baseline_results: list[RequestResult] | None = None
    baseline_stats: BenchmarkStats | None = None
    mismatched = 0
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
            long_prompt_ratio=args.long_prompt_ratio,
            kv_offload_config=baseline_config,
            label="Baseline (RunKV disabled)",
            verbose=args.verbose,
        )

    # Run RunKV benchmark
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
        num_requests=args.num_requests,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_steps=args.max_steps,
        max_num_seqs=args.max_num_seqs,
        use_long_prompts=args.use_long_prompts,
        use_very_long_prompts=args.use_very_long_prompts,
        long_prompt_ratio=args.long_prompt_ratio,
        kv_offload_config=runkv_config,
        label="RunKV enabled",
        verbose=args.verbose,
    )

    # Compare results if baseline was run
    if baseline_results is not None:
        assert baseline_stats is not None
        print("\n" + "=" * 60)
        print("Output Comparison: RunKV vs Baseline")
        print("=" * 60)

        matching, mismatched, mismatch_details = compare_outputs(
            runkv_results, baseline_results
        )

        print(f"Matching outputs:   {matching}")
        print(f"Mismatched outputs: {mismatched}")

        if mismatched > 0:
            print("\nFirst 5 mismatches:")
            for detail in mismatch_details[:5]:
                print(detail)
                print()

        # Performance comparison
        print("\n" + "=" * 60)
        print("Performance Comparison")
        print("=" * 60)
        print(f"{'Metric':<25} {'Baseline':>15} {'RunKV':>15} {'Diff':>15}")
        print("-" * 70)

        def fmt_diff(
            baseline: float, runkv: float, higher_is_better: bool = True
        ) -> str:
            if baseline == 0:
                return "N/A"
            diff_pct = ((runkv - baseline) / baseline) * 100
            if not higher_is_better:
                diff_pct = -diff_pct
            sign = "+" if diff_pct >= 0 else ""
            return f"{sign}{diff_pct:.1f}%"

        baseline_req_s = baseline_stats.throughput_req_per_s
        runkv_req_s = runkv_stats.throughput_req_per_s
        print(
            f"{'Throughput (req/s)':<25} "
            f"{baseline_req_s:>15.2f} "
            f"{runkv_req_s:>15.2f} "
            f"{fmt_diff(baseline_req_s, runkv_req_s):>15}"
        )
        baseline_tok_s = baseline_stats.throughput_tok_per_s
        runkv_tok_s = runkv_stats.throughput_tok_per_s
        print(
            f"{'Throughput (tok/s)':<25} "
            f"{baseline_tok_s:>15.2f} "
            f"{runkv_tok_s:>15.2f} "
            f"{fmt_diff(baseline_tok_s, runkv_tok_s):>15}"
        )
        baseline_avg_ms = baseline_stats.avg_latency_ms
        runkv_avg_ms = runkv_stats.avg_latency_ms
        print(
            f"{'Avg latency (ms)':<25} "
            f"{baseline_avg_ms:>15.2f} "
            f"{runkv_avg_ms:>15.2f} "
            f"{fmt_diff(baseline_avg_ms, runkv_avg_ms, higher_is_better=False):>15}"
        )
        baseline_p99_ms = baseline_stats.p99_latency_ms
        runkv_p99_ms = runkv_stats.p99_latency_ms
        print(
            f"{'P99 latency (ms)':<25} "
            f"{baseline_p99_ms:>15.2f} "
            f"{runkv_p99_ms:>15.2f} "
            f"{fmt_diff(baseline_p99_ms, runkv_p99_ms, higher_is_better=False):>15}"
        )

    # Final summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    success = runkv_stats.failed_requests == 0
    if success:
        print(f"✓ All {runkv_stats.completed_requests} requests completed successfully")
    else:
        print(f"✗ {runkv_stats.failed_requests} requests failed")

    if baseline_results is not None and mismatched > 0:
        print(
            f"⚠ {mismatched} output mismatches detected (may be expected with sampling)"
        )
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
