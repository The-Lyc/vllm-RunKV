# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch
from transformers import AutoTokenizer

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

logger = logging.getLogger(__name__)
_VERBOSE = os.environ.get("VLLM_RUNKV_E2E_VERBOSE", "0") == "1"


# ---------------------------------------------------------------------------
# Fixed seed used everywhere for full reproducibility.
# ---------------------------------------------------------------------------
_SEED = 1234


def _verbose_log(msg: str) -> None:
    if _VERBOSE:
        print(msg, flush=True)
    logger.info(msg)


def _set_all_seeds(seed: int = _SEED) -> None:
    """Pin every RNG source so runs are fully deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (no auto-tuner variance).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Make cuBLAS deterministic (requires workspace config).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a HF tokenizer for decoding token ids into text."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _decode_tokens(
    tokenizer: AutoTokenizer,
    token_ids: list[int],
) -> str:
    """Decode token ids into a human-readable string."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _has_local_weights(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    weight_patterns = [
        "model.safetensors",
        "*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ]
    return any(any(model_dir.glob(pattern)) for pattern in weight_patterns)


def _resolve_test_model() -> str | None:
    env_model = os.environ.get("VLLM_RUNKV_E2E_MODEL", "").strip()
    if env_model:
        env_path = Path(env_model).expanduser()
        if env_path.exists():
            if _has_local_weights(env_path):
                return str(env_path.resolve(strict=False))
            return None
        # Allow explicit HF repo ids when user intentionally passes one.
        return env_model

    local_candidates = [
        # Path("~/hf_models/Qwen3-0.6B"),
        # Path("~/hf_models/TinyLlama-1.1B-Chat-v1.0"),
        # Path("~/hf_models/Qwen2.5-1.5B-Instruct"),
        Path("~/hf_models/opt-1.3b"),
    ]
    for candidate in local_candidates:
        model_dir = candidate.expanduser()
        if _has_local_weights(model_dir):
            return str(model_dir.resolve(strict=False))
    return None


TEST_MODEL = _resolve_test_model()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        TEST_MODEL is None,
        reason=(
            "No local model with weights found. Set VLLM_RUNKV_E2E_MODEL to a valid"
            " local HF model path."
        ),
    ),
]


def _build_engine(*, model_name: str, kv_offload_config: RunKVOffloadConfig):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    engine_args = EngineArgs(
        model=model_name,
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=64,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_log_stats=False,
        seed=_SEED,
        kv_offload_config=kv_offload_config,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    executor_class = Executor.get_class(vllm_config)
    return LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=True,
        usage_context=UsageContext.ENGINE_CONTEXT,
        multiprocess_mode=False,
    )


def _shutdown_engine(engine) -> None:
    """Shut down the engine core (KV transfer, scheduler, etc.)

    NOTE: This only performs the engine-level shutdown.  Callers must
    ``del engine`` (and any other references into the engine, such as
    ``model_runner``) **before** calling ``cleanup_dist_env_and_memory``
    so that ``gc.collect()`` can actually free GPU tensors.
    """
    with contextlib.suppress(Exception):
        engine.engine_core.shutdown()


def _get_host_available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


def _wait_for_host_memory_recovery(
    *,
    target_available_bytes: int,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
    min_recovery_ratio: float = 0.95,
    absolute_slack_bytes: int = 4 * 1024**3,
) -> None:
    """Wait until host available memory recovers near pre-engine baseline."""
    if target_available_bytes <= 0:
        return

    # Use the looser one between ratio and absolute slack.
    threshold = max(
        0,
        min(
            int(target_available_bytes * min_recovery_ratio),
            target_available_bytes - absolute_slack_bytes,
        ),
    )

    deadline = time.monotonic() + timeout_s
    last_available = _get_host_available_memory_bytes()
    while time.monotonic() < deadline:
        last_available = _get_host_available_memory_bytes()
        if last_available >= threshold:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(poll_interval_s)

    raise RuntimeError(
        "Host memory did not recover after engine shutdown. "
        f"target_available={target_available_bytes}, "
        f"threshold={threshold}, "
        f"last_available={last_available}"
    )


def _get_model_runner(engine):
    executor = engine.model_executor
    driver_worker = executor.driver_worker
    worker = driver_worker.worker
    return worker.model_runner


def _run_requests_and_collect_tokens(
    *,
    engine,
    requests: list[tuple[str, str, int]],
    max_steps: int = 4096,
) -> dict[str, list[int]]:
    expected = {rid for rid, _, _ in requests}
    finished: dict[str, list[int]] = {}

    for request_id, prompt, max_tokens in requests:
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
            if out.request_id not in expected or not out.finished:
                continue
            if out.outputs:
                finished[out.request_id] = list(out.outputs[0].token_ids)
            else:
                finished[out.request_id] = []
        if len(finished) == len(expected):
            return finished

    raise TimeoutError(
        f"Timed out after {max_steps} steps, finished={len(finished)}/{len(expected)}"
    )


def _build_requests_single() -> list[tuple[str, str, int]]:
    long_prompt = (
        "Layer recompute correctness test. "
        "This prompt is intentionally long to force multiple KV blocks. "
    ) * 40
    return [("req-single", long_prompt, 24)]


def _build_requests_concurrent() -> list[tuple[str, str, int]]:
    return [
        (
            "req-a",
            (
                "Concurrent request A for layer recompute validation. "
                "Prefix and suffix should produce deterministic tokens. "
            )
            * 30,
            20,
        ),
        (
            "req-b",
            (
                "Concurrent request B with a different history length. "
                "This stresses request-specific logical block ownership. "
            )
            * 22,
            28,
        ),
        (
            "req-c",
            (
                "Concurrent request C includes mixed punctuation and numbers "
                "123 456 789 to vary tokenization patterns. "
            )
            * 18,
            16,
        ),
    ]


def _run_case(
    model_name: str,
    requests: list[tuple[str, str, int]],
    *,
    kv_offload_config: RunKVOffloadConfig,
    expect_runkv: bool,
    expect_layer_recompute: bool = False,
    tokenizer: AutoTokenizer | None = None,
) -> dict[str, list[int]]:
    pre_build_available = _get_host_available_memory_bytes()
    _verbose_log(
        "Starting case: "
        f"runkv={expect_runkv}, layer_recompute={expect_layer_recompute}, "
        f"num_requests={len(requests)}, "
        f"host_available_before={pre_build_available}"
    )

    # Pin all RNG seeds before every engine build for full reproducibility.
    _set_all_seeds()

    try:
        engine = _build_engine(
            model_name=model_name,
            kv_offload_config=kv_offload_config,
        )
    except RuntimeError as exc:
        if "Cannot find any model weights" in str(exc):
            pytest.skip(f"Model weights unavailable for {model_name}: {exc}")
        raise
    try:
        model_runner = _get_model_runner(engine)
        assert model_runner.use_runkv == expect_runkv
        assert model_runner.kv_offload_config.enabled == expect_runkv
        assert model_runner.layer_recompute_enabled == expect_layer_recompute
        result = _run_requests_and_collect_tokens(engine=engine, requests=requests)

        # Build a readable summary: token ids + decoded text.
        token_summary_parts = []
        for rid, toks in sorted(result.items()):
            decoded = _decode_tokens(tokenizer, toks) if tokenizer else "<no tokenizer>"
            token_summary_parts.append(
                f"  {rid}: len={len(toks)}, ids={toks}, text={decoded!r}"
            )
        token_summary = "\n".join(token_summary_parts)
        _verbose_log(
            f"Finished case: "
            f"runkv={expect_runkv}, layer_recompute={expect_layer_recompute}\n"
            f"{token_summary}"
        )
        return result
    finally:
        # --- Proper shutdown sequence ---
        # 1. Engine-level shutdown (KV transfer, scheduler, profiler)
        _shutdown_engine(engine)

        # 2. Delete ALL references into the engine so GC can free GPU tensors
        del model_runner
        del engine

        # 3. Destroy NCCL groups + gc.collect + empty_cache
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                cleanup_dist_env_and_memory,
            )

            cleanup_dist_env_and_memory(shutdown_ray=False)

        # 4. Belt-and-suspenders: another GC + cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Wait for host memory to recover
        _wait_for_host_memory_recovery(target_available_bytes=pre_build_available)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _dump_token_debug(
    *,
    request_id: str,
    baseline_tokens: list[int],
    compare_tokens: list[int],
    tokenizer: AutoTokenizer | None = None,
) -> Path:
    debug_root = Path(
        os.environ.get(
            "VLLM_RUNKV_E2E_DEBUG_DIR", "/tmp/vllm_runkv_layer_recompute_e2e"
        )
    )
    debug_root.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = debug_root / f"{request_id}_tokens_{ts}.json"
    payload: dict = {
        "request_id": request_id,
        "baseline_len": len(baseline_tokens),
        "compare_len": len(compare_tokens),
        "baseline_tokens": baseline_tokens,
        "compare_tokens": compare_tokens,
    }
    if tokenizer is not None:
        payload["baseline_text"] = _decode_tokens(tokenizer, baseline_tokens)
        payload["compare_text"] = _decode_tokens(tokenizer, compare_tokens)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def _format_token_mismatch(
    *,
    request_id: str,
    baseline_tokens: list[int],
    compare_tokens: list[int],
    tokenizer: AutoTokenizer | None = None,
) -> str:
    prefix_len = _common_prefix_len(baseline_tokens, compare_tokens)
    overlap = min(len(baseline_tokens), len(compare_tokens))
    diff_indices = [
        i for i in range(overlap) if baseline_tokens[i] != compare_tokens[i]
    ]
    mismatch_count = len(diff_indices) + abs(len(baseline_tokens) - len(compare_tokens))

    first_diff_idx: int | None = None
    baseline_first: int | None = None
    compare_first: int | None = None
    if prefix_len < overlap:
        first_diff_idx = prefix_len
        baseline_first = baseline_tokens[prefix_len]
        compare_first = compare_tokens[prefix_len]
    elif len(baseline_tokens) != len(compare_tokens):
        first_diff_idx = overlap
        baseline_first = (
            baseline_tokens[overlap] if overlap < len(baseline_tokens) else None
        )
        compare_first = (
            compare_tokens[overlap] if overlap < len(compare_tokens) else None
        )

    preview_indices = diff_indices[:32]
    preview_pairs = [
        {
            "idx": i,
            "baseline": baseline_tokens[i],
            "compare": compare_tokens[i],
        }
        for i in preview_indices
    ]
    dump_path = _dump_token_debug(
        request_id=request_id,
        baseline_tokens=baseline_tokens,
        compare_tokens=compare_tokens,
        tokenizer=tokenizer,
    )

    # Decode full text for both sides so the diff is human-readable.
    baseline_text = (
        _decode_tokens(tokenizer, baseline_tokens) if tokenizer else "<no tokenizer>"
    )
    compare_text = (
        _decode_tokens(tokenizer, compare_tokens) if tokenizer else "<no tokenizer>"
    )

    return (
        f"Token mismatch for {request_id}\n"
        f"baseline_len={len(baseline_tokens)}, compare_len={len(compare_tokens)}\n"
        f"common_prefix_len={prefix_len}, mismatch_count={mismatch_count}\n"
        f"first_diff_idx={first_diff_idx}, "
        f"baseline_first={baseline_first}, compare_first={compare_first}\n"
        f"diff_preview(first {len(preview_pairs)}): {preview_pairs}\n"
        f"baseline_text={baseline_text!r}\n"
        f"compare_text={compare_text!r}\n"
        f"full_tokens_dump={dump_path}"
    )


def _assert_tokens_match(
    baseline: dict[str, list[int]],
    recompute: dict[str, list[int]],
    tokenizer: AutoTokenizer | None = None,
) -> None:
    baseline_keys = set(baseline.keys())
    recompute_keys = set(recompute.keys())
    assert baseline_keys == recompute_keys, (
        "Request id set mismatch: "
        f"baseline_only={sorted(baseline_keys - recompute_keys)}, "
        f"compare_only={sorted(recompute_keys - baseline_keys)}"
    )
    for request_id in baseline:
        baseline_tokens = baseline[request_id]
        compare_tokens = recompute[request_id]
        assert compare_tokens == baseline_tokens, _format_token_mismatch(
            request_id=request_id,
            baseline_tokens=baseline_tokens,
            compare_tokens=compare_tokens,
            tokenizer=tokenizer,
        )


def _vanilla_config() -> RunKVOffloadConfig:
    return RunKVOffloadConfig(enabled=False)


def _runkv_no_recompute_config() -> RunKVOffloadConfig:
    return RunKVOffloadConfig(
        enabled=True,
        num_device_buffers=3,
        gpu_memory_fraction=0.5,
        enable_async_prefetch=True,
        enable_async_offload=True,
        enable_layer_recompute=False,
    )


def _runkv_recompute_config() -> RunKVOffloadConfig:
    return RunKVOffloadConfig(
        enabled=True,
        num_device_buffers=3,
        gpu_memory_fraction=0.5,
        enable_async_prefetch=True,
        enable_async_offload=True,
        enable_layer_recompute=True,
        layer_recompute_io_prefix_blocks=[4],
    )


def test_layer_recompute_single_request_matches_vanilla_and_runkv_baseline():
    if TEST_MODEL is None:
        pytest.skip(
            "No local model weights found. Set VLLM_RUNKV_E2E_MODEL to a local "
            "model directory (or HF repo id if intentionally testing downloads)."
        )
    model = TEST_MODEL
    assert model is not None
    tokenizer = _load_tokenizer(model)
    requests = _build_requests_single()
    vanilla = _run_case(
        model,
        requests,
        kv_offload_config=_vanilla_config(),
        expect_runkv=False,
        expect_layer_recompute=False,
        tokenizer=tokenizer,
    )
    runkv_baseline = _run_case(
        model,
        requests,
        kv_offload_config=_runkv_no_recompute_config(),
        expect_runkv=True,
        expect_layer_recompute=False,
        tokenizer=tokenizer,
    )
    recompute = _run_case(
        model,
        requests,
        kv_offload_config=_runkv_recompute_config(),
        expect_runkv=True,
        expect_layer_recompute=True,
        tokenizer=tokenizer,
    )
    _assert_tokens_match(vanilla, runkv_baseline, tokenizer=tokenizer)
    _assert_tokens_match(vanilla, recompute, tokenizer=tokenizer)


def test_layer_recompute_concurrent_requests_match_vanilla_and_runkv_baseline():
    if TEST_MODEL is None:
        pytest.skip(
            "No local model weights found. Set VLLM_RUNKV_E2E_MODEL to a local "
            "model directory (or HF repo id if intentionally testing downloads)."
        )
    model = TEST_MODEL
    assert model is not None
    tokenizer = _load_tokenizer(model)
    requests = _build_requests_concurrent()
    vanilla = _run_case(
        model,
        requests,
        kv_offload_config=_vanilla_config(),
        expect_runkv=False,
        expect_layer_recompute=False,
        tokenizer=tokenizer,
    )
    runkv_baseline = _run_case(
        model,
        requests,
        kv_offload_config=_runkv_no_recompute_config(),
        expect_runkv=True,
        expect_layer_recompute=False,
        tokenizer=tokenizer,
    )
    recompute = _run_case(
        model,
        requests,
        kv_offload_config=_runkv_recompute_config(),
        expect_runkv=True,
        expect_layer_recompute=True,
        tokenizer=tokenizer,
    )
    _assert_tokens_match(vanilla, runkv_baseline, tokenizer=tokenizer)
    _assert_tokens_match(vanilla, recompute, tokenizer=tokenizer)
