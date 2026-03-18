#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed


@dataclass
class WorkloadSpec:
    seq_lens: list[int]
    query_lens: list[int]


@dataclass
class BenchStats:
    latency_ms: float
    latency_std_ms: float
    flops: float
    tflops: float


@dataclass
class PositionSweepStats:
    name: str
    ctx_len: int
    replay_len: int
    start_block: int
    start_token: int
    total_seq_len: int
    baseline_query_len: int
    replay_query_len: int
    extra_scheduled_len: int
    baseline_ms: float
    replay_ms: float
    delta_ms: float
    delta_pct_vs_baseline: float
    baseline_tflops: float
    replay_tflops: float
    baseline_flops: float
    replay_flops: float
    extra_flops: float


def parse_int_list(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item]


def exact_causal_pairs(query_len: int, seq_len: int) -> int:
    if query_len > seq_len:
        raise ValueError(f"query_len ({query_len}) must be <= seq_len ({seq_len}).")
    return query_len * seq_len - (query_len * (query_len - 1)) // 2


def estimate_attention_flops(
    seq_lens: list[int],
    query_lens: list[int],
    num_query_heads: int,
    head_size: int,
) -> float:
    total_pairs = sum(
        exact_causal_pairs(query_len, seq_len)
        for query_len, seq_len in zip(query_lens, seq_lens, strict=True)
    )
    return float(4 * total_pairs * num_query_heads * head_size)


def estimate_attention_proj_flops(
    query_lens: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> float:
    total_tokens = sum(query_lens)
    hidden_size = num_query_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    q_proj = 2.0 * total_tokens * hidden_size * hidden_size
    k_proj = 2.0 * total_tokens * hidden_size * kv_hidden_size
    v_proj = 2.0 * total_tokens * hidden_size * kv_hidden_size
    out_proj = 2.0 * total_tokens * hidden_size * hidden_size
    return q_proj + k_proj + v_proj + out_proj


def estimate_activation_flops(num_elements: int, activation: str) -> float:
    return float(
        {
            "relu": 1,
            "gelu": 8,
            "silu": 4,
        }[activation]
        * num_elements
    )


def estimate_ffn_flops(
    query_lens: list[int],
    hidden_size: int,
    intermediate_size: int,
    activation: str,
) -> float:
    total_tokens = sum(query_lens)
    fc1 = 2.0 * total_tokens * hidden_size * intermediate_size
    act = estimate_activation_flops(total_tokens * intermediate_size, activation)
    fc2 = 2.0 * total_tokens * intermediate_size * hidden_size
    return fc1 + act + fc2


def sample_start_tokens(
    *,
    max_start: int,
    block_size: int,
    position_samples: int | None,
    position_step_blocks: int,
) -> list[int]:
    if max_start < 0:
        return []
    if position_samples is not None:
        if position_samples <= 0:
            raise ValueError("position_samples must be positive when provided.")
        max_start_block = max_start // block_size
        if max_start_block == 0 or position_samples == 1:
            return [0]
        num_samples = min(position_samples, max_start_block + 1)
        sampled_blocks = {
            round(i * max_start_block / (num_samples - 1)) for i in range(num_samples)
        }
        return sorted(block * block_size for block in sampled_blocks)

    step = max(1, position_step_blocks) * block_size
    start_tokens = list(range(0, max_start + 1, step))
    if not start_tokens or start_tokens[-1] != max_start:
        start_tokens.append(max_start)
    return start_tokens


class FlashAttnBenchRunner:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        softmax_scale: float | None,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.dtype = dtype
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.softmax_scale = softmax_scale or head_size**-0.5
        self.device = torch.device(device)
        set_random_seed(seed)

    def _build_inputs(
        self,
        spec: WorkloadSpec,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        total_q = sum(spec.query_lens)
        max_seq_len = max(spec.seq_lens)
        max_blocks_per_seq = math.ceil(max_seq_len / self.block_size)
        total_blocks = sum(
            math.ceil(seq_len / self.block_size) for seq_len in spec.seq_lens
        )

        query = torch.randn(
            total_q,
            self.num_query_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        key_cache = torch.randn(
            total_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        value_cache = torch.randn_like(key_cache)

        block_table = torch.full(
            (len(spec.seq_lens), max_blocks_per_seq),
            fill_value=0,
            dtype=torch.int32,
            device=self.device,
        )
        next_block = 0
        for row, seq_len in enumerate(spec.seq_lens):
            num_blocks = math.ceil(seq_len / self.block_size)
            block_table[row, :num_blocks] = torch.arange(
                next_block,
                next_block + num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            next_block += num_blocks

        cu_query_lens = torch.zeros(
            len(spec.query_lens) + 1,
            dtype=torch.int32,
            device=self.device,
        )
        cu_query_lens[1:] = torch.tensor(
            spec.query_lens, dtype=torch.int32, device=self.device
        ).cumsum(dim=0, dtype=torch.int32)
        seq_lens = torch.tensor(spec.seq_lens, dtype=torch.int32, device=self.device)
        output = torch.empty_like(query)
        return (
            query,
            key_cache,
            value_cache,
            block_table,
            cu_query_lens,
            seq_lens,
            output,
        )

    @torch.inference_mode()
    def benchmark(
        self,
        spec: WorkloadSpec,
        *,
        warmup: int,
        trials: int,
    ) -> BenchStats:
        (
            query,
            key_cache,
            value_cache,
            block_table,
            cu_query_lens,
            seq_lens,
            output,
        ) = self._build_inputs(spec)

        max_query_len = max(spec.query_lens)
        max_seq_len = max(spec.seq_lens)

        def run() -> None:
            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seq_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.softmax_scale,
                causal=True,
                block_table=block_table,
            )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        samples_ms: list[float] = []
        for _ in range(trials):
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end))

        latency_ms = statistics.fmean(samples_ms)
        latency_std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        flops = estimate_attention_flops(
            spec.seq_lens,
            spec.query_lens,
            self.num_query_heads,
            self.head_size,
        )
        tflops = flops / (latency_ms * 1e-3) / 1e12
        return BenchStats(
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            flops=flops,
            tflops=tflops,
        )


class FullAttentionBenchRunner:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        softmax_scale: float | None,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.dtype = dtype
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.softmax_scale = softmax_scale or head_size**-0.5
        self.device = torch.device(device)
        self.hidden_size = num_query_heads * head_size
        self.kv_hidden_size = num_kv_heads * head_size
        set_random_seed(seed)

    def _build_inputs(
        self,
        spec: WorkloadSpec,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        total_q = sum(spec.query_lens)
        max_seq_len = max(spec.seq_lens)
        max_blocks_per_seq = math.ceil(max_seq_len / self.block_size)
        total_blocks = sum(
            math.ceil(seq_len / self.block_size) for seq_len in spec.seq_lens
        )

        hidden_states = torch.randn(
            total_q,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        q_proj_weight = torch.randn(
            self.hidden_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        k_proj_weight = torch.randn(
            self.kv_hidden_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        v_proj_weight = torch.randn(
            self.kv_hidden_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        out_proj_weight = torch.randn(
            self.hidden_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        key_cache_base = torch.randn(
            total_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        value_cache_base = torch.randn_like(key_cache_base)

        block_table = torch.full(
            (len(spec.seq_lens), max_blocks_per_seq),
            fill_value=0,
            dtype=torch.int32,
            device=self.device,
        )
        next_block = 0
        slot_blocks: list[torch.Tensor] = []
        slot_offsets: list[torch.Tensor] = []
        for row, (seq_len, query_len) in enumerate(
            zip(spec.seq_lens, spec.query_lens, strict=True)
        ):
            num_blocks = math.ceil(seq_len / self.block_size)
            row_blocks = torch.arange(
                next_block,
                next_block + num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            block_table[row, :num_blocks] = row_blocks
            query_positions = torch.arange(
                seq_len - query_len,
                seq_len,
                dtype=torch.int64,
                device=self.device,
            )
            slot_blocks.append(row_blocks[(query_positions // self.block_size).long()])
            slot_offsets.append((query_positions % self.block_size).long())
            next_block += num_blocks

        cu_query_lens = torch.zeros(
            len(spec.query_lens) + 1,
            dtype=torch.int32,
            device=self.device,
        )
        cu_query_lens[1:] = torch.tensor(
            spec.query_lens, dtype=torch.int32, device=self.device
        ).cumsum(dim=0, dtype=torch.int32)
        seq_lens = torch.tensor(spec.seq_lens, dtype=torch.int32, device=self.device)
        slot_block_idx = (
            torch.cat(slot_blocks)
            if slot_blocks
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        slot_offsets_idx = (
            torch.cat(slot_offsets)
            if slot_offsets
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        attn_output = torch.empty(
            total_q,
            self.num_query_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        return (
            hidden_states,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            out_proj_weight,
            key_cache_base,
            value_cache_base,
            block_table,
            cu_query_lens,
            seq_lens,
            slot_block_idx,
            slot_offsets_idx,
            attn_output,
        )

    @torch.inference_mode()
    def benchmark(
        self,
        spec: WorkloadSpec,
        *,
        warmup: int,
        trials: int,
    ) -> BenchStats:
        (
            hidden_states,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            out_proj_weight,
            key_cache_base,
            value_cache_base,
            block_table,
            cu_query_lens,
            seq_lens,
            slot_block_idx,
            slot_offsets_idx,
            attn_output,
        ) = self._build_inputs(spec)

        max_query_len = max(spec.query_lens)
        max_seq_len = max(spec.seq_lens)

        def run() -> None:
            query = F.linear(hidden_states, q_proj_weight).view(
                -1, self.num_query_heads, self.head_size
            )
            key = F.linear(hidden_states, k_proj_weight).view(
                -1, self.num_kv_heads, self.head_size
            )
            value = F.linear(hidden_states, v_proj_weight).view(
                -1, self.num_kv_heads, self.head_size
            )
            key_cache = key_cache_base.clone()
            value_cache = value_cache_base.clone()
            key_cache[slot_block_idx, slot_offsets_idx] = key
            value_cache[slot_block_idx, slot_offsets_idx] = value
            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=attn_output,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seq_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.softmax_scale,
                causal=True,
                block_table=block_table,
            )
            F.linear(attn_output.view(-1, self.hidden_size), out_proj_weight)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        samples_ms: list[float] = []
        for _ in range(trials):
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end))

        latency_ms = statistics.fmean(samples_ms)
        latency_std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        flops = estimate_attention_flops(
            spec.seq_lens,
            spec.query_lens,
            self.num_query_heads,
            self.head_size,
        ) + estimate_attention_proj_flops(
            spec.query_lens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_size,
        )
        tflops = flops / (latency_ms * 1e-3) / 1e12
        return BenchStats(
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            flops=flops,
            tflops=tflops,
        )


class FFNBenchRunner:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        hidden_size: int,
        intermediate_size: int,
        activation: str,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.device = torch.device(device)
        set_random_seed(seed)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    @torch.inference_mode()
    def benchmark(
        self,
        spec: WorkloadSpec,
        *,
        warmup: int,
        trials: int,
    ) -> BenchStats:
        total_tokens = sum(spec.query_lens)
        hidden_states = torch.randn(
            total_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        fc1_weight = torch.randn(
            self.intermediate_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        fc2_weight = torch.randn(
            self.hidden_size,
            self.intermediate_size,
            dtype=self.dtype,
            device=self.device,
        )

        def run() -> None:
            inter = F.linear(hidden_states, fc1_weight)
            inter = self._activation(inter)
            F.linear(inter, fc2_weight)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        samples_ms: list[float] = []
        for _ in range(trials):
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end))

        latency_ms = statistics.fmean(samples_ms)
        latency_std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        flops = estimate_ffn_flops(
            spec.query_lens,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
        )
        tflops = flops / (latency_ms * 1e-3) / 1e12
        return BenchStats(
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            flops=flops,
            tflops=tflops,
        )


def make_position_sweep_specs(
    *,
    ctx_len: int,
    scheduled_len: int,
    replay_len: int,
    block_size: int,
    position_samples: int | None,
    position_step_blocks: int,
) -> list[tuple[int, int, int, int, int, WorkloadSpec, WorkloadSpec]]:
    if replay_len <= 0:
        raise ValueError("replay_len must be positive.")
    if replay_len > ctx_len:
        raise ValueError(f"replay_len={replay_len} must be <= ctx_len={ctx_len}.")
    if scheduled_len <= 0:
        raise ValueError("scheduled_len must be positive.")

    max_start = ctx_len - replay_len
    total_seq_len = ctx_len + scheduled_len
    start_tokens = sample_start_tokens(
        max_start=max_start,
        block_size=block_size,
        position_samples=position_samples,
        position_step_blocks=position_step_blocks,
    )

    specs: list[tuple[int, int, int, int, int, WorkloadSpec, WorkloadSpec]] = []
    for start_token in start_tokens:
        start_block = start_token // block_size
        suffix_len = ctx_len - (start_token + replay_len)
        baseline_query_len = suffix_len + scheduled_len
        replay_query_len = replay_len + suffix_len + scheduled_len
        baseline = WorkloadSpec(
            seq_lens=[total_seq_len],
            query_lens=[baseline_query_len],
        )
        replay_case = WorkloadSpec(
            seq_lens=[total_seq_len],
            query_lens=[replay_query_len],
        )
        specs.append(
            (
                start_block,
                start_token,
                total_seq_len,
                baseline_query_len,
                replay_query_len,
                baseline,
                replay_case,
            )
        )

    return specs


def run_position_sweep(
    runner,
    *,
    ctx_lens: list[int],
    replay_lengths: list[int],
    scheduled_len: int,
    block_size: int,
    position_samples: int | None,
    position_step_blocks: int,
    warmup: int,
    trials: int,
) -> list[PositionSweepStats]:
    rows: list[PositionSweepStats] = []
    for ctx_len in ctx_lens:
        for replay_len in replay_lengths:
            if replay_len > ctx_len:
                continue
            specs = make_position_sweep_specs(
                ctx_len=ctx_len,
                scheduled_len=scheduled_len,
                replay_len=replay_len,
                block_size=block_size,
                position_samples=position_samples,
                position_step_blocks=position_step_blocks,
            )
            for (
                start_block,
                start_token,
                total_seq_len,
                baseline_query_len,
                replay_query_len,
                baseline_spec,
                replay_spec,
            ) in specs:
                baseline = runner.benchmark(
                    baseline_spec,
                    warmup=warmup,
                    trials=trials,
                )
                replay = runner.benchmark(
                    replay_spec,
                    warmup=warmup,
                    trials=trials,
                )
                delta_ms = replay.latency_ms - baseline.latency_ms
                rows.append(
                    PositionSweepStats(
                        name=f"ctx={ctx_len} R={replay_len}@blk={start_block}",
                        ctx_len=ctx_len,
                        replay_len=replay_len,
                        start_block=start_block,
                        start_token=start_token,
                        total_seq_len=total_seq_len,
                        baseline_query_len=baseline_query_len,
                        replay_query_len=replay_query_len,
                        extra_scheduled_len=scheduled_len,
                        baseline_ms=baseline.latency_ms,
                        replay_ms=replay.latency_ms,
                        delta_ms=delta_ms,
                        delta_pct_vs_baseline=100.0 * delta_ms / baseline.latency_ms,
                        baseline_tflops=baseline.tflops,
                        replay_tflops=replay.tflops,
                        baseline_flops=baseline.flops,
                        replay_flops=replay.flops,
                        extra_flops=replay.flops - baseline.flops,
                    )
                )
    return rows


def print_table(title: str, rows: list[PositionSweepStats]) -> None:
    print(f"\n{title}")
    headers = [
        "name",
        "ctx",
        "r_len",
        "blk",
        "tok",
        "base_q",
        "replay_q",
        "base_ms",
        "replay_ms",
        "delta_ms",
        "delta%",
        "base_TF",
        "replay_TF",
    ]
    print(
        f"{headers[0]:<24} {headers[1]:>6} {headers[2]:>6} {headers[3]:>6} "
        f"{headers[4]:>6} {headers[5]:>8} {headers[6]:>8} {headers[7]:>10} "
        f"{headers[8]:>10} {headers[9]:>10} {headers[10]:>9} {headers[11]:>10} "
        f"{headers[12]:>10}"
    )
    print("-" * 154)
    for row in rows:
        print(
            f"{row.name:<24} {row.ctx_len:6d} {row.replay_len:6d} "
            f"{row.start_block:6d} {row.start_token:6d} "
            f"{row.baseline_query_len:8d} {row.replay_query_len:8d} "
            f"{row.baseline_ms:10.3f} {row.replay_ms:10.3f} "
            f"{row.delta_ms:10.3f} {row.delta_pct_vs_baseline:9.2f} "
            f"{row.baseline_tflops:10.3f} {row.replay_tflops:10.3f}"
        )


def print_summary(title: str, rows: list[PositionSweepStats]) -> None:
    if not rows:
        return
    print(f"\n{title} summary:")
    key_pairs = sorted({(row.ctx_len, row.replay_len) for row in rows})
    for ctx_len, replay_len in key_pairs:
        subset = [
            row
            for row in rows
            if row.ctx_len == ctx_len and row.replay_len == replay_len
        ]
        smallest = min(subset, key=lambda row: row.delta_ms)
        largest = max(subset, key=lambda row: row.delta_ms)
        print(
            f"  ctx_len={ctx_len}, replay_len={replay_len}: delta_ms in "
            f"[{smallest.delta_ms:.3f}, {largest.delta_ms:.3f}] from "
            f"blk={smallest.start_block} to blk={largest.start_block}"
        )


def save_results(path: Path, results: dict[str, list[PositionSweepStats]]) -> None:
    payload = {key: [asdict(stat) for stat in value] for key, value in results.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if (
        args.module in ("attention", "all")
        and not is_flash_attn_varlen_func_available()
    ):
        raise RuntimeError(
            "flash_attn_varlen_func is not available in this environment."
        )

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    ctx_lens = parse_int_list(args.ctx_lens) if args.ctx_lens else [args.ctx_len]
    replay_lengths = parse_int_list(args.replay_lengths)
    results: dict[str, list[PositionSweepStats]] = {}

    if args.module in ("attention", "all"):
        attention_scopes = (
            ["flash_core", "full_module"]
            if args.attention_scope == "all"
            else [args.attention_scope]
        )
        for attention_scope in attention_scopes:
            attn_runner_cls = (
                FullAttentionBenchRunner
                if attention_scope == "full_module"
                else FlashAttnBenchRunner
            )
            attn_runner = attn_runner_cls(
                dtype=dtype,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                softmax_scale=args.softmax_scale,
                seed=args.seed,
            )
            result_key = f"attention_{attention_scope}"
            results[result_key] = run_position_sweep(
                attn_runner,
                ctx_lens=ctx_lens,
                replay_lengths=replay_lengths,
                scheduled_len=args.scheduled_len,
                block_size=args.block_size,
                position_samples=args.position_samples,
                position_step_blocks=args.position_step_blocks,
                warmup=args.warmup,
                trials=args.trials,
            )
            attention_title = f"Attention ({attention_scope}) / Replay Position Sweep"
            print_table(attention_title, results[result_key])
            print_summary(f"Attention ({attention_scope})", results[result_key])

    if args.module in ("ffn", "all"):
        hidden_size = args.num_query_heads * args.head_size
        intermediate_size = (
            args.ffn_intermediate_size
            if args.ffn_intermediate_size is not None
            else args.ffn_multiplier * hidden_size
        )
        ffn_runner = FFNBenchRunner(
            dtype=dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=args.ffn_activation,
            seed=args.seed,
        )
        results["ffn"] = run_position_sweep(
            ffn_runner,
            ctx_lens=ctx_lens,
            replay_lengths=replay_lengths,
            scheduled_len=args.scheduled_len,
            block_size=args.block_size,
            position_samples=args.position_samples,
            position_step_blocks=args.position_step_blocks,
            warmup=args.warmup,
            trials=args.trials,
        )
        print_table("FFN / Replay Position Sweep", results["ffn"])
        print_summary("FFN", results["ffn"])

    if args.json_output:
        save_results(Path(args.json_output), results)
        print(f"\nSaved JSON results to {args.json_output}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Profile how replay position and replay length change the extra "
            "overhead when replay and the remaining suffix must form one "
            "contiguous query interval within a single request."
        )
    )
    parser.add_argument(
        "--module",
        choices=["attention", "ffn", "all"],
        default="attention",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=2048,
        help="Existing request length before any extra scheduled tokens.",
    )
    parser.add_argument(
        "--ctx-lens",
        type=str,
        default=None,
        help=(
            "Optional comma-separated ctx_len sweep, e.g. 256,512,1024,2048,4096,"
            "8192. Overrides --ctx-len when provided."
        ),
    )
    parser.add_argument(
        "--scheduled-len",
        type=int,
        default=16,
        help=(
            "Extra scheduled tokens appended after the existing request. "
            "The baseline query interval is suffix+scheduled, and the "
            "replay query interval is replay+suffix+scheduled."
        ),
    )
    parser.add_argument("--replay-lengths", type=str, default="16,32,64,128,256")
    parser.add_argument(
        "--position-samples",
        type=int,
        default=9,
        help=(
            "Uniformly interpolate this many replay start positions over the "
            "valid block range. Set to 0 with --position-step-blocks to fall "
            "back to dense step-based enumeration."
        ),
    )
    parser.add_argument(
        "--position-step-blocks",
        type=int,
        default=1,
        help=(
            "Fallback dense enumeration step in blocks, used only when "
            "--position-samples is 0."
        ),
    )
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["half", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attention-scope",
        choices=["flash_core", "full_module", "all"],
        default="flash_core",
        help=(
            "Measure only the FlashAttention kernel, q/k/v/out projections "
            "plus FlashAttention, or both."
        ),
    )
    parser.add_argument("--softmax-scale", type=float, default=None)
    parser.add_argument("--ffn-intermediate-size", type=int, default=None)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument(
        "--ffn-activation",
        choices=["relu", "gelu", "silu"],
        default="relu",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()
    if args.position_samples == 0:
        args.position_samples = None
    main(args)
