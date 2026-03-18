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

COMMON_FP16_BF16_PEAK_TFLOPS = {
    "A100": 312.0,
    "A800": 312.0,
    "H100 SXM": 989.0,
    "H100 PCIe": 756.0,
    "H200": 989.0,
    "H800": 989.0,
}


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    seq_lens: list[int]
    query_lens: list[int]
    batch_label: str
    avg_seq_len: float
    variance: float


@dataclass
class RunStats:
    name: str
    batch_label: str
    batch_size: int
    avg_seq_len: float
    variance: float
    total_query_tokens: int
    total_kv_tokens: int
    flops: float
    latency_ms: float
    latency_std_ms: float
    actual_tflops: float
    actual_mfu: float | None
    predicted_ms: float | None = None
    predicted_tflops: float | None = None
    predicted_mfu: float | None = None
    error_pct: float | None = None


def parse_int_list(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item]


def infer_peak_tflops(device_name: str) -> float | None:
    for key, value in COMMON_FP16_BF16_PEAK_TFLOPS.items():
        if key in device_name:
            return value
    return None


def exact_causal_pairs(query_len: int, seq_len: int) -> int:
    if query_len > seq_len:
        raise ValueError(f"query_len ({query_len}) must be <= seq_len ({seq_len}).")
    # Bottom-right aligned causal mask used by flash_attn_varlen_func.
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
    # QK^T + P*V, counting a multiply-add as 2 FLOPs.
    return float(4 * total_pairs * num_query_heads * head_size)


def estimate_activation_flops(num_elements: int, activation: str) -> float:
    # These are lightweight approximations; GEMMs dominate overall FFN cost.
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


def format_optional(value: float | None, fmt: str = ".3f") -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def _rebalance_sum(
    values: list[int],
    *,
    target: int,
    min_value: int = 1,
) -> list[int]:
    values = [max(min_value, int(value)) for value in values]
    delta = target - sum(values)
    cursor = 0
    n = len(values)
    while delta != 0 and n > 0:
        idx = cursor % n
        if delta > 0:
            values[idx] += 1
            delta -= 1
        elif values[idx] > min_value:
            values[idx] -= 1
            delta += 1
        cursor += 1
    if delta != 0:
        raise ValueError(
            f"Could not rebalance values to target={target} with min_value={min_value}."
        )
    return values


def print_table(title: str, rows: list[RunStats]) -> None:
    print(f"\n{title}")
    headers = [
        "name",
        "batch",
        "avg_seq",
        "var",
        "q_tokens",
        "ms",
        "TFLOPS",
        "MFU",
        "pred_ms",
        "err%",
    ]
    print(
        f"{headers[0]:<18} {headers[1]:<18} {headers[2]:>8} {headers[3]:>12} "
        f"{headers[4]:>10} {headers[5]:>10} {headers[6]:>10} {headers[7]:>8} "
        f"{headers[8]:>10} {headers[9]:>9}"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row.name:<18} {row.batch_label:<18} "
            f"{row.avg_seq_len:8.1f} {row.variance:12.1f} "
            f"{row.total_query_tokens:10d} {row.latency_ms:10.3f} "
            f"{row.actual_tflops:10.3f} {format_optional(row.actual_mfu):>8} "
            f"{format_optional(row.predicted_ms):>10} "
            f"{format_optional(row.error_pct):>9}"
        )


def print_summary(title: str, rows: list[RunStats]) -> None:
    comparable = [row for row in rows if row.error_pct is not None]
    if not comparable:
        return
    worst = max(comparable, key=lambda row: abs(row.error_pct or 0.0))
    best = min(comparable, key=lambda row: abs(row.error_pct or 0.0))
    print(
        f"\n{title} summary: worst static-prediction error = "
        f"{worst.error_pct:.2f}% on {worst.name}, best = {best.error_pct:.2f}% "
        f"on {best.name}."
    )


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
        peak_tflops: float | None,
    ) -> RunStats:
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
        actual_tflops = flops / (latency_ms * 1e-3) / 1e12
        actual_mfu = actual_tflops / peak_tflops if peak_tflops is not None else None
        return RunStats(
            name=spec.name,
            batch_label=spec.batch_label,
            batch_size=len(spec.seq_lens),
            avg_seq_len=spec.avg_seq_len,
            variance=spec.variance,
            total_query_tokens=sum(spec.query_lens),
            total_kv_tokens=sum(spec.seq_lens),
            flops=flops,
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            actual_tflops=actual_tflops,
            actual_mfu=actual_mfu,
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

    def _build_inputs(
        self,
        spec: WorkloadSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return hidden_states, fc1_weight, fc2_weight

    @torch.inference_mode()
    def benchmark(
        self,
        spec: WorkloadSpec,
        *,
        warmup: int,
        trials: int,
        peak_tflops: float | None,
    ) -> RunStats:
        hidden_states, fc1_weight, fc2_weight = self._build_inputs(spec)

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
        actual_tflops = flops / (latency_ms * 1e-3) / 1e12
        actual_mfu = actual_tflops / peak_tflops if peak_tflops is not None else None
        return RunStats(
            name=spec.name,
            batch_label=spec.batch_label,
            batch_size=len(spec.seq_lens),
            avg_seq_len=spec.avg_seq_len,
            variance=spec.variance,
            total_query_tokens=sum(spec.query_lens),
            total_kv_tokens=sum(spec.seq_lens),
            flops=flops,
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            actual_tflops=actual_tflops,
            actual_mfu=actual_mfu,
        )


class StaticMFUPredictor:
    def __init__(self, calibration: RunStats, peak_tflops: float | None) -> None:
        self.effective_tflops = calibration.actual_tflops
        self.peak_tflops = peak_tflops

    def annotate(self, stat: RunStats) -> RunStats:
        predicted_ms = stat.flops / (self.effective_tflops * 1e12) * 1e3
        predicted_mfu = (
            self.effective_tflops / self.peak_tflops
            if self.peak_tflops is not None
            else None
        )
        stat.predicted_ms = predicted_ms
        stat.predicted_tflops = self.effective_tflops
        stat.predicted_mfu = predicted_mfu
        stat.error_pct = 100.0 * (predicted_ms - stat.latency_ms) / stat.latency_ms
        return stat


def make_nr_sweep_specs(
    *,
    batch_size: int,
    seq_len: int,
    nr_values: list[int],
    tail_seq_len: int,
    num_long_reqs: int,
    long_step: int,
) -> list[WorkloadSpec]:
    max_nr = max(nr_values)
    seq_lens = _exact_mean_multi_tail_distribution(
        batch_size=batch_size,
        mean_seq_len=seq_len,
        tail_seq_len=tail_seq_len,
        num_long_reqs=num_long_reqs,
        long_step=long_step,
        min_seq_len=max_nr,
    )
    variance = statistics.pvariance(seq_lens)
    specs: list[WorkloadSpec] = []
    for nr in nr_values:
        if nr <= 0:
            raise ValueError(
                "Experiment 'nr' requires strictly positive query lengths. "
                f"Got Nr={nr}. flash_attn_varlen_func does not support "
                "zero-query workloads; use Nr>=1, or measure the no-replay "
                "baseline with the full-model profiler instead."
            )
        query_lens = [nr] * batch_size
        specs.append(
            WorkloadSpec(
                name=f"Nr={nr}",
                batch_label=(f"hetero<= {tail_seq_len} q={nr}"),
                seq_lens=seq_lens.copy(),
                query_lens=query_lens,
                avg_seq_len=statistics.fmean(seq_lens),
                variance=variance,
            )
        )
    return specs


def _allocate_budget_to_requests(
    seq_lens: list[int],
    total_budget: int,
    strategy: str,
) -> list[int]:
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative.")
    if total_budget > sum(seq_lens):
        raise ValueError(
            f"total_budget={total_budget} exceeds total available tokens={sum(seq_lens)}."
        )

    query_lens = [0] * len(seq_lens)
    remaining = total_budget

    if strategy == "uniform":
        active = {idx for idx, seq_len in enumerate(seq_lens) if seq_len > 0}
        while remaining > 0 and active:
            progressed = False
            for idx in list(active):
                if remaining == 0:
                    break
                if query_lens[idx] < seq_lens[idx]:
                    query_lens[idx] += 1
                    remaining -= 1
                    progressed = True
                if query_lens[idx] >= seq_lens[idx]:
                    active.discard(idx)
            if not progressed:
                break
    else:
        reverse = strategy == "long_first"
        if strategy not in {"long_first", "short_first"}:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
        order = sorted(
            range(len(seq_lens)),
            key=lambda idx: (seq_lens[idx], idx),
            reverse=reverse,
        )
        for idx in order:
            if remaining == 0:
                break
            take = min(seq_lens[idx], remaining)
            query_lens[idx] = take
            remaining -= take

    if remaining != 0:
        raise AssertionError(
            f"Failed to allocate the full replay budget. Remaining={remaining}."
        )
    return query_lens


def make_budget_allocation_specs(
    *,
    batch_size: int,
    mean_seq_len: int,
    tail_seq_len: int,
    replay_budget: int,
    num_long_reqs: int,
    long_step: int,
) -> list[WorkloadSpec]:
    seq_lens = _exact_mean_multi_tail_distribution(
        batch_size=batch_size,
        mean_seq_len=mean_seq_len,
        tail_seq_len=tail_seq_len,
        num_long_reqs=num_long_reqs,
        long_step=long_step,
    )
    variance = statistics.pvariance(seq_lens)

    specs: list[WorkloadSpec] = []
    for strategy in ("uniform", "long_first", "short_first"):
        query_lens = _allocate_budget_to_requests(
            seq_lens=seq_lens,
            total_budget=replay_budget,
            strategy=strategy,
        )
        specs.append(
            WorkloadSpec(
                name=strategy,
                batch_label=(
                    f"budget={replay_budget} q[{min(query_lens)},{max(query_lens)}]"
                ),
                seq_lens=seq_lens.copy(),
                query_lens=query_lens,
                avg_seq_len=statistics.fmean(seq_lens),
                variance=variance,
            )
        )
    return specs


def _exact_mean_multi_tail_distribution(
    *,
    batch_size: int,
    mean_seq_len: int,
    tail_seq_len: int,
    num_long_reqs: int,
    long_step: int,
    min_seq_len: int = 1,
) -> list[int]:
    total = batch_size * mean_seq_len
    if tail_seq_len < mean_seq_len:
        raise ValueError("tail_seq_len must be >= mean_seq_len.")
    if tail_seq_len > total:
        raise ValueError("tail_seq_len is too large for the requested mean.")
    if min_seq_len > mean_seq_len:
        raise ValueError(
            f"min_seq_len={min_seq_len} cannot exceed mean_seq_len={mean_seq_len}."
        )
    if not (1 <= num_long_reqs < batch_size):
        raise ValueError(
            f"num_long_reqs must be in [1, {batch_size - 1}], got {num_long_reqs}."
        )

    num_short = batch_size - num_long_reqs
    min_short = max(1, mean_seq_len // 16, min_seq_len)
    max_long_budget = total - num_short * min_short
    if max_long_budget < num_long_reqs:
        raise ValueError("Mean sequence length is too small for the requested setup.")

    step = max(1, long_step)
    long_lens = [max(1, tail_seq_len - step * idx) for idx in range(num_long_reqs)]
    long_total = sum(long_lens)
    if long_total > max_long_budget:
        scale = max_long_budget / long_total
        long_lens = [
            max(min_short + 1, int(round(length * scale))) for length in long_lens
        ]
        long_lens = _rebalance_sum(
            long_lens,
            target=max_long_budget,
            min_value=min_short + 1,
        )

    short_total = total - sum(long_lens)
    short_avg = short_total / num_short
    short_amplitude = min(
        max(1, int(short_avg // 3)),
        max(int(short_avg) - min_short, 0),
    )
    ranks = [idx - (num_short - 1) / 2 for idx in range(num_short)]
    denom = max(abs(rank) for rank in ranks) or 1.0
    short_lens = [
        max(
            min_short,
            int(round(short_avg + short_amplitude * rank / denom)),
        )
        for rank in ranks
    ]
    short_lens = _rebalance_sum(short_lens, target=short_total, min_value=min_short)

    lengths = long_lens + short_lens
    if sum(lengths) != total:
        raise AssertionError(
            f"Constructed lengths do not preserve the exact mean budget: {sum(lengths)} vs {total}."
        )
    return lengths


def make_variance_sweep_specs(
    *,
    batch_size: int,
    mean_seq_len: int,
    tail_seq_lens: list[int],
    num_long_reqs: int,
    long_step: int,
) -> list[WorkloadSpec]:
    specs: list[WorkloadSpec] = []

    uniform = [mean_seq_len] * batch_size
    specs.append(
        WorkloadSpec(
            name="uniform",
            batch_label=f"all={mean_seq_len}",
            seq_lens=uniform,
            query_lens=uniform.copy(),
            avg_seq_len=float(mean_seq_len),
            variance=statistics.pvariance(uniform),
        )
    )

    for tail_seq_len in tail_seq_lens:
        if tail_seq_len == mean_seq_len:
            continue
        seq_lens = _exact_mean_multi_tail_distribution(
            batch_size=batch_size,
            mean_seq_len=mean_seq_len,
            tail_seq_len=tail_seq_len,
            num_long_reqs=num_long_reqs,
            long_step=long_step,
        )
        variance = statistics.pvariance(seq_lens)
        short_lens = seq_lens[num_long_reqs:]
        specs.append(
            WorkloadSpec(
                name=f"tail={tail_seq_len}",
                batch_label=(
                    f"{num_long_reqs}x<= {tail_seq_len}"
                    f"+rest[{min(short_lens)},{max(short_lens)}]"
                ),
                seq_lens=seq_lens,
                query_lens=seq_lens.copy(),
                avg_seq_len=statistics.fmean(seq_lens),
                variance=variance,
            )
        )
    return specs


def run_nr_experiment(
    runner: FlashAttnBenchRunner,
    *,
    batch_size: int,
    seq_len: int,
    nr_values: list[int],
    tail_seq_len: int,
    num_long_reqs: int,
    long_step: int,
    calibration_nr: int,
    warmup: int,
    trials: int,
    peak_tflops: float | None,
) -> list[RunStats]:
    specs = make_nr_sweep_specs(
        batch_size=batch_size,
        seq_len=seq_len,
        nr_values=nr_values,
        tail_seq_len=tail_seq_len,
        num_long_reqs=num_long_reqs,
        long_step=long_step,
    )
    stats = [
        runner.benchmark(
            spec,
            warmup=warmup,
            trials=trials,
            peak_tflops=peak_tflops,
        )
        for spec in specs
    ]
    calibration = next(
        (stat for stat in stats if stat.name == f"Nr={calibration_nr}"),
        None,
    )
    if calibration is None:
        raise ValueError(
            f"calibration_nr={calibration_nr} not found in sweep {nr_values}."
        )
    predictor = StaticMFUPredictor(calibration, peak_tflops)
    return [predictor.annotate(stat) for stat in stats]


def run_variance_experiment(
    runner: FlashAttnBenchRunner,
    *,
    batch_size: int,
    mean_seq_len: int,
    tail_seq_lens: list[int],
    num_long_reqs: int,
    long_step: int,
    warmup: int,
    trials: int,
    peak_tflops: float | None,
) -> list[RunStats]:
    specs = make_variance_sweep_specs(
        batch_size=batch_size,
        mean_seq_len=mean_seq_len,
        tail_seq_lens=tail_seq_lens,
        num_long_reqs=num_long_reqs,
        long_step=long_step,
    )
    stats = [
        runner.benchmark(
            spec,
            warmup=warmup,
            trials=trials,
            peak_tflops=peak_tflops,
        )
        for spec in specs
    ]
    predictor = StaticMFUPredictor(stats[0], peak_tflops)
    return [predictor.annotate(stat) for stat in stats]


def run_budget_allocation_experiment(
    runner,
    *,
    batch_size: int,
    mean_seq_len: int,
    tail_seq_len: int,
    replay_budget: int,
    num_long_reqs: int,
    long_step: int,
    warmup: int,
    trials: int,
    peak_tflops: float | None,
) -> list[RunStats]:
    specs = make_budget_allocation_specs(
        batch_size=batch_size,
        mean_seq_len=mean_seq_len,
        tail_seq_len=tail_seq_len,
        replay_budget=replay_budget,
        num_long_reqs=num_long_reqs,
        long_step=long_step,
    )
    stats = [
        runner.benchmark(
            spec,
            warmup=warmup,
            trials=trials,
            peak_tflops=peak_tflops,
        )
        for spec in specs
    ]
    calibration = next((stat for stat in stats if stat.name == "uniform"), None)
    if calibration is None:
        raise AssertionError(
            "Budget-allocation experiment requires uniform calibration."
        )
    predictor = StaticMFUPredictor(calibration, peak_tflops)
    return [predictor.annotate(stat) for stat in stats]


def save_results(path: Path, results: dict[str, list[RunStats]]) -> None:
    payload = {key: [asdict(stat) for stat in value] for key, value in results.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_ncu_hint(script_path: Path, args) -> None:
    metrics = ",".join(
        [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ]
    )
    base = f"ncu --target-processes all --metrics {metrics} python {script_path}"
    print("\nNsight Compute hints:")
    print(
        f"  N_r sweep: {base} --experiment nr --batch-size {args.batch_size} "
        f"--seq-len {args.seq_len} --nr-tail-seq-len {args.nr_tail_seq_len} "
        f"--nr-values {args.nr_values} --num-long-reqs {args.num_long_reqs} "
        f"--long-step {args.long_step} "
        f"--warmup 1 --trials 1"
    )
    print(
        f"  Variance sweep: {base} --experiment variance --batch-size {args.batch_size} "
        f"--mean-seq-len {args.mean_seq_len} --tail-seq-lens {args.tail_seq_lens} "
        f"--num-long-reqs {args.num_long_reqs} --long-step {args.long_step} "
        f"--warmup 1 --trials 1"
    )
    print(
        f"  Budget sweep: {base} --experiment budget --batch-size {args.batch_size} "
        f"--mean-seq-len {args.mean_seq_len} --budget-tail-seq-len "
        f"{args.budget_tail_seq_len} --replay-budget {args.replay_budget} "
        f"--num-long-reqs {args.num_long_reqs} --long-step {args.long_step} "
        f"--warmup 1 --trials 1"
    )


def run_selected_experiments(
    *,
    module_label: str,
    runner,
    args,
    peak_tflops: float | None,
) -> dict[str, list[RunStats]]:
    results: dict[str, list[RunStats]] = {}

    if args.experiment in ("nr", "all"):
        nr_values = parse_int_list(args.nr_values)
        calibration_nr = args.calibration_nr or max(nr_values)
        results["nr"] = run_nr_experiment(
            runner,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            nr_values=nr_values,
            tail_seq_len=args.nr_tail_seq_len,
            num_long_reqs=args.num_long_reqs,
            long_step=args.long_step,
            calibration_nr=calibration_nr,
            warmup=args.warmup,
            trials=args.trials,
            peak_tflops=peak_tflops,
        )
        print_table(f"{module_label} / Experiment 1: N_r Sensitivity", results["nr"])
        print_summary(f"{module_label} / Experiment 1", results["nr"])

    if args.experiment in ("variance", "all"):
        tail_seq_lens = parse_int_list(args.tail_seq_lens)
        results["variance"] = run_variance_experiment(
            runner,
            batch_size=args.batch_size,
            mean_seq_len=args.mean_seq_len,
            tail_seq_lens=tail_seq_lens,
            num_long_reqs=args.num_long_reqs,
            long_step=args.long_step,
            warmup=args.warmup,
            trials=args.trials,
            peak_tflops=peak_tflops,
        )
        print_table(
            f"{module_label} / Experiment 2: Batch Variance Trap",
            results["variance"],
        )
        print_summary(f"{module_label} / Experiment 2", results["variance"])
        if args.mean_seq_len == 1024 and 8192 in tail_seq_lens:
            print(
                "\nNote: an exact-mean multi-tail replacement is used for the "
                "high-variance case. The script now builds multiple long requests "
                "with descending lengths plus a varied short-request tail while "
                "preserving the exact average length."
            )

    if args.experiment in ("budget", "all"):
        results["budget"] = run_budget_allocation_experiment(
            runner,
            batch_size=args.batch_size,
            mean_seq_len=args.mean_seq_len,
            tail_seq_len=args.budget_tail_seq_len,
            replay_budget=args.replay_budget,
            num_long_reqs=args.num_long_reqs,
            long_step=args.long_step,
            warmup=args.warmup,
            trials=args.trials,
            peak_tflops=peak_tflops,
        )
        print_table(
            f"{module_label} / Experiment 3: Replay Budget Allocation",
            results["budget"],
        )
        print_summary(f"{module_label} / Experiment 3", results["budget"])

    return results


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
    if args.nr_tail_seq_len is None:
        args.nr_tail_seq_len = max(
            args.seq_len + args.long_step * max(args.num_long_reqs - 1, 1),
            args.seq_len * 2,
        )

    device_name = torch.cuda.get_device_name(0)
    peak_tflops = args.peak_tflops
    if peak_tflops is None:
        peak_tflops = infer_peak_tflops(device_name)
        if peak_tflops is not None:
            print(
                f"Inferred peak TFLOPS={peak_tflops:.1f} for {device_name}. "
                "Pass --peak-tflops to override."
            )
        else:
            print(
                f"Could not infer peak TFLOPS for {device_name}. "
                "MFU columns will be reported as n/a. "
                "Pass --peak-tflops for real MFU."
            )

    results: dict[str, list[RunStats]] = {}
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    if args.module in ("attention", "all"):
        attn_runner = FlashAttnBenchRunner(
            dtype=dtype,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            softmax_scale=args.softmax_scale,
            seed=args.seed,
        )
        module_results = run_selected_experiments(
            module_label="Attention",
            runner=attn_runner,
            args=args,
            peak_tflops=peak_tflops,
        )
        results.update(
            {f"attention_{key}": value for key, value in module_results.items()}
        )

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
        module_results = run_selected_experiments(
            module_label="FFN",
            runner=ffn_runner,
            args=args,
            peak_tflops=peak_tflops,
        )
        results.update({f"ffn_{key}": value for key, value in module_results.items()})

    if args.json_output:
        save_results(Path(args.json_output), results)
        print(f"\nSaved JSON results to {args.json_output}")

    if args.print_ncu_hint:
        print_ncu_hint(Path(__file__), args)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Profile how attention / FFN actual MFU and TFLOPS deviate from "
            "a TightLLM-style static MFU predictor."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=["nr", "variance", "budget", "all"],
        default="all",
    )
    parser.add_argument(
        "--module",
        choices=["attention", "ffn", "all"],
        default="attention",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--nr-tail-seq-len", type=int, default=None)
    parser.add_argument("--mean-seq-len", type=int, default=1024)
    parser.add_argument("--nr-values", type=str, default="16,32,64,128,256,512")
    parser.add_argument("--tail-seq-lens", type=str, default="2048,4096,8192")
    parser.add_argument("--replay-budget", type=int, default=256)
    parser.add_argument("--budget-tail-seq-len", type=int, default=8192)
    parser.add_argument("--num-long-reqs", type=int, default=4)
    parser.add_argument("--long-step", type=int, default=256)
    parser.add_argument("--calibration-nr", type=int, default=None)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--ffn-intermediate-size", type=int, default=None)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument(
        "--ffn-activation",
        choices=["relu", "gelu", "silu"],
        default="relu",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["half", "bfloat16"], default="bfloat16")
    parser.add_argument("--softmax-scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--print-ncu-hint", action="store_true")
    main(parser.parse_args())
