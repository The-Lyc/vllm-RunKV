#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Microbenchmark: local linearity of replay compute cost vs. replay budget (Model B).

Model-B workload
~~~~~~~~~~~~~~~~
For each (ctx_len, anchor_block) the suffix is FIXED at
``[anchor_block * block_size, ctx_len)``.  Replay grows leftward from the
anchor, so the replay region is
``[(anchor_block - replay_blocks) * block_size, anchor_block * block_size)``.

  baseline  : query_len = suffix_len + scheduled_len   (R=0, constant per anchor)
  replay(R) : query_len = R * block_size + suffix_len + scheduled_len
  delta_ms  = replay_ms(R) - baseline_ms

Locally-linear delta_ms in R supports the Newton-Secant slope assumption
used by RunKV to adapt replay budget without exhaustive profiling.

Model presets (--model-preset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  opt-1.3b  hidden=2048  MHA 32q/32kv  head=64   ffn=8192  relu
  opt-2.7b  hidden=2560  MHA 32q/32kv  head=80   ffn=10240 relu
  opt-13b   hidden=5120  MHA 40q/40kv  head=128  ffn=20480 relu
  opt-30b   hidden=7168  MHA 56q/56kv  head=128  ffn=28672 relu

Context lengths tested: 1k / 2k / 4k / 8k  (--ctx-lens 1024,2048,4096,8192)
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPreset:
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    ffn_intermediate_size: int
    ffn_activation: str
    description: str

    @property
    def hidden_size(self) -> int:
        return self.num_query_heads * self.head_size


MODEL_PRESETS: dict[str, ModelPreset] = {
    "opt-1.3b": ModelPreset(
        num_query_heads=32, num_kv_heads=32, head_size=64,
        ffn_intermediate_size=8192, ffn_activation="relu",
        description="OPT-1.3B (hidden=2048, MHA 32q/32kv, head=64)",
    ),
    "opt-2.7b": ModelPreset(
        num_query_heads=32, num_kv_heads=32, head_size=80,
        ffn_intermediate_size=10240, ffn_activation="relu",
        description="OPT-2.7B (hidden=2560, MHA 32q/32kv, head=80)",
    ),
    "opt-13b": ModelPreset(
        num_query_heads=40, num_kv_heads=40, head_size=128,
        ffn_intermediate_size=20480, ffn_activation="relu",
        description="OPT-13B (hidden=5120, MHA 40q/40kv, head=128)",
    ),
    "opt-30b": ModelPreset(
        num_query_heads=56, num_kv_heads=56, head_size=128,
        ffn_intermediate_size=28672, ffn_activation="relu",
        description="OPT-30B (hidden=7168, MHA 56q/56kv, head=128)",
    ),
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

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


def aggregate_bench_stats(samples: list[BenchStats]) -> BenchStats:
    """Combine equal-sized repeated benchmark estimates into one estimate."""
    if not samples:
        raise ValueError("samples must not be empty.")
    latency_ms = statistics.fmean(sample.latency_ms for sample in samples)
    second_moment = statistics.fmean(
        sample.latency_std_ms**2 + sample.latency_ms**2 for sample in samples
    )
    latency_std_ms = math.sqrt(max(0.0, second_moment - latency_ms**2))
    flops = samples[0].flops
    tflops = flops / (latency_ms * 1e-3) / 1e12
    return BenchStats(
        latency_ms=latency_ms,
        latency_std_ms=latency_std_ms,
        flops=flops,
        tflops=tflops,
    )


@dataclass
class BudgetSweepRaw:
    """One (anchor, replay_blocks) measurement point."""
    module: str
    ctx_len: int
    scheduled_len: int
    anchor_block: int       # suffix starts here (block index)
    anchor_token: int       # anchor_block * block_size
    suffix_len: int         # ctx_len - anchor_token  (fixed per anchor)
    replay_blocks: int      # R  (0 → baseline, 1..max → replay budget)
    baseline_ms: float      # latency of suffix+scheduled (constant per anchor)
    baseline_std_ms: float
    replay_ms: float        # latency of replay+suffix+scheduled
    replay_std_ms: float
    delta_ms: float         # replay_ms - baseline_ms


@dataclass
class BudgetSweepFit:
    """Local linear fit result for one sliding window."""
    module: str
    ctx_len: int
    scheduled_len: int
    anchor_block: int
    window_start_blocks: int
    window_end_blocks: int
    n_points: int
    slope_ms_per_block: float
    intercept_ms: float
    r2: float
    max_abs_residual_ms: float
    cv_marginal_delta: float    # CV of marginal latency in ms per replay block


def parse_int_list(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item]


def parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item]


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



# ---------------------------------------------------------------------------
# Model-B sweep logic
# ---------------------------------------------------------------------------

def make_anchor_blocks(
    ctx_blocks: int,
    anchor_fractions: list[float],
) -> list[int]:
    """Convert fractions to concrete block indices, clamped to [1, ctx_blocks]."""
    seen: set[int] = set()
    result: list[int] = []
    for frac in sorted(anchor_fractions):
        ab = max(1, min(ctx_blocks, round(frac * ctx_blocks)))
        if ab not in seen:
            seen.add(ab)
            result.append(ab)
    return result


def run_budget_sweep(
    runner,
    *,
    module: str,
    ctx_lens: list[int],
    scheduled_len: int,
    anchor_fractions: list[float],
    block_size: int,
    max_replay_blocks: int,
    replay_step_blocks: int,
    sweep_repeats: int,
    warmup: int,
    trials: int,
    seed: int,
) -> list[BudgetSweepRaw]:
    """
    For every (ctx_len, anchor_block), randomly order and sweep
    replay_blocks = 0, step, 2*step, …, min(anchor_block, max_replay_blocks).

    The zero-budget point is the baseline shared by all replay_blocks values at
    the same anchor. Randomizing the order breaks systematic correlation between
    increasing replay budget and time-dependent GPU state drift.
    Multiple repeated randomized sweeps can be used for stronger evidence.
    """
    if max_replay_blocks < 0:
        raise ValueError("max_replay_blocks must be non-negative.")
    if replay_step_blocks <= 0:
        raise ValueError("replay_step_blocks must be positive.")
    if sweep_repeats <= 0:
        raise ValueError("sweep_repeats must be positive.")

    rows: list[BudgetSweepRaw] = []
    rng = random.Random(seed)
    for ctx_len in ctx_lens:
        ctx_blocks = ctx_len // block_size
        anchor_blocks = make_anchor_blocks(ctx_blocks, anchor_fractions)
        for anchor_block in anchor_blocks:
            suffix_len = ctx_len - anchor_block * block_size
            total_seq_len = ctx_len + scheduled_len
            baseline_query_len = suffix_len + scheduled_len

            sweep_max = min(anchor_block, max_replay_blocks)
            rb_vals = list(range(0, sweep_max + 1, replay_step_blocks))
            if rb_vals[-1] != sweep_max:
                rb_vals.append(sweep_max)

            print(
                f"  [{module}] ctx={ctx_len} anchor_blk={anchor_block}"
                f" (frac≈{anchor_block/ctx_blocks:.2f})"
                f" suffix={suffix_len} sweep R=0..{sweep_max}"
                f" step={replay_step_blocks}"
                f" n_points={len(rb_vals)}"
                f" randomized_repeats={sweep_repeats}"
            )

            per_budget: dict[int, list[BenchStats]] = {rb: [] for rb in rb_vals}
            for _ in range(sweep_repeats):
                measurement_order = rb_vals.copy()
                rng.shuffle(measurement_order)
                for rb in measurement_order:
                    replay_query_len = rb * block_size + baseline_query_len
                    replay_spec = WorkloadSpec(
                        seq_lens=[total_seq_len],
                        query_lens=[replay_query_len],
                    )
                    per_budget[rb].append(
                        runner.benchmark(replay_spec, warmup=warmup, trials=trials)
                    )

            baseline_stats = aggregate_bench_stats(per_budget[0])
            for rb in rb_vals:
                replay_stats = aggregate_bench_stats(per_budget[rb])
                rows.append(BudgetSweepRaw(
                    module=module,
                    ctx_len=ctx_len,
                    scheduled_len=scheduled_len,
                    anchor_block=anchor_block,
                    anchor_token=anchor_block * block_size,
                    suffix_len=suffix_len,
                    replay_blocks=rb,
                    baseline_ms=baseline_stats.latency_ms,
                    baseline_std_ms=baseline_stats.latency_std_ms,
                    replay_ms=replay_stats.latency_ms,
                    replay_std_ms=replay_stats.latency_std_ms,
                    delta_ms=replay_stats.latency_ms - baseline_stats.latency_ms,
                ))
    return rows


# ---------------------------------------------------------------------------
# Post-processing: local linear fits
# ---------------------------------------------------------------------------

def compute_local_fits(
    raw_rows: list[BudgetSweepRaw],
    window_blocks: int,
) -> list[BudgetSweepFit]:
    """
    For each (module, ctx_len, anchor_block) group, slide a window of
    `window_blocks` consecutive data points over the sorted replay_blocks
    sequence and fit delta_ms = slope * replay_blocks + intercept.

    Also computes cv_marginal_delta over marginal latency per replay block as
    a measure of how stable the per-block cost is within the window.
    """
    if window_blocks < 2:
        raise ValueError("window_blocks must be at least 2.")

    fits: list[BudgetSweepFit] = []
    key_fn = lambda r: (r.module, r.ctx_len, r.anchor_block)  # noqa: E731
    for key, grp in groupby(sorted(raw_rows, key=key_fn), key=key_fn):
        module, ctx_len, anchor_block = key
        subset = sorted(grp, key=lambda r: r.replay_blocks)
        if len(subset) < 2:
            continue
        scheduled_len = subset[0].scheduled_len
        x = np.array([r.replay_blocks for r in subset], dtype=np.float64)
        y = np.array([r.delta_ms      for r in subset], dtype=np.float64)
        marginal = np.diff(y) / np.diff(x)  # ms per replay block

        n = len(subset)
        for w0 in range(0, n - window_blocks + 1):
            w1 = w0 + window_blocks          # exclusive end
            wx, wy = x[w0:w1], y[w0:w1]

            coeffs = np.polyfit(wx, wy, 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            residuals = wy - np.polyval(coeffs, wx)
            ss_res = float(np.dot(residuals, residuals))
            ss_tot = float(np.sum((wy - wy.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            max_abs_res = float(np.max(np.abs(residuals)))

            wm = marginal[w0 : w1 - 1]
            if wm.size > 0:
                mean_m = float(np.mean(wm))
                cv = float(np.std(wm)) / (abs(mean_m) + 1e-9)
            else:
                cv = float("nan")

            fits.append(BudgetSweepFit(
                module=module,
                ctx_len=ctx_len,
                scheduled_len=scheduled_len,
                anchor_block=anchor_block,
                window_start_blocks=int(wx[0]),
                window_end_blocks=int(wx[-1]),
                n_points=window_blocks,
                slope_ms_per_block=slope,
                intercept_ms=intercept,
                r2=r2,
                max_abs_residual_ms=max_abs_res,
                cv_marginal_delta=cv,
            ))
    return fits


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(rows[0])])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def save_json_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_raw_table(title: str, rows: list[BudgetSweepRaw]) -> None:
    if not rows:
        return
    sep = "=" * 96
    print(f"\n{sep}\n  {title}\n{sep}")
    print(
        f"{'module':<18} {'ctx':>6} {'anc_blk':>8} {'frac':>5} "
        f"{'suffix':>7} {'R_blk':>6} "
        f"{'base_ms':>9} {'base_std':>8} "
        f"{'rply_ms':>9} {'rply_std':>8} "
        f"{'delta_ms':>9}"
    )
    print("-" * 96)
    for r in rows:
        frac = r.anchor_token / r.ctx_len if r.ctx_len else 0
        print(
            f"{r.module:<18} {r.ctx_len:6d} {r.anchor_block:8d} {frac:5.2f} "
            f"{r.suffix_len:7d} {r.replay_blocks:6d} "
            f"{r.baseline_ms:9.3f} {r.baseline_std_ms:8.4f} "
            f"{r.replay_ms:9.3f} {r.replay_std_ms:8.4f} "
            f"{r.delta_ms:9.3f}"
        )


def print_fit_summary(title: str, fits: list[BudgetSweepFit]) -> None:
    if not fits:
        return
    win = fits[0].n_points
    sep = "=" * 100
    print(f"\n{sep}\n  {title}  (window={win} pts)\n{sep}")
    key_fn = lambda f: (f.module, f.ctx_len, f.anchor_block)  # noqa: E731
    for key, grp in groupby(sorted(fits, key=key_fn), key=key_fn):
        module, ctx_len, anchor_block = key
        subset = list(grp)
        r2s = [f.r2 for f in subset]
        sls = [f.slope_ms_per_block for f in subset]
        cvs = [f.cv_marginal_delta  for f in subset
               if not math.isnan(f.cv_marginal_delta)]
        cv_str = (
            f"cv_marg [{min(cvs):.3f}, {max(cvs):.3f}]"
            if cvs else "cv_marg n/a"
        )
        print(
            f"  {module:<18} ctx={ctx_len:5d}  anchor={anchor_block:4d}  |"
            f"  R²  [{min(r2s):.3f}, {max(r2s):.3f}]"
            f"  slope [{min(sls):.4f}, {max(sls):.4f}] ms/blk"
            f"  {cv_str}"
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



def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.module in ("flash_core", "full_attention", "all") and not is_flash_attn_varlen_func_available():
        raise RuntimeError("flash_attn_varlen_func is not available in this environment.")

    # ---- resolve model parameters ----
    if args.model_preset:
        preset = MODEL_PRESETS[args.model_preset]
        print(f"Using preset {args.model_preset!r}: {preset.description}")
        num_query_heads = preset.num_query_heads
        num_kv_heads    = preset.num_kv_heads
        head_size       = preset.head_size
        ffn_intermediate_size = preset.ffn_intermediate_size
        ffn_activation  = preset.ffn_activation
    else:
        num_query_heads = args.num_query_heads
        num_kv_heads    = args.num_kv_heads
        head_size       = args.head_size
        ffn_intermediate_size = (
            args.ffn_intermediate_size
            if args.ffn_intermediate_size is not None
            else args.ffn_multiplier * num_query_heads * head_size
        )
        ffn_activation  = args.ffn_activation

    dtype            = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    ctx_lens         = parse_int_list(args.ctx_lens)
    anchor_fractions = parse_float_list(args.anchor_fractions)
    hidden_size      = num_query_heads * head_size

    modules_to_run = (
        ["flash_core", "full_attention", "ffn"]
        if args.module == "all"
        else [args.module]
    )

    all_raw:  list[BudgetSweepRaw] = []
    all_fits: list[BudgetSweepFit] = []

    for mod in modules_to_run:
        print(f"\n{'#' * 80}\n#  Module: {mod}\n{'#' * 80}")

        if mod == "flash_core":
            runner = FlashAttnBenchRunner(
                dtype=dtype,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=args.block_size,
                softmax_scale=args.softmax_scale,
                seed=args.seed,
            )
        elif mod == "full_attention":
            runner = FullAttentionBenchRunner(
                dtype=dtype,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=args.block_size,
                softmax_scale=args.softmax_scale,
                seed=args.seed,
            )
        else:  # ffn
            runner = FFNBenchRunner(
                dtype=dtype,
                hidden_size=hidden_size,
                intermediate_size=ffn_intermediate_size,
                activation=ffn_activation,
                seed=args.seed,
            )

        raw = run_budget_sweep(
            runner,
            module=mod,
            ctx_lens=ctx_lens,
            scheduled_len=args.scheduled_len,
            anchor_fractions=anchor_fractions,
            block_size=args.block_size,
            max_replay_blocks=args.max_replay_blocks,
            replay_step_blocks=args.replay_step_blocks,
            sweep_repeats=args.sweep_repeats,
            warmup=args.warmup,
            trials=args.trials,
            seed=args.seed,
        )
        all_raw.extend(raw)
        print_raw_table(f"{mod} — raw measurements", raw)

        fits = compute_local_fits(raw, window_blocks=args.fit_window_blocks)
        all_fits.extend(fits)
        print_fit_summary(f"{mod} — local linear fits", fits)

    # ---- save outputs ----
    if args.output_dir:
        out = Path(args.output_dir)
        raw_path = out / "raw.csv"
        fit_path = out / "local_fits.csv"
        save_csv(raw_path, all_raw)
        save_csv(fit_path, all_fits)
        print(f"\nSaved raw measurements  → {raw_path}")
        print(f"Saved local-fit results → {fit_path}")
        if args.save_json:
            save_json_results(
                out / "raw.json",
                {"raw": [asdict(r) for r in all_raw]},
            )
            save_json_results(
                out / "local_fits.json",
                {"fits": [asdict(f) for f in all_fits]},
            )
            print(f"Saved JSON             → {out}/{{raw,local_fits}}.json")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Microbenchmark: local linearity of replay compute cost vs. replay "
            "budget (Model B).\n\n"
            "The suffix start (anchor) is fixed per sweep; replay grows leftward "
            "from the anchor so that baseline query_len is constant and delta_ms "
            "directly measures the extra cost of R replay blocks."
        )
    )

    # ---- module ----
    parser.add_argument(
        "--module",
        choices=["flash_core", "full_attention", "ffn", "all"],
        default="flash_core",
        help=(
            "flash_core: bare flash-attn kernel only.  "
            "full_attention: q/k/v/o projections + flash-attn.  "
            "ffn: two-layer FFN.  "
            "all: run all three."
        ),
    )

    # ---- model preset ----
    preset_help = "  |  ".join(
        f"{k}: {v.description}" for k, v in MODEL_PRESETS.items()
    )
    parser.add_argument(
        "--model-preset",
        choices=list(MODEL_PRESETS),
        default=None,
        help=(
            "OPT per-layer shape preset (overrides manual head/FFN args).  "
            f"{preset_help}"
        ),
    )

    # ---- context lengths ----
    parser.add_argument(
        "--ctx-lens",
        type=str,
        default="1024,2048,4096,8192",
        help="Comma-separated context lengths (tokens) to sweep.",
    )
    parser.add_argument(
        "--scheduled-len",
        type=int,
        default=16,
        help="Number of newly-scheduled tokens (part of the fixed baseline query).",
    )

    # ---- anchor / replay-budget sweep ----
    parser.add_argument(
        "--anchor-fractions",
        type=str,
        default="0.25,0.50,0.75",
        help=(
            "Comma-separated fractions of ctx_blocks at which the suffix starts "
            "(= replay end). E.g. 0.25 → anchor at 25%% of context, "
            "max_replay_blocks = 25%% * ctx_blocks."
        ),
    )
    parser.add_argument(
        "--max-replay-blocks",
        type=int,
        default=128,
        help="Maximum replay budget (blocks) to sweep per anchor.",
    )
    parser.add_argument(
        "--replay-step-blocks",
        type=int,
        default=1,
        help="Step size (blocks) for the dense replay-budget sweep.",
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=1,
        help=(
            "Number of randomized budget sweeps to aggregate per anchor. "
            "Use 3 or more when collecting evidence for local smoothness."
        ),
    )

    # ---- local-fit window ----
    parser.add_argument(
        "--fit-window-blocks",
        type=int,
        default=16,
        help=(
            "Sliding-window size (number of data points) for the local linear fit. "
            "Corresponds to how many consecutive replay_blocks values are fitted."
        ),
    )

    # ---- architecture (used when no preset) ----
    parser.add_argument("--num-query-heads",     type=int,   default=64)
    parser.add_argument("--num-kv-heads",        type=int,   default=8)
    parser.add_argument("--head-size",           type=int,   default=128)
    parser.add_argument("--block-size",          type=int,   default=16)
    parser.add_argument("--dtype",               choices=["half", "bfloat16"], default="bfloat16")
    parser.add_argument("--softmax-scale",       type=float, default=None)
    parser.add_argument("--ffn-intermediate-size", type=int, default=None)
    parser.add_argument("--ffn-multiplier",      type=int,   default=4)
    parser.add_argument(
        "--ffn-activation",
        choices=["relu", "gelu", "silu"],
        default="silu",
    )

    # ---- benchmark tuning ----
    parser.add_argument("--warmup",  type=int, default=10)
    parser.add_argument("--trials",  type=int, default=30,
                        help="Increase to ≥30 to reduce noise on marginal_delta.")
    parser.add_argument("--seed",    type=int, default=0)

    # ---- output ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write raw.csv and local_fits.csv (created if absent).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Also write raw.json and local_fits.json in --output-dir.",
    )

    args = parser.parse_args()
    main(args)
