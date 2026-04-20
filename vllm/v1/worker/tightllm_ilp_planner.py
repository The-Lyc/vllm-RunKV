# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TightLLM ILP-based replay plan provider for KV offloading.

Implements the KV distributor from TightLLM (Hu et al., IEEE TC 2025).
Instead of using a runtime feedback controller (like FeedbackReplayPlanProvider),
this provider uses *offline-profiled* MFU and bandwidth data to analytically
compute the optimal number of recomputation blocks per step, minimizing:

    max(T_compute(replay_blocks), T_transfer(replay_blocks))

Key equations (adapted from TightLLM Section IV-B, Eq 2-5):
    T_attn  = flops_attn(Nr) / (peak_FLOPS * MFU_attn(Nr))
    T_ffn   = flops_ffn(Nr)  / (peak_FLOPS * MFU_ffn(Nr))
    T_cache = bytes_to_load(Nr) / PCIe_bandwidth

    optimal Nr = argmin_{Nr} max(T_attn + T_ffn, T_cache)

The provider plugs into the existing ``ReplayPlanProvider`` protocol and
reuses the existing budget allocator and plan builder infrastructure.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch.cuda.nvtx as _nvtx

from vllm.logger import init_logger
from vllm.v1.profiling.tightllm_offline_profiler import TightLLMProfileData
from vllm.v1.worker.opt_dynamic_replay import (
    LayerReplayPlan,
    StaticReplayPlanProvider,
    _allocate_budget_to_requests,
    _coerce_int32_array,
    compute_layer_replay_plan_for_layer,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ILP solver — TightLLM KV Distributor (Hu et al., IEEE TC 2025, §IV-B)
#
# Solves the integer programming problem from Eq 2:
#
#   min_{Nr ∈ Z}  max( T_attn(Nr) + T_ffn(Nr),  T_cache(Nr) )
#   s.t.  0 ≤ Nr ≤ total_replayable_blocks
#
# Because MFU varies with the effective sequence length (paper Fig 7),
# the objective is non-linear, so we solve by exhaustive enumeration
# over the 1-D integer feasible region — equivalent to the ILP solver
# (Gurobi [36]) used in the paper.
# ---------------------------------------------------------------------------


def _compute_times(
    replay_blocks: int,
    *,
    profile: TightLLMProfileData,
    total_context_blocks: int,
    total_scheduled_tokens: int,
    avg_context_len: int,
    block_size: int,
) -> tuple[float, float]:
    """Compute T_compute and T_transfer for a given Nr (replay_blocks).

    Implements TightLLM Eq 3–5 (KV-only offloading, no disk tier).

    Eq 3 — Attention compute time:
        T_attn = Nr · 2·d_attn · (4·d_model + (n_ctx + Nr))
                 / (FLOPS × MFU_attn)

    Eq 4 — FFN compute time:
        T_ffn  = Nr · 4·d_model·d_ffn / (FLOPS × MFU_ffn)

    Eq 5 — KV cache transfer time (CPU-only, Cd=0, Cc=1):
        T_cache = 4·(N_ctx − Nr)·d_model / Bdw_cg

    Returns:
        (t_compute_s, t_transfer_s)  — both in seconds.
    """
    replay_tokens = replay_blocks * block_size
    num_actual_tokens = replay_tokens + total_scheduled_tokens

    if num_actual_tokens <= 0:
        return 0.0, float("inf")

    H = profile.hidden_size
    F_dim = profile.ffn_dim
    n_heads = profile.num_attention_heads
    d_head = profile.head_dim
    peak = profile.gpu_peak_flops

    # MFU lookup — varies with effective sequence length (paper Fig 7).
    mfu_attn = max(profile.lookup_mfu_attn(num_actual_tokens), 1e-6)
    mfu_ffn = max(profile.lookup_mfu_ffn(num_actual_tokens), 1e-6)

    # --- Attention FLOPs  (Eq 3) ---
    # QKV projection:    bs·Nr · 2·d_model · 3·d_attn  = N · 6·H²
    # Output projection:  bs·Nr · 2·d_attn · d_model    = N · 2·H²
    # Attention scores:   bs·Nr · 2·d_attn · (n_ctx+Nr) = N · 2·H·ctx
    #   (d_attn = n_heads·d_head = H for standard MHA)
    flops_attn = num_actual_tokens * (
        8 * H * H  # QKV + output proj
        + 2 * n_heads * avg_context_len * d_head  # attention scores
    )

    # --- FFN FLOPs  (Eq 4) ---
    # Two linear layers: bs·Nr · 4·d_model·d_ffn
    flops_ffn = 4 * num_actual_tokens * H * F_dim

    t_compute = 0.0
    if peak > 0:
        t_compute = flops_attn / (peak * mfu_attn) + flops_ffn / (peak * mfu_ffn)

    # --- Transfer time  (Eq 5, CPU→GPU only, no disk tier) ---
    # bytes_KV = 4·(N_ctx − Nr)·d_model  (factor 4 = K+V × FP16 bytes)
    io_blocks = max(total_context_blocks - replay_blocks, 0)
    bytes_per_block = (
        block_size * 2 * profile.num_kv_heads * d_head * profile.dtype_bytes
    )
    bytes_to_transfer = io_blocks * bytes_per_block

    t_transfer = 0.0
    if profile.pcie_bandwidth_h2d > 0:
        t_transfer = bytes_to_transfer / profile.pcie_bandwidth_h2d

    return t_compute, t_transfer


def solve_optimal_replay_blocks(
    profile: TightLLMProfileData,
    total_replayable_blocks: int,
    total_context_blocks: int,
    total_scheduled_tokens: int,
    avg_context_len: int,
    block_size: int,
) -> tuple[int, float, float]:
    """Solve the ILP for optimal Nr (replay_blocks).

    TightLLM §IV-B, Eq 2:
        min_{Nr}  max( T_attn(Nr) + T_ffn(Nr),  T_cache(Nr) )
        s.t.  0 ≤ Nr ≤ total_replayable_blocks,  Nr ∈ Z

    Because MFU is a non-linear function of Nr (paper Fig 7), the
    problem is solved by exhaustive enumeration over the integer
    feasible region — the standard approach for 1-D integer programs
    and equivalent to using an ILP solver (Gurobi [36] in the paper).

    Returns:
        (optimal_replay_blocks, t_compute, t_transfer)
    """
    if total_replayable_blocks <= 0:
        return 0, 0.0, 0.0

    best_rb = 0
    best_cost = float("inf")
    best_tc = 0.0
    best_tt = 0.0

    for nr in range(total_replayable_blocks + 1):
        tc, tt = _compute_times(
            nr,
            profile=profile,
            total_context_blocks=total_context_blocks,
            total_scheduled_tokens=total_scheduled_tokens,
            avg_context_len=avg_context_len,
            block_size=block_size,
        )
        cost = max(tc, tt)
        if cost < best_cost:
            best_cost = cost
            best_rb = nr
            best_tc = tc
            best_tt = tt

    return best_rb, best_tc, best_tt


# ---------------------------------------------------------------------------
# Replay plan provider
# ---------------------------------------------------------------------------


@dataclass
class TightLLMReplayPlanProvider:
    """TightLLM ILP-based replay plan provider.

    At each step's ``begin_step()``, uses the offline-profiled timing model
    to analytically compute the optimal replay block budget.  Then
    ``get_layer_plan()`` distributes the budget across requests and layers
    using the same infrastructure as the feedback planner.

    Config fields on RunKVOffloadConfig:
        layer_recompute_planner = "tightllm"
        tightllm_profile_path  = "/path/to/profile.json"
    """

    profile: TightLLMProfileData
    io_prefix_blocks: list[int]

    # Optional: combine ILP with feedback correction.
    # When enabled, the ILP prediction is used as initial budget, and the
    # runtime imbalance signal is used to apply a small additive correction.
    enable_feedback_correction: bool = False
    feedback_correction_gain: float = 0.15  # blocks per ms of imbalance
    feedback_correction_max: int = 4  # max |correction| in blocks

    # Fallback static provider (used when no replayable blocks)
    static_provider: StaticReplayPlanProvider = field(init=False)

    # Current step state
    _ilp_budget_blocks: int = field(init=False, default=0)
    _feedback_correction_blocks: int = field(init=False, default=0)
    _current_budget_blocks: int = field(init=False, default=0)
    _current_replayable_per_req: np.ndarray | None = field(init=False, default=None)
    _current_total_replayable: int = field(init=False, default=0)
    _step_count: int = field(init=False, default=0)

    # Diagnostics
    _last_t_compute: float = field(init=False, default=0.0)
    _last_t_transfer: float = field(init=False, default=0.0)
    _last_ilp_optimal: int = field(init=False, default=0)

    # Imbalance history — accumulated for all layers, for post-run statistics.
    # Separate lists for decode-only steps vs steps containing prefill requests.
    _imbalance_history: list[float] = field(init=False, default_factory=list)
    _imbalance_decode_only: list[float] = field(init=False, default_factory=list)
    _imbalance_has_prefill: list[float] = field(init=False, default_factory=list)
    _current_step_is_decode_only: bool = field(init=False, default=True)
    # Per-step ILP budget history (one entry per step)
    _budget_history: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.static_provider = StaticReplayPlanProvider(
            io_prefix_blocks=list(self.io_prefix_blocks)
        )

    # ---- ReplayPlanProvider protocol ----

    def begin_step(self, **metadata: Any) -> None:
        _nvtx.range_push("tightllm:begin_step")
        self._step_count += 1
        self._feedback_correction_blocks = 0  # reset per-step correction

        computed_lens = _coerce_int32_array(
            "computed_lens",
            metadata.get("computed_lens", np.zeros(0, dtype=np.int32)),
        )
        scheduled_lens = _coerce_int32_array(
            "scheduled_lens",
            metadata.get("scheduled_lens", np.zeros(0, dtype=np.int32)),
        )
        num_blocks_per_row = _coerce_int32_array(
            "num_blocks_per_row",
            metadata.get("num_blocks_per_row", np.zeros(0, dtype=np.int32)),
        )
        block_size = int(metadata.get("block_size", 0))
        if block_size <= 0:
            _nvtx.range_pop()
            return

        num_reqs = len(computed_lens)

        # Phase detection: decode-only if all requests have computed_lens > 0
        # (i.e. no fresh prefill requests in this batch)
        if num_reqs > 0:
            self._current_step_is_decode_only = bool(
                np.all(computed_lens > 0)
            )
        else:
            self._current_step_is_decode_only = True

        if num_reqs == 0:
            self._current_budget_blocks = 0
            self._current_replayable_per_req = np.zeros(0, dtype=np.int32)
            self._current_total_replayable = 0
            _nvtx.range_pop()
            return

        # Compute per-request replayable blocks
        computed_blocks = (
            (computed_lens.astype(np.int64) + block_size - 1) // block_size
        ).astype(np.int32)
        replayable_blocks_per_req = np.minimum(
            computed_blocks, num_blocks_per_row
        ).astype(np.int32)
        total_replayable_blocks = int(replayable_blocks_per_req.sum())

        self._current_replayable_per_req = replayable_blocks_per_req
        self._current_total_replayable = total_replayable_blocks

        if total_replayable_blocks == 0:
            self._current_budget_blocks = 0
            self._ilp_budget_blocks = 0
            _nvtx.range_pop()
            return

        total_context_blocks = int(computed_blocks.sum())
        total_scheduled_tokens = int(scheduled_lens.sum())
        avg_context_len = int(computed_lens.mean()) if num_reqs > 0 else 0

        # ---- ILP solve (TightLLM KV distributor) ----
        _nvtx.range_push("tightllm:ilp_solve")
        optimal_blocks, tc, tt = solve_optimal_replay_blocks(
            profile=self.profile,
            total_replayable_blocks=total_replayable_blocks,
            total_context_blocks=total_context_blocks,
            total_scheduled_tokens=total_scheduled_tokens,
            avg_context_len=avg_context_len,
            block_size=block_size,
        )
        _nvtx.range_pop()  # tightllm:ilp_solve

        self._ilp_budget_blocks = optimal_blocks
        self._current_budget_blocks = optimal_blocks
        self._last_t_compute = tc
        self._last_t_transfer = tt
        self._last_ilp_optimal = optimal_blocks
        self._budget_history.append(optimal_blocks)

        if self._step_count <= 3 or self._step_count % 50 == 0:
            logger.debug(
                "TightLLM ILP step=%d: optimal_blocks=%d/%d  "
                "T_compute=%.3fms  T_transfer=%.3fms  reqs=%d  "
                "avg_ctx=%d  sched_tok=%d",
                self._step_count,
                optimal_blocks,
                total_replayable_blocks,
                tc * 1000,
                tt * 1000,
                num_reqs,
                avg_context_len,
                total_scheduled_tokens,
            )
        _nvtx.range_pop()  # tightllm:begin_step

    def observe_layer_feedback(
        self, layer_idx: int, imbalance_ms: float
    ) -> None:
        """Record runtime imbalance and optionally apply additive correction.

        imbalance_ms > 0  → transfer is slower → should increase replay
        imbalance_ms < 0  → compute is slower  → should decrease replay

        The imbalance value is always recorded for post-run diagnostics,
        regardless of whether feedback correction is enabled.
        """
        _nvtx.range_push(f"tightllm:observe_feedback:L{layer_idx}")
        val = float(imbalance_ms)
        self._imbalance_history.append(val)

        # Route to decode-only or has-prefill bucket
        if self._current_step_is_decode_only:
            self._imbalance_decode_only.append(val)
        else:
            self._imbalance_has_prefill.append(val)

        if not self.enable_feedback_correction:
            _nvtx.range_pop()
            return

        correction = imbalance_ms * self.feedback_correction_gain
        correction = max(
            -self.feedback_correction_max,
            min(self.feedback_correction_max, correction),
        )
        self._feedback_correction_blocks += int(round(correction))

        new_budget = self._ilp_budget_blocks + self._feedback_correction_blocks
        new_budget = max(0, min(new_budget, self._current_total_replayable))
        self._current_budget_blocks = new_budget
        _nvtx.range_pop()  # tightllm:observe_feedback

    def get_layer_plan(
        self,
        layer_idx: int,
        num_reqs: int,
        computed_lens: np.ndarray,
        scheduled_lens: np.ndarray,
        logical_block_tables: np.ndarray,
        block_size: int,
        mapper_mapping: dict[int, int],
        prev_layer_plan: LayerReplayPlan | None,
    ) -> LayerReplayPlan:
        if (
            self._current_replayable_per_req is None
            or self._current_total_replayable == 0
        ):
            return self.static_provider.get_layer_plan(
                layer_idx=layer_idx,
                num_reqs=num_reqs,
                computed_lens=computed_lens,
                scheduled_lens=scheduled_lens,
                logical_block_tables=logical_block_tables,
                block_size=block_size,
                mapper_mapping=mapper_mapping,
                prev_layer_plan=prev_layer_plan,
            )

        # Fast path: once the plan reaches steady state (prev plan had no CPU
        # fills), all subsequent layers with the same budget produce an
        # identical plan.  Skip the full numpy rebuild when feedback correction
        # is disabled (budget is constant across layers within a step).
        if (
            not self.enable_feedback_correction
            and prev_layer_plan is not None
            and prev_layer_plan.cpu_fill_token_count == 0
        ):
            return prev_layer_plan

        # Distribute ILP-optimal budget across requests
        replayable = self._current_replayable_per_req[:num_reqs]
        allocated = _allocate_budget_to_requests(
            budget_blocks=self._current_budget_blocks,
            replayable_blocks_per_req=replayable,
        )

        # Convert to per-request replay start tokens
        computed_lens_i32 = _coerce_int32_array("computed_lens", computed_lens)
        computed_blocks = (
            (computed_lens_i32.astype(np.int64) + block_size - 1) // block_size
        ).astype(np.int32)
        replay_start_blocks = np.maximum(computed_blocks - allocated, 0)
        desired_replay_start_tokens = (
            replay_start_blocks.astype(np.int64) * block_size
        ).astype(np.int32)

        return compute_layer_replay_plan_for_layer(
            layer_idx=layer_idx,
            desired_replay_start_tokens=desired_replay_start_tokens,
            computed_lens=computed_lens,
            scheduled_lens=scheduled_lens,
            logical_block_tables=logical_block_tables,
            block_size=block_size,
            mapper_mapping=mapper_mapping,
            prev_layer_plan=prev_layer_plan,
        )

    # ---- Diagnostics ----

    @staticmethod
    def _phase_stats(hist: list[float], label: str) -> dict[str, Any]:
        """Compute summary stats for a single phase bucket."""
        if not hist:
            return {f"{label}_count": 0}
        abs_hist = [abs(v) for v in hist]
        return {
            f"{label}_count": len(hist),
            f"{label}_mean_ms": statistics.mean(hist),
            f"{label}_stdev_ms": (
                statistics.stdev(hist) if len(hist) >= 2 else 0.0
            ),
            f"{label}_abs_mean_ms": statistics.mean(abs_hist),
            f"{label}_abs_max_ms": max(abs_hist),
            f"{label}_median_ms": statistics.median(hist),
            f"{label}_p95_ms": sorted(abs_hist)[int(len(abs_hist) * 0.95)],
            f"{label}_positive_ratio": (
                sum(1 for v in hist if v > 0) / len(hist)
            ),
        }

    def get_imbalance_stats(self) -> dict[str, Any]:
        """Return summary statistics of accumulated imbalance observations.

        Designed to be called once after inference completes.
        Includes overall stats plus decode-only and has-prefill breakdowns.
        """
        hist = self._imbalance_history
        if not hist:
            return {"count": 0}
        abs_hist = [abs(v) for v in hist]
        result: dict[str, Any] = {
            "provider": "TightLLMReplayPlanProvider",
            "count": len(hist),
            "mean_ms": statistics.mean(hist),
            "stdev_ms": statistics.stdev(hist) if len(hist) >= 2 else 0.0,
            "abs_mean_ms": statistics.mean(abs_hist),
            "abs_max_ms": max(abs_hist),
            "median_ms": statistics.median(hist),
            "p95_ms": sorted(abs_hist)[int(len(abs_hist) * 0.95)],
            "positive_ratio": sum(1 for v in hist if v > 0) / len(hist),
            "budget_mean": (
                statistics.mean(self._budget_history)
                if self._budget_history
                else 0.0
            ),
            "budget_stdev": (
                statistics.stdev(self._budget_history)
                if len(self._budget_history) >= 2
                else 0.0
            ),
        }
        # Phase breakdowns
        result.update(self._phase_stats(self._imbalance_decode_only, "decode"))
        result.update(self._phase_stats(self._imbalance_has_prefill, "prefill"))
        return result

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "provider": "TightLLMReplayPlanProvider",
            "step_count": self._step_count,
            "ilp_budget_blocks": self._ilp_budget_blocks,
            "feedback_correction_blocks": self._feedback_correction_blocks,
            "current_budget_blocks": self._current_budget_blocks,
            "current_total_replayable": self._current_total_replayable,
            "last_t_compute_ms": self._last_t_compute * 1000,
            "last_t_transfer_ms": self._last_t_transfer * 1000,
            "enable_feedback_correction": self.enable_feedback_correction,
        }
