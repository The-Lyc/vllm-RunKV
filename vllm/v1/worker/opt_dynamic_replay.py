# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import statistics as _statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.cuda.nvtx as _nvtx


@dataclass
class LayerReplayPlan:
    kv_replay_start_per_req: np.ndarray
    computed_lens_per_req: np.ndarray
    prev_gpu_start_per_req: np.ndarray
    replay_blocks_per_req: np.ndarray
    replay_block_count: int
    skip_logical_block_ids: np.ndarray
    per_req_replay_block_ranges: np.ndarray
    cpu_fill_token_count: int
    gpu_reuse_token_count: int
    replay_token_count: int
    scheduled_token_count: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    combined_replay_indices: torch.Tensor
    combined_scheduled_indices: torch.Tensor
    cpu_fill_positions: np.ndarray
    cpu_fill_logical_ids: np.ndarray
    cpu_fill_block_offsets: np.ndarray
    gpu_reuse_slice_per_req: list[tuple[int, int]]


@dataclass
class FeedbackPlannerBatchFingerprint:
    """Lightweight step summary used to decide whether cross-step state can
    still be trusted for the next batch.

    This intentionally stores only a compact fingerprint instead of the raw
    step metadata. The authoritative per-step metadata remains owned by
    LayerRecomputeManager.
    """

    num_reqs: int
    req_ids: tuple[str, ...]
    total_computed_tokens: int
    total_scheduled_tokens: int
    total_replayable_blocks: int


@dataclass
class FeedbackPlannerProbeState:
    """Controller-side probe bookkeeping.

    Later steps will use this state to remember whether the planner recently
    perturbed the replay budget in order to estimate a local
    budget->imbalance slope.
    """

    active: bool = False
    reference_budget_blocks: int | None = None
    reference_imbalance_ms: float | None = None


@dataclass
class FeedbackPlannerControllerState:
    """Cross-step planner state that survives across execute_model steps.

    This is the state we actually want to carry from one step to the next.
    It does not duplicate raw token/block metadata already owned by
    LayerRecomputeManager. Instead it keeps only controller variables and a
    compact fingerprint of the last batch they were derived from.
    """

    global_budget_blocks: int = 0
    estimated_local_gain: float | None = None
    last_budget_blocks: int | None = None
    last_imbalance_ms: float | None = None
    probe_state: FeedbackPlannerProbeState = field(
        default_factory=FeedbackPlannerProbeState
    )
    reinit_generation: int = 0
    step_batch_fingerprint: FeedbackPlannerBatchFingerprint | None = None


@dataclass
class FeedbackPlannerStepSummary:
    """Planner-owned summary derived from the current step metadata.

    The manager still owns the full logical block tables and token-level step
    metadata. The planner only keeps the compact per-request quantities needed
    for later budget/allocation logic.
    """

    replayable_blocks_per_req: np.ndarray
    total_replayable_blocks: int


@dataclass
class FeedbackControllerLayerUpdate:
    """Per-layer record of the controller budget update from feedback.

    Captures a snapshot of the budget before/after the update, the imbalance
    that triggered it, and the controller parameters used.  These records are
    kept in memory by the provider and optionally forwarded to the profiler
    when debug or profiling output is enabled.
    """

    budget_before: int
    budget_after: int
    imbalance_ms: float
    gain_used: float | None = None
    raw_delta: float | None = None
    clipped_delta: float | None = None
    action: str = "deadband"

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "imbalance_ms": self.imbalance_ms,
            "gain_used": self.gain_used,
            "raw_delta": self.raw_delta,
            "clipped_delta": self.clipped_delta,
            "action": self.action,
        }


@runtime_checkable
class ReplayPlanProvider(Protocol):
    def begin_step(self, **metadata: Any) -> None: ...

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
    ) -> LayerReplayPlan: ...


def _coerce_int32_array(name: str, values: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    # coerce inputs such as computed_lens into 1D, non-negative np arrays
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={array.shape}.")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array.astype(np.int32, copy=False)


def _align_replay_start_tokens(
    desired_replay_start_tokens: int | np.ndarray,
    computed_lens: np.ndarray,
    block_size: int,
) -> np.ndarray:
    desired = np.asarray(desired_replay_start_tokens, dtype=np.int64)
    if desired.ndim == 0:
        desired = np.full(
            computed_lens.shape,
            int(desired.item()),
            dtype=np.int64,
        )
    elif desired.shape != computed_lens.shape:
        raise ValueError(
            "desired_replay_start_tokens must be a scalar or match computed_lens "
            f"shape {computed_lens.shape}, got {desired.shape}."
        )

    clipped = np.minimum(desired, computed_lens.astype(np.int64))
    clipped = np.maximum(clipped, 0)
    aligned = ((clipped // block_size) * block_size).astype(np.int64)
    # Preserve the exact computed_len when the caller requests a replay start
    # at or beyond the end of the already-computed prefix. This represents
    # "no replay" for the request; rounding down to the previous block would
    # incorrectly force replay of the current partial block.
    no_replay_mask = clipped >= computed_lens.astype(np.int64)
    aligned[no_replay_mask] = computed_lens.astype(np.int64)[no_replay_mask]
    return aligned.astype(np.int32)


def _lookup_slot_and_block(
    *,
    req_idx: int,
    position: int,
    logical_block_tables: np.ndarray,
    block_size: int,
    mapper_mapping: Mapping[int, int],
) -> tuple[int, int, int]:
    block_idx = position // block_size
    if block_idx >= logical_block_tables.shape[1]:
        raise ValueError(
            "logical_block_tables does not contain enough blocks for "
            f"req_idx={req_idx}, position={position}, block_idx={block_idx}."
        )

    logical_id = int(logical_block_tables[req_idx, block_idx])
    if logical_id < 0:
        raise ValueError(
            "logical_block_tables contains an invalid block id for "
            f"req_idx={req_idx}, position={position}, block_idx={block_idx}."
        )

    if logical_id not in mapper_mapping:
        raise KeyError(f"logical block id {logical_id} is missing from mapper_mapping.")

    block_offset = position % block_size
    slot = int(mapper_mapping[logical_id]) * block_size + block_offset
    return logical_id, block_offset, slot


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compute_layer_replay_plan_for_layer(
    *,
    layer_idx: int,
    desired_replay_start_tokens: int | np.ndarray,
    computed_lens: np.ndarray,
    scheduled_lens: np.ndarray,
    logical_block_tables: np.ndarray,
    block_size: int,
    mapper_mapping: Mapping[int, int],
    prev_layer_plan: LayerReplayPlan | None,
) -> LayerReplayPlan:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    computed_lens_i32 = _coerce_int32_array("computed_lens", computed_lens)
    scheduled_lens_i32 = _coerce_int32_array("scheduled_lens", scheduled_lens)
    if computed_lens_i32.shape != scheduled_lens_i32.shape:
        raise ValueError(
            "computed_lens and scheduled_lens must have the same shape, got "
            f"{computed_lens_i32.shape} and {scheduled_lens_i32.shape}."
        )

    num_reqs = computed_lens_i32.shape[0]
    if logical_block_tables.shape[0] != num_reqs:
        raise ValueError(
            "logical_block_tables row count must match num_reqs, got "
            f"{logical_block_tables.shape[0]} and {num_reqs}."
        )

    kv_replay_start_per_req = _align_replay_start_tokens(
        desired_replay_start_tokens=desired_replay_start_tokens,
        computed_lens=computed_lens_i32,
        block_size=block_size,
    )

    if prev_layer_plan is None:
        prev_gpu_start_per_req = computed_lens_i32.copy()
    else:
        prev_gpu_start_per_req = np.minimum(
            np.maximum(
                prev_layer_plan.kv_replay_start_per_req.astype(np.int64),
                0,
            ),
            computed_lens_i32.astype(np.int64),
        ).astype(np.int32)

    replay_lens_per_req = (
        computed_lens_i32.astype(np.int64) - kv_replay_start_per_req.astype(np.int64)
    ).astype(np.int32)
    cpu_fill_end_per_req = np.minimum(prev_gpu_start_per_req, computed_lens_i32)
    cpu_fill_lens_per_req = np.maximum(
        cpu_fill_end_per_req.astype(np.int64)
        - kv_replay_start_per_req.astype(np.int64),
        0,
    ).astype(np.int32)
    gpu_reuse_start_per_req = np.maximum(
        kv_replay_start_per_req, prev_gpu_start_per_req
    )
    gpu_reuse_lens_per_req = np.maximum(
        computed_lens_i32.astype(np.int64) - gpu_reuse_start_per_req.astype(np.int64),
        0,
    ).astype(np.int32)

    if not np.array_equal(
        replay_lens_per_req,
        cpu_fill_lens_per_req + gpu_reuse_lens_per_req,
    ):
        raise AssertionError(
            f"layer_idx={layer_idx} replay length decomposition is inconsistent."
        )

    cpu_fill_positions: list[int] = []
    cpu_fill_logical_ids: list[int] = []
    cpu_fill_block_offsets: list[int] = []
    replay_blocks_per_req: list[int] = []
    skip_logical_block_ids: list[int] = []
    per_req_replay_block_ranges: list[tuple[int, int]] = []
    slot_mapping_values: list[int] = []
    combined_replay_indices: list[int] = []
    combined_scheduled_indices: list[int] = []
    gpu_reuse_slice_per_req: list[tuple[int, int]] = []
    query_start_loc = [0]

    prev_replay_prefix = 0
    combined_prefix = 0
    for req_idx in range(num_reqs):
        replay_start = int(kv_replay_start_per_req[req_idx])
        computed_len = int(computed_lens_i32[req_idx])
        scheduled_len = int(scheduled_lens_i32[req_idx])
        prev_gpu_start = int(prev_gpu_start_per_req[req_idx])
        cpu_fill_end = min(prev_gpu_start, computed_len)
        replay_len = int(replay_lens_per_req[req_idx])
        gpu_reuse_len = int(gpu_reuse_lens_per_req[req_idx])
        prev_replay_len = max(computed_len - prev_gpu_start, 0)
        computed_block_end = _ceil_div(computed_len, block_size)
        if replay_len == 0:
            replay_start_block = computed_block_end
        else:
            replay_start_block = replay_start // block_size

        per_req_replay_block_ranges.append((replay_start_block, computed_block_end))
        replay_blocks_per_req.append(max(computed_block_end - replay_start_block, 0))
        for block_idx in range(replay_start_block, computed_block_end):
            logical_id = int(logical_block_tables[req_idx, block_idx])
            if logical_id >= 0:
                skip_logical_block_ids.append(logical_id)

        gpu_reuse_start = max(replay_start, prev_gpu_start)
        gpu_slice_start = prev_replay_prefix + max(gpu_reuse_start - prev_gpu_start, 0)
        gpu_reuse_slice_per_req.append(
            (gpu_slice_start, gpu_slice_start + gpu_reuse_len)
        )
        prev_replay_prefix += prev_replay_len

        replay_segment_end = combined_prefix + replay_len
        combined_replay_indices.extend(range(combined_prefix, replay_segment_end))
        combined_scheduled_indices.extend(
            range(replay_segment_end, replay_segment_end + scheduled_len)
        )

        for position in range(replay_start, computed_len):
            logical_id, block_offset, slot = _lookup_slot_and_block(
                req_idx=req_idx,
                position=position,
                logical_block_tables=logical_block_tables,
                block_size=block_size,
                mapper_mapping=mapper_mapping,
            )
            slot_mapping_values.append(slot)
            if position < cpu_fill_end:
                cpu_fill_positions.append(position)
                cpu_fill_logical_ids.append(logical_id)
                cpu_fill_block_offsets.append(block_offset)

        for position in range(computed_len, computed_len + scheduled_len):
            _, _, slot = _lookup_slot_and_block(
                req_idx=req_idx,
                position=position,
                logical_block_tables=logical_block_tables,
                block_size=block_size,
                mapper_mapping=mapper_mapping,
            )
            slot_mapping_values.append(slot)

        combined_prefix = replay_segment_end + scheduled_len
        query_start_loc.append(combined_prefix)

    query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32)
    slot_mapping_tensor = torch.tensor(slot_mapping_values, dtype=torch.int64)
    combined_replay_indices_tensor = torch.tensor(
        combined_replay_indices, dtype=torch.int64
    )
    combined_scheduled_indices_tensor = torch.tensor(
        combined_scheduled_indices, dtype=torch.int64
    )
    replay_blocks_per_req_array = np.asarray(replay_blocks_per_req, dtype=np.int32)
    skip_logical_block_ids_array = np.asarray(
        sorted(set(skip_logical_block_ids)),
        dtype=np.int32,
    )
    per_req_replay_block_ranges_array = np.asarray(
        per_req_replay_block_ranges,
        dtype=np.int32,
    ).reshape(num_reqs, 2)

    scheduled_token_count = int(scheduled_lens_i32.sum())
    replay_token_count = int(replay_lens_per_req.sum())
    replay_block_count = int(replay_blocks_per_req_array.sum())
    num_actual_tokens = replay_token_count + scheduled_token_count
    if slot_mapping_tensor.shape[0] != num_actual_tokens:
        raise AssertionError(
            f"layer_idx={layer_idx} built {slot_mapping_tensor.shape[0]} slots for "
            f"{num_actual_tokens} actual tokens."
        )

    return LayerReplayPlan(
        kv_replay_start_per_req=kv_replay_start_per_req,
        computed_lens_per_req=computed_lens_i32,
        prev_gpu_start_per_req=prev_gpu_start_per_req,
        replay_blocks_per_req=replay_blocks_per_req_array,
        replay_block_count=replay_block_count,
        skip_logical_block_ids=skip_logical_block_ids_array,
        per_req_replay_block_ranges=per_req_replay_block_ranges_array,
        cpu_fill_token_count=len(cpu_fill_positions),
        gpu_reuse_token_count=int(gpu_reuse_lens_per_req.sum()),
        replay_token_count=replay_token_count,
        scheduled_token_count=scheduled_token_count,
        num_actual_tokens=num_actual_tokens,
        max_query_len=int((replay_lens_per_req + scheduled_lens_i32).max(initial=0)),
        query_start_loc=query_start_loc_tensor,
        slot_mapping=slot_mapping_tensor,
        combined_replay_indices=combined_replay_indices_tensor,
        combined_scheduled_indices=combined_scheduled_indices_tensor,
        cpu_fill_positions=np.asarray(cpu_fill_positions, dtype=np.int32),
        cpu_fill_logical_ids=np.asarray(cpu_fill_logical_ids, dtype=np.int32),
        cpu_fill_block_offsets=np.asarray(cpu_fill_block_offsets, dtype=np.int32),
        gpu_reuse_slice_per_req=gpu_reuse_slice_per_req,
    )


def _allocate_budget_to_requests(
    budget_blocks: int,
    replayable_blocks_per_req: np.ndarray,
) -> np.ndarray:
    """Short-request-first greedy budget allocator.

    Distributes *budget_blocks* replay blocks across requests, prioritising
    requests with fewer replayable blocks (ascending order, stable tie-break
    by request index).  Each request receives a contiguous suffix allocation
    of at most its full replayable block count.

    Args:
        budget_blocks: Total replay block budget to distribute.
        replayable_blocks_per_req: Per-request upper bound on allocatable
            replay blocks (1-D int array, length = num_reqs).

    Returns:
        allocated_blocks_per_req: Per-request allocation in the original
            request order (same length as *replayable_blocks_per_req*).
    """
    num_reqs = len(replayable_blocks_per_req)
    allocated = np.zeros(num_reqs, dtype=np.int32)
    if budget_blocks <= 0 or num_reqs == 0:
        return allocated

    # Stable sort ensures deterministic tie-breaking by original req_idx.
    sorted_indices = np.argsort(replayable_blocks_per_req, kind="stable")

    remaining = int(budget_blocks)
    for idx in sorted_indices:
        if remaining <= 0:
            break
        cap = int(replayable_blocks_per_req[idx])
        alloc = min(remaining, cap)
        allocated[idx] = alloc
        remaining -= alloc

    return allocated


@dataclass
class StaticReplayPlanProvider:
    io_prefix_blocks: list[int]

    def begin_step(self, **metadata: Any) -> None:
        del metadata

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
        if num_reqs != len(computed_lens):
            raise ValueError(
                f"num_reqs={num_reqs} does not match computed_lens "
                f"length={len(computed_lens)}."
            )

        prefix_blocks = (
            int(self.io_prefix_blocks[layer_idx])
            if layer_idx < len(self.io_prefix_blocks)
            else 0
        )
        return compute_layer_replay_plan_for_layer(
            layer_idx=layer_idx,
            desired_replay_start_tokens=prefix_blocks * int(block_size),
            computed_lens=computed_lens,
            scheduled_lens=scheduled_lens,
            logical_block_tables=logical_block_tables,
            block_size=block_size,
            mapper_mapping=mapper_mapping,
            prev_layer_plan=prev_layer_plan,
        )


@dataclass
class FeedbackReplayPlanProvider:
    io_prefix_blocks: list[int]
    dry_run: bool = False

    # --- Controller hyperparameters ---
    # Imbalance values within [-deadband_ms, +deadband_ms] are considered
    # balanced and do not trigger a budget update.
    deadband_ms: float = 0.5
    # Multiplicative damping factor applied to the raw Newton step before
    # clipping.  Values in (0, 1) make the controller more conservative;
    # 1.0 means no damping.
    damping: float = 0.3
    # Maximum absolute budget change (in blocks) per single layer update.
    # Prevents the controller from making excessively large jumps.
    max_step_blocks: int = 4
    # Fallback value of d(imbalance_ms)/d(budget_blocks) used when no
    # secant-based gain estimate is available yet.  Negative because
    # increasing budget adds compute time and thus *decreases* imbalance.
    default_gain: float = -0.1

    static_provider: StaticReplayPlanProvider = field(init=False)
    controller_state: FeedbackPlannerControllerState = field(
        init=False, default_factory=FeedbackPlannerControllerState
    )
    current_step_summary: FeedbackPlannerStepSummary | None = field(
        init=False, default=None
    )
    last_feedback_by_layer: dict[int, float] = field(init=False, default_factory=dict)
    # Per-layer controller update records for the *current* step.
    # Cleared at the start of each step.  Used for debug snapshots and
    # forwarded to the profiler when profiling output is enabled.
    _layer_controller_updates: dict[int, FeedbackControllerLayerUpdate] = field(
        init=False, default_factory=dict
    )
    begin_step_count: int = field(init=False, default=0)
    observe_feedback_count: int = field(init=False, default=0)
    # Whether the budget has been seeded to total_replayable_blocks at least
    # once.  Deferred until the first step where replayable blocks > 0 so
    # that prefill steps (computed_lens == 0) do not seed the budget to 0.
    _budget_seeded: bool = field(init=False, default=False)

    # Imbalance history — accumulated for all layers, for post-run statistics
    _imbalance_history: list[float] = field(init=False, default_factory=list)
    # Per-step budget history
    _budget_history: list[int] = field(init=False, default_factory=list)
    # Phase-separated imbalance: decode-only vs steps containing prefill
    _imbalance_decode_only: list[float] = field(init=False, default_factory=list)
    _imbalance_has_prefill: list[float] = field(init=False, default_factory=list)
    _current_step_is_decode_only: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.static_provider = StaticReplayPlanProvider(
            io_prefix_blocks=list(self.io_prefix_blocks)
        )

    def begin_step(self, **metadata: Any) -> None:
        # Step-boundary metadata is still owned by LayerRecomputeManager.
        # The planner only derives compact summaries from it and stores the
        # cross-step controller state that later steps will evolve.
        _nvtx.range_push("feedback:begin_step")
        self.begin_step_count += 1
        req_ids = tuple(str(req_id) for req_id in metadata.get("req_ids", ()))
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
            raise ValueError(
                "FeedbackReplayPlanProvider.begin_step requires block_size."
            )

        # Phase detection: decode-only if all requests have computed_lens > 0
        num_reqs = len(computed_lens)
        if num_reqs > 0:
            self._current_step_is_decode_only = bool(np.all(computed_lens > 0))
        else:
            self._current_step_is_decode_only = True

        replayable_blocks_per_req = np.minimum(
            (computed_lens.astype(np.int64) + block_size - 1) // block_size,
            num_blocks_per_row,
        ).astype(np.int32)
        total_replayable_blocks = int(replayable_blocks_per_req.sum())
        self.current_step_summary = FeedbackPlannerStepSummary(
            replayable_blocks_per_req=replayable_blocks_per_req,
            total_replayable_blocks=total_replayable_blocks,
        )
        self.controller_state.step_batch_fingerprint = FeedbackPlannerBatchFingerprint(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            total_computed_tokens=int(computed_lens.sum()),
            total_scheduled_tokens=int(scheduled_lens.sum()),
            total_replayable_blocks=total_replayable_blocks,
        )

        # Budget initialisation / clamping.
        # We seed the budget at total_replayable_blocks on the first step
        # where replayable blocks > 0, so the controller can converge
        # downward from the maximum.  The first step is typically prefill
        # where computed_lens == 0 and replayable blocks == 0 — seeding
        # there would lock the budget at 0 before the controller has any
        # chance to act.
        cs = self.controller_state
        if not self._budget_seeded and total_replayable_blocks > 0:
            cs.global_budget_blocks = total_replayable_blocks
            self._budget_seeded = True
        else:
            cs.global_budget_blocks = max(
                0, min(cs.global_budget_blocks, total_replayable_blocks)
            )

        self._budget_history.append(cs.global_budget_blocks)
        self.last_feedback_by_layer.clear()
        self._layer_controller_updates.clear()
        _nvtx.range_pop()  # feedback:begin_step

    def observe_layer_feedback(self, layer_idx: int, imbalance_ms: float) -> None:
        """Update the controller state based on the observed layer imbalance.

        The controller uses a damped Newton / secant-style update:
          1. If the imbalance falls within the deadband, no budget change.
          2. Otherwise, estimate the local gain d(imbalance)/d(budget) via the
             secant rule when a prior (budget, imbalance) pair is available.
          3. Compute a raw Newton step: delta = -imbalance / gain.
          4. Apply damping and step-size clipping.
          5. Update ``global_budget_blocks`` and record the update snapshot.

        In the current (Step 8 / dry-run) phase the updated budget is *not*
        fed back into the actual replay plan; that happens in Step 10.
        """
        _nvtx.range_push(f"feedback:controller_update:L{layer_idx}")
        self.observe_feedback_count += 1
        imbalance_value = float(imbalance_ms)
        self._imbalance_history.append(imbalance_value)

        # Route to decode-only or has-prefill bucket
        if self._current_step_is_decode_only:
            self._imbalance_decode_only.append(imbalance_value)
        else:
            self._imbalance_has_prefill.append(imbalance_value)

        layer_key = int(layer_idx)
        self.last_feedback_by_layer[layer_key] = imbalance_value

        cs = self.controller_state
        budget_before = cs.global_budget_blocks

        # --- Deadband: don't react to small imbalance ---
        if abs(imbalance_value) < self.deadband_ms:
            self._layer_controller_updates[layer_key] = FeedbackControllerLayerUpdate(
                budget_before=budget_before,
                budget_after=budget_before,
                imbalance_ms=imbalance_value,
                action="deadband",
            )
            cs.last_imbalance_ms = imbalance_value
            _nvtx.range_pop()  # feedback:controller_update
            return

        # --- Secant-based gain estimation ---
        # When the budget actually changed between the last observation and now,
        # we can use the (delta_budget, delta_imbalance) pair to refine the
        # local gain estimate.  When the budget did NOT change, fall back to
        # ``default_gain`` so the controller always reacts to imbalance —
        # a stale ``estimated_local_gain`` from a noisy secant can otherwise
        # lock the controller in a zero-delta dead zone.
        gain: float | None = None
        if (
            cs.last_budget_blocks is not None
            and cs.last_imbalance_ms is not None
            and cs.last_budget_blocks != budget_before
        ):
            delta_b = budget_before - cs.last_budget_blocks
            delta_d = imbalance_value - cs.last_imbalance_ms
            if abs(delta_b) > 0:
                new_gain = delta_d / delta_b
                # Only accept the estimate when the magnitude is sensible.
                if abs(new_gain) > 1e-6:
                    gain = new_gain
                    cs.estimated_local_gain = gain

        # Use the freshly estimated secant gain when available; otherwise
        # always fall back to ``default_gain``.  We intentionally do NOT
        # reuse a stale ``estimated_local_gain`` here — the secant estimate
        # is only trustworthy in the same update where it was computed.
        effective_gain = gain if gain is not None else self.default_gain

        # --- Newton step with damping and clipping ---
        raw_delta = -imbalance_value / effective_gain
        clipped_delta = max(
            -self.max_step_blocks,
            min(self.max_step_blocks, self.damping * raw_delta),
        )
        new_budget = budget_before + int(round(clipped_delta))

        # Clamp to [0, total_replayable_blocks].
        max_budget = (
            self.current_step_summary.total_replayable_blocks
            if self.current_step_summary is not None
            else budget_before
        )
        new_budget = max(0, min(new_budget, max_budget))

        # --- Record the update ---
        self._layer_controller_updates[layer_key] = FeedbackControllerLayerUpdate(
            budget_before=budget_before,
            budget_after=new_budget,
            imbalance_ms=imbalance_value,
            gain_used=effective_gain,
            raw_delta=raw_delta,
            clipped_delta=clipped_delta,
            action="update",
        )

        cs.last_budget_blocks = budget_before
        cs.last_imbalance_ms = imbalance_value
        cs.global_budget_blocks = new_budget
        _nvtx.range_pop()  # feedback:controller_update

    def get_layer_controller_update(
        self,
        layer_idx: int,
    ) -> FeedbackControllerLayerUpdate | None:
        """Return the controller update record for *layer_idx* in the current
        step, or ``None`` if no feedback has been observed for that layer yet.
        """
        return self._layer_controller_updates.get(int(layer_idx))

    @staticmethod
    def _phase_stats(hist: list[float], label: str) -> dict[str, Any]:
        """Compute summary stats for a single phase bucket."""
        if not hist:
            return {f"{label}_count": 0}
        abs_hist = [abs(v) for v in hist]
        return {
            f"{label}_count": len(hist),
            f"{label}_mean_ms": _statistics.mean(hist),
            f"{label}_stdev_ms": (_statistics.stdev(hist) if len(hist) >= 2 else 0.0),
            f"{label}_abs_mean_ms": _statistics.mean(abs_hist),
            f"{label}_abs_max_ms": max(abs_hist),
            f"{label}_median_ms": _statistics.median(hist),
            f"{label}_p95_ms": sorted(abs_hist)[int(len(abs_hist) * 0.95)],
            f"{label}_positive_ratio": (sum(1 for v in hist if v > 0) / len(hist)),
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
            "provider": "FeedbackReplayPlanProvider",
            "count": len(hist),
            "mean_ms": _statistics.mean(hist),
            "stdev_ms": _statistics.stdev(hist) if len(hist) >= 2 else 0.0,
            "abs_mean_ms": _statistics.mean(abs_hist),
            "abs_max_ms": max(abs_hist),
            "median_ms": _statistics.median(hist),
            "p95_ms": sorted(abs_hist)[int(len(abs_hist) * 0.95)],
            "positive_ratio": sum(1 for v in hist if v > 0) / len(hist),
            "budget_mean": (
                _statistics.mean(self._budget_history) if self._budget_history else 0.0
            ),
            "budget_stdev": (
                _statistics.stdev(self._budget_history)
                if len(self._budget_history) >= 2
                else 0.0
            ),
        }
        # Phase breakdowns
        result.update(self._phase_stats(self._imbalance_decode_only, "decode"))
        result.update(self._phase_stats(self._imbalance_has_prefill, "prefill"))
        return result

    def get_debug_snapshot(self) -> dict[str, Any]:
        step_summary = None
        if self.current_step_summary is not None:
            step_summary = {
                "replayable_blocks_per_req": (
                    self.current_step_summary.replayable_blocks_per_req.tolist()
                ),
                "total_replayable_blocks": (
                    self.current_step_summary.total_replayable_blocks
                ),
            }
        fingerprint = self.controller_state.step_batch_fingerprint
        return {
            "provider": type(self).__name__,
            "dry_run": self.dry_run,
            "begin_step_count": self.begin_step_count,
            "observe_feedback_count": self.observe_feedback_count,
            "controller_state": {
                "global_budget_blocks": self.controller_state.global_budget_blocks,
                "estimated_local_gain": self.controller_state.estimated_local_gain,
                "last_budget_blocks": self.controller_state.last_budget_blocks,
                "last_imbalance_ms": self.controller_state.last_imbalance_ms,
                "probe_state": {
                    "active": self.controller_state.probe_state.active,
                    "reference_budget_blocks": (
                        self.controller_state.probe_state.reference_budget_blocks
                    ),
                    "reference_imbalance_ms": (
                        self.controller_state.probe_state.reference_imbalance_ms
                    ),
                },
                "reinit_generation": self.controller_state.reinit_generation,
                "step_batch_fingerprint": (
                    None
                    if fingerprint is None
                    else {
                        "num_reqs": fingerprint.num_reqs,
                        "req_ids": list(fingerprint.req_ids),
                        "total_computed_tokens": fingerprint.total_computed_tokens,
                        "total_scheduled_tokens": fingerprint.total_scheduled_tokens,
                        "total_replayable_blocks": fingerprint.total_replayable_blocks,
                    }
                ),
            },
            "current_step_summary": step_summary,
            "last_feedback_by_layer": dict(self.last_feedback_by_layer),
            "layer_controller_updates": {
                layer_idx: update.to_dict()
                for layer_idx, update in self._layer_controller_updates.items()
            },
        }

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
        # In dry-run mode or when no step summary is available, fall back to
        # the static provider so that the actual execution remains unchanged.
        if self.dry_run or self.current_step_summary is None:
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

        # --- Budget-based per-request allocation (Step 9) ---
        # Use the short-request-first greedy allocator to distribute the
        # controller's global budget across requests in the current batch.
        replayable = self.current_step_summary.replayable_blocks_per_req[:num_reqs]
        allocated = _allocate_budget_to_requests(
            budget_blocks=self.controller_state.global_budget_blocks,
            replayable_blocks_per_req=replayable,
        )

        # Convert per-request block allocation to per-request replay start
        # token positions.  The replay region is always a contiguous suffix:
        #   [replay_start_token, computed_len)
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


@dataclass
class RandomReplayPlanProvider:
    num_layers: int
    max_tokens: int | None = None
    max_blocks: int | None = None
    seed: int = 42
    _desired_start_tokens_by_block_size: dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.max_tokens is None and self.max_blocks is None:
            raise ValueError("Either max_tokens or max_blocks must be provided.")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative.")
        if self.max_blocks is not None and self.max_blocks < 0:
            raise ValueError("max_blocks must be non-negative.")

    def _get_random_start_tokens(self, block_size: int) -> np.ndarray:
        cached = self._desired_start_tokens_by_block_size.get(block_size)
        if cached is not None:
            return cached

        max_tokens = self.max_tokens
        if max_tokens is None:
            assert self.max_blocks is not None
            max_tokens = self.max_blocks * block_size

        rng = np.random.default_rng(self.seed)
        random_tokens = rng.integers(
            low=0,
            high=int(max_tokens) + 1,
            size=self.num_layers,
            dtype=np.int32,
        )
        self._desired_start_tokens_by_block_size[block_size] = random_tokens
        return random_tokens

    def begin_step(self, **metadata: Any) -> None:
        del metadata

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
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} is out of range for "
                f"num_layers={self.num_layers}."
            )
        if num_reqs != len(computed_lens):
            raise ValueError(
                f"num_reqs={num_reqs} does not match computed_lens "
                f"length={len(computed_lens)}."
            )

        desired_start_tokens = int(self._get_random_start_tokens(block_size)[layer_idx])
        return compute_layer_replay_plan_for_layer(
            layer_idx=layer_idx,
            desired_replay_start_tokens=desired_start_tokens,
            computed_lens=computed_lens,
            scheduled_lens=scheduled_lens,
            logical_block_tables=logical_block_tables,
            block_size=block_size,
            mapper_mapping=mapper_mapping,
            prev_layer_plan=prev_layer_plan,
        )


@dataclass
class OPTDynamicReplayRuntime:
    num_layers: int
    cpu_hs_store: torch.Tensor
    replay_plan_provider: ReplayPlanProvider
    layer_recompute_manager: Any | None = None
    scheduled_req_indices: np.ndarray | None = None
    scheduled_positions: np.ndarray | None = None
    _layer_plans: list[LayerReplayPlan | None] = field(init=False)
    _per_layer_attn_metadata: list[dict[str, Any] | None] = field(init=False)
    _step_anchor_event: torch.cuda.Event | None = field(init=False, default=None)
    _layer_end_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_start_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_load_ready_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_load_start_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_cpu_fill_ready_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_forward_start_events: list[torch.cuda.Event | None] = field(init=False)
    _layer_imbalance_ms: list[float | None] = field(init=False)

    def __post_init__(self) -> None:
        self._layer_plans = [None] * self.num_layers
        self._per_layer_attn_metadata = [None] * self.num_layers
        self._layer_end_events = [None] * self.num_layers
        self._layer_start_events = [None] * self.num_layers
        self._layer_load_ready_events = [None] * self.num_layers
        self._layer_load_start_events = [None] * self.num_layers
        self._layer_cpu_fill_ready_events = [None] * self.num_layers
        self._layer_forward_start_events = [None] * self.num_layers
        self._layer_imbalance_ms = [None] * self.num_layers

    def get_layer_plan(self, layer_idx: int) -> LayerReplayPlan:
        plan = self._layer_plans[layer_idx]
        assert plan is not None
        return plan

    def set_layer_plan(self, layer_idx: int, plan: LayerReplayPlan) -> None:
        self._layer_plans[layer_idx] = plan

    def set_layer_metadata(self, layer_idx: int, metadata: dict[str, Any]) -> None:
        self._per_layer_attn_metadata[layer_idx] = metadata

    def get_layer_metadata(self, layer_idx: int) -> dict[str, Any]:
        metadata = self._per_layer_attn_metadata[layer_idx]
        assert metadata is not None
        return metadata

    def current_layer_plan(self, layer_idx: int) -> LayerReplayPlan | None:
        return self._layer_plans[layer_idx]

    def current_layer_metadata(self, layer_idx: int) -> dict[str, Any] | None:
        return self._per_layer_attn_metadata[layer_idx]

    def set_layer_end_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Records the exact point where this layer's forward finishes on the
        # compute stream.
        self._layer_end_events[layer_idx] = event

    def get_layer_end_event(self, layer_idx: int) -> torch.cuda.Event | None:
        return self._layer_end_events[layer_idx]

    def set_layer_start_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Records the exact point where this layer's forward begins on the
        # compute stream (just before layer(hidden_states)).
        self._layer_start_events[layer_idx] = event

    def get_layer_start_event(self, layer_idx: int) -> torch.cuda.Event | None:
        return self._layer_start_events[layer_idx]

    def set_step_anchor_event(self, event: torch.cuda.Event | None) -> None:
        # Common time base for cross-stream timing. Later event timestamps are
        # converted into absolute times by measuring against this anchor.
        self._step_anchor_event = event

    def get_step_anchor_event(self) -> torch.cuda.Event | None:
        return self._step_anchor_event

    def set_layer_load_ready_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Raw producer-side completion time for KV block prefetch on load_stream.
        self._layer_load_ready_events[layer_idx] = event

    def get_layer_load_ready_event(self, layer_idx: int) -> torch.cuda.Event | None:
        return self._layer_load_ready_events[layer_idx]

    def set_layer_load_start_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Records the start of H2D DMA for this layer on load_stream.
        self._layer_load_start_events[layer_idx] = event

    def get_layer_load_start_event(self, layer_idx: int) -> torch.cuda.Event | None:
        return self._layer_load_start_events[layer_idx]

    def set_layer_cpu_fill_ready_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Raw producer-side completion time for CPU-fill H2D on the dedicated
        # H2D stream. This may be None when the layer has no CPU-fill replay.
        self._layer_cpu_fill_ready_events[layer_idx] = event

    def get_layer_cpu_fill_ready_event(
        self,
        layer_idx: int,
    ) -> torch.cuda.Event | None:
        return self._layer_cpu_fill_ready_events[layer_idx]

    def set_layer_forward_start_event(
        self,
        layer_idx: int,
        event: torch.cuda.Event | None,
    ) -> None:
        # Records the point right before layer(hidden_states) on the compute
        # stream — after cpu_fill sync and tensor scatter, so it measures
        # pure attention + FFN time (unlike layer_start which includes
        # replay assembly overhead).
        self._layer_forward_start_events[layer_idx] = event

    def get_layer_forward_start_event(self, layer_idx: int) -> torch.cuda.Event | None:
        return self._layer_forward_start_events[layer_idx]

    def set_layer_imbalance_ms(
        self, layer_idx: int, imbalance_ms: float | None
    ) -> None:
        self._layer_imbalance_ms[layer_idx] = (
            None if imbalance_ms is None else float(imbalance_ms)
        )

    def get_layer_imbalance_ms(self, layer_idx: int) -> float | None:
        return self._layer_imbalance_ms[layer_idx]

    def set_capture_token_metadata(
        self,
        *,
        req_indices: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        req_indices_np = np.asarray(req_indices, dtype=np.int64)
        positions_np = np.asarray(positions, dtype=np.int64)
        if req_indices_np.ndim != 1 or positions_np.ndim != 1:
            raise ValueError("req_indices and positions must both be 1D arrays.")
        if req_indices_np.shape != positions_np.shape:
            raise ValueError(
                "req_indices and positions must have the same shape, got "
                f"{req_indices_np.shape} and {positions_np.shape}."
            )
        self.scheduled_req_indices = req_indices_np
        self.scheduled_positions = positions_np

    def load_cpu_fill(self, layer_idx: int, plan: LayerReplayPlan) -> torch.Tensor:
        manager = self.layer_recompute_manager
        if manager is None:
            raise AssertionError("Dynamic replay runtime requires a layer manager.")

        prefetched = manager.sync_cpu_fill_h2d(layer_idx)
        if prefetched is not None:
            return prefetched

        return manager.load_cpu_fill_h2d(
            layer_idx=layer_idx,
            cpu_fill_positions=plan.cpu_fill_positions,
            cpu_fill_logical_ids=plan.cpu_fill_logical_ids,
            cpu_fill_block_offsets=plan.cpu_fill_block_offsets,
        )

    def capture_scheduled_layer_input(
        self,
        *,
        target_layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> None:
        if target_layer_idx >= self.num_layers:
            return

        manager = self.layer_recompute_manager
        if manager is None:
            raise AssertionError("Dynamic replay runtime requires a layer manager.")
        if self.scheduled_req_indices is None or self.scheduled_positions is None:
            raise AssertionError(
                "Dynamic replay runtime requires scheduled req_indices and "
                "positions before capture."
            )
        if hidden_states.shape[0] != self.scheduled_req_indices.shape[0]:
            raise ValueError(
                "scheduled hidden_states row count must match scheduled token "
                f"metadata, got {hidden_states.shape[0]} and "
                f"{self.scheduled_req_indices.shape[0]}."
            )

        manager.capture_layer_input_d2h(
            layer_idx=target_layer_idx,
            hidden_states=hidden_states,
            req_indices=self.scheduled_req_indices,
            positions=self.scheduled_positions,
        )
