# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


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
    return ((clipped // block_size) * block_size).astype(np.int32)


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
        replay_start_block = replay_start // block_size
        computed_block_end = _ceil_div(computed_len, block_size)

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
    static_provider: StaticReplayPlanProvider = field(init=False)
    controller_state: FeedbackPlannerControllerState = field(
        init=False, default_factory=FeedbackPlannerControllerState
    )
    current_step_summary: FeedbackPlannerStepSummary | None = field(
        init=False, default=None
    )
    last_feedback_by_layer: dict[int, float] = field(init=False, default_factory=dict)
    begin_step_count: int = field(init=False, default=0)
    observe_feedback_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.static_provider = StaticReplayPlanProvider(
            io_prefix_blocks=list(self.io_prefix_blocks)
        )

    def begin_step(self, **metadata: Any) -> None:
        # Step-boundary metadata is still owned by LayerRecomputeManager.
        # The planner only derives compact summaries from it and stores the
        # cross-step controller state that later steps will evolve.
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
            raise ValueError(
                "FeedbackReplayPlanProvider.begin_step requires block_size."
            )

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
        self.last_feedback_by_layer.clear()

    def observe_layer_feedback(self, layer_idx: int, imbalance_ms: float) -> None:
        # TODO: incorporate the observed feedback into future replay plan decisions.
        # For now, we just record the feedback by layer.
        self.observe_feedback_count += 1
        imbalance_value = float(imbalance_ms)
        self.last_feedback_by_layer[int(layer_idx)] = imbalance_value
        self.controller_state.last_imbalance_ms = imbalance_value

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
        # TODO: use the recorded metadata and feedback to make dynamic decisions
        # about the replay plan(including replay budget and per-request allocation)
        # for the current layer. For now, we just delegate to a static provider
        # that uses fixed prefix block settings.
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

    def __post_init__(self) -> None:
        self._layer_plans = [None] * self.num_layers
        self._per_layer_attn_metadata = [None] * self.num_layers

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
