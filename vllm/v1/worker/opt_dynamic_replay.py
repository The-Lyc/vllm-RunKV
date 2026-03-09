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


@runtime_checkable
class ReplayPlanProvider(Protocol):
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

    scheduled_token_count = int(scheduled_lens_i32.sum())
    replay_token_count = int(replay_lens_per_req.sum())
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
