# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass
class OPTDynamicReplayRuntime:
    num_layers: int
    cpu_hs_store: torch.Tensor
    replay_plan_provider: ReplayPlanProvider
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

    def current_layer_plan(self, layer_idx: int) -> LayerReplayPlan | None:
        return self._layer_plans[layer_idx]
