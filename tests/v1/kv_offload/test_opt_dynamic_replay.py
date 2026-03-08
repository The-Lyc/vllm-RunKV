# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.forward_context import ForwardContext
from vllm.v1.worker.opt_dynamic_replay import (
    LayerReplayPlan,
    OPTDynamicReplayRuntime,
    ReplayPlanProvider,
)


class _DummyReplayPlanProvider:
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
        del (
            layer_idx,
            num_reqs,
            computed_lens,
            scheduled_lens,
            logical_block_tables,
            block_size,
            mapper_mapping,
            prev_layer_plan,
        )
        raise NotImplementedError


def _make_layer_plan() -> LayerReplayPlan:
    return LayerReplayPlan(
        kv_replay_start_per_req=np.array([0, 4], dtype=np.int32),
        computed_lens_per_req=np.array([8, 12], dtype=np.int32),
        prev_gpu_start_per_req=np.array([8, 12], dtype=np.int32),
        cpu_fill_token_count=3,
        gpu_reuse_token_count=5,
        replay_token_count=8,
        scheduled_token_count=2,
        num_actual_tokens=10,
        max_query_len=6,
        query_start_loc=torch.tensor([0, 4, 10], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64),
        combined_replay_indices=torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64
        ),
        combined_scheduled_indices=torch.tensor([8, 9], dtype=torch.int64),
        cpu_fill_positions=np.array([4, 5, 6], dtype=np.int32),
        cpu_fill_logical_ids=np.array([10, 10, 11], dtype=np.int32),
        cpu_fill_block_offsets=np.array([0, 1, 0], dtype=np.int32),
        gpu_reuse_slice_per_req=[(0, 3), (3, 5)],
    )


def test_opt_dynamic_replay_runtime_initializes_empty_per_layer_state() -> None:
    provider: ReplayPlanProvider = _DummyReplayPlanProvider()
    runtime = OPTDynamicReplayRuntime(
        num_layers=3,
        cpu_hs_store=torch.empty(2, 4, 8),
        replay_plan_provider=provider,
    )

    assert runtime._layer_plans == [None, None, None]
    assert runtime._per_layer_attn_metadata == [None, None, None]
    assert runtime.current_layer_plan(1) is None


def test_opt_dynamic_replay_runtime_stores_layer_plan_and_metadata() -> None:
    runtime = OPTDynamicReplayRuntime(
        num_layers=2,
        cpu_hs_store=torch.empty(1, 2, 3),
        replay_plan_provider=_DummyReplayPlanProvider(),
    )
    plan = _make_layer_plan()
    metadata = {"layer_1": object()}

    runtime.set_layer_plan(1, plan)
    runtime.set_layer_metadata(1, metadata)

    assert runtime.get_layer_plan(1) is plan
    assert runtime.get_layer_metadata(1) is metadata
    assert runtime.current_layer_plan(0) is None
    assert runtime.current_layer_metadata(0) is None
    assert runtime._per_layer_attn_metadata[1] is metadata


def test_opt_dynamic_replay_runtime_rejects_get_on_unset_layer() -> None:
    runtime = OPTDynamicReplayRuntime(
        num_layers=1,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
    )

    with pytest.raises(AssertionError):
        runtime.get_layer_plan(0)


def test_forward_context_accepts_dynamic_replay_runtime() -> None:
    runtime = OPTDynamicReplayRuntime(
        num_layers=1,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
    )
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
        layer_recompute_runtime=runtime,
    )

    assert forward_context.layer_recompute_runtime is runtime
