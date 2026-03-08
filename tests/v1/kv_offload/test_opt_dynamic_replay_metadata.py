# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.opt_dynamic_replay import (
    LayerReplayPlan,
    compute_layer_replay_plan_for_layer,
)


@dataclass
class _BuiltMetadata:
    common_attn_metadata: CommonAttentionMetadata
    build_index: int


class _DummyBuilder:
    def __init__(self) -> None:
        self.build_calls: list[CommonAttentionMetadata] = []

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> _BuiltMetadata:
        assert common_prefix_len == 0
        assert fast_build is False
        self.build_calls.append(common_attn_metadata)
        return _BuiltMetadata(
            common_attn_metadata=common_attn_metadata,
            build_index=len(self.build_calls),
        )


class _DummyAttnGroup:
    def __init__(self, layer_names: list[str], builder: _DummyBuilder) -> None:
        self.layer_names = layer_names
        self._builder = builder

    def get_metadata_builder(self, ubatch_id: int = 0) -> _DummyBuilder:
        assert ubatch_id == 0
        return self._builder


def _make_runner(builder: _DummyBuilder) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.attn_groups = [[_DummyAttnGroup(["model.layers.0.attn"], builder)]]
    return runner


def _make_layer_plan(
    *,
    query_start_loc: list[int],
    slot_mapping: list[int],
    num_actual_tokens: int,
    max_query_len: int,
) -> LayerReplayPlan:
    replay_token_count = num_actual_tokens - 2
    return LayerReplayPlan(
        kv_replay_start_per_req=np.array([4, 4], dtype=np.int32),
        computed_lens_per_req=np.array([8, 8], dtype=np.int32),
        prev_gpu_start_per_req=np.array([8, 8], dtype=np.int32),
        cpu_fill_token_count=replay_token_count,
        gpu_reuse_token_count=0,
        replay_token_count=replay_token_count,
        scheduled_token_count=2,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
        combined_replay_indices=torch.arange(replay_token_count, dtype=torch.int64),
        combined_scheduled_indices=torch.arange(
            replay_token_count, num_actual_tokens, dtype=torch.int64
        ),
        cpu_fill_positions=np.arange(replay_token_count, dtype=np.int32),
        cpu_fill_logical_ids=np.arange(replay_token_count, dtype=np.int32),
        cpu_fill_block_offsets=np.arange(replay_token_count, dtype=np.int32),
        gpu_reuse_slice_per_req=[(0, 0), (0, 0)],
    )


def _make_real_plan(
    *,
    desired_replay_start_tokens: int,
    prev_layer_plan: LayerReplayPlan | None,
) -> LayerReplayPlan:
    return compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=desired_replay_start_tokens,
        computed_lens=np.array([12], dtype=np.int32),
        scheduled_lens=np.array([1], dtype=np.int32),
        logical_block_tables=np.array([[10, 11, 12, 13]], dtype=np.int32),
        block_size=4,
        mapper_mapping={10: 0, 11: 1, 12: 2, 13: 3},
        prev_layer_plan=prev_layer_plan,
    )


def test_layer_attn_metadata_reuse_when_plan_is_unchanged() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    plan = _make_layer_plan(
        query_start_loc=[0, 3, 6],
        slot_mapping=[10, 11, 12, 20, 21, 22],
        num_actual_tokens=6,
        max_query_len=3,
    )
    base_seq_lens = torch.tensor([8, 8], dtype=torch.int32)
    block_table_tensor = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    metadata_0 = runner._build_layer_attn_metadata(
        layer_idx=0,
        plan=plan,
        prev_plan=None,
        prev_metadata=None,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=8,
        block_table_tensor=block_table_tensor,
        num_reqs=2,
    )
    metadata_1 = runner._build_layer_attn_metadata(
        layer_idx=1,
        plan=_make_layer_plan(
            query_start_loc=[0, 3, 6],
            slot_mapping=[10, 11, 12, 20, 21, 22],
            num_actual_tokens=6,
            max_query_len=3,
        ),
        prev_plan=plan,
        prev_metadata=metadata_0,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=8,
        block_table_tensor=block_table_tensor,
        num_reqs=2,
    )

    assert metadata_1 is metadata_0
    assert len(builder.build_calls) == 1


def test_build_layer_attn_metadata_builds_common_metadata_for_current_plan() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    plan = _make_layer_plan(
        query_start_loc=[0, 4, 7],
        slot_mapping=[30, 31, 32, 33, 40, 41, 42],
        num_actual_tokens=7,
        max_query_len=4,
    )
    base_seq_lens = torch.tensor([10, 9], dtype=torch.int32)
    block_table_tensor = torch.tensor([[5, 6, 7], [8, 9, -1]], dtype=torch.int32)

    metadata = runner._build_layer_attn_metadata(
        layer_idx=2,
        plan=plan,
        prev_plan=None,
        prev_metadata=None,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=10,
        block_table_tensor=block_table_tensor,
        num_reqs=2,
    )

    built = metadata["model.layers.0.attn"]
    common = built.common_attn_metadata
    assert len(builder.build_calls) == 1
    assert built.build_index == 1
    assert torch.equal(common.query_start_loc_cpu, plan.query_start_loc)
    assert torch.equal(common.query_start_loc, plan.query_start_loc)
    assert torch.equal(common.seq_lens, base_seq_lens)
    assert torch.equal(common._seq_lens_cpu, base_seq_lens)
    assert torch.equal(
        common._num_computed_tokens_cpu,
        torch.tensor([6, 6], dtype=torch.int32),
    )
    assert common.num_actual_tokens == 7
    assert common.max_query_len == 4
    assert common.max_seq_len == 10
    assert torch.equal(common.slot_mapping, plan.slot_mapping)
    assert torch.equal(common.block_table_tensor, block_table_tensor)


def test_build_layer_attn_metadata_rebuilds_when_plan_changes() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    plan_0 = _make_layer_plan(
        query_start_loc=[0, 3, 6],
        slot_mapping=[1, 2, 3, 4, 5, 6],
        num_actual_tokens=6,
        max_query_len=3,
    )
    plan_1 = _make_layer_plan(
        query_start_loc=[0, 4, 7],
        slot_mapping=[1, 2, 3, 4, 5, 6, 7],
        num_actual_tokens=7,
        max_query_len=4,
    )
    base_seq_lens = torch.tensor([8, 8], dtype=torch.int32)
    block_table_tensor = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    metadata_0 = runner._build_layer_attn_metadata(
        layer_idx=0,
        plan=plan_0,
        prev_plan=None,
        prev_metadata=None,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=8,
        block_table_tensor=block_table_tensor,
        num_reqs=2,
    )
    metadata_1 = runner._build_layer_attn_metadata(
        layer_idx=1,
        plan=plan_1,
        prev_plan=plan_0,
        prev_metadata=metadata_0,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=8,
        block_table_tensor=block_table_tensor,
        num_reqs=2,
    )

    assert metadata_1 is not metadata_0
    assert len(builder.build_calls) == 2


def test_build_layer_attn_metadata_ignores_ratio_if_window_is_same() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    prev_plan_full_gpu_reuse = _make_real_plan(
        desired_replay_start_tokens=4,
        prev_layer_plan=None,
    )
    prev_plan_partial_gpu_reuse = _make_real_plan(
        desired_replay_start_tokens=8,
        prev_layer_plan=None,
    )
    current_plan_full_gpu_reuse = _make_real_plan(
        desired_replay_start_tokens=4,
        prev_layer_plan=prev_plan_full_gpu_reuse,
    )
    current_plan_partial_gpu_reuse = _make_real_plan(
        desired_replay_start_tokens=4,
        prev_layer_plan=prev_plan_partial_gpu_reuse,
    )
    base_seq_lens = torch.tensor([12], dtype=torch.int32)
    block_table_tensor = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    metadata_0 = runner._build_layer_attn_metadata(
        layer_idx=0,
        plan=current_plan_full_gpu_reuse,
        prev_plan=None,
        prev_metadata=None,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=12,
        block_table_tensor=block_table_tensor,
        num_reqs=1,
    )
    metadata_1 = runner._build_layer_attn_metadata(
        layer_idx=1,
        plan=current_plan_partial_gpu_reuse,
        prev_plan=current_plan_full_gpu_reuse,
        prev_metadata=metadata_0,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=12,
        block_table_tensor=block_table_tensor,
        num_reqs=1,
    )

    assert current_plan_full_gpu_reuse.cpu_fill_token_count == 0
    assert current_plan_full_gpu_reuse.gpu_reuse_token_count == 8
    assert current_plan_partial_gpu_reuse.cpu_fill_token_count == 4
    assert current_plan_partial_gpu_reuse.gpu_reuse_token_count == 4
    assert torch.equal(
        current_plan_full_gpu_reuse.query_start_loc,
        current_plan_partial_gpu_reuse.query_start_loc,
    )
    assert torch.equal(
        current_plan_full_gpu_reuse.slot_mapping,
        current_plan_partial_gpu_reuse.slot_mapping,
    )
    assert metadata_1 is metadata_0
    assert len(builder.build_calls) == 1


def test_build_layer_attn_metadata_changes_when_replay_window_changes() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    base_seq_lens = torch.tensor([12], dtype=torch.int32)
    block_table_tensor = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    shorter_replay_plan = _make_real_plan(
        desired_replay_start_tokens=8,
        prev_layer_plan=None,
    )
    longer_replay_plan = _make_real_plan(
        desired_replay_start_tokens=4,
        prev_layer_plan=None,
    )

    metadata_0 = runner._build_layer_attn_metadata(
        layer_idx=0,
        plan=shorter_replay_plan,
        prev_plan=None,
        prev_metadata=None,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=12,
        block_table_tensor=block_table_tensor,
        num_reqs=1,
    )
    metadata_1 = runner._build_layer_attn_metadata(
        layer_idx=1,
        plan=longer_replay_plan,
        prev_plan=shorter_replay_plan,
        prev_metadata=metadata_0,
        base_seq_lens=base_seq_lens,
        base_max_seq_len=12,
        block_table_tensor=block_table_tensor,
        num_reqs=1,
    )

    assert shorter_replay_plan.replay_token_count == 4
    assert longer_replay_plan.replay_token_count == 8
    assert not torch.equal(
        shorter_replay_plan.query_start_loc,
        longer_replay_plan.query_start_loc,
    )
    assert metadata_1 is not metadata_0
    assert len(builder.build_calls) == 2


def test_build_layer_attn_metadata_validates_base_seq_lens_shape() -> None:
    builder = _DummyBuilder()
    runner = _make_runner(builder)
    plan = _make_layer_plan(
        query_start_loc=[0, 3, 6],
        slot_mapping=[1, 2, 3, 4, 5, 6],
        num_actual_tokens=6,
        max_query_len=3,
    )

    with pytest.raises(ValueError, match="base_seq_lens must be 1D"):
        runner._build_layer_attn_metadata(
            layer_idx=0,
            plan=plan,
            prev_plan=None,
            prev_metadata=None,
            base_seq_lens=torch.tensor([[8, 8]], dtype=torch.int32),
            base_max_seq_len=8,
            block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            num_reqs=2,
        )
