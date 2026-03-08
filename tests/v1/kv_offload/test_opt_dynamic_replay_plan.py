# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch

from vllm.v1.worker.opt_dynamic_replay import (
    RandomReplayPlanProvider,
    ReplayPlanProvider,
    StaticReplayPlanProvider,
    compute_layer_replay_plan_for_layer,
)


def _make_test_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    computed_lens = np.array([10, 6], dtype=np.int32)
    scheduled_lens = np.array([2, 1], dtype=np.int32)
    logical_block_tables = np.array(
        [
            [100, 101, 102],
            [200, 201, -1],
        ],
        dtype=np.int32,
    )
    mapper_mapping = {
        100: 0,
        101: 1,
        102: 2,
        200: 3,
        201: 4,
    }
    return computed_lens, scheduled_lens, logical_block_tables, mapper_mapping


def test_compute_layer_replay_plan_aligns_token_start_down_to_block_boundary() -> None:
    computed_lens = np.array([48], dtype=np.int32)
    scheduled_lens = np.array([1], dtype=np.int32)
    logical_block_tables = np.array([[10, 11, 12, 13, 14]], dtype=np.int32)
    mapper_mapping = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4}

    plan_a = compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=19,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=16,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )
    plan_b = compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=17,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=16,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )

    assert np.array_equal(
        plan_a.kv_replay_start_per_req, np.array([16], dtype=np.int32)
    )
    assert np.array_equal(
        plan_a.kv_replay_start_per_req, plan_b.kv_replay_start_per_req
    )
    assert torch.equal(plan_a.query_start_loc, plan_b.query_start_loc)
    assert torch.equal(plan_a.slot_mapping, plan_b.slot_mapping)
    assert torch.equal(plan_a.combined_replay_indices, plan_b.combined_replay_indices)
    assert torch.equal(
        plan_a.combined_scheduled_indices, plan_b.combined_scheduled_indices
    )


def test_compute_layer_replay_plan_builds_slot_mapping_and_pack_indices() -> None:
    computed_lens, scheduled_lens, logical_block_tables, mapper_mapping = (
        _make_test_inputs()
    )

    plan = compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=8,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )

    assert np.array_equal(
        plan.kv_replay_start_per_req,
        np.array([8, 4], dtype=np.int32),
    )
    assert np.array_equal(
        plan.prev_gpu_start_per_req,
        np.array([10, 6], dtype=np.int32),
    )
    assert plan.cpu_fill_token_count == 4
    assert plan.gpu_reuse_token_count == 0
    assert plan.replay_token_count == 4
    assert plan.scheduled_token_count == 3
    assert plan.num_actual_tokens == 7
    assert plan.max_query_len == 4
    assert torch.equal(plan.query_start_loc, torch.tensor([0, 4, 7], dtype=torch.int32))
    assert torch.equal(
        plan.slot_mapping,
        torch.tensor([8, 9, 10, 11, 16, 17, 18], dtype=torch.int64),
    )
    assert torch.equal(
        plan.combined_replay_indices,
        torch.tensor([0, 1, 4, 5], dtype=torch.int64),
    )
    assert torch.equal(
        plan.combined_scheduled_indices,
        torch.tensor([2, 3, 6], dtype=torch.int64),
    )
    assert np.array_equal(
        plan.cpu_fill_positions, np.array([8, 9, 4, 5], dtype=np.int32)
    )
    assert np.array_equal(
        plan.cpu_fill_logical_ids,
        np.array([102, 102, 201, 201], dtype=np.int32),
    )
    assert np.array_equal(
        plan.cpu_fill_block_offsets,
        np.array([0, 1, 0, 1], dtype=np.int32),
    )
    assert plan.gpu_reuse_slice_per_req == [(0, 0), (0, 0)]


def test_compute_layer_replay_plan_reuses_gpu_suffix_when_current_layer_shorter() -> (
    None
):
    computed_lens = np.array([12], dtype=np.int32)
    scheduled_lens = np.array([1], dtype=np.int32)
    logical_block_tables = np.array([[10, 11, 12, 13]], dtype=np.int32)
    mapper_mapping = {10: 0, 11: 1, 12: 2, 13: 3}

    prev_plan = compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=4,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )
    curr_plan = compute_layer_replay_plan_for_layer(
        layer_idx=1,
        desired_replay_start_tokens=8,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=prev_plan,
    )

    assert np.array_equal(
        curr_plan.kv_replay_start_per_req, np.array([8], dtype=np.int32)
    )
    assert np.array_equal(
        curr_plan.prev_gpu_start_per_req, np.array([4], dtype=np.int32)
    )
    assert curr_plan.cpu_fill_token_count == 0
    assert curr_plan.gpu_reuse_token_count == 4
    assert curr_plan.gpu_reuse_slice_per_req == [(4, 8)]
    assert np.array_equal(curr_plan.cpu_fill_positions, np.array([], dtype=np.int32))


def test_compute_layer_replay_plan_cpu_fills_prefix_when_current_layer_is_longer() -> (
    None
):
    computed_lens = np.array([12], dtype=np.int32)
    scheduled_lens = np.array([1], dtype=np.int32)
    logical_block_tables = np.array([[10, 11, 12, 13]], dtype=np.int32)
    mapper_mapping = {10: 0, 11: 1, 12: 2, 13: 3}

    prev_plan = compute_layer_replay_plan_for_layer(
        layer_idx=0,
        desired_replay_start_tokens=8,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )
    curr_plan = compute_layer_replay_plan_for_layer(
        layer_idx=1,
        desired_replay_start_tokens=4,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=prev_plan,
    )

    assert np.array_equal(
        curr_plan.kv_replay_start_per_req, np.array([4], dtype=np.int32)
    )
    assert np.array_equal(
        curr_plan.prev_gpu_start_per_req, np.array([8], dtype=np.int32)
    )
    assert curr_plan.cpu_fill_token_count == 4
    assert curr_plan.gpu_reuse_token_count == 4
    assert np.array_equal(
        curr_plan.cpu_fill_positions,
        np.array([4, 5, 6, 7], dtype=np.int32),
    )
    assert curr_plan.gpu_reuse_slice_per_req == [(0, 4)]


def test_static_replay_plan_provider_matches_protocol_and_uses_block_prefixes() -> None:
    computed_lens, scheduled_lens, logical_block_tables, mapper_mapping = (
        _make_test_inputs()
    )
    provider = StaticReplayPlanProvider(io_prefix_blocks=[2, 1])

    assert isinstance(provider, ReplayPlanProvider)

    plan = provider.get_layer_plan(
        layer_idx=1,
        num_reqs=2,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )

    assert np.array_equal(
        plan.kv_replay_start_per_req,
        np.array([4, 4], dtype=np.int32),
    )


def test_random_replay_plan_provider_is_deterministic_and_block_aligned() -> None:
    computed_lens, scheduled_lens, logical_block_tables, mapper_mapping = (
        _make_test_inputs()
    )
    provider_a = RandomReplayPlanProvider(num_layers=3, max_tokens=15, seed=7)
    provider_b = RandomReplayPlanProvider(num_layers=3, max_tokens=15, seed=7)

    plan_a = provider_a.get_layer_plan(
        layer_idx=2,
        num_reqs=2,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )
    plan_b = provider_b.get_layer_plan(
        layer_idx=2,
        num_reqs=2,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=4,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )

    assert np.array_equal(
        plan_a.kv_replay_start_per_req, plan_b.kv_replay_start_per_req
    )
    assert np.all(plan_a.kv_replay_start_per_req % 4 == 0)
