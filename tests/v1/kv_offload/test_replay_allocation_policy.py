# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-request replay budget redistribution policies.

Covers `_resize_allocation_from_seed` with both policies:
  * "spread" (legacy): short-request-first incremental fill.
  * "concentrate": all-or-nothing redistribution that minimises attention
    FLOPs by filling the cheapest (smallest un-replayed prefix) requests
    first and draining the most expensive allocations first.

Cross-step sticky semantics must hold for both policies:
  * Unchanged budget -> allocation unchanged, element by element.
  * Only the budget delta is redistributed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm.v1.worker.opt_dynamic_replay import (
    FeedbackReplayPlanProvider,
    _allocate_budget_to_requests,
    _resize_allocation_from_seed,
)


def _resize(
    seed: list[int],
    budget: int,
    replayable: list[int],
    policy: str,
) -> list[int]:
    return _resize_allocation_from_seed(
        seed_allocated_blocks_per_req=np.array(seed, dtype=np.int32),
        budget_blocks=budget,
        replayable_blocks_per_req=np.array(replayable, dtype=np.int32),
        allocation_policy=policy,
    ).tolist()


# ---------------------------------------------------------------------------
# Sticky invariants shared by both policies.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["spread", "concentrate"])
def test_unchanged_budget_preserves_allocation_exactly(policy: str) -> None:
    seed = [3, 0, 7, 2]
    replayable = [10, 5, 8, 6]
    assert _resize(seed, sum(seed), replayable, policy) == seed


@pytest.mark.parametrize("policy", ["spread", "concentrate"])
def test_budget_clamped_to_total_replayable(policy: str) -> None:
    allocated = _resize([1, 1], 100, [4, 3], policy)
    assert sum(allocated) == 7
    assert allocated == [4, 3]


@pytest.mark.parametrize("policy", ["spread", "concentrate"])
def test_negative_budget_clamped_to_zero(policy: str) -> None:
    assert _resize([2, 2], -5, [4, 4], policy) == [0, 0]


@pytest.mark.parametrize("policy", ["spread", "concentrate"])
def test_seed_clamped_to_replayable_capacity(policy: str) -> None:
    # A shrunk capacity (e.g. fewer replayable blocks this step) clips the
    # seed before redistribution.
    allocated = _resize([5, 5], 6, [3, 10], policy)
    assert sum(allocated) == 6
    assert allocated[0] <= 3


@pytest.mark.parametrize("policy", ["spread", "concentrate"])
def test_allocation_never_exceeds_capacity(policy: str) -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(1, 9))
        replayable = rng.integers(0, 20, size=n).astype(np.int32)
        seed = np.minimum(
            rng.integers(0, 20, size=n).astype(np.int32), replayable
        )
        budget = int(rng.integers(0, int(replayable.sum()) + 5))
        allocated = _resize_allocation_from_seed(
            seed_allocated_blocks_per_req=seed,
            budget_blocks=budget,
            replayable_blocks_per_req=replayable,
            allocation_policy=policy,
        )
        assert (allocated >= 0).all()
        assert (allocated <= replayable).all()
        assert int(allocated.sum()) == min(budget, int(replayable.sum()))


def test_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="allocation_policy"):
        _resize([0], 0, [1], "unknown")


# ---------------------------------------------------------------------------
# Concentrate policy shape.
# ---------------------------------------------------------------------------


def test_concentrate_increase_fills_smallest_remaining_first() -> None:
    # remaining = replayable - allocated = [2, 6, 9].
    # +5 blocks: fill req0 fully (+2), then req1 (+3 of its 6).
    allocated = _resize([8, 4, 1], 18, [10, 10, 10], "concentrate")
    assert allocated == [10, 7, 1]


def test_concentrate_increase_tie_breaks_by_shorter_request() -> None:
    # Equal remaining (4 each); shorter replayable wins the delta.
    allocated = _resize([0, 6], 12, [4, 10], "concentrate")
    assert allocated == [4, 8]


def test_concentrate_decrease_drains_largest_remaining_first() -> None:
    # remaining = [7, 1, 4]; -4 blocks: drain req0 (largest remaining)
    # entirely (3), then req2 (1 of its 6).
    allocated = _resize([3, 9, 6], 14, [10, 10, 10], "concentrate")
    assert allocated == [0, 9, 5]


def test_concentrate_decrease_tie_breaks_by_longer_request() -> None:
    # Both fully allocated (remaining 0); the longer request loses first.
    allocated = _resize([4, 10], 11, [4, 10], "concentrate")
    assert allocated == [4, 7]


def test_concentrate_converges_to_short_first_greedy_shape() -> None:
    # Starting from a fully spread-out allocation, a sequence of decreases
    # under "concentrate" must converge to the same all-or-nothing shape as
    # the short-request-first greedy allocator used by TightLLM.
    replayable = np.array([8, 16, 4, 12], dtype=np.int32)
    allocated = replayable.copy()  # seeded at full replay
    for budget in (30, 24, 18, 12):
        allocated = _resize_allocation_from_seed(
            seed_allocated_blocks_per_req=allocated,
            budget_blocks=budget,
            replayable_blocks_per_req=replayable,
            allocation_policy="concentrate",
        )
    greedy = _allocate_budget_to_requests(
        budget_blocks=12, replayable_blocks_per_req=replayable
    )
    assert allocated.tolist() == greedy.tolist()


def test_concentrate_minimises_partial_allocations() -> None:
    # After any single resize, at most one request that was untouched by the
    # seed should end up partially allocated.
    allocated = _resize([0, 0, 0, 0], 10, [4, 4, 4, 4], "concentrate")
    partial = [a for a, r in zip(allocated, [4, 4, 4, 4]) if 0 < a < r]
    assert len(partial) <= 1
    assert allocated == [4, 4, 2, 0]


def test_spread_policy_preserves_legacy_behaviour() -> None:
    # Legacy: increases fill remaining capacity in short-request-first
    # order of *replayable* (not remaining).
    allocated = _resize([1, 1, 1], 9, [4, 8, 6], "spread")
    assert allocated == [4, 1, 4]
    # Legacy decrease removes from the reverse order (longest first).
    allocated = _resize([4, 8, 6], 12, [4, 8, 6], "spread")
    assert allocated == [4, 2, 6]


# ---------------------------------------------------------------------------
# Provider wiring.
# ---------------------------------------------------------------------------


def test_provider_rejects_unknown_allocation_policy() -> None:
    with pytest.raises(ValueError, match="allocation_policy"):
        FeedbackReplayPlanProvider(
            io_prefix_blocks=[0], allocation_policy="bogus"
        )


def test_provider_defaults_to_spread_and_reports_in_snapshot() -> None:
    provider = FeedbackReplayPlanProvider(io_prefix_blocks=[0])
    assert provider.allocation_policy == "spread"
    assert provider.get_debug_snapshot()["allocation_policy"] == "spread"


def test_provider_concentrate_policy_used_in_layer_plans() -> None:
    provider = FeedbackReplayPlanProvider(
        io_prefix_blocks=[0], allocation_policy="concentrate"
    )
    block_size = 2
    computed_lens = np.array([8, 8], dtype=np.int32)
    scheduled_lens = np.array([1, 1], dtype=np.int32)
    num_blocks_per_row = np.array([5, 5], dtype=np.int32)
    logical_block_tables = np.array(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32
    )
    mapper_mapping = {i: i for i in range(10)}

    provider.begin_step(
        req_ids=("a", "b"),
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        num_blocks_per_row=num_blocks_per_row,
        block_size=block_size,
    )
    # First replayable step seeds budget at full replay (8 blocks).
    # Shrink to 6: concentrate drains one request by 2 blocks instead of
    # taking 1 block from each.
    provider.controller_state.global_budget_blocks = 6
    plan = provider.get_layer_plan(
        layer_idx=0,
        num_reqs=2,
        computed_lens=computed_lens,
        scheduled_lens=scheduled_lens,
        logical_block_tables=logical_block_tables,
        block_size=block_size,
        mapper_mapping=mapper_mapping,
        prev_layer_plan=None,
    )
    starts = plan.kv_replay_start_per_req.tolist()
    # One request keeps full replay (start 0), the other gives up 2 blocks
    # (start 4 tokens); the spread policy would produce [2, 2] instead.
    assert sorted(starts) == [0, 4]
