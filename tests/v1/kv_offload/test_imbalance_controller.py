# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the state-machine imbalance controller.

See docs/design/imbalance_state_machine_controller.md §9.1.
"""
from __future__ import annotations

import random

import pytest

from vllm.v1.worker.imbalance_controller import (
    ImbalanceController,
    ImbalanceControllerConfig,
    SMState,
    _hint_from_delta,
)


def _mk_controller(**overrides) -> ImbalanceController:
    cfg = ImbalanceControllerConfig(**overrides)
    return ImbalanceController(config=cfg)


def _mk_tracking_controller() -> ImbalanceController:
    ctl = _mk_controller(
        deadband_ms=0.5,
        tracking_settle_layers=100,
        tracking_damping=1.0,
        tracking_initial_step_cap_blocks=10,
        tracking_step_cap_increment_blocks=5,
        tracking_max_step_blocks=20,
    )
    ctl.state = SMState.TRACKING
    ctl.gain = -0.1
    return ctl


def test_hint_from_delta_buckets():
    assert _hint_from_delta(0) == "unchanged"
    assert _hint_from_delta(1) == "small_delta"
    assert _hint_from_delta(-1) == "small_delta"
    assert _hint_from_delta(2) == "significant_delta"
    assert _hint_from_delta(-10) == "significant_delta"


def test_pure_steady_stays_steady_with_zero_delta():
    """Long run of sub-ms noise stays in STEADY, emits 'unchanged' most of the time."""
    rng = random.Random(0)
    ctl = _mk_controller(
        window_size=3,
        sigma_baseline_ms=0.5,
        stdev_trigger_multiple=3.0,
        deadband_ms=0.5,
    )
    hints = []
    states = set()
    for _ in range(200):
        y = rng.gauss(0.0, 0.24)
        d = ctl.observe(imbalance_ms=y, current_budget=10)
        hints.append(d.plan_change_hint)
        states.add(d.state)
    assert SMState.STEADY in states
    assert SMState.TRANSIT not in states
    assert SMState.TRACKING not in states
    # At least 80% of hints should be "unchanged" in a clean steady state.
    unchanged_ratio = hints.count("unchanged") / len(hints)
    assert unchanged_ratio > 0.8, f"unchanged ratio too low: {unchanged_ratio}"


@pytest.mark.parametrize(("imbalance_ms", "expected_delta"), [(2.0, 1), (-2.0, -1)])
def test_steady_nudge_matches_imbalance_sign(
    imbalance_ms: float, expected_delta: int
):
    """With negative plant gain, the corrective budget delta matches y's sign."""
    ctl = _mk_controller(deadband_ms=0.5, steady_small_step_blocks=1)

    decision = ctl.observe(imbalance_ms=imbalance_ms, current_budget=10)

    assert decision.state == SMState.STEADY
    assert decision.delta_budget == expected_delta


def test_single_outlier_triggers_transit_then_false_alarm_back_to_steady():
    ctl = _mk_controller(window_size=3, sigma_baseline_ms=0.5)
    # Seed a clean steady state.
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    assert ctl.state == SMState.STEADY
    # One big spike.
    ctl.observe(imbalance_ms=50.0, current_budget=10)
    # Should be in TRANSIT now (stdev blew up).
    assert ctl.state == SMState.TRANSIT
    # Continue with clean steady samples; false-alarm path should return STEADY.
    saw_steady_after_transit = False
    for _ in range(10):
        d = ctl.observe(imbalance_ms=0.0, current_budget=10)
        if d.state == SMState.STEADY:
            saw_steady_after_transit = True
            break
    assert saw_steady_after_transit


@pytest.mark.parametrize(("shift_ms", "expected_probe"), [(15.0, 2), (-15.0, -2)])
def test_real_shift_transitions_to_tracking_with_probe(
    shift_ms: float, expected_probe: int
):
    """A signed regime shift should enter TRACKING with a same-sign probe."""
    ctl = _mk_controller(
        window_size=3,
        sigma_baseline_ms=0.5,
        stdev_trigger_multiple=3.0,
        delta_threshold_ms=1.0,
        probe_size_blocks=2,
    )
    # Clean steady at 0.
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    # Sustained shifted-imbalance regime.
    got_probe = False
    for _ in range(10):
        d = ctl.observe(imbalance_ms=shift_ms, current_budget=10)
        if d.state == SMState.TRACKING and abs(d.delta_budget) == 2:
            got_probe = True
            # For gain < 0, -imbalance / gain has the same sign as imbalance.
            assert d.delta_budget == expected_probe
            # If the probe has not produced a measurable gain yet, the
            # sign-only fallback must preserve the same corrective direction.
            fallback = ctl.observe(imbalance_ms=shift_ms, current_budget=10)
            assert fallback.delta_budget * shift_ms > 0
            break
    assert got_probe


def test_tracking_converges_back_to_steady():
    ctl = _mk_controller(
        window_size=3,
        sigma_baseline_ms=0.5,
        deadband_ms=0.5,
        tracking_settle_layers=3,
    )
    # Prime with steady then shift.
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    for _ in range(10):
        d = ctl.observe(imbalance_ms=15.0, current_budget=10)
        if d.state == SMState.TRACKING:
            break
    assert ctl.state == SMState.TRACKING
    # Now supply converged samples (|y|<deadband) for several layers.
    deadband_decisions = []
    for _ in range(5):
        d = ctl.observe(imbalance_ms=0.1, current_budget=10)
        deadband_decisions.append(d)
    assert all(d.delta_budget == 0 for d in deadband_decisions)
    assert ctl.state == SMState.STEADY


def test_transit_timeout_forces_tracking():
    ctl = _mk_controller(
        window_size=3,
        sigma_baseline_ms=0.5,
        transit_timeout_layers=4,
    )
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    # Persistent high stdev: alternate big swings.
    saw_tracking = False
    for i in range(20):
        y = 30.0 if i % 2 == 0 else -30.0
        d = ctl.observe(imbalance_ms=y, current_budget=10)
        if d.state == SMState.TRACKING:
            saw_tracking = True
            break
    assert saw_tracking


def test_gain_sign_guard_rejects_positive_gain_in_rls():
    """Wrong-sign samples must not poison the gain estimate."""
    ctl = _mk_controller(
        window_size=3,
        sigma_baseline_ms=0.5,
        tracking_settle_layers=100,  # don't let it converge during the test
    )
    # Push into TRACKING via a real shift.
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    for _ in range(10):
        d = ctl.observe(imbalance_ms=15.0, current_budget=10)
        if d.state == SMState.TRACKING:
            break
    assert ctl.state == SMState.TRACKING
    # Feed deliberately wrong-sign samples: budget↑ together with imbalance↑.
    budget = 10
    for _ in range(10):
        budget += 2
        ctl.observe(imbalance_ms=budget * 1.0, current_budget=budget)  # positive gain
    # Controller gain should remain None or negative (never positive).
    assert ctl.gain is None or ctl.gain < 0


def test_hint_matches_delta():
    """ControlDecision.plan_change_hint is always derived from delta_budget."""
    ctl = _mk_controller()
    for _ in range(50):
        d = ctl.observe(imbalance_ms=0.0, current_budget=5)
        assert d.plan_change_hint == _hint_from_delta(d.delta_budget)


@pytest.mark.parametrize("imbalance_ms", [10.0, -10.0])
def test_tracking_step_cap_progresses_after_same_direction_saturation(
    imbalance_ms: float,
):
    ctl = _mk_tracking_controller()

    decisions = [
        ctl.observe(imbalance_ms=imbalance_ms, current_budget=50)
        for _ in range(4)
    ]

    direction = 1 if imbalance_ms > 0 else -1
    assert [d.delta_budget for d in decisions] == [
        direction * 10,
        direction * 15,
        direction * 20,
        direction * 20,
    ]
    assert [d.step_cap_blocks for d in decisions] == [10, 15, 20, 20]
    assert all(d.step_cap_saturated for d in decisions)
    assert [d.step_cap_saturation_streak for d in decisions] == [1, 2, 3, 4]


def test_tracking_step_cap_resets_on_direction_change():
    ctl = _mk_tracking_controller()

    first = ctl.observe(imbalance_ms=10.0, current_budget=50)
    escalated = ctl.observe(imbalance_ms=10.0, current_budget=50)
    reversed_direction = ctl.observe(imbalance_ms=-10.0, current_budget=50)

    assert first.step_cap_blocks == 10
    assert escalated.step_cap_blocks == 15
    assert reversed_direction.delta_budget == -10
    assert reversed_direction.step_cap_blocks == 10
    assert reversed_direction.step_cap_saturated
    assert reversed_direction.step_cap_saturation_streak == 1


def test_tracking_step_cap_resets_after_non_saturated_update():
    ctl = _mk_tracking_controller()

    ctl.observe(imbalance_ms=10.0, current_budget=50)
    escalated = ctl.observe(imbalance_ms=10.0, current_budget=50)
    non_saturated = ctl.observe(imbalance_ms=0.6, current_budget=50)
    next_saturated = ctl.observe(imbalance_ms=10.0, current_budget=50)

    assert escalated.step_cap_blocks == 15
    assert non_saturated.delta_budget == 6
    assert non_saturated.step_cap_blocks == 20
    assert not non_saturated.step_cap_saturated
    assert non_saturated.step_cap_saturation_streak == 0
    assert next_saturated.delta_budget == 10
    assert next_saturated.step_cap_blocks == 10
    assert next_saturated.step_cap_saturation_streak == 1


def test_tracking_step_cap_resets_in_deadband():
    ctl = _mk_tracking_controller()

    ctl.observe(imbalance_ms=10.0, current_budget=50)
    ctl.observe(imbalance_ms=10.0, current_budget=50)
    in_deadband = ctl.observe(imbalance_ms=0.0, current_budget=50)
    next_saturated = ctl.observe(imbalance_ms=10.0, current_budget=50)

    assert in_deadband.delta_budget == 0
    # No cap is applied because deadband observations do not actuate.
    assert in_deadband.step_cap_blocks is None
    assert not in_deadband.step_cap_saturated
    assert in_deadband.step_cap_saturation_streak == 0
    assert next_saturated.delta_budget == 10
    assert next_saturated.step_cap_blocks == 10


def test_tracking_deadband_freezes_budget_while_confirming_settle():
    ctl = _mk_controller(
        deadband_ms=0.5,
        tracking_settle_layers=3,
        tracking_damping=1.0,
        tracking_initial_step_cap_blocks=10,
        tracking_max_step_blocks=20,
    )
    ctl.state = SMState.TRACKING
    # This deliberately steep gain would turn a residual just inside the
    # deadband into a cap-sized Newton step without the no-actuation guard.
    ctl.gain = -0.01

    first = ctl.observe(imbalance_ms=0.49, current_budget=50)
    second = ctl.observe(imbalance_ms=-0.49, current_budget=50)
    settled = ctl.observe(imbalance_ms=0.1, current_budget=50)

    assert [first.delta_budget, second.delta_budget, settled.delta_budget] == [
        0,
        0,
        0,
    ]
    assert first.plan_change_hint == "unchanged"
    assert second.plan_change_hint == "unchanged"
    assert first.state == SMState.TRACKING
    assert second.state == SMState.TRACKING
    assert settled.state == SMState.STEADY


def test_tracking_resumes_correction_after_leaving_deadband():
    ctl = _mk_controller(
        deadband_ms=0.5,
        tracking_settle_layers=3,
        tracking_damping=1.0,
        tracking_initial_step_cap_blocks=10,
        tracking_max_step_blocks=20,
    )
    ctl.state = SMState.TRACKING
    ctl.gain = -0.1

    frozen = ctl.observe(imbalance_ms=0.1, current_budget=50)
    settle_count_while_frozen = ctl.tracking_settle_count
    resumed = ctl.observe(imbalance_ms=1.0, current_budget=50)

    assert frozen.delta_budget == 0
    assert settle_count_while_frozen == 1
    assert ctl.tracking_settle_count == 0
    assert resumed.delta_budget == 10
    assert resumed.plan_change_hint == "significant_delta"
    assert resumed.state == SMState.TRACKING


def test_tracking_deadband_still_consumes_probe_measurement():
    ctl = _mk_controller(deadband_ms=0.5, tracking_settle_layers=3)
    ctl.state = SMState.TRACKING
    ctl._pending_probe_delta = 2
    ctl._probe_pre_imbalance = 2.0
    ctl._probe_pre_budget = 10

    decision = ctl.observe(imbalance_ms=0.2, current_budget=12)

    assert decision.delta_budget == 0
    assert ctl._pending_probe_delta == 0
    assert ctl._probe_pre_imbalance is None
    assert ctl._probe_pre_budget is None
    assert ctl.gain == pytest.approx(-0.9)


def test_reset_returns_to_clean_steady():
    ctl = _mk_controller()
    for _ in range(5):
        ctl.observe(imbalance_ms=0.0, current_budget=10)
    ctl.observe(imbalance_ms=50.0, current_budget=10)
    assert ctl.state == SMState.TRANSIT
    ctl.reset()
    assert ctl.state == SMState.STEADY
    assert ctl.old_baseline_ms is None
    assert ctl.gain is None
    assert len(ctl._window) == 0
    assert ctl._tracking_step_cap_blocks == 10
    assert ctl._tracking_cap_direction == 0
    assert not ctl._last_tracking_step_saturated
    assert ctl._tracking_saturation_streak == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
