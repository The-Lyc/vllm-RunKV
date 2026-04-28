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


def test_real_shift_transitions_to_tracking_with_probe():
    """Steady at 0, then jumps to +15 ms — should enter TRACKING with a probe."""
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
    # Sustained high-imbalance regime.
    got_probe = False
    for _ in range(10):
        d = ctl.observe(imbalance_ms=15.0, current_budget=10)
        if d.state == SMState.TRACKING and abs(d.delta_budget) == 2:
            got_probe = True
            # Probe sign must oppose imbalance sign because gain<0.
            assert d.delta_budget < 0
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
    for _ in range(5):
        d = ctl.observe(imbalance_ms=0.1, current_budget=10)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
