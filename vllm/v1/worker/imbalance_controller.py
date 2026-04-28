# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-machine-based imbalance controller for dynamic replay.

Implements the design in docs/design/imbalance_state_machine_controller.md:
three states (STEADY, TRANSIT, TRACKING) with Δbudget-driven plan-change
hints, replacing the per-layer Newton secant controller.
"""
from __future__ import annotations

import math
import statistics as _stat
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SMState(str, Enum):
    STEADY = "steady"
    TRANSIT = "transit"
    TRACKING = "tracking"


# Plan-change hint derived from Δbudget, consumed by the pre_hook to decide
# whether to reuse / lightly rebuild / fully rebuild plan(L+1).
PlanChangeHint = str  # one of: "unchanged", "small_delta", "significant_delta"


def _hint_from_delta(delta_blocks: int) -> PlanChangeHint:
    a = abs(int(delta_blocks))
    if a == 0:
        return "unchanged"
    if a <= 1:
        return "small_delta"
    return "significant_delta"


@dataclass
class ImbalanceControllerConfig:
    # Decision window length (§7.2).
    window_size: int = 3
    # Variance floor for stdev-based triggers (ms).
    sigma_baseline_ms: float = 0.5
    # STEADY → TRANSIT trigger: stdev > M * sigma_baseline.
    stdev_trigger_multiple: float = 3.0
    # |mean − old_baseline| < delta_threshold → virtual alarm.
    delta_threshold_ms: float = 1.0
    # STEADY / TRACKING settle deadband (ms).
    deadband_ms: float = 0.5
    # STEADY nudge step (blocks).
    steady_small_step_blocks: int = 1
    # TRACKING first-step probe size (blocks).
    probe_size_blocks: int = 2
    # TRACKING max step size (blocks).
    tracking_max_step_blocks: int = 10
    # TRACKING Newton step damping.
    tracking_damping: float = 0.6
    # TRANSIT force-exit timeout (layers).
    transit_timeout_layers: int = 6
    # TRACKING convergence condition: consecutive layers in deadband.
    tracking_settle_layers: int = 3
    # TRACKING hard timeout (layers).
    tracking_timeout_layers: int = 20
    # EWMA alpha for baseline in STEADY.
    baseline_ewma_alpha: float = 0.1
    # RLS pool FIFO length (pairs).
    rls_window: int = 6
    # Gain sign physical invariant: budget↑ → imbalance↓ ⇒ gain < 0.
    gain_sign: int = -1


@dataclass
class ControlDecision:
    """Output of the state machine for a single layer observation."""

    delta_budget: int
    plan_change_hint: PlanChangeHint
    state: SMState
    # Diagnostic fields (optional, for observability / debugging).
    window_mean_ms: float | None = None
    window_stdev_ms: float | None = None
    old_baseline_ms: float | None = None
    gain_used: float | None = None
    action: str = "sm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_budget": self.delta_budget,
            "plan_change_hint": self.plan_change_hint,
            "sm_state": self.state.value,
            "window_mean_ms": self.window_mean_ms,
            "window_stdev_ms": self.window_stdev_ms,
            "old_baseline_ms": self.old_baseline_ms,
            "gain_used": self.gain_used,
            "action": self.action,
        }


@dataclass
class ImbalanceController:
    """Three-state imbalance controller.

    Usage:
        ctl = ImbalanceController(ImbalanceControllerConfig())
        decision = ctl.observe(imbalance_ms=2.3, current_budget=12)
    """

    config: ImbalanceControllerConfig = field(default_factory=ImbalanceControllerConfig)

    state: SMState = field(default=SMState.STEADY, init=False)
    # Rolling imbalance window (size up to K).
    _window: deque[float] = field(init=False)
    # EWMA of baseline imbalance while in STEADY.
    baseline_ewma: float = field(default=0.0, init=False)
    _baseline_initialized: bool = field(default=False, init=False)
    # Frozen baseline at moment of entering TRANSIT.
    old_baseline_ms: float | None = field(default=None, init=False)
    # Frozen budget at moment of entering TRANSIT.
    frozen_budget: int | None = field(default=None, init=False)
    # Counter of layers since entering TRANSIT.
    layers_in_transit: int = field(default=0, init=False)
    # Counter of layers since entering TRACKING.
    layers_in_tracking: int = field(default=0, init=False)
    # Consecutive layers with |imbalance|<deadband inside TRACKING.
    tracking_settle_count: int = field(default=0, init=False)
    # Probe bookkeeping (first TRACKING step only sets delta, no gain yet).
    _pending_probe_delta: int = field(default=0, init=False)
    _probe_pre_imbalance: float | None = field(default=None, init=False)
    _probe_pre_budget: int | None = field(default=None, init=False)
    # RLS FIFO of (budget, imbalance) pairs used for OLS gain estimation.
    _rls_pool: deque[tuple[int, float]] = field(init=False)
    # Current gain estimate (ms per block).  None until first valid measurement.
    gain: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=int(self.config.window_size))
        self._rls_pool = deque(maxlen=int(self.config.rls_window))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def reset(self) -> None:
        """Reset to STEADY and drop all accumulated state.

        Intended to be called by an external regime classifier on regime
        switch.  The caller is responsible for also swapping in the new
        regime's baseline/σ parameters if any.
        """
        self.state = SMState.STEADY
        self._window.clear()
        self.baseline_ewma = 0.0
        self._baseline_initialized = False
        self.old_baseline_ms = None
        self.frozen_budget = None
        self.layers_in_transit = 0
        self.layers_in_tracking = 0
        self.tracking_settle_count = 0
        self._pending_probe_delta = 0
        self._probe_pre_imbalance = None
        self._probe_pre_budget = None
        self._rls_pool.clear()
        self.gain = None

    def observe(
        self,
        imbalance_ms: float,
        current_budget: int,
    ) -> ControlDecision:
        """Process one layer observation, return control decision.

        The caller is expected to apply ``decision.delta_budget`` to the
        global budget and pass ``decision.plan_change_hint`` down to the
        pre_hook.
        """
        cfg = self.config
        y = float(imbalance_ms)
        self._window.append(y)

        # Attach any in-flight probe measurement to the RLS pool first so
        # downstream gain estimation in TRACKING can use it.
        self._record_rls_sample(current_budget, y)

        if self.state == SMState.STEADY:
            delta = self._step_steady(y)
        elif self.state == SMState.TRANSIT:
            delta = self._step_transit(y, current_budget)
        else:  # TRACKING
            delta = self._step_tracking(y, current_budget)

        mean = _safe_mean(self._window)
        stdev = _safe_stdev(self._window)
        return ControlDecision(
            delta_budget=int(delta),
            plan_change_hint=_hint_from_delta(delta),
            state=self.state,
            window_mean_ms=mean,
            window_stdev_ms=stdev,
            old_baseline_ms=self.old_baseline_ms,
            gain_used=self.gain,
        )

    # ------------------------------------------------------------
    # STEADY
    # ------------------------------------------------------------

    def _step_steady(self, y: float) -> int:
        cfg = self.config
        # Maintain baseline EWMA while in STEADY.
        if not self._baseline_initialized:
            self.baseline_ewma = y
            self._baseline_initialized = True
        else:
            a = cfg.baseline_ewma_alpha
            self.baseline_ewma = (1.0 - a) * self.baseline_ewma + a * y

        # Check TRANSIT trigger only once the window is full.
        if len(self._window) >= cfg.window_size:
            sd = _safe_stdev(self._window)
            if sd is not None and sd > cfg.stdev_trigger_multiple * cfg.sigma_baseline_ms:
                self._enter_transit(current_budget_override=None)
                return 0  # budget frozen in TRANSIT

        # Proportional nudge: ±1 block if |y| outside deadband, else 0.
        if abs(y) < cfg.deadband_ms:
            return 0
        # sign(−y) because budget↑ → imbalance↓.
        return -int(math.copysign(cfg.steady_small_step_blocks, y)) if y != 0 else 0

    # ------------------------------------------------------------
    # TRANSIT
    # ------------------------------------------------------------

    def _enter_transit(self, current_budget_override: int | None) -> None:
        self.state = SMState.TRANSIT
        self.old_baseline_ms = self.baseline_ewma if self._baseline_initialized else None
        self.frozen_budget = current_budget_override
        self.layers_in_transit = 0

    def _step_transit(self, y: float, current_budget: int) -> int:
        cfg = self.config
        self.layers_in_transit += 1
        # Window must become "pure" first (all K samples observed since
        # entering TRANSIT).  Only then do we evaluate exit conditions.
        if self.layers_in_transit >= cfg.window_size:
            sd = _safe_stdev(self._window)
            mean = _safe_mean(self._window) or 0.0
            stdev_ok = sd is not None and sd < cfg.stdev_trigger_multiple * cfg.sigma_baseline_ms
            if stdev_ok:
                old = self.old_baseline_ms if self.old_baseline_ms is not None else 0.0
                if abs(mean - old) < cfg.delta_threshold_ms:
                    # False alarm — return to STEADY, keep baseline.
                    self.state = SMState.STEADY
                    self.old_baseline_ms = None
                    self.frozen_budget = None
                    self.layers_in_transit = 0
                    return 0
                # Real shift: enter TRACKING and issue probe.
                return self._enter_tracking(mean, current_budget)
            if self.layers_in_transit > cfg.transit_timeout_layers:
                # Forced TRACKING.
                return self._enter_tracking(mean, current_budget)
        return 0  # still frozen

    # ------------------------------------------------------------
    # TRACKING
    # ------------------------------------------------------------

    def _enter_tracking(self, window_mean: float, current_budget: int) -> int:
        cfg = self.config
        self.state = SMState.TRACKING
        self.layers_in_tracking = 0
        self.tracking_settle_count = 0
        # Probe step: use only the sign, not the numeric gain.
        # window_mean > 0 means imbalance was positive → need budget↑ → Δ > 0.
        # Since gain < 0, sign(Δ) = −sign(mean).
        if window_mean == 0.0:
            # Degenerate case: default to positive probe.
            sign = 1
        else:
            sign = -1 if window_mean > 0 else 1
        delta = sign * cfg.probe_size_blocks
        self._pending_probe_delta = int(delta)
        self._probe_pre_imbalance = window_mean
        self._probe_pre_budget = current_budget
        return delta

    def _step_tracking(self, y: float, current_budget: int) -> int:
        cfg = self.config
        self.layers_in_tracking += 1

        # Settle check.
        if abs(y) < cfg.deadband_ms:
            self.tracking_settle_count += 1
        else:
            self.tracking_settle_count = 0
        if self.tracking_settle_count >= cfg.tracking_settle_layers:
            # Converged — adopt current window mean as new baseline.
            self.baseline_ewma = _safe_mean(self._window) or y
            self._baseline_initialized = True
            self.state = SMState.STEADY
            self.old_baseline_ms = None
            self.frozen_budget = None
            self.layers_in_tracking = 0
            self.tracking_settle_count = 0
            return 0

        # Hard timeout.
        if self.layers_in_tracking > cfg.tracking_timeout_layers:
            self.baseline_ewma = _safe_mean(self._window) or y
            self._baseline_initialized = True
            self.state = SMState.STEADY
            self.old_baseline_ms = None
            self.frozen_budget = None
            self.layers_in_tracking = 0
            self.tracking_settle_count = 0
            return 0

        # If we just completed a probe, compute measured gain from the
        # probe pair before doing a Newton step.
        if self._pending_probe_delta != 0:
            if (
                self._probe_pre_imbalance is not None
                and self._pending_probe_delta != 0
            ):
                g = (y - self._probe_pre_imbalance) / float(self._pending_probe_delta)
                if abs(g) > 1e-9 and (g < 0 if cfg.gain_sign < 0 else g > 0):
                    self.gain = g
            self._pending_probe_delta = 0
            self._probe_pre_imbalance = None
            self._probe_pre_budget = None

        # Refresh gain from RLS pool if we have enough samples.
        g_rls = self._ols_gain()
        if g_rls is not None and (
            (cfg.gain_sign < 0 and g_rls < 0) or (cfg.gain_sign > 0 and g_rls > 0)
        ):
            self.gain = g_rls

        if self.gain is None or abs(self.gain) < 1e-9:
            # No trustworthy gain yet — issue another small sign-only step.
            sign = -1 if y > 0 else (1 if y < 0 else 0)
            return sign * cfg.steady_small_step_blocks

        raw = -y / self.gain
        damped = cfg.tracking_damping * raw
        clipped = max(-cfg.tracking_max_step_blocks, min(cfg.tracking_max_step_blocks, damped))
        return int(round(clipped))

    # ------------------------------------------------------------
    # RLS pool / OLS gain
    # ------------------------------------------------------------

    def _record_rls_sample(self, budget: int, imbalance: float) -> None:
        # Only accumulate once the pool exists.
        self._rls_pool.append((int(budget), float(imbalance)))

    def _ols_gain(self) -> float | None:
        if len(self._rls_pool) < 3:
            return None
        xs = [b for (b, _) in self._rls_pool]
        ys = [y for (_, y) in self._rls_pool]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        num = 0.0
        den = 0.0
        for x, yv in zip(xs, ys):
            dx = x - mean_x
            num += dx * (yv - mean_y)
            den += dx * dx
        if den < 1e-9:
            return None
        return num / den


# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------


def _safe_mean(window: deque[float]) -> float | None:
    if not window:
        return None
    return _stat.fmean(window)


def _safe_stdev(window: deque[float]) -> float | None:
    if len(window) < 2:
        return None
    return _stat.stdev(window)
