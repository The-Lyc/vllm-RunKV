#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.models.opt import OPTAttention, OPTDecoderLayer
    from vllm.v1.worker.opt_dynamic_replay import (
        LayerReplayPlan,
        OPTDynamicReplayRuntime,
    )

logger = init_logger(__name__)

OPT_COMPONENT_MFU_PROFILER_KEY = "opt_component_mfu_profiler"


def get_opt_component_mfu_profiler() -> OPTComponentMFUStepProfiler | None:
    if not is_forward_context_available():
        return None
    return get_forward_context().additional_kwargs.get(OPT_COMPONENT_MFU_PROFILER_KEY)


class OPTComponentMFUStepProfiler:
    def __init__(
        self,
        *,
        output_path: str | None,
        step_idx: int,
        rank: int,
        model_name: str,
        total_scheduled_tokens: int,
        num_reqs: int,
    ) -> None:
        self.output_path = (
            Path(output_path).expanduser() if output_path is not None else None
        )
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_idx = step_idx
        self.rank = rank
        self.model_name = model_name
        self.total_scheduled_tokens = total_scheduled_tokens
        self.num_reqs = num_reqs
        self._layer_imbalance_ms: dict[int, float] = {}
        # Per-layer controller update snapshots forwarded from the feedback
        # planner provider.  Only populated when planner == "feedback".
        self._layer_controller_updates: dict[int, dict[str, Any]] = {}
        self._dynamic_replay_runtime: OPTDynamicReplayRuntime | None = None

    def attach_dynamic_replay_runtime(
        self,
        runtime: OPTDynamicReplayRuntime | None,
    ) -> None:
        self._dynamic_replay_runtime = runtime

    def set_layer_imbalance_ms(self, layer_idx: int, imbalance_ms: float) -> None:
        self._layer_imbalance_ms[int(layer_idx)] = float(imbalance_ms)

    def set_layer_controller_update(
        self,
        layer_idx: int,
        update: dict[str, Any],
    ) -> None:
        """Record the feedback controller's budget update for *layer_idx*.

        Called from the pre-hook after ``observe_layer_feedback`` so the
        profiler can include budget dynamics in its JSONL output.
        """
        self._layer_controller_updates[int(layer_idx)] = update

    @contextmanager
    def profile_attention(
        self,
        layer: OPTAttention,
        hidden_states: torch.Tensor,
    ):
        del layer, hidden_states
        yield

    @contextmanager
    def profile_ffn(
        self,
        layer: OPTDecoderLayer,
        hidden_states: torch.Tensor,
    ):
        del layer, hidden_states
        yield

    def finish_step(self) -> None:
        if self._dynamic_replay_runtime is not None:
            torch.cuda.synchronize()

        if self.output_path is not None:
            self._write_step_record()

        self._layer_imbalance_ms.clear()
        self._layer_controller_updates.clear()
        self._dynamic_replay_runtime = None

    def _write_step_record(self) -> None:
        """Write one JSONL line (step record) and one flat line per layer."""
        import json

        layer_records = self._build_layer_records()

        step_record: dict[str, Any] = {
            "step": self.step_idx,
            "rank": self.rank,
            "model_name": self.model_name,
            "num_reqs": self.num_reqs,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "layers": layer_records,
        }

        assert self.output_path is not None
        # Main JSONL: one line per step (nested layers list)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(step_record) + "\n")

        # Flat JSONL: one line per (step, layer) — easier for per-layer analysis
        flat_path = self.output_path.with_suffix("").with_suffix(".flat.jsonl")
        with open(flat_path, "a") as f:
            for lr in layer_records:
                flat_rec = {
                    "step": self.step_idx,
                    "rank": self.rank,
                    "model_name": self.model_name,
                    "num_reqs": self.num_reqs,
                    "total_scheduled_tokens": self.total_scheduled_tokens,
                    **lr,
                }
                f.write(json.dumps(flat_rec) + "\n")

    def _build_layer_records(self) -> list[dict[str, Any]]:
        per_layer: dict[int, dict[str, Any]] = {}

        runtime = self._dynamic_replay_runtime
        if runtime is not None:
            step_anchor = runtime.get_step_anchor_event()

            for layer_idx in range(runtime.num_layers):
                plan = runtime.current_layer_plan(layer_idx)
                replay_ratio = None
                replay_token_count = None
                num_actual_tokens = None
                num_tokens = None
                cpu_fill_token_count = None
                gpu_reuse_token_count = None
                if plan is not None:
                    replay_ratio, replay_token_count, num_actual_tokens = (
                        _replay_stats_from_plan(plan)
                    )
                    num_tokens = int(plan.scheduled_token_count)
                    cpu_fill_token_count = int(plan.cpu_fill_token_count)
                    gpu_reuse_token_count = int(plan.gpu_reuse_token_count)

                layer_entry: dict[str, Any] = {
                    "layer_idx": layer_idx,
                    "next_layer_idx": layer_idx + 1,
                    "compute_start_ms_from_anchor": None,
                    "forward_start_ms_from_anchor": None,
                    "compute_end_ms_from_anchor": None,
                    "load_start_ms_from_anchor": None,
                    "load_ready_ms_from_anchor": None,
                    "kv_ready_ms_from_anchor": None,
                    "hs_ready_ms_from_anchor": None,
                    "imbalance_ms": runtime.get_layer_imbalance_ms(layer_idx),
                    # Feedback controller budget update for this layer.
                    # None when planner != "feedback" or no feedback observed.
                    "controller_update": self._layer_controller_updates.get(layer_idx),
                    "replay_ratio": replay_ratio,
                    "replay_token_count": replay_token_count,
                    "cpu_fill_token_count": cpu_fill_token_count,
                    "gpu_reuse_token_count": gpu_reuse_token_count,
                    "num_actual_tokens": num_actual_tokens,
                    "num_tokens": num_tokens,
                }
                per_layer[layer_idx] = layer_entry

                if step_anchor is not None:
                    # compute_start from layer_start_event (on compute stream)
                    compute_start = runtime.get_layer_start_event(layer_idx)
                    if compute_start is not None:
                        layer_entry["compute_start_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(compute_start)
                        )

                    # forward_start: right before layer() call, after
                    # cpu_fill sync and tensor scatter (on compute stream)
                    forward_start = runtime.get_layer_forward_start_event(layer_idx)
                    if forward_start is not None:
                        layer_entry["forward_start_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(forward_start)
                        )

                    # compute_end from layer_end_event (on compute stream)
                    compute_end = runtime.get_layer_end_event(layer_idx)
                    if compute_end is not None:
                        layer_entry["compute_end_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(compute_end)
                        )

                # load_start and load_ready for this layer (on load stream)
                if step_anchor is not None:
                    load_start = runtime.get_layer_load_start_event(layer_idx)
                    if load_start is not None:
                        layer_entry["load_start_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(load_start)
                        )

                # load_ready for the *next* layer (on load stream)
                next_layer_idx = layer_idx + 1
                if next_layer_idx >= runtime.num_layers:
                    layer_entry["next_layer_idx"] = None
                    continue

                if step_anchor is not None:
                    load_ready = runtime.get_layer_load_ready_event(next_layer_idx)
                    hs_ready = runtime.get_layer_cpu_fill_ready_event(next_layer_idx)
                    # Record separate KV-DMA and HS-DMA ready timestamps
                    if load_ready is not None:
                        layer_entry["kv_ready_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(load_ready)
                        )
                    if hs_ready is not None:
                        layer_entry["hs_ready_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(hs_ready)
                        )
                    # Combined: whichever finishes last
                    final_ready = hs_ready or load_ready
                    if final_ready is not None:
                        layer_entry["load_ready_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(final_ready)
                        )

        return [per_layer[layer_idx] for layer_idx in sorted(per_layer)]


def _replay_stats_from_plan(plan: LayerReplayPlan) -> tuple[float, int, int]:
    num_actual_tokens = int(plan.num_actual_tokens)
    replay_token_count = int(plan.replay_token_count)
    replay_ratio = (
        replay_token_count / num_actual_tokens if num_actual_tokens > 0 else 0.0
    )
    return replay_ratio, replay_token_count, num_actual_tokens
