#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
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
        output_path: str,
        step_idx: int,
        rank: int,
        model_name: str,
        total_scheduled_tokens: int,
        num_reqs: int,
    ) -> None:
        self.output_path = Path(output_path).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_idx = step_idx
        self.rank = rank
        self.model_name = model_name
        self.total_scheduled_tokens = total_scheduled_tokens
        self.num_reqs = num_reqs
        self._layer_imbalance_ms: dict[int, float] = {}
        self._dynamic_replay_runtime: OPTDynamicReplayRuntime | None = None

    def attach_dynamic_replay_runtime(
        self,
        runtime: OPTDynamicReplayRuntime | None,
    ) -> None:
        self._dynamic_replay_runtime = runtime

    def set_layer_imbalance_ms(self, layer_idx: int, imbalance_ms: float) -> None:
        self._layer_imbalance_ms[int(layer_idx)] = float(imbalance_ms)

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

        layer_records = self._build_layer_records()

        payload = {
            "step": self.step_idx,
            "rank": self.rank,
            "model_name": self.model_name,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "num_reqs": self.num_reqs,
            "layers": layer_records,
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
        flat_path = self.output_path.with_name(
            f"{self.output_path.stem}.flat{self.output_path.suffix}"
        )
        with flat_path.open("a", encoding="utf-8") as f:
            for layer_record in layer_records:
                flat_row = {
                    "step": self.step_idx,
                    "rank": self.rank,
                    "model_name": self.model_name,
                    "total_scheduled_tokens": self.total_scheduled_tokens,
                    "num_reqs": self.num_reqs,
                    **layer_record,
                }
                json.dump(flat_row, f, sort_keys=True)
                f.write("\n")

        self._layer_imbalance_ms.clear()
        self._dynamic_replay_runtime = None

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
                if plan is not None:
                    replay_ratio, replay_token_count, num_actual_tokens = (
                        _replay_stats_from_plan(plan)
                    )
                    num_tokens = int(plan.scheduled_token_count)

                layer_entry: dict[str, Any] = {
                    "layer_idx": layer_idx,
                    "next_layer_idx": layer_idx + 1,
                    "compute_end_ms_from_anchor": None,
                    "load_ready_ms_from_anchor": None,
                    "imbalance_ms": runtime.get_layer_imbalance_ms(layer_idx),
                    "replay_ratio": replay_ratio,
                    "replay_token_count": replay_token_count,
                    "num_actual_tokens": num_actual_tokens,
                    "num_tokens": num_tokens,
                }
                per_layer[layer_idx] = layer_entry

                # compute_end from layer_end_event (on compute stream)
                if step_anchor is not None:
                    compute_end = runtime.get_layer_end_event(layer_idx)
                    if compute_end is not None:
                        layer_entry["compute_end_ms_from_anchor"] = float(
                            step_anchor.elapsed_time(compute_end)
                        )

                # load_ready for the *next* layer (on load stream)
                next_layer_idx = layer_idx + 1
                if next_layer_idx >= runtime.num_layers:
                    layer_entry["next_layer_idx"] = None
                    continue

                if step_anchor is not None:
                    load_ready = runtime.get_layer_load_ready_event(next_layer_idx)
                    hs_ready = runtime.get_layer_cpu_fill_ready_event(next_layer_idx)
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
