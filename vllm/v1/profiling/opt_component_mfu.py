#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.models.opt import OPTAttention, OPTDecoderLayer
    from vllm.v1.worker.opt_dynamic_replay import LayerReplayPlan

logger = init_logger(__name__)

OPT_COMPONENT_MFU_PROFILER_KEY = "opt_component_mfu_profiler"


@dataclass
class _PendingRecord:
    component: str
    layer_idx: int
    num_tokens: int
    replay_token_count: int
    num_actual_tokens: int
    replay_ratio: float
    flops: float
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


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
        peak_tflops: float | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_idx = step_idx
        self.rank = rank
        self.model_name = model_name
        self.total_scheduled_tokens = total_scheduled_tokens
        self.num_reqs = num_reqs
        self.peak_tflops = peak_tflops
        self._records: list[_PendingRecord] = []

    @contextmanager
    def profile_attention(
        self,
        layer: OPTAttention,
        hidden_states: torch.Tensor,
    ):
        with self._profile_component(
            component="attention",
            layer_idx=layer.layer_idx,
            num_tokens=_num_tokens(hidden_states),
            flops=self._estimate_attention_flops(layer, hidden_states),
            hidden_states=hidden_states,
        ):
            yield

    @contextmanager
    def profile_ffn(
        self,
        layer: OPTDecoderLayer,
        hidden_states: torch.Tensor,
    ):
        with self._profile_component(
            component="ffn",
            layer_idx=layer.layer_idx,
            num_tokens=_num_tokens(hidden_states),
            flops=self._estimate_ffn_flops(layer, hidden_states),
            hidden_states=hidden_states,
        ):
            yield

    @contextmanager
    def _profile_component(
        self,
        *,
        component: str,
        layer_idx: int,
        num_tokens: int,
        flops: float,
        hidden_states: torch.Tensor,
    ):
        if hidden_states.device.type != "cuda" or num_tokens <= 0:
            yield
            return

        replay_ratio, replay_token_count, num_actual_tokens = self._get_replay_stats(
            layer_idx
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            self._records.append(
                _PendingRecord(
                    component=component,
                    layer_idx=layer_idx,
                    num_tokens=num_tokens,
                    replay_token_count=replay_token_count,
                    num_actual_tokens=num_actual_tokens,
                    replay_ratio=replay_ratio,
                    flops=flops,
                    start_event=start_event,
                    end_event=end_event,
                )
            )

    def finish_step(self) -> None:
        if not self._records:
            return

        torch.cuda.synchronize()
        layer_records: list[dict[str, Any]] = []
        for record in self._records:
            elapsed_ms = record.start_event.elapsed_time(record.end_event)
            tflops = _compute_tflops(record.flops, elapsed_ms)
            entry = {
                "component": record.component,
                "layer_idx": record.layer_idx,
                "num_tokens": record.num_tokens,
                "replay_token_count": record.replay_token_count,
                "num_actual_tokens": record.num_actual_tokens,
                "replay_ratio": record.replay_ratio,
                "time_ms": elapsed_ms,
                "flops": record.flops,
                "tflops": tflops,
                "mfu": (
                    tflops / self.peak_tflops
                    if tflops is not None and self.peak_tflops
                    else None
                ),
            }
            layer_records.append(entry)

        payload = {
            "step": self.step_idx,
            "rank": self.rank,
            "model_name": self.model_name,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "num_reqs": self.num_reqs,
            "attention": self._aggregate_component(layer_records, "attention"),
            "ffn": self._aggregate_component(layer_records, "ffn"),
            "layers": layer_records,
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")

        self._records.clear()

    def _aggregate_component(
        self,
        layer_records: list[dict[str, Any]],
        component: str,
    ) -> dict[str, Any]:
        records = [
            record for record in layer_records if record["component"] == component
        ]
        if not records:
            return {
                "calls": 0,
                "time_ms": 0.0,
                "flops": 0.0,
                "tflops": None,
                "mfu": None,
                "weighted_replay_ratio": 0.0,
                "num_tokens": 0,
                "replay_token_count": 0,
            }

        total_time_ms = sum(record["time_ms"] for record in records)
        total_flops = sum(record["flops"] for record in records)
        total_num_tokens = sum(record["num_tokens"] for record in records)
        total_replay_tokens = sum(record["replay_token_count"] for record in records)
        weighted_replay_ratio = (
            sum(record["replay_ratio"] * record["flops"] for record in records)
            / total_flops
            if total_flops > 0
            else 0.0
        )
        tflops = _compute_tflops(total_flops, total_time_ms)
        return {
            "calls": len(records),
            "time_ms": total_time_ms,
            "flops": total_flops,
            "tflops": tflops,
            "mfu": (
                tflops / self.peak_tflops
                if tflops is not None and self.peak_tflops
                else None
            ),
            "weighted_replay_ratio": weighted_replay_ratio,
            "num_tokens": total_num_tokens,
            "replay_token_count": total_replay_tokens,
        }

    def _estimate_attention_flops(
        self,
        layer: OPTAttention,
        hidden_states: torch.Tensor,
    ) -> float:
        num_tokens = _num_tokens(hidden_states)
        qkv_flops = (
            2.0
            * num_tokens
            * layer.qkv_proj.input_size
            * layer.qkv_proj.output_size_per_partition
        )
        out_flops = (
            2.0
            * num_tokens
            * layer.out_proj.input_size_per_partition
            * layer.out_proj.output_size
        )

        attn_metadata = self._get_attention_metadata(layer.attn.layer_name)
        if attn_metadata is None:
            return qkv_flops + out_flops

        query_start_loc = _to_int_list(
            getattr(
                attn_metadata,
                "query_start_loc_cpu",
                getattr(attn_metadata, "query_start_loc", None),
            )
        )
        seq_lens = _to_int_list(
            getattr(
                attn_metadata,
                "_seq_lens_cpu",
                getattr(attn_metadata, "seq_lens", None),
            )
        )
        if query_start_loc is None or seq_lens is None or len(query_start_loc) < 2:
            return qkv_flops + out_flops

        attn_core_flops = 0.0
        for query_start, query_end, kv_len in zip(
            query_start_loc[:-1], query_start_loc[1:], seq_lens
        ):
            query_len = max(query_end - query_start, 0)
            if query_len == 0 or kv_len <= 0:
                continue
            query_len = min(query_len, kv_len)
            valid_pairs = query_len * (2 * kv_len - query_len + 1) / 2.0
            attn_core_flops += 4.0 * valid_pairs * layer.num_heads * layer.head_dim

        return qkv_flops + attn_core_flops + out_flops

    def _estimate_ffn_flops(
        self,
        layer: OPTDecoderLayer,
        hidden_states: torch.Tensor,
    ) -> float:
        num_tokens = _num_tokens(hidden_states)
        fc1_flops = (
            2.0
            * num_tokens
            * layer.fc1.input_size
            * layer.fc1.output_size_per_partition
        )
        fc2_flops = (
            2.0
            * num_tokens
            * layer.fc2.input_size_per_partition
            * layer.fc2.output_size
        )
        return fc1_flops + fc2_flops

    def _get_attention_metadata(self, layer_name: str) -> Any | None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            return attn_metadata.get(layer_name)
        return None

    def _get_replay_stats(self, layer_idx: int) -> tuple[float, int, int]:
        forward_context = get_forward_context()
        runtime = forward_context.layer_recompute_runtime
        if runtime is None:
            return 0.0, 0, 0

        plan = runtime.current_layer_plan(layer_idx)
        if plan is None:
            return 0.0, 0, 0
        return _replay_stats_from_plan(plan)


def _compute_tflops(flops: float, elapsed_ms: float) -> float | None:
    if elapsed_ms <= 0:
        return None
    return flops / (elapsed_ms * 1e-3) / 1e12


def _num_tokens(hidden_states: torch.Tensor) -> int:
    if hidden_states.ndim == 2:
        return int(hidden_states.shape[0])
    return int(hidden_states.numel() // hidden_states.shape[-1])


def _replay_stats_from_plan(plan: LayerReplayPlan) -> tuple[float, int, int]:
    num_actual_tokens = int(plan.num_actual_tokens)
    replay_token_count = int(plan.replay_token_count)
    replay_ratio = (
        replay_token_count / num_actual_tokens if num_actual_tokens > 0 else 0.0
    )
    return replay_ratio, replay_token_count, num_actual_tokens


def _to_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return [int(v) for v in values.detach().cpu().tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return None
