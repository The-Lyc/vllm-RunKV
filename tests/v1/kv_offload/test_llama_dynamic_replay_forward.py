# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.models import llama as llama_module
from vllm.model_executor.models.llama import LlamaModel
from vllm.v1.worker.opt_dynamic_replay import (
    LayerReplayPlan,
    OPTDynamicReplayRuntime,
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


class _FakePPGroup:
    is_first_rank = True
    is_last_rank = True
    rank_in_group = 0
    world_size = 1


class _RecordingLlamaLayer(nn.Module):
    def __init__(self, *, delta: float, expected_metadata: dict[str, object]):
        super().__init__()
        self.delta = delta
        self.expected_metadata = expected_metadata
        self.positions: list[torch.Tensor] = []
        self.hidden_states: list[torch.Tensor] = []
        self.residuals: list[torch.Tensor | None] = []

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert get_forward_context().attn_metadata is self.expected_metadata
        self.positions.append(positions.detach().clone())
        self.hidden_states.append(hidden_states.detach().clone())
        self.residuals.append(None if residual is None else residual.detach().clone())
        logical_input = hidden_states if residual is None else hidden_states + residual
        return logical_input + self.delta, torch.zeros_like(logical_input)


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, None


class _FakeLayerRecomputeManager:
    def __init__(self, cpu_fill_by_layer: dict[int, torch.Tensor]):
        self.cpu_fill_by_layer = {
            layer_idx: tensor.clone() for layer_idx, tensor in cpu_fill_by_layer.items()
        }
        self.capture_calls: list[tuple[int, torch.Tensor, np.ndarray, np.ndarray]] = []
        self.load_calls: list[int] = []

    def sync_cpu_fill_h2d(self, layer_idx: int) -> torch.Tensor | None:
        del layer_idx
        return None

    def load_cpu_fill_h2d(
        self,
        *,
        layer_idx: int,
        cpu_fill_positions: np.ndarray,
        cpu_fill_logical_ids: np.ndarray,
        cpu_fill_block_offsets: np.ndarray,
    ) -> torch.Tensor:
        del cpu_fill_positions, cpu_fill_logical_ids, cpu_fill_block_offsets
        self.load_calls.append(layer_idx)
        return self.cpu_fill_by_layer[layer_idx].clone()

    def capture_layer_input_d2h(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        req_indices: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        self.capture_calls.append(
            (
                layer_idx,
                hidden_states.detach().clone(),
                np.asarray(req_indices).copy(),
                np.asarray(positions).copy(),
            )
        )


def _make_layer_plan(
    *,
    kv_replay_start_per_req: list[int],
    computed_lens_per_req: list[int],
    prev_gpu_start_per_req: list[int],
    cpu_fill_token_count: int,
    gpu_reuse_token_count: int,
    replay_token_count: int,
    scheduled_token_count: int,
    num_actual_tokens: int,
    query_start_loc: list[int],
    combined_positions: list[int],
    combined_replay_indices: list[int],
    combined_scheduled_indices: list[int],
    cpu_fill_positions: list[int],
    cpu_fill_logical_ids: list[int],
    cpu_fill_block_offsets: list[int],
    gpu_reuse_slice_per_req: list[tuple[int, int]],
    max_query_len: int,
) -> LayerReplayPlan:
    return LayerReplayPlan(
        kv_replay_start_per_req=np.asarray(kv_replay_start_per_req, dtype=np.int32),
        computed_lens_per_req=np.asarray(computed_lens_per_req, dtype=np.int32),
        prev_gpu_start_per_req=np.asarray(prev_gpu_start_per_req, dtype=np.int32),
        replay_blocks_per_req=np.zeros(
            len(kv_replay_start_per_req), dtype=np.int32
        ),
        replay_block_count=0,
        skip_logical_block_ids=np.empty(0, dtype=np.int32),
        per_req_replay_block_ranges=np.zeros(
            (len(kv_replay_start_per_req), 2), dtype=np.int32
        ),
        cpu_fill_token_count=cpu_fill_token_count,
        gpu_reuse_token_count=gpu_reuse_token_count,
        replay_token_count=replay_token_count,
        scheduled_token_count=scheduled_token_count,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        slot_mapping=torch.arange(num_actual_tokens, dtype=torch.int64),
        combined_positions=torch.tensor(combined_positions, dtype=torch.int64),
        combined_replay_indices=torch.tensor(
            combined_replay_indices, dtype=torch.int64
        ),
        combined_scheduled_indices=torch.tensor(
            combined_scheduled_indices, dtype=torch.int64
        ),
        cpu_fill_positions=np.asarray(cpu_fill_positions, dtype=np.int32),
        cpu_fill_logical_ids=np.asarray(cpu_fill_logical_ids, dtype=np.int32),
        cpu_fill_block_offsets=np.asarray(cpu_fill_block_offsets, dtype=np.int32),
        gpu_reuse_slice_per_req=gpu_reuse_slice_per_req,
    )


def _build_model(monkeypatch, num_layers: int) -> LlamaModel:
    monkeypatch.setattr(llama_module, "get_pp_group", lambda: _FakePPGroup())
    model = LlamaModel.__new__(LlamaModel)
    nn.Module.__init__(model)
    model.config = type("_Config", (), {"hidden_size": 1})()
    model.start_layer = 0
    model.end_layer = num_layers
    model.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    model.norm = _FinalNorm()
    model.aux_hidden_state_layers = ()
    model._dynamic_replay_forward_logged = False
    model._dynamic_replay_nonzero_replay_logged = False
    return model


def test_llama_layer_zero_capture_uses_independent_snapshot() -> None:
    hidden_states = torch.tensor([[1.0], [2.0]])

    snapshot = LlamaModel._materialize_layer_input(hidden_states, residual=None)
    hidden_states.add_(10.0)

    assert torch.equal(snapshot, torch.tensor([[1.0], [2.0]]))


def test_llama_dynamic_replay_preserves_positions_and_residual_stream(
    monkeypatch,
) -> None:
    model = _build_model(monkeypatch, num_layers=2)
    metadata_0 = {"layer_0": object()}
    metadata_1 = {"layer_1": object()}
    layer_0 = _RecordingLlamaLayer(delta=1000.0, expected_metadata=metadata_0)
    layer_1 = _RecordingLlamaLayer(delta=2000.0, expected_metadata=metadata_1)
    model.layers = nn.ModuleList([layer_0, layer_1])

    runtime = OPTDynamicReplayRuntime(
        num_layers=2,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
        layer_recompute_manager=_FakeLayerRecomputeManager(
            {
                0: torch.tensor([[11.0], [21.0]]),
                1: torch.tensor([[30.0], [40.0]]),
            }
        ),
    )
    runtime.set_capture_token_metadata(
        req_indices=np.array([0, 1], dtype=np.int64),
        positions=np.array([2, 2], dtype=np.int64),
    )
    runtime.set_layer_plan(
        0,
        _make_layer_plan(
            kv_replay_start_per_req=[1, 1],
            computed_lens_per_req=[2, 2],
            prev_gpu_start_per_req=[2, 2],
            cpu_fill_token_count=2,
            gpu_reuse_token_count=0,
            replay_token_count=2,
            scheduled_token_count=2,
            num_actual_tokens=4,
            query_start_loc=[0, 2, 4],
            combined_positions=[1, 2, 1, 2],
            combined_replay_indices=[0, 2],
            combined_scheduled_indices=[1, 3],
            cpu_fill_positions=[1, 1],
            cpu_fill_logical_ids=[10, 11],
            cpu_fill_block_offsets=[1, 1],
            gpu_reuse_slice_per_req=[(0, 0), (0, 0)],
            max_query_len=2,
        ),
    )
    runtime.set_layer_metadata(0, metadata_0)
    runtime.set_layer_plan(
        1,
        _make_layer_plan(
            kv_replay_start_per_req=[0, 0],
            computed_lens_per_req=[2, 2],
            prev_gpu_start_per_req=[1, 1],
            cpu_fill_token_count=2,
            gpu_reuse_token_count=2,
            replay_token_count=4,
            scheduled_token_count=2,
            num_actual_tokens=6,
            query_start_loc=[0, 3, 6],
            combined_positions=[0, 1, 2, 0, 1, 2],
            combined_replay_indices=[0, 1, 3, 4],
            combined_scheduled_indices=[2, 5],
            cpu_fill_positions=[0, 0],
            cpu_fill_logical_ids=[10, 11],
            cpu_fill_block_offsets=[0, 0],
            gpu_reuse_slice_per_req=[(0, 1), (1, 2)],
            max_query_len=3,
        ),
    )
    runtime.set_layer_metadata(1, metadata_1)

    base_attn_metadata = {"base": object()}
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata=base_attn_metadata,
        virtual_engine=0,
        layer_recompute_runtime=runtime,
    )

    with override_forward_context(forward_context):
        output = model._forward_dynamic_replay(
            positions=torch.tensor([2, 2], dtype=torch.long),
            hidden_states=torch.tensor([[100.0], [200.0]]),
            residual=None,
        )

    assert torch.equal(
        layer_0.hidden_states[0],
        torch.tensor([[11.0], [100.0], [21.0], [200.0]]),
    )
    assert layer_0.residuals[0] is None
    assert torch.equal(layer_0.positions[0], torch.tensor([1, 2, 1, 2]))

    assert torch.equal(
        layer_1.hidden_states[0],
        torch.tensor([[30.0], [1011.0], [1100.0], [40.0], [1021.0], [1200.0]]),
    )
    assert layer_1.residuals[0] is None
    assert torch.equal(layer_1.positions[0], torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(output, torch.tensor([[3100.0], [3200.0]]))
    assert forward_context.attn_metadata is base_attn_metadata

    manager = runtime.layer_recompute_manager
    assert manager is not None
    assert manager.load_calls == [0, 1]
    assert len(manager.capture_calls) == 2
    assert manager.capture_calls[0][0] == 0
    assert torch.equal(manager.capture_calls[0][1], torch.tensor([[100.0], [200.0]]))
    assert manager.capture_calls[1][0] == 1
    assert torch.equal(manager.capture_calls[1][1], torch.tensor([[1100.0], [1200.0]]))


def test_llama_dynamic_replay_zero_replay_keeps_native_residual_path(
    monkeypatch,
) -> None:
    model = _build_model(monkeypatch, num_layers=1)
    metadata = {"layer_0": object()}
    layer = _RecordingLlamaLayer(delta=5.0, expected_metadata=metadata)
    model.layers = nn.ModuleList([layer])

    runtime = OPTDynamicReplayRuntime(
        num_layers=1,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
        layer_recompute_manager=_FakeLayerRecomputeManager({}),
    )
    runtime.set_capture_token_metadata(
        req_indices=np.array([0, 1], dtype=np.int64),
        positions=np.array([4, 5], dtype=np.int64),
    )
    runtime.set_layer_plan(
        0,
        _make_layer_plan(
            kv_replay_start_per_req=[1, 1],
            computed_lens_per_req=[1, 1],
            prev_gpu_start_per_req=[1, 1],
            cpu_fill_token_count=0,
            gpu_reuse_token_count=0,
            replay_token_count=0,
            scheduled_token_count=2,
            num_actual_tokens=2,
            query_start_loc=[0, 1, 2],
            combined_positions=[4, 5],
            combined_replay_indices=[],
            combined_scheduled_indices=[0, 1],
            cpu_fill_positions=[],
            cpu_fill_logical_ids=[],
            cpu_fill_block_offsets=[],
            gpu_reuse_slice_per_req=[(0, 0), (0, 0)],
            max_query_len=1,
        ),
    )
    runtime.set_layer_metadata(0, metadata)

    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"base": object()},
        virtual_engine=0,
        layer_recompute_runtime=runtime,
    )
    positions = torch.tensor([4, 5], dtype=torch.long)
    residual = torch.tensor([[10.0], [20.0]])
    with override_forward_context(forward_context):
        output = model._forward_dynamic_replay(
            positions=positions,
            hidden_states=torch.tensor([[1.0], [2.0]]),
            residual=residual,
        )

    assert torch.equal(layer.positions[0], positions)
    assert torch.equal(layer.residuals[0], residual)
    assert torch.equal(output, torch.tensor([[16.0], [27.0]]))
    manager = runtime.layer_recompute_manager
    assert manager is not None
    assert manager.load_calls == []
    assert len(manager.capture_calls) == 1
    assert torch.equal(manager.capture_calls[0][1], torch.tensor([[11.0], [22.0]]))
