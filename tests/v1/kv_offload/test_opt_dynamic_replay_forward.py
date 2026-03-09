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
from vllm.model_executor.models import opt as opt_module
from vllm.model_executor.models.opt import OPTDecoder
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


class _RecordingLayer(nn.Module):
    def __init__(
        self, *, layer_idx: int, delta: float, expected_metadata: dict[str, object]
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.delta = delta
        self.expected_metadata = expected_metadata
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().attn_metadata is self.expected_metadata
        self.inputs.append(hidden_states.detach().clone())
        return hidden_states + self.delta


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
    combined_replay_indices: list[int],
    combined_scheduled_indices: list[int],
    cpu_fill_positions: list[int],
    cpu_fill_logical_ids: list[int],
    cpu_fill_block_offsets: list[int],
    gpu_reuse_slice_per_req: list[tuple[int, int]],
    max_query_len: int = 3,
) -> LayerReplayPlan:
    return LayerReplayPlan(
        kv_replay_start_per_req=np.asarray(kv_replay_start_per_req, dtype=np.int32),
        computed_lens_per_req=np.asarray(computed_lens_per_req, dtype=np.int32),
        prev_gpu_start_per_req=np.asarray(prev_gpu_start_per_req, dtype=np.int32),
        cpu_fill_token_count=cpu_fill_token_count,
        gpu_reuse_token_count=gpu_reuse_token_count,
        replay_token_count=replay_token_count,
        scheduled_token_count=scheduled_token_count,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        slot_mapping=torch.arange(num_actual_tokens, dtype=torch.int64),
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


def _build_decoder(monkeypatch, num_layers: int) -> OPTDecoder:
    monkeypatch.setattr(opt_module, "get_pp_group", lambda: _FakePPGroup())
    decoder = OPTDecoder.__new__(OPTDecoder)
    nn.Module.__init__(decoder)
    decoder.config = type("_Config", (), {"hidden_size": 1})()
    decoder.start_layer = 0
    decoder.end_layer = num_layers
    decoder.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    decoder.final_layer_norm = None
    decoder.project_out = None
    return decoder


def test_opt_decoder_dynamic_replay_interleaves_cpu_fill_and_gpu_reuse(
    monkeypatch,
) -> None:
    decoder = _build_decoder(monkeypatch, num_layers=2)
    metadata_0 = {"layer_0": object()}
    metadata_1 = {"layer_1": object()}
    fake_layer_0 = _RecordingLayer(
        layer_idx=0, delta=1000.0, expected_metadata=metadata_0
    )
    fake_layer_1 = _RecordingLayer(
        layer_idx=1, delta=2000.0, expected_metadata=metadata_1
    )
    decoder.layers = nn.ModuleList([fake_layer_0, fake_layer_1])

    runtime = OPTDynamicReplayRuntime(
        num_layers=2,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
        layer_recompute_manager=_FakeLayerRecomputeManager(
            {
                0: torch.tensor([[10.0], [11.0], [20.0], [21.0]]),
                1: torch.tensor([[30.0], [40.0]]),
            }
        ),
    )
    runtime.set_capture_token_metadata(
        req_indices=np.array([0, 1], dtype=np.int64),
        positions=np.array([6, 7], dtype=np.int64),
    )
    runtime.set_layer_plan(
        0,
        _make_layer_plan(
            kv_replay_start_per_req=[0, 0],
            computed_lens_per_req=[2, 2],
            prev_gpu_start_per_req=[2, 2],
            cpu_fill_token_count=4,
            gpu_reuse_token_count=0,
            replay_token_count=4,
            scheduled_token_count=2,
            num_actual_tokens=6,
            query_start_loc=[0, 3, 6],
            combined_replay_indices=[0, 1, 3, 4],
            combined_scheduled_indices=[2, 5],
            cpu_fill_positions=[0, 1, 0, 1],
            cpu_fill_logical_ids=[10, 10, 11, 11],
            cpu_fill_block_offsets=[0, 1, 0, 1],
            gpu_reuse_slice_per_req=[(0, 0), (0, 0)],
        ),
    )
    runtime.set_layer_metadata(0, metadata_0)
    runtime.set_layer_plan(
        1,
        _make_layer_plan(
            kv_replay_start_per_req=[2, 2],
            computed_lens_per_req=[4, 4],
            prev_gpu_start_per_req=[3, 3],
            cpu_fill_token_count=2,
            gpu_reuse_token_count=2,
            replay_token_count=4,
            scheduled_token_count=2,
            num_actual_tokens=6,
            query_start_loc=[0, 3, 6],
            combined_replay_indices=[0, 1, 3, 4],
            combined_scheduled_indices=[2, 5],
            cpu_fill_positions=[2, 2],
            cpu_fill_logical_ids=[12, 13],
            cpu_fill_block_offsets=[0, 0],
            gpu_reuse_slice_per_req=[(1, 2), (3, 4)],
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
    inputs_embeds = torch.tensor([[100.0], [200.0]])
    positions = torch.zeros(2, dtype=torch.long)

    del positions
    with override_forward_context(forward_context):
        output = decoder._forward_dynamic_replay(inputs_embeds)

    assert torch.equal(
        fake_layer_0.inputs[0],
        torch.tensor([[10.0], [11.0], [100.0], [20.0], [21.0], [200.0]]),
    )
    assert torch.equal(
        fake_layer_1.inputs[0],
        torch.tensor([[30.0], [1011.0], [1100.0], [40.0], [1021.0], [1200.0]]),
    )
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
    np.testing.assert_array_equal(manager.capture_calls[1][2], np.array([0, 1]))
    np.testing.assert_array_equal(manager.capture_calls[1][3], np.array([6, 7]))


def test_opt_decoder_dynamic_replay_handles_zero_replay_path(monkeypatch) -> None:
    decoder = _build_decoder(monkeypatch, num_layers=1)
    metadata_0 = {"layer_0": object()}
    fake_layer = _RecordingLayer(layer_idx=0, delta=5.0, expected_metadata=metadata_0)
    decoder.layers = nn.ModuleList([fake_layer])

    runtime = OPTDynamicReplayRuntime(
        num_layers=1,
        cpu_hs_store=torch.empty(1, 1, 1),
        replay_plan_provider=_DummyReplayPlanProvider(),
        layer_recompute_manager=_FakeLayerRecomputeManager({}),
    )
    runtime.set_capture_token_metadata(
        req_indices=np.array([0, 1], dtype=np.int64),
        positions=np.array([2, 3], dtype=np.int64),
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
            combined_replay_indices=[],
            combined_scheduled_indices=[0, 1],
            cpu_fill_positions=[],
            cpu_fill_logical_ids=[],
            cpu_fill_block_offsets=[],
            gpu_reuse_slice_per_req=[(0, 0), (0, 0)],
            max_query_len=1,
        ),
    )
    runtime.set_layer_metadata(0, metadata_0)

    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"base": object()},
        virtual_engine=0,
        layer_recompute_runtime=runtime,
    )

    with override_forward_context(forward_context):
        output = decoder._forward_dynamic_replay(torch.tensor([[1.0], [2.0]]))

    assert torch.equal(fake_layer.inputs[0], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(output, torch.tensor([[6.0], [7.0]]))
    manager = runtime.layer_recompute_manager
    assert manager is not None
    assert manager.load_calls == []
    assert len(manager.capture_calls) == 1
    assert manager.capture_calls[0][0] == 0
