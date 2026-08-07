# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""

from collections.abc import Iterable
from contextlib import contextmanager
from itertools import islice

import numpy as np
import torch
import torch.cuda.nvtx as _nvtx
from torch import nn
from transformers import LlamaConfig

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        llama_4_scaling_config = getattr(config, "llama_4_scaling", None)
        self.do_llama_4_scaling = llama_4_scaling_config is not None
        if self.do_llama_4_scaling:
            self.llama_4_scaling_original_max_position_embeddings = (
                llama_4_scaling_config["original_max_position_embeddings"]
            )
            self.llama_4_scaling_beta = llama_4_scaling_config["beta"]

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            if hasattr(config, "target_layer_count"):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), (
                f"effective_layer_idx: {effective_layer_idx} "
                f"is out of bounds for layer_types: {layer_types}"
            )

            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        # Llama4 scaling
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1
            + torch.floor(
                positions / self.llama_4_scaling_original_max_position_embeddings
            )
        )
        # Broadcast over head_dim
        return scaling.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.do_llama_4_scaling:
            attn_scale = self._get_llama_4_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        return vllm_config.quant_config


def llama_model_invariants(
    input_ids, positions, intermediate_tensors=None, inputs_embeds=None
):
    """Shape invariants for Llama model compilation, those are translated to
    runtime assertions for unbacked dynamic shapes and are compiled away for
    backed"""
    if input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])


@support_torch_compile(shape_invariants=llama_model_invariants)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()
        self._dynamic_replay_forward_logged = False
        self._dynamic_replay_nonzero_replay_logged = False

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @contextmanager
    def _with_runtime_attn_metadata(
        self,
        *,
        layer_idx: int,
    ):
        forward_context = get_forward_context()
        prev_attn_metadata = forward_context.attn_metadata
        runtime = forward_context.layer_recompute_runtime
        assert runtime is not None
        forward_context.attn_metadata = runtime.get_layer_metadata(layer_idx)
        try:
            yield
        finally:
            forward_context.attn_metadata = prev_attn_metadata

    @staticmethod
    def _materialize_layer_input(
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the logical residual-stream value at a decoder boundary."""
        if residual is None:
            # Layer 0 aliases `residual` to its input and later updates that
            # residual in-place. Dynamic replay captures this tensor on a D2H
            # stream before entering the layer, so keep an independent snapshot
            # to prevent the copy from racing with the fused residual update.
            return hidden_states.clone()
        return hidden_states + residual

    def _assemble_replay_layer_inputs(
        self,
        *,
        plan,
        cpu_fill_layer_inputs: torch.Tensor | None,
        prev_layer_replay_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = None
        device = None
        if cpu_fill_layer_inputs is not None:
            dtype = cpu_fill_layer_inputs.dtype
            device = cpu_fill_layer_inputs.device
        elif prev_layer_replay_inputs is not None:
            dtype = prev_layer_replay_inputs.dtype
            device = prev_layer_replay_inputs.device
        else:
            raise AssertionError("Replay assembly requires at least one input.")

        cpu_fill_cursor = 0
        replay_segments: list[torch.Tensor] = []
        cpu_fill_lens_per_req = (
            np.minimum(plan.prev_gpu_start_per_req, plan.computed_lens_per_req)
            - plan.kv_replay_start_per_req
        ).clip(min=0)

        for req_idx, (gpu_start, gpu_end) in enumerate(plan.gpu_reuse_slice_per_req):
            req_segments: list[torch.Tensor] = []

            cpu_fill_len = int(cpu_fill_lens_per_req[req_idx])
            if cpu_fill_len > 0:
                assert cpu_fill_layer_inputs is not None
                req_segments.append(
                    cpu_fill_layer_inputs[
                        cpu_fill_cursor : cpu_fill_cursor + cpu_fill_len
                    ]
                )
                cpu_fill_cursor += cpu_fill_len

            gpu_reuse_len = int(gpu_end - gpu_start)
            if gpu_reuse_len > 0:
                if prev_layer_replay_inputs is None:
                    raise AssertionError(
                        "gpu_reuse requires previous layer replay inputs."
                    )
                req_segments.append(prev_layer_replay_inputs[gpu_start:gpu_end])

            if req_segments:
                replay_segments.append(torch.cat(req_segments, dim=0))

        if (
            cpu_fill_layer_inputs is not None
            and cpu_fill_cursor != cpu_fill_layer_inputs.shape[0]
        ):
            raise AssertionError(
                "Replay assembly did not consume all cpu_fill layer inputs."
            )

        if replay_segments:
            replay_layer_inputs = torch.cat(replay_segments, dim=0)
        else:
            replay_layer_inputs = torch.empty(
                (0, self.config.hidden_size),
                dtype=dtype,
                device=device,
            )

        if replay_layer_inputs.shape[0] != plan.replay_token_count:
            raise AssertionError(
                f"Expected {plan.replay_token_count} replay tokens but assembled "
                f"{replay_layer_inputs.shape[0]}."
            )
        return replay_layer_inputs

    def _forward_dynamic_replay(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise NotImplementedError(
                "Dynamic replay currently requires pipeline parallel size 1."
            )

        runtime = get_forward_context().layer_recompute_runtime
        assert runtime is not None

        scheduled_hidden_states = hidden_states
        scheduled_residual = residual
        scheduled_layer_inputs = self._materialize_layer_input(
            scheduled_hidden_states, scheduled_residual
        )
        replay_layer_inputs: torch.Tensor | None = None
        aux_hidden_states: list[torch.Tensor] = []

        if not self._dynamic_replay_forward_logged:
            logger.info(
                "Llama dynamic replay forward entered: start_layer=%d, "
                "end_layer=%d, scheduled_tokens=%d, hidden_size=%d.",
                self.start_layer,
                self.end_layer,
                int(hidden_states.shape[0]),
                int(hidden_states.shape[-1]),
            )
            self._dynamic_replay_forward_logged = True

        runtime.capture_scheduled_layer_input(
            target_layer_idx=self.start_layer,
            hidden_states=scheduled_layer_inputs,
        )

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            relative_idx = layer_idx - self.start_layer
            if relative_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(scheduled_layer_inputs)

            plan = runtime.get_layer_plan(layer_idx)
            has_next_layer = layer_idx + 1 < self.end_layer
            _nvtx.range_push(f"runkv:layer_compute:L{layer_idx}")

            layer_start_event: torch.cuda.Event | None = None
            if scheduled_hidden_states.device.type == "cuda":
                layer_start_event = torch.cuda.Event(enable_timing=True)
                layer_start_event.record()
            runtime.set_layer_start_event(layer_idx, layer_start_event)

            if plan.replay_token_count == 0:
                fwd_start_event: torch.cuda.Event | None = None
                if scheduled_hidden_states.device.type == "cuda":
                    fwd_start_event = torch.cuda.Event(enable_timing=True)
                    fwd_start_event.record()
                runtime.set_layer_forward_start_event(layer_idx, fwd_start_event)

                with self._with_runtime_attn_metadata(layer_idx=layer_idx):
                    scheduled_hidden_states, scheduled_residual = layer(
                        positions,
                        scheduled_hidden_states,
                        scheduled_residual,
                    )
                replay_layer_inputs = None
                if has_next_layer:
                    scheduled_layer_inputs = self._materialize_layer_input(
                        scheduled_hidden_states, scheduled_residual
                    )
            else:
                if not self._dynamic_replay_nonzero_replay_logged:
                    logger.info(
                        "Llama dynamic replay active on layer %d: "
                        "replay_tokens=%d (cpu_fill=%d, gpu_reuse=%d), "
                        "scheduled_tokens=%d, num_actual_tokens=%d.",
                        layer_idx,
                        int(plan.replay_token_count),
                        int(plan.cpu_fill_token_count),
                        int(plan.gpu_reuse_token_count),
                        int(plan.scheduled_token_count),
                        int(plan.num_actual_tokens),
                    )
                    self._dynamic_replay_nonzero_replay_logged = True

                cpu_fill_layer_inputs = None
                if plan.cpu_fill_token_count > 0:
                    cpu_fill_layer_inputs = runtime.load_cpu_fill(layer_idx, plan)

                replay_layer_inputs = self._assemble_replay_layer_inputs(
                    plan=plan,
                    cpu_fill_layer_inputs=cpu_fill_layer_inputs,
                    prev_layer_replay_inputs=replay_layer_inputs,
                )

                replay_indices = plan.combined_replay_indices
                scheduled_indices = plan.combined_scheduled_indices
                combined_layer_inputs = torch.empty(
                    (plan.num_actual_tokens, scheduled_layer_inputs.shape[-1]),
                    dtype=scheduled_layer_inputs.dtype,
                    device=scheduled_layer_inputs.device,
                )
                combined_layer_inputs[replay_indices] = replay_layer_inputs
                combined_layer_inputs[scheduled_indices] = scheduled_layer_inputs

                fwd_start_event = None
                if combined_layer_inputs.device.type == "cuda":
                    fwd_start_event = torch.cuda.Event(enable_timing=True)
                    fwd_start_event.record()
                runtime.set_layer_forward_start_event(layer_idx, fwd_start_event)

                with self._with_runtime_attn_metadata(layer_idx=layer_idx):
                    combined_hidden_states, combined_residual = layer(
                        plan.combined_positions,
                        combined_layer_inputs,
                        None,
                    )

                if has_next_layer:
                    combined_layer_outputs = self._materialize_layer_input(
                        combined_hidden_states, combined_residual
                    )
                    replay_layer_inputs = combined_layer_outputs.index_select(
                        0, replay_indices
                    )
                    scheduled_layer_inputs = combined_layer_outputs.index_select(
                        0, scheduled_indices
                    )
                else:
                    replay_layer_inputs = None
                scheduled_hidden_states = combined_hidden_states.index_select(
                    0, scheduled_indices
                )
                scheduled_residual = combined_residual.index_select(
                    0, scheduled_indices
                )

            _nvtx.range_pop()

            layer_end_event: torch.cuda.Event | None = None
            if scheduled_hidden_states.device.type == "cuda":
                layer_end_event = torch.cuda.Event(enable_timing=True)
                layer_end_event.record()
            runtime.set_layer_end_event(layer_idx, layer_end_event)

            if (
                runtime.async_plan_build_enabled
                and layer_idx + 2 < self.end_layer
            ):
                runtime.submit_speculative_build(
                    target_layer_idx=layer_idx + 2,
                    current_plan=runtime.get_layer_plan(layer_idx + 1),
                )

            if has_next_layer:
                runtime.capture_scheduled_layer_input(
                    target_layer_idx=layer_idx + 1,
                    hidden_states=scheduled_layer_inputs,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": scheduled_hidden_states,
                    "residual": scheduled_residual,
                }
            )

        scheduled_hidden_states, _ = self.norm(
            scheduled_hidden_states, scheduled_residual
        )
        if aux_hidden_states:
            return scheduled_hidden_states, aux_hidden_states
        return scheduled_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        runtime = get_forward_context().layer_recompute_runtime
        if runtime is not None:
            return self._forward_dynamic_replay(positions, hidden_states, residual)

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Override to return default layers for Llama

        Note: The GPU model runner will override this with layers from
        the speculative config if available, providing dynamic configuration.
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights
        )

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int, attn_out: int):
            attn_in = self.config.head_dim * n_heads

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        # If using quantized model in mistral format,
        # quantization scales (qscale_weight) also need to be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_key_value_heads, self.config.hidden_size
            )
        elif (
            "wk" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads, 1)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_attention_heads, self.config.hidden_size
            )
        elif (
            "wq" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads, 1)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = f"{item}.{next_item}" if next_item is not None else None

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight


class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass
