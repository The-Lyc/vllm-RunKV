# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
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
"""Inference-only OPT model compatible with HuggingFace weights."""

from collections.abc import Iterable
from contextlib import contextmanager
from itertools import islice

import numpy as np
import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.v1.profiling.opt_component_mfu import get_opt_component_mfu_profiler

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.layer_idx = extract_layer_index(prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        profiler = get_opt_component_mfu_profiler()
        if profiler is None:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            attn_output = self.attn(q, k, v)
            output, _ = self.out_proj(attn_output)
            return output

        with profiler.profile_attention(self, hidden_states):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            attn_output = self.attn(q, k, v)
            output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: OPTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.layer_idx = extract_layer_index(prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        profiler = get_opt_component_mfu_profiler()
        if profiler is None:
            hidden_states, _ = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
        else:
            with profiler.profile_ffn(self, hidden_states):
                hidden_states, _ = self.fc1(hidden_states)
                hidden_states = self.activation_fn(hidden_states)
                hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):
    def __init__(
        self,
        config: OPTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.project_out",
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.project_in",
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OPTDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self._dynamic_replay_forward_logged = False
        self._dynamic_replay_nonzero_replay_logged = False

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

    def _assemble_replay_hidden_states(
        self,
        *,
        plan,
        cpu_fill_hidden_states: torch.Tensor | None,
        prev_layer_replay_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_size = self.config.hidden_size
        dtype = None
        device = None
        if cpu_fill_hidden_states is not None:
            dtype = cpu_fill_hidden_states.dtype
            device = cpu_fill_hidden_states.device
        elif prev_layer_replay_hidden_states is not None:
            dtype = prev_layer_replay_hidden_states.dtype
            device = prev_layer_replay_hidden_states.device
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
                assert cpu_fill_hidden_states is not None
                req_segments.append(
                    cpu_fill_hidden_states[
                        cpu_fill_cursor : cpu_fill_cursor + cpu_fill_len
                    ]
                )
                cpu_fill_cursor += cpu_fill_len

            gpu_reuse_len = int(gpu_end - gpu_start)
            if gpu_reuse_len > 0:
                if prev_layer_replay_hidden_states is None:
                    raise AssertionError(
                        "gpu_reuse requires previous layer replay hidden states."
                    )
                req_segments.append(prev_layer_replay_hidden_states[gpu_start:gpu_end])

            if req_segments:
                replay_segments.append(torch.cat(req_segments, dim=0))

        if (
            cpu_fill_hidden_states is not None
            and cpu_fill_cursor != cpu_fill_hidden_states.shape[0]
        ):
            raise AssertionError(
                "Replay assembly did not consume all cpu_fill hidden states."
            )

        if replay_segments:
            replay_hidden_states = torch.cat(replay_segments, dim=0)
        else:
            replay_hidden_states = torch.empty(
                (0, hidden_size),
                dtype=dtype,
                device=device,
            )

        if replay_hidden_states.shape[0] != plan.replay_token_count:
            raise AssertionError(
                f"Expected {plan.replay_token_count} replay tokens but assembled "
                f"{replay_hidden_states.shape[0]}."
            )
        return replay_hidden_states

    def _forward_dynamic_replay(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | IntermediateTensors:
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise NotImplementedError(
                "Dynamic replay currently requires pipeline parallel size 1."
            )

        forward_context = get_forward_context()
        runtime = forward_context.layer_recompute_runtime
        assert runtime is not None

        scheduled_hidden_states = hidden_states
        replay_hidden_states: torch.Tensor | None = None
        if not self._dynamic_replay_forward_logged:
            logger.info(
                "OPT dynamic replay forward entered: start_layer=%d, end_layer=%d, "
                "scheduled_tokens=%d, hidden_size=%d.",
                self.start_layer,
                self.end_layer,
                int(hidden_states.shape[0]),
                int(hidden_states.shape[-1]),
            )
            self._dynamic_replay_forward_logged = True
        runtime.capture_scheduled_layer_input(
            target_layer_idx=self.start_layer,
            hidden_states=scheduled_hidden_states,
        )

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            plan = runtime.get_layer_plan(layer_idx)

            if plan.replay_token_count == 0:
                with self._with_runtime_attn_metadata(layer_idx=layer_idx):
                    scheduled_hidden_states = layer(scheduled_hidden_states)
                replay_hidden_states = None
            else:
                if not self._dynamic_replay_nonzero_replay_logged:
                    logger.info(
                        "OPT dynamic replay active on layer %d: replay_tokens=%d "
                        "(cpu_fill=%d, gpu_reuse=%d), scheduled_tokens=%d, "
                        "num_actual_tokens=%d.",
                        layer_idx,
                        int(plan.replay_token_count),
                        int(plan.cpu_fill_token_count),
                        int(plan.gpu_reuse_token_count),
                        int(plan.scheduled_token_count),
                        int(plan.num_actual_tokens),
                    )
                    self._dynamic_replay_nonzero_replay_logged = True
                cpu_fill_hidden_states = None
                if plan.cpu_fill_token_count > 0:
                    cpu_fill_hidden_states = runtime.load_cpu_fill(layer_idx, plan)

                replay_hidden_states = self._assemble_replay_hidden_states(
                    plan=plan,
                    cpu_fill_hidden_states=cpu_fill_hidden_states,
                    prev_layer_replay_hidden_states=replay_hidden_states,
                )

                replay_indices = plan.combined_replay_indices.to(
                    scheduled_hidden_states.device, dtype=torch.long
                )
                scheduled_indices = plan.combined_scheduled_indices.to(
                    scheduled_hidden_states.device, dtype=torch.long
                )
                combined_hidden_states = torch.empty(
                    (plan.num_actual_tokens, scheduled_hidden_states.shape[-1]),
                    dtype=scheduled_hidden_states.dtype,
                    device=scheduled_hidden_states.device,
                )
                combined_hidden_states[replay_indices] = replay_hidden_states
                combined_hidden_states[scheduled_indices] = scheduled_hidden_states

                with self._with_runtime_attn_metadata(layer_idx=layer_idx):
                    # commit replay version computation to the computing stream
                    combined_hidden_states = layer(combined_hidden_states)

                replay_hidden_states = combined_hidden_states.index_select(
                    0, replay_indices
                )
                scheduled_hidden_states = combined_hidden_states.index_select(
                    0, scheduled_indices
                )

            layer_end_event: torch.cuda.Event | None = None
            if scheduled_hidden_states.device.type == "cuda":
                # Record the exact end of this layer's forward so later steps
                # can compare it against the next layer's ready timestamp.
                layer_end_event = torch.cuda.Event(enable_timing=True)
                layer_end_event.record()
            runtime.set_layer_end_event(layer_idx, layer_end_event)

            runtime.capture_scheduled_layer_input(
                target_layer_idx=layer_idx + 1,
                hidden_states=scheduled_hidden_states,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": scheduled_hidden_states})
        if self.final_layer_norm is not None:
            scheduled_hidden_states = self.final_layer_norm(scheduled_hidden_states)
        if self.project_out is not None:
            scheduled_hidden_states, _ = self.project_out(scheduled_hidden_states)
        return scheduled_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        runtime = get_forward_context().layer_recompute_runtime
        if runtime is not None:
            return self._forward_dynamic_replay(hidden_states)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


@support_torch_compile
class OPTModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.decoder = OPTDecoder(
            config, cache_config, quant_config, prefix=f"{prefix}.decoder"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.decoder(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
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


class OPTForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "decoder.": "model.decoder.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.word_embed_proj_dim,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.weight"] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
