# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm._custom_ops as custom_ops
from vllm.attention.layer import Attention
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import PagedBlockMapper

logger = init_logger(__name__)


@dataclass
class _PendingLayerWrite:
    hs_cpu: torch.Tensor
    logical_ids: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray


@dataclass
class _PrefetchedLayerInputs:
    hs_gpu: torch.Tensor
    pos_gpu: torch.Tensor
    slot_gpu: torch.Tensor
    ready_event: torch.cuda.Event | None


class LayerRecomputeManager:
    """Owns hidden-state snapshots and recompute helpers for RunKV hybrid IO."""

    def __init__(
        self,
        *,
        device: torch.device,
        kv_offload_config: RunKVOffloadConfig,
        kv_cache_config: KVCacheConfig,
        hidden_size: int,
        block_size: int,
        num_layers: int,
        model: torch.nn.Module,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.model = model
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.num_layers = int(num_layers)
        self.num_cpu_blocks = int(kv_cache_config.num_blocks)

        self.enable_layer_recompute = bool(kv_offload_config.enable_layer_recompute)
        self.layer_recompute_io_prefix_blocks = list(
            kv_offload_config.layer_recompute_io_prefix_blocks
        )
        self.layer_recompute_measure_overhead = bool(
            kv_offload_config.layer_recompute_measure_overhead
        )
        self.pin_memory = self.device.type == "cuda"
        # Preallocate full CPU hidden-state store, one block-aligned slot per KV
        # logical block, so KV and HS capacities always match exactly.
        self.cpu_attn_inputs_by_layer: list[torch.Tensor] = [
            torch.empty(
                self.num_cpu_blocks,
                self.block_size,
                self.hidden_size,
                dtype=dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            for _ in range(self.num_layers)
        ]
        self.cpu_block_positions = torch.full(
            (self.num_cpu_blocks, self.block_size),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.cpu_block_valid_lens = torch.zeros(
            self.num_cpu_blocks,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        hs_bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        hs_bytes_per_block = (
            self.num_layers * self.block_size * self.hidden_size * hs_bytes_per_elem
        )
        hs_total_bytes = self.num_cpu_blocks * hs_bytes_per_block
        meta_total_bytes = (
            self.cpu_block_positions.numel() * self.cpu_block_positions.element_size()
            + self.cpu_block_valid_lens.numel()
            * self.cpu_block_valid_lens.element_size()
        )
        logger.info_once(
            "RunKV layer recompute HS store: num_blocks=%d, total_size=%.2f GiB, "
            "hs_bytes_per_block=%d, block_size_tokens=%d, hidden_size=%d, "
            "num_layers=%d, dtype=%s, meta_size=%.2f MiB",
            self.num_cpu_blocks,
            hs_total_bytes / (1024**3),
            hs_bytes_per_block,
            self.block_size,
            self.hidden_size,
            self.num_layers,
            str(dtype),
            meta_total_bytes / (1024**2),
            scope="global",
        )

        self.logical_id_owner_req_id: dict[int, str] = {}

        self._step_req_indices_np: np.ndarray | None = None
        self._step_positions_np: np.ndarray | None = None
        self._step_logical_ids_np: np.ndarray | None = None
        self._step_block_offsets_np: np.ndarray | None = None
        # logical block id -> block index within its owner request for current step.
        self._step_logical_block_indices: dict[int, int] = {}

        self._layernorm_hook_handles: list[Any] = []
        self._pending_writes_by_layer: dict[int, list[_PendingLayerWrite]] = (
            defaultdict(list)
        )
        self._hs_d2h_events_by_layer: dict[int, list[torch.cuda.Event]] = defaultdict(
            list
        )
        self._prefetched_inputs_by_layer: dict[int, _PrefetchedLayerInputs] = {}
        self._hs_d2h_stream: torch.cuda.Stream | None = None
        self._hs_h2d_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda":
            self._hs_d2h_stream = torch.cuda.Stream(device=self.device)
            self._hs_h2d_stream = torch.cuda.Stream(device=self.device)

        (
            self._decoder_layers,
            self._layernorm_module_name,
            self._recompute_model_type,
        ) = self._resolve_model_layers()
        if len(self._decoder_layers) != self.num_layers:
            logger.warning(
                "LayerRecomputeManager initialized with num_layers=%d but model "
                "contains %d decoder layers for recompute.",
                self.num_layers,
                len(self._decoder_layers),
            )

    def register_layernorm_hooks(self, gpu_model_runner: Any) -> None:
        """Attach hooks to the decoder layer norm before self-attention."""
        self.remove_layernorm_hooks()
        for layer_idx, decoder_layer in enumerate(self._decoder_layers):
            layernorm_module = getattr(decoder_layer, self._layernorm_module_name, None)
            if layernorm_module is None:
                raise ValueError(
                    "Layer recompute hook registration failed: decoder layer "
                    f"{layer_idx} missing {self._layernorm_module_name!r}."
                )
            hook = functools.partial(
                self._layernorm_hook,
                layer_idx=layer_idx,
                gpu_model_runner=gpu_model_runner,
            )
            handle = layernorm_module.register_forward_hook(hook)
            self._layernorm_hook_handles.append(handle)

    def remove_layernorm_hooks(self) -> None:
        for handle in self._layernorm_hook_handles:
            handle.remove()
        self._layernorm_hook_handles.clear()

    def begin_step(
        self,
        *,
        req_ids: list[str],
        req_indices_np: np.ndarray,
        positions_np: np.ndarray,
        logical_ids_np: np.ndarray,
        block_offsets_np: np.ndarray,
        logical_block_table_np: np.ndarray,
        num_blocks_per_row: np.ndarray,
    ) -> None:
        """Cache current-step token metadata and reset ownership-changed blocks."""
        self._prefetched_inputs_by_layer.clear()
        self._step_req_indices_np = req_indices_np
        self._step_positions_np = positions_np
        self._step_logical_ids_np = logical_ids_np
        self._step_block_offsets_np = block_offsets_np
        self._step_logical_block_indices.clear()

        num_reqs = min(len(req_ids), logical_block_table_np.shape[0])
        for row in range(num_reqs):
            req_id = req_ids[row]
            row_blocks = int(num_blocks_per_row[row])
            for col in range(row_blocks):
                logical_id = int(logical_block_table_np[row, col])
                if logical_id < 0 or logical_id >= self.num_cpu_blocks:
                    continue
                self._step_logical_block_indices[logical_id] = col
                prev_owner = self.logical_id_owner_req_id.get(logical_id)
                if prev_owner != req_id:
                    self.logical_id_owner_req_id[logical_id] = req_id
                    self.cpu_block_positions[logical_id, :].fill_(-1)
                    self.cpu_block_valid_lens[logical_id] = 0

    def compute_skip_block_ids_for_layer(
        self,
        *,
        layer_idx: int,
        gid: int,
        mapper: PagedBlockMapper,
        dirty_blocks: set[int],
        io_prefix_blocks: list[int],
    ) -> set[int]:
        del gid, dirty_blocks
        prefix_blocks = 0
        if layer_idx < len(io_prefix_blocks):
            prefix_blocks = int(io_prefix_blocks[layer_idx])

        skip_block_ids: set[int] = set()
        for logical_id in mapper.mapping:
            block_idx = self._step_logical_block_indices.get(logical_id)
            if block_idx is None:
                continue
            if block_idx >= prefix_blocks:
                skip_block_ids.add(logical_id)

        return skip_block_ids

    def prefetch_recompute_inputs_for_layer(
        self,
        *,
        layer_idx: int,
        layer_name: str,
        gid: int,
        mapper: PagedBlockMapper,
        skip_block_ids: set[int],
    ) -> None:
        """Stage suffix hidden states/positions/slots to GPU for recompute."""
        del layer_name, gid
        if self.device.type != "cuda" or not skip_block_ids:
            return
        if layer_idx in self._prefetched_inputs_by_layer:
            return
        prefetched = self._build_prefetched_inputs(
            layer_idx=layer_idx,
            mapper=mapper,
            skip_block_ids=skip_block_ids,
        )
        if prefetched is not None:
            self._prefetched_inputs_by_layer[layer_idx] = prefetched

    def recompute_kv_for_layer(
        self,
        *,
        layer_idx: int,
        layer_name: str,
        gid: int,
        mapper: PagedBlockMapper,
        attn_module: Attention,
        skip_block_ids: set[int],
    ) -> None:
        """Recompute K/V for skipped blocks and write them into staging cache."""
        del layer_name, gid
        if not skip_block_ids:
            return
        if self.device.type != "cuda":
            return

        prefetched = self._prefetched_inputs_by_layer.pop(layer_idx, None)
        if prefetched is None:
            prefetched = self._build_prefetched_inputs(
                layer_idx=layer_idx,
                mapper=mapper,
                skip_block_ids=skip_block_ids,
            )
            if prefetched is None:
                return
        if prefetched.ready_event is not None:
            torch.cuda.current_stream(device=self.device).wait_event(
                prefetched.ready_event
            )

        hs_gpu = prefetched.hs_gpu
        pos_gpu = prefetched.pos_gpu
        slot_gpu = prefetched.slot_gpu

        k, v = self._project_kv_for_recompute(
            layer_idx=layer_idx,
            hs_gpu=hs_gpu,
            pos_gpu=pos_gpu,
        )

        forward_context = get_forward_context()
        kv_cache = attn_module.kv_cache[forward_context.virtual_engine]
        key_cache, value_cache = kv_cache.unbind(0)

        custom_ops.reshape_and_cache_flash(
            k,
            v,
            key_cache,
            value_cache,
            slot_gpu,
            attn_module.kv_cache_dtype,
            attn_module._k_scale,
            attn_module._v_scale,
        )

    def _build_prefetched_inputs(
        self,
        *,
        layer_idx: int,
        mapper: PagedBlockMapper,
        skip_block_ids: set[int],
    ) -> _PrefetchedLayerInputs | None:
        hs_chunks: list[torch.Tensor] = []
        pos_chunks: list[torch.Tensor] = []
        slot_chunks: list[torch.Tensor] = []
        layer_store = self.cpu_attn_inputs_by_layer[layer_idx]

        for logical_id in sorted(skip_block_ids):
            if logical_id < 0 or logical_id >= self.num_cpu_blocks:
                continue
            slot = mapper.mapping.get(logical_id)
            if slot is None:
                continue
            block_hs = layer_store[logical_id]

            valid_len = int(self.cpu_block_valid_lens[logical_id].item())
            if valid_len <= 0:
                continue
            valid_len = min(valid_len, mapper.block_size)

            positions = self.cpu_block_positions[logical_id, :valid_len]
            if torch.any(positions < 0):
                continue

            hs_chunks.append(block_hs[:valid_len, :])
            pos_chunks.append(positions.to(torch.int64))
            slot_chunks.append(
                torch.arange(valid_len, dtype=torch.int64) + slot * mapper.block_size
            )

        if not hs_chunks:
            return None

        hs_cat_cpu = torch.cat(hs_chunks, dim=0)
        pos_cat_cpu = torch.cat(pos_chunks, dim=0)
        slot_cat_cpu = torch.cat(slot_chunks, dim=0)

        if self.pin_memory and not hs_cat_cpu.is_pinned():
            hs_cat_cpu = hs_cat_cpu.pin_memory()
        if self.pin_memory and not pos_cat_cpu.is_pinned():
            pos_cat_cpu = pos_cat_cpu.pin_memory()
        if self.pin_memory and not slot_cat_cpu.is_pinned():
            slot_cat_cpu = slot_cat_cpu.pin_memory()

        if self._hs_h2d_stream is not None:
            with torch.cuda.stream(self._hs_h2d_stream):
                hs_gpu = hs_cat_cpu.to(self.device, non_blocking=True)
                pos_gpu = pos_cat_cpu.to(self.device, non_blocking=True)
                slot_gpu = slot_cat_cpu.to(self.device, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(self._hs_h2d_stream)
        else:
            hs_gpu = hs_cat_cpu.to(self.device)
            pos_gpu = pos_cat_cpu.to(self.device)
            slot_gpu = slot_cat_cpu.to(self.device)
            ready_event = None

        return _PrefetchedLayerInputs(
            hs_gpu=hs_gpu,
            pos_gpu=pos_gpu,
            slot_gpu=slot_gpu,
            ready_event=ready_event,
        )

    def sync_hs_d2h(self) -> None:
        """Synchronize pending D2H copies and materialize hidden-state snapshots."""
        for events in self._hs_d2h_events_by_layer.values():
            for event in events:
                event.synchronize()

        for layer_idx, pending_writes in self._pending_writes_by_layer.items():
            layer_store = self.cpu_attn_inputs_by_layer[layer_idx]
            for pending in pending_writes:
                logical_ids_t = torch.from_numpy(
                    pending.logical_ids.astype(np.int64, copy=False)
                )
                offsets_t = torch.from_numpy(
                    pending.offsets.astype(np.int64, copy=False)
                )
                positions_t = torch.from_numpy(
                    pending.positions.astype(np.int32, copy=False)
                )

                layer_store[logical_ids_t, offsets_t, :] = pending.hs_cpu

                self.cpu_block_positions[logical_ids_t, offsets_t] = positions_t

                unique_ids, inverse = np.unique(
                    pending.logical_ids, return_inverse=True
                )
                max_valid_lens = np.zeros(unique_ids.shape[0], dtype=np.int32)
                np.maximum.at(
                    max_valid_lens,
                    inverse,
                    pending.offsets.astype(np.int32, copy=False) + 1,
                )
                unique_ids_t = torch.from_numpy(unique_ids.astype(np.int64, copy=False))
                max_valid_lens_t = torch.from_numpy(max_valid_lens)
                self.cpu_block_valid_lens[unique_ids_t] = torch.maximum(
                    self.cpu_block_valid_lens[unique_ids_t],
                    max_valid_lens_t,
                )

        self._hs_d2h_events_by_layer.clear()
        self._pending_writes_by_layer.clear()

    def _layernorm_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
        *,
        layer_idx: int,
        gpu_model_runner: Any,
    ) -> None:
        del module, inputs

        positions_np, logical_ids_np, block_offsets_np = self._get_step_arrays(
            gpu_model_runner
        )
        if positions_np is None or logical_ids_np is None or block_offsets_np is None:
            return

        hidden_states_normed = output if isinstance(output, torch.Tensor) else output[0]
        if hidden_states_normed is None:
            return

        selected_indices = np.arange(positions_np.shape[0], dtype=np.int64)
        logical_ids = logical_ids_np.astype(np.int64, copy=True)
        offsets = block_offsets_np.astype(np.int64, copy=True)
        positions = positions_np.astype(np.int32, copy=True)

        token_indices = torch.from_numpy(selected_indices).to(
            hidden_states_normed.device, dtype=torch.long
        )
        hs_gpu = hidden_states_normed.index_select(0, token_indices).contiguous()

        hs_cpu: torch.Tensor
        if self._hs_d2h_stream is not None and hs_gpu.is_cuda:
            with torch.cuda.stream(self._hs_d2h_stream):
                hs_cpu = hs_gpu.to("cpu", non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._hs_d2h_stream)
            self._hs_d2h_events_by_layer[layer_idx].append(event)
        else:
            hs_cpu = hs_gpu.to("cpu")

        self._pending_writes_by_layer[layer_idx].append(
            _PendingLayerWrite(
                hs_cpu=hs_cpu,
                logical_ids=logical_ids,
                offsets=offsets,
                positions=positions,
            )
        )

    def _get_step_arrays(
        self, gpu_model_runner: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if (
            self._step_positions_np is not None
            and self._step_logical_ids_np is not None
            and self._step_block_offsets_np is not None
        ):
            return (
                self._step_positions_np,
                self._step_logical_ids_np,
                self._step_block_offsets_np,
            )

        return (
            getattr(gpu_model_runner, "_lr_positions_np", None),
            getattr(gpu_model_runner, "_lr_logical_ids_np", None),
            getattr(gpu_model_runner, "_lr_block_offsets_np", None),
        )

    def _resolve_model_layers(self) -> tuple[list[Any], str, str]:
        model_root = getattr(self.model, "model", None)
        if model_root is None:
            raise ValueError("Layer recompute currently expects model.model to exist.")

        llama_layers = getattr(model_root, "layers", None)
        if llama_layers is not None:
            return list(llama_layers), "input_layernorm", "llama"

        decoder = getattr(model_root, "decoder", None)
        opt_layers = None if decoder is None else getattr(decoder, "layers", None)
        if opt_layers is not None:
            return list(opt_layers), "self_attn_layer_norm", "opt"

        raise ValueError(
            "Layer recompute currently supports llama/opt decoder layouts only."
        )

    def _project_kv_for_recompute(
        self,
        *,
        layer_idx: int,
        hs_gpu: torch.Tensor,
        pos_gpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_impl = self._decoder_layers[layer_idx].self_attn

        if self._recompute_model_type == "llama":
            qkv, _ = attn_impl.qkv_proj(hs_gpu)
            q, k, v = qkv.split(
                [attn_impl.q_size, attn_impl.kv_size, attn_impl.kv_size], dim=-1
            )
            q, k = attn_impl.rotary_emb(pos_gpu, q, k)
            del q
            k = k.view(-1, attn_impl.num_kv_heads, attn_impl.head_dim)
            v = v.view(-1, attn_impl.num_kv_heads, attn_impl.head_dim)
            return k, v

        if self._recompute_model_type == "opt":
            qkv, _ = attn_impl.qkv_proj(hs_gpu)
            _, k, v = qkv.chunk(chunks=3, dim=-1)
            k = k.view(-1, attn_impl.num_heads, attn_impl.head_dim)
            v = v.view(-1, attn_impl.num_heads, attn_impl.head_dim)
            return k, v

        raise ValueError(
            f"Unsupported recompute model type: {self._recompute_model_type!r}."
        )
