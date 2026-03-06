# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RunKV KV-cache debug instrumentation.

Enable via environment variable:
    VLLM_RUNKV_DEBUG=1   ->  per-token index + KV checksum (L2 norm + sum)
    VLLM_RUNKV_DEBUG=2   ->  level 1 + per-block KV checksum

Output: one JSONL file per engine instance at
    runkv_debug_{engine_tag}_{pid}.jsonl

Each JSON line represents one decode step's worth of information.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RUNKV_DEBUG_LEVEL: int = int(os.environ.get("VLLM_RUNKV_DEBUG", "0"))


def runkv_debug_enabled() -> bool:
    return _RUNKV_DEBUG_LEVEL >= 1


def runkv_debug_level() -> int:
    return _RUNKV_DEBUG_LEVEL


# ---------------------------------------------------------------------------
# Data classes for structured logging
# ---------------------------------------------------------------------------


@dataclass
class TokenKVChecksum:
    """KV checksum for a single token at a specific layer."""

    position: int  # token position in the sequence
    block_id: int  # block id used to locate this token's KV
    offset_in_block: int  # offset within the block
    k_norm: float  # L2 norm of key vector
    v_norm: float  # L2 norm of value vector
    k_sum: float  # sum of key vector
    v_sum: float  # sum of value vector
    # first 4 elements of key and value (for spot-check)
    k_head: list[float] = field(default_factory=list)
    v_head: list[float] = field(default_factory=list)


@dataclass
class TokenIndexInfo:
    """Index information for a single token in a request."""

    position: int  # absolute position in the sequence
    logical_block_id: int  # logical block ID (CPU-side)
    offset_in_block: int  # offset within block
    slot_mapping_value: int  # the slot mapping value for this token
    # runkv-only fields
    staging_slot: int | None = None  # GPU staging slot (runkv only)


@dataclass
class RequestStepInfo:
    """Per-request information for a single decode step."""

    req_id: str
    seq_len: int  # current sequence length after this step
    num_new_tokens: int  # tokens being processed this step (usually 1)
    # Full block table for this request (logical IDs)
    logical_block_table: list[int] = field(default_factory=list)
    # runkv: staging-slot block table
    staging_block_table: list[int] | None = None
    # Per-token index info (only for tokens processed this step)
    token_indices: list[dict] = field(default_factory=list)
    # runkv: logical_id -> staging_slot mapping
    mapping_snapshot: dict[int, int] | None = None


@dataclass
class LayerKVRecord:
    """KV checksum record for one layer, one request, this step's tokens."""

    layer_name: str
    layer_idx: int
    req_id: str
    phase: str  # "after_attention" / "after_h2d" / "cpu_side"
    token_checksums: list[dict] = field(default_factory=list)


@dataclass
class StepRecord:
    """Complete record for a single engine step."""

    step_id: int
    engine_tag: str  # "vanilla" / "runkv" / "recompute"
    timestamp: float
    num_reqs: int
    num_tokens: int
    requests: list[dict] = field(default_factory=list)
    layer_kv_records: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class DebugWriter:
    """Writes JSONL debug records to a file."""

    def __init__(self, engine_tag: str, output_dir: str | None = None):
        self.engine_tag = engine_tag
        if output_dir is None:
            output_dir = os.environ.get(
                "VLLM_RUNKV_DEBUG_DIR",
                os.path.join(os.getcwd(), "runkv_debug_logs"),
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"runkv_debug_{engine_tag}_{os.getpid()}.jsonl"
        self.filepath = self.output_dir / fname
        # Truncate on open (new run)
        with self.filepath.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "type": "header",
                        "engine_tag": engine_tag,
                        "pid": os.getpid(),
                        "timestamp": time.time(),
                        "debug_level": _RUNKV_DEBUG_LEVEL,
                    }
                )
                + "\n"
            )

    def write_step(self, record: StepRecord) -> None:
        obj = asdict(record)
        obj["type"] = "step"
        with self.filepath.open("a") as f:
            f.write(json.dumps(obj, default=_json_default) + "\n")

    def close(self) -> None:
        return


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# KV checksum helpers
# ---------------------------------------------------------------------------


def compute_token_kv_checksum(
    kv_cache: torch.Tensor,
    block_id: int,
    offset_in_block: int,
    position: int,
    blocks_dim: int = 1,
    head_elems: int = 4,
) -> TokenKVChecksum:
    """Compute checksum of a single token's KV from the KV cache tensor.

    Expected KV cache shape (FlashAttention):
        (2, num_blocks, block_size, num_kv_heads, head_size)
        where dim 0 = [key, value], dim 1 = blocks_dim

    Args:
        kv_cache: The KV cache tensor.
        block_id: Block index to select from.
        offset_in_block: Token offset within the block.
        position: Token position (for metadata).
        blocks_dim: Which dimension indexes blocks.
        head_elems: How many leading elements to extract for spot-check.
    """
    # kv_cache has shape (2, num_blocks, block_size, num_kv_heads, head_size)
    # We need kv_cache[:, block_id, offset_in_block, :, :]
    # Use select on blocks_dim
    try:
        # Select the block -> (2, block_size, num_kv_heads, head_size)
        block_data = kv_cache.select(blocks_dim, block_id)
        # block_data now has shape (2, block_size, num_kv_heads, head_size)
        # after selecting along blocks_dim=1
        # The first dim is still KV (0=key, 1=value)
        # We need the token at offset_in_block in the block_size dim
        # After .select(blocks_dim, ...), the dims shift:
        #   original: (2, num_blocks, block_size, num_kv_heads, head_size)
        #   after select(1, block_id): (2, block_size, num_kv_heads, head_size)
        # So offset_in_block is at dim 1 of block_data
        k_vec = block_data[0, offset_in_block].float().flatten()
        v_vec = block_data[1, offset_in_block].float().flatten()

        k_norm = k_vec.norm().item()
        v_norm = v_vec.norm().item()
        k_sum = k_vec.sum().item()
        v_sum = v_vec.sum().item()
        k_head = k_vec[:head_elems].tolist()
        v_head = v_vec[:head_elems].tolist()

        return TokenKVChecksum(
            position=position,
            block_id=block_id,
            offset_in_block=offset_in_block,
            k_norm=k_norm,
            v_norm=v_norm,
            k_sum=k_sum,
            v_sum=v_sum,
            k_head=k_head,
            v_head=v_head,
        )
    except Exception:
        return TokenKVChecksum(
            position=position,
            block_id=block_id,
            offset_in_block=offset_in_block,
            k_norm=float("nan"),
            v_norm=float("nan"),
            k_sum=float("nan"),
            v_sum=float("nan"),
            k_head=[],
            v_head=[],
        )


def compute_per_block_checksums(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    blocks_dim: int = 1,
) -> dict[int, dict[str, float]]:
    """Compute per-block KV checksums (level 2 debug).

    Returns mapping:
        block_id -> {"k_norm": ..., "v_norm": ..., "k_sum": ..., "v_sum": ...}
    """
    result = {}
    for bid in block_ids:
        try:
            block_data = kv_cache.select(blocks_dim, bid).float()
            k_block = block_data[0]  # (block_size, num_kv_heads, head_size)
            v_block = block_data[1]
            result[bid] = {
                "k_norm": k_block.norm().item(),
                "v_norm": v_block.norm().item(),
                "k_sum": k_block.sum().item(),
                "v_sum": v_block.sum().item(),
            }
        except Exception:
            result[bid] = {
                "k_norm": float("nan"),
                "v_norm": float("nan"),
                "k_sum": float("nan"),
                "v_sum": float("nan"),
            }
    return result


# ---------------------------------------------------------------------------
# Step-level debug collection helpers
# ---------------------------------------------------------------------------


def build_request_step_info_vanilla(
    *,
    req_id: str,
    req_idx: int,
    seq_len: int,
    num_scheduled: int,
    block_table_np: np.ndarray,
    num_blocks: int,
    positions: np.ndarray,
    slot_mapping_np: np.ndarray,
    token_offset: int,
    block_size: int,
) -> RequestStepInfo:
    """Build RequestStepInfo for a vanilla engine request."""
    logical_bt = block_table_np[req_idx, :num_blocks].tolist()

    token_indices = []
    for t in range(num_scheduled):
        pos = int(positions[token_offset + t])
        blk_idx = pos // block_size
        offset = pos % block_size
        logical_id = (
            int(block_table_np[req_idx, blk_idx]) if blk_idx < num_blocks else -1
        )
        sm_val = int(slot_mapping_np[token_offset + t])
        token_indices.append(
            asdict(
                TokenIndexInfo(
                    position=pos,
                    logical_block_id=logical_id,
                    offset_in_block=offset,
                    slot_mapping_value=sm_val,
                )
            )
        )

    return RequestStepInfo(
        req_id=req_id,
        seq_len=seq_len,
        num_new_tokens=num_scheduled,
        logical_block_table=logical_bt,
        token_indices=token_indices,
    )


def build_request_step_info_runkv(
    *,
    req_id: str,
    req_idx: int,
    seq_len: int,
    num_scheduled: int,
    logical_block_table_np: np.ndarray,
    num_logical_blocks: int,
    paged_block_table: torch.Tensor,
    paged_slot_mapping: torch.Tensor,
    positions: np.ndarray,
    token_offset: int,
    block_size: int,
    mapping: dict[int, int],
    dirty_blocks: set[int],
) -> RequestStepInfo:
    """Build RequestStepInfo for a runkv engine request."""
    logical_bt = logical_block_table_np[req_idx, :num_logical_blocks].tolist()
    pbt = paged_block_table[req_idx].cpu().tolist()
    # Trim trailing -1s
    staging_bt = [x for x in pbt if x >= 0]

    token_indices = []
    for t in range(num_scheduled):
        pos = int(positions[token_offset + t])
        blk_idx = pos // block_size
        offset = pos % block_size
        logical_id = (
            int(logical_block_table_np[req_idx, blk_idx])
            if blk_idx < num_logical_blocks
            else -1
        )
        sm_val = int(paged_slot_mapping[token_offset + t].cpu().item())
        staging_slot = mapping.get(logical_id)

        token_indices.append(
            asdict(
                TokenIndexInfo(
                    position=pos,
                    logical_block_id=logical_id,
                    offset_in_block=offset,
                    slot_mapping_value=sm_val,
                    staging_slot=staging_slot,
                )
            )
        )

    return RequestStepInfo(
        req_id=req_id,
        seq_len=seq_len,
        num_new_tokens=num_scheduled,
        logical_block_table=logical_bt,
        staging_block_table=staging_bt,
        token_indices=token_indices,
        mapping_snapshot=dict(mapping),
    )


def build_layer_kv_record_vanilla(
    *,
    layer_name: str,
    layer_idx: int,
    req_id: str,
    kv_cache: torch.Tensor,
    block_table_np: np.ndarray,
    req_idx: int,
    positions: list[int],
    block_size: int,
    num_blocks: int,
    blocks_dim: int,
) -> LayerKVRecord:
    """Build per-token KV checksums for a vanilla engine layer."""
    checksums = []
    for pos in positions:
        blk_idx = pos // block_size
        offset = pos % block_size
        if blk_idx >= num_blocks:
            continue
        block_id = int(block_table_np[req_idx, blk_idx])
        cksum = compute_token_kv_checksum(
            kv_cache,
            block_id,
            offset,
            pos,
            blocks_dim=blocks_dim,
        )
        checksums.append(asdict(cksum))

    return LayerKVRecord(
        layer_name=layer_name,
        layer_idx=layer_idx,
        req_id=req_id,
        phase="after_attention",
        token_checksums=checksums,
    )


def build_layer_kv_record_runkv(
    *,
    layer_name: str,
    layer_idx: int,
    req_id: str,
    gpu_buffer: torch.Tensor,
    cpu_cache: torch.Tensor,
    logical_block_table_np: np.ndarray,
    req_idx: int,
    positions: list[int],
    block_size: int,
    num_logical_blocks: int,
    mapping: dict[int, int],
    blocks_dim: int,
    phase: str,
) -> list[LayerKVRecord]:
    """Build per-token KV checksums for runkv (both GPU staging and CPU).

    Returns a list with 1 or 2 records:
        - GPU staging buffer checksums (phase as given)
        - CPU cache checksums (phase="cpu_side")
    """
    records = []

    # GPU staging buffer
    gpu_checksums = []
    for pos in positions:
        blk_idx = pos // block_size
        offset = pos % block_size
        if blk_idx >= num_logical_blocks:
            continue
        logical_id = int(logical_block_table_np[req_idx, blk_idx])
        staging_slot = mapping.get(logical_id)
        if staging_slot is None:
            continue
        cksum = compute_token_kv_checksum(
            gpu_buffer,
            staging_slot,
            offset,
            pos,
            blocks_dim=blocks_dim,
        )
        gpu_checksums.append(asdict(cksum))

    records.append(
        LayerKVRecord(
            layer_name=layer_name,
            layer_idx=layer_idx,
            req_id=req_id,
            phase=phase,
            token_checksums=gpu_checksums,
        )
    )

    # CPU cache (to verify batch copy correctness)
    cpu_checksums = []
    for pos in positions:
        blk_idx = pos // block_size
        offset = pos % block_size
        if blk_idx >= num_logical_blocks:
            continue
        logical_id = int(logical_block_table_np[req_idx, blk_idx])
        cksum = compute_token_kv_checksum(
            cpu_cache,
            logical_id,
            offset,
            pos,
            blocks_dim=blocks_dim,
        )
        cpu_checksums.append(asdict(cksum))

    records.append(
        LayerKVRecord(
            layer_name=layer_name,
            layer_idx=layer_idx,
            req_id=req_id,
            phase="cpu_side",
            token_checksums=cpu_checksums,
        )
    )

    return records
