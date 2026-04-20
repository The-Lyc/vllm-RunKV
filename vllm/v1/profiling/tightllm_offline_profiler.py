#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Offline profiler for TightLLM ILP planner.

Profiles three quantities required by the TightLLM KV distributor:
  1. MFU_attn(seq_len) — Attention layer Model FLOPs Utilization curve
  2. MFU_ffn(seq_len)  — FFN layer MFU curve
  3. PCIe H2D bandwidth — Effective bandwidth for KV block transfers
  4. GPU peak FLOPS     — From device properties / benchmark

Reference: TightLLM (Hu et al., IEEE TC 2025), Section IV-B, Eq 3-5.

Usage:
    python -m vllm.v1.profiling.tightllm_offline_profiler \
        --model /path/to/opt-2.7b \
        --output tightllm_profile.json \
        --seq-lengths 128 256 512 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Profile data container
# ---------------------------------------------------------------------------


@dataclass
class TightLLMProfileData:
    """Offline profiled data for TightLLM ILP planner.

    Stores MFU look-up tables (keyed by effective sequence length),
    effective PCIe bandwidth, GPU peak FLOPS, and model architecture
    parameters needed for the analytical FLOPS model.
    """

    # MFU look-up tables: {effective_seq_len: mfu_ratio}
    mfu_attn_by_seqlen: dict[int, float] = field(default_factory=dict)
    mfu_ffn_by_seqlen: dict[int, float] = field(default_factory=dict)

    # PCIe H2D effective bandwidth (bytes/s)
    pcie_bandwidth_h2d: float = 0.0

    # GPU peak FLOPS (FP16/BF16 tensor-core throughput)
    gpu_peak_flops: float = 0.0

    # Model architecture constants
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    ffn_dim: int = 0  # intermediate_size / ffn_dim
    num_layers: int = 0
    dtype_bytes: int = 2  # FP16 / BF16

    # Interpolation arrays (built from the dicts above)
    _mfu_attn_seqlens: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([1.0])
    )
    _mfu_attn_values: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([0.5])
    )
    _mfu_ffn_seqlens: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([1.0])
    )
    _mfu_ffn_values: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([0.5])
    )

    def __post_init__(self) -> None:
        self._build_interpolation_tables()

    def _build_interpolation_tables(self) -> None:
        for attr_dict, attr_sl, attr_val in (
            ("mfu_attn_by_seqlen", "_mfu_attn_seqlens", "_mfu_attn_values"),
            ("mfu_ffn_by_seqlen", "_mfu_ffn_seqlens", "_mfu_ffn_values"),
        ):
            d = getattr(self, attr_dict)
            if d:
                seqlens = sorted(d.keys())
                setattr(self, attr_sl, np.array(seqlens, dtype=np.float64))
                setattr(
                    self,
                    attr_val,
                    np.array([d[s] for s in seqlens], dtype=np.float64),
                )

    def lookup_mfu_attn(self, effective_seqlen: int) -> float:
        """Linearly interpolate MFU_attn for *effective_seqlen*."""
        return float(
            np.interp(effective_seqlen, self._mfu_attn_seqlens, self._mfu_attn_values)
        )

    def lookup_mfu_ffn(self, effective_seqlen: int) -> float:
        """Linearly interpolate MFU_ffn for *effective_seqlen*."""
        return float(
            np.interp(effective_seqlen, self._mfu_ffn_seqlens, self._mfu_ffn_values)
        )

    # ---- Serialisation ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mfu_attn_by_seqlen": {str(k): v for k, v in self.mfu_attn_by_seqlen.items()},
            "mfu_ffn_by_seqlen": {str(k): v for k, v in self.mfu_ffn_by_seqlen.items()},
            "pcie_bandwidth_h2d": self.pcie_bandwidth_h2d,
            "gpu_peak_flops": self.gpu_peak_flops,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "num_layers": self.num_layers,
            "dtype_bytes": self.dtype_bytes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("TightLLM profile saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TightLLMProfileData:
        with open(path) as f:
            data = json.load(f)
        return cls(
            mfu_attn_by_seqlen={int(k): v for k, v in data["mfu_attn_by_seqlen"].items()},
            mfu_ffn_by_seqlen={int(k): v for k, v in data["mfu_ffn_by_seqlen"].items()},
            pcie_bandwidth_h2d=data["pcie_bandwidth_h2d"],
            gpu_peak_flops=data["gpu_peak_flops"],
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_kv_heads=data.get("num_kv_heads", data["num_attention_heads"]),
            head_dim=data["head_dim"],
            ffn_dim=data["ffn_dim"],
            num_layers=data["num_layers"],
            dtype_bytes=data.get("dtype_bytes", 2),
        )


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def profile_gpu_peak_flops(
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Benchmark GPU peak FLOPS via large matmuls."""
    M = N = K = 4096
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0
    flops = 2 * M * N * K * iters
    return flops / elapsed_s


def profile_pcie_bandwidth(
    device: torch.device,
    block_size: int = 16,
    num_kv_heads: int = 32,
    head_dim: int = 80,
    dtype: torch.dtype = torch.float16,
    num_blocks_list: list[int] | None = None,
) -> float:
    """Profile effective PCIe H2D bandwidth via pinned-memory copies."""
    if num_blocks_list is None:
        num_blocks_list = [16, 32, 64, 128, 256, 512]

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    bandwidths: list[float] = []

    for num_blocks in num_blocks_list:
        total_elements = num_blocks * block_size * 2 * num_kv_heads * head_dim
        total_bytes = total_elements * dtype_bytes

        cpu_tensor = torch.randn(total_elements, dtype=dtype).pin_memory()
        gpu_tensor = torch.empty(total_elements, device=device, dtype=dtype)

        # Warmup
        for _ in range(5):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            torch.cuda.synchronize()

        iters = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end.record()
        torch.cuda.synchronize()

        elapsed_s = start.elapsed_time(end) / 1000.0
        bw = (total_bytes * iters) / elapsed_s
        bandwidths.append(bw)

    # Return median bandwidth
    return float(sorted(bandwidths)[len(bandwidths) // 2])


def profile_mfu_attn(
    device: torch.device,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    peak_flops: float,
    seq_lengths: list[int],
    dtype: torch.dtype = torch.float16,
) -> dict[int, float]:
    """Profile attention MFU at discrete sequence lengths.

    Runs a QKV-projection → scaled-dot-product-attention → output-projection
    micro-benchmark and computes MFU = actual_FLOPS / peak_FLOPS.
    """
    mfu_by_seqlen: dict[int, float] = {}

    for seq_len in seq_lengths:
        if seq_len <= 0:
            continue
        try:
            x = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
            w_qkv = torch.randn(hidden_size, 3 * hidden_size, device=device, dtype=dtype)
            w_out = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)

            def _run():
                qkv = torch.matmul(x, w_qkv)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(
                    1, seq_len, hidden_size
                )
                return torch.matmul(attn_out, w_out)

            # Warmup
            for _ in range(3):
                _run()
            torch.cuda.synchronize()

            iters = 10
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _run()
            end.record()
            torch.cuda.synchronize()

            elapsed_s = start.elapsed_time(end) / 1000.0

            # Theoretical attention FLOPs per forward:
            #   QKV proj : 2 * seq * H * 3H = 6*seq*H^2
            #   QK^T     : 2 * nheads * seq * seq * head_dim
            #   Score*V  : 2 * nheads * seq * seq * head_dim  (implicit in SDPA)
            #   Out proj : 2 * seq * H * H
            flops_per = (
                6 * seq_len * hidden_size * hidden_size
                + 4 * num_heads * seq_len * seq_len * head_dim  # QK^T + score*V
                + 2 * seq_len * hidden_size * hidden_size
            )
            actual = flops_per * iters / elapsed_s
            mfu_by_seqlen[seq_len] = min(actual / peak_flops, 1.0)

            del x, w_qkv, w_out
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at seq_len=%d during attention profiling, skipping.", seq_len)
            torch.cuda.empty_cache()
            break

    return mfu_by_seqlen


def profile_mfu_ffn(
    device: torch.device,
    hidden_size: int,
    ffn_dim: int,
    peak_flops: float,
    seq_lengths: list[int],
    dtype: torch.dtype = torch.float16,
) -> dict[int, float]:
    """Profile FFN MFU at discrete sequence lengths."""
    mfu_by_seqlen: dict[int, float] = {}

    for seq_len in seq_lengths:
        if seq_len <= 0:
            continue
        try:
            x = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
            w1 = torch.randn(hidden_size, ffn_dim, device=device, dtype=dtype)
            w2 = torch.randn(ffn_dim, hidden_size, device=device, dtype=dtype)

            def _run():
                h = torch.matmul(x, w1)
                h = F.relu(h)
                return torch.matmul(h, w2)

            for _ in range(3):
                _run()
            torch.cuda.synchronize()

            iters = 10
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _run()
            end.record()
            torch.cuda.synchronize()

            elapsed_s = start.elapsed_time(end) / 1000.0
            # FFN FLOPs: 2 * seq * H * F + 2 * seq * F * H = 4 * seq * H * F
            flops_per = 4 * seq_len * hidden_size * ffn_dim
            actual = flops_per * iters / elapsed_s
            mfu_by_seqlen[seq_len] = min(actual / peak_flops, 1.0)

            del x, w1, w2
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at seq_len=%d during FFN profiling, skipping.", seq_len)
            torch.cuda.empty_cache()
            break

    return mfu_by_seqlen


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_offline_profile(
    model_name_or_path: str,
    output_path: str,
    seq_lengths: list[int] | None = None,
    device: str = "cuda:0",
    block_size: int = 16,
) -> TightLLMProfileData:
    """Run complete offline profiling and save results.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        output_path: Where to save the profile JSON.
        seq_lengths: Sequence lengths at which to measure MFU.
        device: CUDA device string.
        block_size: KV cache block size (for bandwidth profiling).

    Returns:
        The populated ``TightLLMProfileData``.
    """
    from transformers import AutoConfig

    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    config = AutoConfig.from_pretrained(model_name_or_path)
    hidden_size: int = config.hidden_size
    num_heads: int = config.num_attention_heads
    num_kv_heads: int = getattr(config, "num_key_value_heads", num_heads)
    head_dim: int = hidden_size // num_heads
    ffn_dim: int = getattr(config, "ffn_dim", None) or getattr(
        config, "intermediate_size", 4 * hidden_size
    )
    num_layers: int = config.num_hidden_layers

    torch_device = torch.device(device)
    dtype = torch.float16

    print(f"Model : {model_name_or_path}")
    print(
        f"  H={hidden_size}  heads={num_heads}  kv_heads={num_kv_heads}  "
        f"head_dim={head_dim}  ffn_dim={ffn_dim}  layers={num_layers}"
    )

    # 1. GPU peak FLOPS
    print("\n[1/4] GPU peak FLOPS ...")
    peak_flops = profile_gpu_peak_flops(torch_device, dtype)
    print(f"  {peak_flops:.2e}  ({peak_flops / 1e12:.1f} TFLOPS)")

    # 2. PCIe bandwidth
    print("\n[2/4] PCIe H2D bandwidth ...")
    bandwidth = profile_pcie_bandwidth(
        torch_device,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    print(f"  {bandwidth:.2e} B/s  ({bandwidth / 1e9:.1f} GB/s)")

    # 3. Attention MFU
    print(f"\n[3/4] Attention MFU  seq_lengths={seq_lengths}")
    mfu_attn = profile_mfu_attn(
        torch_device, hidden_size, num_heads, head_dim, peak_flops, seq_lengths, dtype
    )
    for sl, val in sorted(mfu_attn.items()):
        print(f"  seq_len={sl:>6d}  MFU_attn={val:.4f}")

    # 4. FFN MFU
    print(f"\n[4/4] FFN MFU  seq_lengths={seq_lengths}")
    mfu_ffn = profile_mfu_ffn(
        torch_device, hidden_size, ffn_dim, peak_flops, seq_lengths, dtype
    )
    for sl, val in sorted(mfu_ffn.items()):
        print(f"  seq_len={sl:>6d}  MFU_ffn ={val:.4f}")

    profile_data = TightLLMProfileData(
        mfu_attn_by_seqlen=mfu_attn,
        mfu_ffn_by_seqlen=mfu_ffn,
        pcie_bandwidth_h2d=bandwidth,
        gpu_peak_flops=peak_flops,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    )
    profile_data.save(output_path)
    return profile_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TightLLM offline profiler — collects MFU and bandwidth data"
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to profile MFU at",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_offline_profile(
        model_name_or_path=args.model,
        output_path=args.output,
        seq_lengths=args.seq_lengths,
        device=args.device,
        block_size=args.block_size,
    )
