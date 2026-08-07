# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.v1.profiling.tightllm_offline_profiler import (
    TightLLMProfileData,
)
from vllm.v1.worker.tightllm_ilp_planner import _compute_times


@pytest.mark.parametrize(
    (
        "model_type",
        "num_kv_heads",
        "qkv_coefficient",
        "score_coefficient",
        "mlp_coefficient",
    ),
    [
        pytest.param("opt", 4, 6.0, 4.0, 4.0, id="opt-current"),
        pytest.param("opt", 4, 6.0, 2.0, 4.0, id="opt-legacy"),
        pytest.param("llama", 4, 6.0, 4.0, 6.0, id="llama-mha"),
        pytest.param("llama", 1, 3.0, 4.0, 6.0, id="llama-gqa"),
    ],
)
def test_compute_times_uses_model_coefficients_and_actual_token_ffn_mfu(
    model_type: str,
    num_kv_heads: int,
    qkv_coefficient: float,
    score_coefficient: float,
    mlp_coefficient: float,
) -> None:
    hidden_size = 16
    num_heads = 4
    head_dim = 4
    ffn_dim = 64
    block_size = 8
    replay_blocks = 2
    scheduled_tokens = 4
    num_actual_tokens = replay_blocks * block_size + scheduled_tokens
    avg_context_len = 64
    total_context_blocks = 10
    peak_flops = 1.0e6
    bandwidth = 2.0e5

    profile = TightLLMProfileData(
        # Attention still uses the average context length.
        mfu_attn_by_seqlen={avg_context_len: 0.5},
        # If the planner incorrectly used avg_context_len for the FFN too,
        # this would select 1.0 rather than the expected 0.25.
        mfu_ffn_by_seqlen={num_actual_tokens: 0.25, avg_context_len: 1.0},
        pcie_bandwidth_h2d=bandwidth,
        gpu_peak_flops=peak_flops,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        num_layers=2,
        model_type=model_type,
        attention_qkv_flops_coefficient=qkv_coefficient,
        attention_output_flops_coefficient=2.0,
        attention_score_flops_coefficient=score_coefficient,
        mlp_flops_coefficient=mlp_coefficient,
    )

    t_compute, t_transfer = _compute_times(
        replay_blocks,
        profile=profile,
        total_context_blocks=total_context_blocks,
        total_scheduled_tokens=scheduled_tokens,
        avg_context_len=avg_context_len,
        block_size=block_size,
    )

    expected_attn_flops = num_actual_tokens * (
        (2 * qkv_coefficient + 2.0) * hidden_size**2
        + score_coefficient
        * num_heads
        * avg_context_len
        * head_dim
    )
    expected_mlp_flops = (
        mlp_coefficient * num_actual_tokens * hidden_size * ffn_dim
    )
    expected_compute = (
        expected_attn_flops / (peak_flops * 0.5)
        + expected_mlp_flops / (peak_flops * 0.25)
    )
    expected_transfer_bytes = (
        (total_context_blocks - replay_blocks)
        * block_size
        * 2
        * num_kv_heads
        * head_dim
        * profile.dtype_bytes
    )

    assert t_compute == pytest.approx(expected_compute)
    assert t_transfer == pytest.approx(
        expected_transfer_bytes / bandwidth
    )
