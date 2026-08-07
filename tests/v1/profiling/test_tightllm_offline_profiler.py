# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.profiling import tightllm_offline_profiler as profiler


def _opt_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="opt",
        hidden_size=2048,
        num_attention_heads=32,
        ffn_dim=8192,
        num_hidden_layers=24,
        activation_function="relu",
        layer_norm_eps=1e-5,
    )


def _llama_config(*, num_kv_heads: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="llama",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=num_kv_heads,
        head_dim=128,
        intermediate_size=11008,
        num_hidden_layers=32,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )


def test_resolve_opt_spec_preserves_legacy_coefficients() -> None:
    spec = profiler.resolve_model_profile_spec(_opt_config())

    assert spec.model_type == "opt"
    assert spec.hidden_size == 2048
    assert spec.num_kv_heads == 32
    assert spec.head_dim == 64
    assert spec.ffn_dim == 8192
    assert spec.attention_qkv_flops_coefficient == 6.0
    assert spec.attention_output_flops_coefficient == 2.0
    assert spec.attention_score_flops_coefficient == 4.0
    assert spec.mlp_flops_coefficient == 4.0
    assert spec.uses_rms_norm is False
    assert spec.uses_swiglu is False


def test_resolve_llama2_7b_spec_uses_mha_rmsnorm_and_swiglu() -> None:
    spec = profiler.resolve_model_profile_spec(_llama_config())

    assert spec.model_type == "llama"
    assert spec.hidden_size == 4096
    assert spec.num_attention_heads == spec.num_kv_heads == 32
    assert spec.head_dim == 128
    assert spec.ffn_dim == 11008
    assert spec.num_layers == 32
    assert spec.norm_eps == 1e-5
    assert spec.attention_qkv_flops_coefficient == 6.0
    assert spec.attention_output_flops_coefficient == 2.0
    assert spec.attention_score_flops_coefficient == 4.0
    assert spec.mlp_flops_coefficient == 6.0
    assert spec.uses_rms_norm is True
    assert spec.uses_swiglu is True


def test_resolve_llama_gqa_adjusts_qkv_projection_coefficient() -> None:
    spec = profiler.resolve_model_profile_spec(_llama_config(num_kv_heads=8))

    assert spec.num_attention_heads == 32
    assert spec.num_kv_heads == 8
    assert spec.attention_qkv_flops_coefficient == 3.0


def test_old_profile_schema_loads_with_opt_defaults(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "mfu_attn_by_seqlen": {"128": 0.4},
                "mfu_ffn_by_seqlen": {"128": 0.5},
                "pcie_bandwidth_h2d": 1.0e10,
                "gpu_peak_flops": 1.0e14,
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "head_dim": 64,
                "ffn_dim": 8192,
                "num_layers": 24,
            }
        )
    )

    profile = profiler.TightLLMProfileData.load(path)

    assert profile.model_type == "opt"
    assert profile.num_kv_heads == 32
    assert profile.attention_qkv_flops_coefficient == 6.0
    assert profile.attention_output_flops_coefficient == 2.0
    assert profile.attention_score_flops_coefficient == 2.0
    assert profile.mlp_flops_coefficient == 4.0
    assert profile.lookup_mfu_attn(128) == 0.4


def test_llama_profile_schema_round_trip(tmp_path) -> None:
    path = tmp_path / "llama.json"
    expected = profiler.TightLLMProfileData(
        mfu_attn_by_seqlen={128: 0.4, 256: 0.6},
        mfu_ffn_by_seqlen={128: 0.5, 256: 0.7},
        pcie_bandwidth_h2d=1.0e10,
        gpu_peak_flops=1.0e14,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        ffn_dim=11008,
        num_layers=32,
        model_type="llama",
        attention_qkv_flops_coefficient=6.0,
        attention_output_flops_coefficient=2.0,
        attention_score_flops_coefficient=4.0,
        mlp_flops_coefficient=6.0,
    )

    expected.save(path)
    actual = profiler.TightLLMProfileData.load(path)

    assert actual.model_type == "llama"
    assert actual.mfu_attn_by_seqlen == expected.mfu_attn_by_seqlen
    assert actual.mfu_ffn_by_seqlen == expected.mfu_ffn_by_seqlen
    assert actual.attention_qkv_flops_coefficient == 6.0
    assert actual.attention_output_flops_coefficient == 2.0
    assert actual.attention_score_flops_coefficient == 4.0
    assert actual.mlp_flops_coefficient == 6.0


def test_llama_swiglu_microbenchmark_matches_three_projection_math() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4)
    norm_weight = torch.randn(4)
    w_gate_up = torch.randn(4, 10)
    w_down = torch.randn(5, 4)

    actual = profiler._run_mlp_microbenchmark(
        x,
        w_gate_up,
        w_down,
        model_type="llama",
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )

    normalized = F.rms_norm(x, (4,), norm_weight, 1e-5)
    gate, up = torch.matmul(normalized, w_gate_up).chunk(2, dim=-1)
    expected = torch.matmul(F.silu(gate) * up, w_down)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape


def test_opt_relu_microbenchmark_preserves_two_projection_math() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4)
    w_in = torch.randn(4, 5)
    w_out = torch.randn(5, 4)

    actual = profiler._run_mlp_microbenchmark(
        x,
        w_in,
        w_out,
        model_type="opt",
    )
    expected = torch.matmul(F.relu(torch.matmul(x, w_in)), w_out)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("num_kv_heads", "qkv_size"), [(2, 12), (1, 8)])
def test_llama_attention_microbenchmark_supports_mha_and_gqa(
    num_kv_heads: int,
    qkv_size: int,
) -> None:
    torch.manual_seed(3)
    x = torch.randn(1, 3, 4)
    norm_weight = torch.ones(4)
    w_qkv = torch.randn(4, qkv_size)
    w_out = torch.randn(4, 4)

    actual = profiler._run_attention_microbenchmark(
        x,
        w_qkv,
        w_out,
        model_type="llama",
        num_heads=2,
        num_kv_heads=num_kv_heads,
        head_dim=2,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )

    normalized = F.rms_norm(x, (4,), norm_weight, 1e-5)
    expected = profiler._run_attention_microbenchmark(
        normalized,
        w_qkv,
        w_out,
        model_type="opt",
        num_heads=2,
        num_kv_heads=num_kv_heads,
        head_dim=2,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


def test_architecture_aware_flop_formulas() -> None:
    seq_len = 8
    hidden_size = 16
    num_heads = 4
    head_dim = 4
    ffn_dim = 32

    attn = profiler.attention_flops_per_forward(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_flops_coefficient=6.0,
        output_flops_coefficient=2.0,
        score_flops_coefficient=4.0,
    )
    opt_mlp = profiler.mlp_flops_per_forward(
        seq_len=seq_len,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        mlp_flops_coefficient=4.0,
    )
    llama_mlp = profiler.mlp_flops_per_forward(
        seq_len=seq_len,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        mlp_flops_coefficient=6.0,
    )

    assert attn == (
        8 * seq_len * hidden_size**2
        + 4 * num_heads * seq_len**2 * head_dim
    )
    assert opt_mlp == 4 * seq_len * hidden_size * ffn_dim
    assert llama_mlp == 6 * seq_len * hidden_size * ffn_dim


def test_run_offline_profile_wires_llama_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _llama_config(),
    )
    monkeypatch.setattr(profiler, "profile_gpu_peak_flops", lambda *args: 1.0e14)
    monkeypatch.setattr(
        profiler,
        "profile_pcie_bandwidth",
        lambda *args, **kwargs: 1.0e10,
    )

    calls: dict[str, dict] = {}

    def fake_attn(*args, **kwargs):
        calls["attn"] = kwargs
        return {128: 0.4}

    def fake_ffn(*args, **kwargs):
        calls["ffn"] = kwargs
        return {128: 0.5}

    monkeypatch.setattr(profiler, "profile_mfu_attn", fake_attn)
    monkeypatch.setattr(profiler, "profile_mfu_ffn", fake_ffn)

    output_path = tmp_path / "llama-profile.json"
    result = profiler.run_offline_profile(
        "llama-test",
        str(output_path),
        seq_lengths=[128],
        device="cpu",
    )

    assert calls["attn"]["model_type"] == "llama"
    assert calls["attn"]["num_kv_heads"] == 32
    assert calls["attn"]["qkv_flops_coefficient"] == 6.0
    assert calls["attn"]["score_flops_coefficient"] == 4.0
    assert calls["ffn"]["model_type"] == "llama"
    assert calls["ffn"]["mlp_flops_coefficient"] == 6.0
    assert result.model_type == "llama"
    assert result.ffn_dim == 11008
    assert result.attention_score_flops_coefficient == 4.0
    assert output_path.exists()
