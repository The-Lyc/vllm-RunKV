# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.offline_inference import (
    opt_replay_component_mfu,
    run_llama_feedback_observation,
    run_opt_feedback_observation,
    run_tightllm_observation,
)


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


@pytest.mark.parametrize(
    ("defaults", "expected_family", "expected_word", "expected_prefix", "nsys_prefix"),
    [
        pytest.param(
            run_opt_feedback_observation.OPT_OBSERVATION_DEFAULTS,
            "opt",
            "replay",
            "opt_component_mfu",
            "opt_gap_",
            id="legacy-opt",
        ),
        pytest.param(
            run_llama_feedback_observation.LLAMA_OBSERVATION_DEFAULTS,
            "llama",
            "the",
            "llama_runkv_component",
            "llama_gap_",
            id="llama",
        ),
    ],
)
def test_feedback_observation_defaults_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defaults: run_opt_feedback_observation.FeedbackObservationDefaults,
    expected_family: str,
    expected_word: str,
    expected_prefix: str,
    nsys_prefix: str,
) -> None:
    for name in (
        "MODEL",
        "PROMPT_WORD",
        "COMPONENT_ARTIFACT_PREFIX",
        "NSYS_OUTPUT_STEM",
        "PLANNER",
    ):
        monkeypatch.delenv(name, raising=False)

    output_dir = tmp_path / expected_family
    manifest_path = tmp_path / f"{expected_family}.manifest.json"
    nsys_dir = tmp_path / "nsys"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MANIFEST_FILE", str(manifest_path))
    monkeypatch.setenv("RUN_TAG", "unit")
    monkeypatch.setenv("ENABLE_NSYS", "1")
    monkeypatch.setenv("NSYS_CMD", "nsys-test")
    monkeypatch.setenv("NSYS_OUTPUT_DIR", str(nsys_dir))

    captured: dict[str, object] = {}

    def fake_run(command, *, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check

    monkeypatch.setattr(run_opt_feedback_observation.subprocess, "run", fake_run)

    run_opt_feedback_observation.run_feedback_observation(
        defaults,
        passthrough_args=["--profile"],
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == "nsys-test"
    assert _option_value(command, "--model-family") == expected_family
    assert _option_value(command, "--prompt-word") == expected_word
    assert _option_value(command, "--component-artifact-prefix") == expected_prefix
    assert command[-1] == "--profile"
    assert Path(_option_value(command, "-o")).name.startswith(nsys_prefix)
    assert captured["check"] is True

    manifest = json.loads(manifest_path.read_text())
    expected_glob = str(output_dir / f"{expected_prefix}_*_unit.jsonl")
    expected_flat_glob = str(output_dir / f"{expected_prefix}_*_unit.flat.jsonl")
    assert manifest["model_family"] == expected_family
    assert manifest["prompt_word"] == expected_word
    assert manifest["component_artifact_prefix"] == expected_prefix
    assert manifest["mfu_jsonl_glob"] == expected_glob
    assert manifest["mfu_flat_jsonl_glob"] == expected_flat_glob
    assert manifest["component_jsonl_glob"] == expected_glob
    assert manifest["component_flat_jsonl_glob"] == expected_flat_glob


def test_prompt_word_environment_override_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RUN_TAG", "unit")
    monkeypatch.setenv("ENABLE_NSYS", "0")
    monkeypatch.setenv("PROMPT_WORD", "custom")
    monkeypatch.delenv("MANIFEST_FILE", raising=False)

    captured: dict[str, list[str]] = {}

    def fake_run(command, *, env, check):
        del env, check
        captured["command"] = command

    monkeypatch.setattr(run_opt_feedback_observation.subprocess, "run", fake_run)
    run_opt_feedback_observation.run_feedback_observation(
        run_llama_feedback_observation.LLAMA_OBSERVATION_DEFAULTS,
        passthrough_args=[],
    )

    assert _option_value(captured["command"], "--prompt-word") == "custom"


def test_tightllm_observation_builds_llama_command_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    profile_path = tmp_path / "llama-profile.json"
    profile_path.write_text("{}\n")

    for name in (
        "PROMPT_WORD",
        "COMPONENT_ARTIFACT_PREFIX",
        "ENABLE_NSYS",
        "NSYS_OUTPUT_STEM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_FAMILY", "llama")
    monkeypatch.setenv("MODEL", "/data/models/Llama-2-7b-hf-8k")
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MANIFEST_FILE", str(manifest_path))
    monkeypatch.setenv("RUN_TAG", "unit")
    monkeypatch.setenv("TIGHTLLM_PROFILE_PATH", str(profile_path))
    monkeypatch.setattr(run_tightllm_observation.sys, "argv", ["runner"])

    captured: dict[str, list[str]] = {}

    def fake_run(command, *, env, check):
        del env
        assert check is True
        captured["command"] = command

    monkeypatch.setattr(
        run_tightllm_observation.subprocess,
        "run",
        fake_run,
    )

    run_tightllm_observation.main()

    command = captured["command"]
    assert _option_value(command, "--model-family") == "llama"
    assert _option_value(command, "--prompt-word") == "the"
    assert (
        _option_value(command, "--component-artifact-prefix")
        == "llama_tightllm_component"
    )
    assert _option_value(command, "--tightllm-profile-path") == str(profile_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_family"] == "llama"
    assert manifest["prompt_word"] == "the"
    assert manifest["component_artifact_prefix"] == "llama_tightllm_component"
    assert manifest["mfu_jsonl_glob"].endswith(
        "/llama_tightllm_component_*_unit.jsonl"
    )


def test_build_prompts_keeps_opt_default_and_accepts_llama_word() -> None:
    opt_prompt = opt_replay_component_mfu.build_prompts(1, 3)[0]
    llama_prompt = opt_replay_component_mfu.build_prompts(1, 3, "the")[0]

    assert opt_prompt.endswith("replay replay replay")
    assert llama_prompt.endswith("the the the")


def test_legacy_disable_component_flag_is_still_accepted() -> None:
    args = opt_replay_component_mfu.parse_args(
        ["--disable-opt-component-mfu-profiling"]
    )

    assert args.disable_opt_component_mfu_profiling is True
    assert args.model_family == "opt"
    assert args.prompt_word == "replay"
    assert args.component_artifact_prefix == "opt_component_mfu"


def test_llama_tightllm_reaches_engine_with_model_aware_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "llama-tightllm.json"
    llama_args = opt_replay_component_mfu.parse_args(
        [
            "--model-family",
            "llama",
            "--planner",
            "tightllm",
            "--tightllm-profile-path",
            str(profile_path),
            "--prefix-blocks",
            "1",
            "--num-prompts",
            "1",
            "--prompt-words",
            "1",
            "--max-tokens",
            "1",
            "--disable-opt-component-mfu-profiling",
            "--disable-nvtx-scopes",
        ]
    )
    monkeypatch.setattr(
        opt_replay_component_mfu,
        "parse_args",
        lambda: llama_args,
    )

    captured: dict[str, object] = {}
    fake_engine = object()

    def fake_build_engine(**kwargs):
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_run_prompts(engine, prompts, **kwargs):
        captured["engine"] = engine
        captured["prompts"] = prompts
        captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        opt_replay_component_mfu,
        "build_engine",
        fake_build_engine,
    )
    monkeypatch.setattr(
        opt_replay_component_mfu,
        "run_prompts_with_engine",
        fake_run_prompts,
    )

    opt_replay_component_mfu.main()

    engine_kwargs = captured["engine_kwargs"]
    assert isinstance(engine_kwargs, dict)
    offload_config = engine_kwargs["kv_offload_config"]
    assert offload_config["layer_recompute_planner"] == "tightllm"
    assert offload_config["tightllm_profile_path"] == str(profile_path)
    assert captured["engine"] is fake_engine


@pytest.mark.parametrize(
    ("model_type", "supported"),
    [("opt", True), ("llama", True), ("qwen2", False), (None, False)],
)
def test_component_timing_model_gate(model_type: str | None, supported: bool) -> None:
    from vllm.v1.worker.gpu_model_runner import (
        _supports_runkv_component_timing,
    )

    assert _supports_runkv_component_timing(model_type) is supported
