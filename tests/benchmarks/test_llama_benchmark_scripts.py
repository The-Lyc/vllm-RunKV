# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_llama_benchmark_pipeline as llama_pipeline
from scripts import run_staged_resource_benchmark as staged_pipeline
from tools.analyze_staged_resource_comparison import _discover_run_inputs

ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_llama_pipeline_dry_run_uses_llama_runner_and_prompt_word() -> None:
    result = _run(
        "scripts/run_llama_benchmark_pipeline.py",
        "--dry-run",
        "--disable-nsys",
        "--skip-analysis",
        "--num-prompts",
        "1",
        "--prompt-words",
        "64",
        "--max-tokens",
        "2",
    )

    assert result.returncode == 0, result.stderr
    assert "run_llama_feedback_observation.py" in result.stdout
    assert "run_tightllm_observation.py" in result.stdout
    assert "env PROMPT_WORD=the" in result.stdout
    assert "TightLLM env MODEL_FAMILY=llama" in result.stdout
    assert "TightLLM env TIGHTLLM_PROFILE_PATH=" in result.stdout


@pytest.mark.parametrize("prefix_blocks", ["baseline", "16,32", "-1"])
def test_llama_pipeline_rejects_non_scalar_prefix_blocks(
    prefix_blocks: str,
) -> None:
    result = _run(
        "scripts/run_llama_benchmark_pipeline.py",
        "--dry-run",
        "--disable-nsys",
        "--skip-analysis",
        "--prefix-blocks",
        prefix_blocks,
    )

    assert result.returncode != 0
    assert "--prefix-blocks" in result.stderr


def test_llama_normal_batch_config_dry_run() -> None:
    result = _run(
        "scripts/run_llama_benchmark_batch.py",
        "configs/benchmark_batch_llama2_7b.json",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run_llama_benchmark_pipeline.py") == 4
    assert "--prompt-word the" in result.stdout
    assert (
        "--tightllm-profile-path "
        "exp_results/tightllm_profiles/ubuntu/Llama-2-7b-hf-8k.json"
        in result.stdout
    )
    assert "--skip-tightllm" not in result.stdout


def test_llama_pipeline_collects_model_neutral_manifest_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_observation(
        step_name: str,
        _command: list[str],
        env: dict[str, str],
        *,
        flux_contender: bool,
    ) -> None:
        assert step_name == "Llama RunKV observation"
        assert flux_contender is False
        run_dir = Path(env["OUTPUT_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)
        step_path = run_dir / "llama_runkv_component_128_unit.jsonl"
        flat_path = run_dir / "llama_runkv_component_128_unit.flat.jsonl"
        step_path.write_text("{}\n")
        flat_path.write_text("{}\n")
        Path(env["MANIFEST_FILE"]).write_text(
            json.dumps(
                {
                    "mfu_jsonl_glob": str(
                        run_dir / "llama_runkv_component_*_unit.jsonl"
                    ),
                    "mfu_flat_jsonl_glob": str(
                        run_dir
                        / "llama_runkv_component_*_unit.flat.jsonl"
                    ),
                    "nsys_report": None,
                }
            )
        )

    monkeypatch.setattr(
        llama_pipeline,
        "_run_observation",
        fake_run_observation,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llama_benchmark_pipeline.py",
            "--run-tag",
            "unit",
            "--output-root",
            str(tmp_path),
            "--disable-nsys",
            "--skip-analysis",
            "--skip-tightllm",
            "--num-prompts",
            "1",
            "--prompt-words",
            "64",
            "--max-tokens",
            "2",
        ],
    )

    assert llama_pipeline.main() == 0

    manifest = json.loads((tmp_path / "unit" / "pipeline_manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert artifacts["component_jsonl"].endswith("_unit.jsonl")
    assert artifacts["component_flat_jsonl"].endswith("_unit.flat.jsonl")


def test_llama_pipeline_collects_paired_runkv_and_tightllm_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "llama-profile.json"
    profile_path.write_text('{"model_type": "llama"}\n')
    launched: list[str] = []

    def fake_run_observation(
        step_name: str,
        _command: list[str],
        env: dict[str, str],
        *,
        flux_contender: bool,
    ) -> None:
        assert flux_contender is False
        launched.append(step_name)
        run_dir = Path(env["OUTPUT_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)
        prefix = env["COMPONENT_ARTIFACT_PREFIX"]
        step_path = run_dir / f"{prefix}_128_unit.jsonl"
        flat_path = run_dir / f"{prefix}_128_unit.flat.jsonl"
        step_path.write_text("{}\n")
        flat_path.write_text("{}\n")
        Path(env["MANIFEST_FILE"]).write_text(
            json.dumps(
                {
                    "mfu_jsonl_glob": str(
                        run_dir / f"{prefix}_*_unit.jsonl"
                    ),
                    "mfu_flat_jsonl_glob": str(
                        run_dir / f"{prefix}_*_unit.flat.jsonl"
                    ),
                    "nsys_report": None,
                }
            )
        )

    monkeypatch.setattr(
        llama_pipeline,
        "_run_observation",
        fake_run_observation,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llama_benchmark_pipeline.py",
            "--run-tag",
            "unit",
            "--output-root",
            str(tmp_path),
            "--tightllm-profile-path",
            str(profile_path),
            "--disable-nsys",
            "--skip-analysis",
            "--num-prompts",
            "1",
            "--prompt-words",
            "64",
            "--max-tokens",
            "2",
        ],
    )

    assert llama_pipeline.main() == 0

    manifest = json.loads(
        (tmp_path / "unit" / "pipeline_manifest.json").read_text()
    )
    assert launched == [
        "Llama RunKV observation",
        "Llama TightLLM observation",
    ]
    assert set(manifest["systems"]) == {"runkv", "tightllm"}
    assert "llama_runkv_component" in (
        manifest["systems"]["runkv"]["component_jsonl"]
    )
    assert "llama_tightllm_component" in (
        manifest["systems"]["tightllm"]["component_jsonl"]
    )


@pytest.mark.parametrize(
    "config_name,pressure_kind",
    [
        ("staged_resource_benchmark_llama2_7b_io.json", "io"),
        ("staged_resource_benchmark_llama2_7b_sm.json", "sm"),
    ],
)
def test_llama_staged_batch_configs_dry_run(
    config_name: str,
    pressure_kind: str,
) -> None:
    result = _run(
        "scripts/run_llama_staged_resource_benchmark_batch.py",
        f"configs/{config_name}",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run_llama_staged_resource_benchmark.py") == 4
    assert "--prompt-word the" in result.stdout
    assert f"--resource-pressure-kind {pressure_kind}" in result.stdout
    assert "--skip-tightllm" not in result.stdout
    assert (
        "--tightllm-profile-path "
        + str(
            ROOT
            / "exp_results/tightllm_profiles/ubuntu/Llama-2-7b-hf-8k.json"
        )
        in result.stdout
    )


def test_llama_staged_pipeline_reports_llama_profile_command(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "missing-profile.json"
    result = _run(
        "scripts/run_llama_staged_resource_benchmark.py",
        "--skip-runkv",
        "--skip-analysis",
        "--disable-nsys",
        "--tightllm-profile-path",
        str(profile_path),
    )

    assert result.returncode != 0
    assert "Llama TightLLM profile not found" in result.stderr
    assert "--seq-lengths 128 256 512 1024 2048 4096 8192" in result.stderr
    assert "16384" not in result.stderr


def test_staged_analysis_discovers_model_neutral_component_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "r0"
    run_dir.mkdir()
    step_path = run_dir / "llama_runkv_component_32_unit.jsonl"
    flat_path = run_dir / "llama_runkv_component_32_unit.flat.jsonl"
    step_path.write_text("{}\n")
    flat_path.write_text("{}\n")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mfu_jsonl_glob": str(
                    run_dir / "llama_runkv_component_*_unit.jsonl"
                ),
                "mfu_flat_jsonl_glob": str(
                    run_dir / "llama_runkv_component_*_unit.flat.jsonl"
                ),
            }
        )
    )

    [inputs] = _discover_run_inputs("runkv-feedback", [str(run_dir)])

    assert inputs.mfu_steps == [step_path]
    assert inputs.mfu_flat == [flat_path]


def test_staged_per_layer_analysis_accepts_single_runkv_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run_step(
        _name: str,
        command: list[str],
        _env: dict[str, str],
        *,
        manifest_path: Path | None,
        log_path: Path | None,
    ) -> int:
        del manifest_path, log_path
        commands.append(command)
        return 0

    monkeypatch.setattr(staged_pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(
        staged_pipeline,
        "PER_LAYER_ANALYSIS_ROOT",
        tmp_path / "analysis",
    )
    args = argparse.Namespace(
        skip_analysis=False,
        skip_per_layer_analysis=False,
        enable_nsys=True,
        run_tag="unit",
        skip_warmup_steps=1,
        compute_stream=7,
        dma_tol_ms=2.0,
        num_prompts="1",
        max_tokens="2",
        runkv_run_dir=[],
        tightllm_run_dir=[],
    )
    result = {
        "system": "runkv-feedback",
        "run_dir": str(tmp_path / "r0"),
        "artifacts": {
            "mfu_flat_jsonl": str(tmp_path / "component.flat.jsonl"),
            "nsys_sqlite": str(tmp_path / "run.sqlite"),
        },
    }

    output_dirs = staged_pipeline._run_per_layer_analysis(
        args=args,
        pattern_name="io",
        runkv_results=[result],
        tightllm_results=[],
    )

    assert len(output_dirs) == 1
    assert len(commands) == 1
    assert "--runkv-mfu" in commands[0]
    assert "--runkv-sqlite" in commands[0]
    assert "--tightllm-mfu" not in commands[0]
