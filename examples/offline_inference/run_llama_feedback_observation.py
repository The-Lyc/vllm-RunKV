#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run a RunKV feedback observation with Llama-safe defaults.

This is the Llama counterpart of ``run_opt_feedback_observation.py``. All
runtime knobs remain environment-driven so the normal and staged benchmark
pipelines can invoke both entry points in the same way.

Llama-specific defaults:

* ``MODEL=/data/models/Llama-2-7b-hf-8k``
* ``PROMPT_WORD=the`` (a stable one-token-like synthetic prompt unit)
* component artifacts named ``llama_runkv_component_*.jsonl``
* Nsight Systems stems beginning with ``llama_gap_``

The legacy ``ENABLE_OPT_COMPONENT_MFU_PROFILING`` environment variable is
still accepted; ``ENABLE_COMPONENT_TIMING_PROFILING`` is the preferred
model-neutral spelling.
"""

from __future__ import annotations

from pathlib import Path

if __package__:
    from .run_opt_feedback_observation import (
        FeedbackObservationDefaults,
        run_feedback_observation,
    )
else:
    from run_opt_feedback_observation import (  # type: ignore[no-redef]
        FeedbackObservationDefaults,
        run_feedback_observation,
    )


LLAMA_OBSERVATION_DEFAULTS = FeedbackObservationDefaults(
    model_family="llama",
    model="/data/models/Llama-2-7b-hf-8k",
    output_dir=str(
        Path(__file__).resolve().parents[2]
        / "exp_results"
        / "llama_feedback_observation"
    ),
    prompt_word="the",
    component_artifact_prefix="llama_runkv_component",
    nsys_prefix="llama_gap",
    display_name="Llama",
)


def main() -> None:
    run_feedback_observation(LLAMA_OBSERVATION_DEFAULTS)


if __name__ == "__main__":
    main()
