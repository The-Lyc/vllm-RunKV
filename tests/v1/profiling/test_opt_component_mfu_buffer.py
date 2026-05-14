# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the buffered emission path in OPTComponentMFUStepProfiler.

The profiler's per-step path must NOT touch the filesystem; records go to
in-memory buffers owned by the model runner and are flushed once at
process exit.  These tests pin that contract so a future refactor can't
silently reintroduce per-step IO.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.v1.profiling.opt_component_mfu import OPTComponentMFUStepProfiler


def _make_profiler(
    output_path: str,
    step_idx: int,
    step_buf: list[dict[str, Any]],
    flat_buf: list[dict[str, Any]],
) -> OPTComponentMFUStepProfiler:
    return OPTComponentMFUStepProfiler(
        output_path=output_path,
        step_idx=step_idx,
        rank=0,
        model_name="opt-test",
        total_scheduled_tokens=32,
        num_reqs=4,
        step_record_buffer=step_buf,
        flat_record_buffer=flat_buf,
    )


def test_finish_step_does_not_write_to_disk(tmp_path) -> None:
    out = tmp_path / "mfu.jsonl"
    flat = tmp_path / "mfu.flat.jsonl"
    step_buf: list[dict[str, Any]] = []
    flat_buf: list[dict[str, Any]] = []

    profiler = _make_profiler(str(out), step_idx=0, step_buf=step_buf, flat_buf=flat_buf)
    # _dynamic_replay_runtime stays None — _build_layer_records returns [],
    # finish_step still appends the step-level record.
    profiler.finish_step()

    # Critical contract: nothing on disk during the inference loop.
    assert not out.exists()
    assert not flat.exists()
    # Step record buffered; no per-layer records (runtime not attached).
    assert len(step_buf) == 1
    assert step_buf[0]["step"] == 0
    assert step_buf[0]["layers"] == []
    assert flat_buf == []


def test_buffer_grows_per_step(tmp_path) -> None:
    out = tmp_path / "mfu.jsonl"
    step_buf: list[dict[str, Any]] = []
    flat_buf: list[dict[str, Any]] = []

    for step_idx in range(5):
        _make_profiler(
            str(out), step_idx=step_idx, step_buf=step_buf, flat_buf=flat_buf
        ).finish_step()

    assert [r["step"] for r in step_buf] == [0, 1, 2, 3, 4]
    # No file IO has happened yet.
    assert not out.exists()


def test_finish_step_noops_without_buffer(tmp_path) -> None:
    """When no buffer is injected, finish_step must not crash and must not
    write to disk on its own (the deferred-flush contract).
    """
    out = tmp_path / "mfu.jsonl"
    profiler = OPTComponentMFUStepProfiler(
        output_path=str(out),
        step_idx=0,
        rank=0,
        model_name="opt-test",
        total_scheduled_tokens=1,
        num_reqs=1,
        step_record_buffer=None,
        flat_record_buffer=None,
    )
    profiler.finish_step()
    assert not out.exists()


def test_records_are_json_serializable(tmp_path) -> None:
    """Records that go into the buffer must round-trip through json.dumps —
    otherwise the atexit flush would explode on shutdown.
    """
    out = tmp_path / "mfu.jsonl"
    step_buf: list[dict[str, Any]] = []
    flat_buf: list[dict[str, Any]] = []
    _make_profiler(
        str(out), step_idx=7, step_buf=step_buf, flat_buf=flat_buf
    ).finish_step()
    json.dumps(step_buf[0])
    # Also ensure the parent dir was created by the profiler (still desired
    # so the eventual flush doesn't fail on a missing path).
    assert os.path.isdir(tmp_path)
