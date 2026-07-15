# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource pressure controller used by staged RunKV experiments.

This module intentionally stays Python-only.  It provides four modes:

* thread: background IO/SM pressure worker.
* inline: pressure injected from RunKV pre-hooks.
* throttle: IO-only stream-side bandwidth throttle.
* process: external CUDA processes that physically contend for PCIe or SM time.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from vllm.v1.worker.io_bandwidth_throttle import (
    IOBandwidthThrottle,
    calibrate_cycles_per_ms,
    set_throttle,
)


PressureKind = Literal["io", "sm"]
PressureClock = Literal["step", "time"]
PressureMode = Literal["thread", "inline", "throttle", "process"]


def _pcie_contention_worker(
    *,
    device: str,
    buffer_mb: int,
    direction: Literal["h2d", "d2h", "bidirectional"],
    stop_event: mp.synchronize.Event,
    active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
) -> None:
    """Saturate PCIe from an independent process when ``active_event`` is set."""
    torch.cuda.set_device(torch.device(device))
    cuda_device = torch.device(device)
    n_bytes = max(1, int(buffer_mb)) * 1024 * 1024
    h_cpu = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    d_buf = torch.empty(n_bytes, dtype=torch.uint8, device=cuda_device)
    d_buf_b = None
    if direction in ("d2h", "bidirectional"):
        d_buf_b = torch.empty(n_bytes, dtype=torch.uint8, device=cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)

    with torch.cuda.stream(stream):
        if direction in ("h2d", "bidirectional"):
            d_buf.copy_(h_cpu, non_blocking=True)
        if direction in ("d2h", "bidirectional"):
            assert d_buf_b is not None
            h_cpu.copy_(d_buf_b, non_blocking=True)
    stream.synchronize()
    ready_event.set()

    while not stop_event.is_set():
        if not active_event.wait(timeout=0.05):
            continue
        copies_since_sync = 0
        with torch.cuda.stream(stream):
            while (
                not stop_event.is_set()
                and active_event.is_set()
                and copies_since_sync < 16
            ):
                if direction in ("h2d", "bidirectional"):
                    d_buf.copy_(h_cpu, non_blocking=True)
                if direction in ("d2h", "bidirectional"):
                    assert d_buf_b is not None
                    h_cpu.copy_(d_buf_b, non_blocking=True)
                copies_since_sync += 1
        stream.synchronize()


def _sm_contention_worker(
    *,
    device: str,
    matrix_size: int,
    dtype: Literal["float16", "bfloat16"],
    stop_event: mp.synchronize.Event,
    active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
) -> None:
    """Saturate SM/tensor-core compute from an independent process."""
    torch.cuda.set_device(torch.device(device))
    cuda_device = torch.device(device)
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    n = max(1, int(matrix_size))
    a = torch.randn((n, n), dtype=torch_dtype, device=cuda_device)
    b = torch.randn((n, n), dtype=torch_dtype, device=cuda_device)
    out = torch.empty((n, n), dtype=torch_dtype, device=cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)

    with torch.inference_mode(), torch.cuda.stream(stream):
        torch.mm(a, b, out=out)
    stream.synchronize()
    ready_event.set()

    while not stop_event.is_set():
        if not active_event.wait(timeout=0.05):
            continue
        iters_since_sync = 0
        with torch.inference_mode(), torch.cuda.stream(stream):
            while (
                not stop_event.is_set()
                and active_event.is_set()
                and iters_since_sync < 8
            ):
                torch.mm(a, b, out=out)
                iters_since_sync += 1
        stream.synchronize()


@dataclass
class ResourcePressureConfig:
    kind: PressureKind
    clock: PressureClock
    pattern: str
    log_path: str | None = None
    step_log_path: str | None = None
    device: str = "cuda:0"
    buffer_mb: int = 256
    direction: Literal["h2d", "d2h", "bidirectional"] = "h2d"
    matrix_size: int = 4096
    dtype: Literal["float16", "bfloat16"] = "float16"
    window_s: float = 0.25
    period_ms: float = 100.0
    max_fraction: float = 0.5
    io_calibration_s: float = 0.5
    io_max_gbps: float | None = None
    mode: PressureMode = "thread"
    inline_layer_period_ms: float = 5.0


class ResourcePressureController:
    def __init__(self, config: ResourcePressureConfig) -> None:
        self.config = config
        self.inline_mode = config.mode == "inline"
        self._device = torch.device(config.device)
        self._pattern = self._parse_pattern(config.pattern)
        self._start_s: float | None = None
        self._current_target = 0.0
        self._target_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._log_file = None
        self._csv_writer: csv.DictWriter | None = None
        self._step_log_file = None
        self._cycles_per_ms: float | None = None
        self._io_capacity_gbps: float | None = None
        self._h_cpu: torch.Tensor | None = None
        self._d_buf: torch.Tensor | None = None
        self._d_buf_b: torch.Tensor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._throttle: IOBandwidthThrottle | None = None
        self._mp_context = mp.get_context("spawn")
        self._processes: list[mp.Process] = []
        self._process_stop_events: list[mp.synchronize.Event] = []
        self._process_active_events: list[mp.synchronize.Event] = []
        self._process_worker_count = 0
        self._max_process_worker_count = 0
        self._current_actual_target = 0.0

    @staticmethod
    def _parse_pattern(pattern: str) -> list[tuple[float, float]]:
        stages: list[tuple[float, float]] = []
        for chunk in pattern.split(","):
            raw = chunk.strip()
            if not raw:
                continue
            start_s, target_s = raw.split(":", 1)
            stages.append((float(start_s), float(target_s)))
        if not stages:
            stages.append((0.0, 0.0))
        stages.sort(key=lambda item: item[0])
        return stages

    def prepare(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Resource pressure requires CUDA.")
        torch.cuda.set_device(self._device)
        self._stream = torch.cuda.Stream(device=self._device)
        if self.config.kind == "io":
            if self.config.io_max_gbps and self.config.io_max_gbps > 0:
                self._io_capacity_gbps = float(self.config.io_max_gbps)
            if (
                self.config.mode != "process"
                or self._io_capacity_gbps is None
                and self.config.io_calibration_s > 0
            ):
                self._prepare_io_buffers()
            if self._io_capacity_gbps is None and self.config.io_calibration_s > 0:
                self._io_capacity_gbps = self._calibrate_io_gbps()
        elif self.config.kind == "sm":
            self._cycles_per_ms = calibrate_cycles_per_ms(self._device)

        if self.config.mode == "inline" and self._cycles_per_ms is None:
            self._cycles_per_ms = calibrate_cycles_per_ms(self._device)

        if self.config.mode == "throttle":
            if self.config.kind != "io":
                raise ValueError("resource pressure mode 'throttle' only supports IO.")
            self._throttle = IOBandwidthThrottle()
        if self.config.mode == "process":
            if self.config.kind == "io" and self._io_capacity_gbps is None:
                raise ValueError(
                    "IO process pressure needs a PCIe baseline. "
                    "Set --resource-pressure-io-max-gbps, or keep "
                    "--resource-pressure-io-calibration-s > 0."
                )
            self._max_process_worker_count = self._max_process_workers_for_pattern()
            self._set_process_pool_size(self._max_process_worker_count)

        self._open_logs()

    def _prepare_io_buffers(self) -> None:
        n_bytes = max(1, int(self.config.buffer_mb)) * 1024 * 1024
        self._h_cpu = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
        self._d_buf = torch.empty(n_bytes, dtype=torch.uint8, device=self._device)
        if self.config.direction in ("d2h", "bidirectional"):
            self._d_buf_b = torch.empty(n_bytes, dtype=torch.uint8, device=self._device)

    def _calibrate_io_gbps(self) -> float:
        assert self._h_cpu is not None and self._d_buf is not None
        assert self._stream is not None
        end_s = time.perf_counter() + float(self.config.io_calibration_s)
        bytes_done = 0
        with torch.cuda.stream(self._stream):
            while time.perf_counter() < end_s:
                self._d_buf.copy_(self._h_cpu, non_blocking=True)
                bytes_done += self._h_cpu.numel()
        self._stream.synchronize()
        elapsed = max(float(self.config.io_calibration_s), 1e-6)
        return bytes_done / elapsed / 1e9

    def _open_logs(self) -> None:
        if self.config.log_path:
            path = Path(self.config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("w", newline="")
            self._csv_writer = csv.DictWriter(
                self._log_file,
                fieldnames=[
                    "elapsed_s",
                    "kind",
                    "mode",
                    "target",
                    "actual",
                    "unit",
                ],
            )
            self._csv_writer.writeheader()
        if self.config.step_log_path:
            path = Path(self.config.step_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._step_log_file = path.open("w")

    def start(self) -> None:
        self._start_s = time.perf_counter()
        self._stop_event.clear()
        if self.config.mode == "thread":
            target = self._target_for_clock(0.0)
            self._set_target(target)
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="runkv-resource-pressure",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        self._set_process_worker_count(0)
        self._set_process_pool_size(0)
        if self._throttle is not None:
            self._throttle.disable()
            set_throttle(None)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._step_log_file is not None:
            self._step_log_file.close()
            self._step_log_file = None

    def elapsed_s(self) -> float:
        if self._start_s is None:
            return 0.0
        return time.perf_counter() - self._start_s

    def before_step(self, step: int) -> dict[str, float | int | str]:
        value = float(step) if self.config.clock == "step" else self.elapsed_s()
        target = self._target_for_clock(value)
        self._set_target(target)
        step_start_s = self.elapsed_s()
        return {
            "resource_pressure_kind": self.config.kind,
            "resource_pressure_mode": self.config.mode,
            "resource_pressure_clock": self.config.clock,
            "resource_pressure_step": int(step),
            "resource_pressure_clock_value": value,
            "resource_pressure_target": float(target),
            "resource_pressure_actual_target": float(self._current_actual_target),
            "resource_pressure_process_workers": int(self._process_worker_count),
            "resource_pressure_step_start_s": step_start_s,
            "step_start_s": step_start_s,
        }

    def after_step(
        self,
        *,
        step_id: int,
        step_start_s: float | None,
        step_end_s: float,
        output_count: int,
        pending_count: int,
    ) -> None:
        if self._step_log_file is None:
            return
        record = {
            "resource_pressure_step": int(step_id),
            "step_start_s": step_start_s,
            "step_end_s": step_end_s,
            "target": self._get_target(),
            "actual_target": self._current_actual_target,
            "kind": self.config.kind,
            "mode": self.config.mode,
            "process_workers": self._process_worker_count,
            "output_count": int(output_count),
            "pending_count": int(pending_count),
        }
        self._step_log_file.write(json.dumps(record) + "\n")
        self._step_log_file.flush()

    def inject_pre_prefetch_io(self, layer_idx: int) -> None:
        del layer_idx
        if self.config.kind != "io" or self.config.mode != "inline":
            return
        target = self._get_target()
        if target <= 0:
            return
        bytes_to_copy = int(target * self.config.inline_layer_period_ms * 1e6)
        self._submit_io_bytes(bytes_to_copy, synchronize=False)

    def inject_pre_attention_sm(self, layer_idx: int) -> None:
        del layer_idx
        if self.config.kind != "sm" or self.config.mode != "inline":
            return
        target = self._get_target()
        if target <= 0:
            return
        cycles_per_ms = self._cycles_per_ms
        if cycles_per_ms is None:
            return
        active_ms = self.config.inline_layer_period_ms * min(target, 100.0) / 100.0
        cycles = int(active_ms * cycles_per_ms)
        if cycles > 0:
            torch.cuda._sleep(cycles)

    def _target_for_clock(self, value: float) -> float:
        target = self._pattern[0][1]
        for start, stage_target in self._pattern:
            if value >= start:
                target = stage_target
            else:
                break
        if self.config.kind == "io" and self._io_capacity_gbps is not None:
            if self.config.mode != "process":
                target = min(target, self._io_capacity_gbps * self.config.max_fraction)
        if self.config.kind == "sm" and self.config.mode != "process":
            target = min(target, 100.0 * self.config.max_fraction)
        return max(0.0, float(target))

    def _set_target(self, target: float) -> None:
        with self._target_lock:
            self._current_target = float(target)
        if self.config.mode == "process":
            actual = self._set_process_worker_count(
                self._process_worker_count_for_target(target)
            )
            self._current_actual_target = actual
            return
        if self.config.mode == "throttle" and self._throttle is not None:
            if target > 0:
                if self._throttle.enabled:
                    self._throttle.set_target(target)
                else:
                    self._throttle.enable(target, self._device)
                    set_throttle(self._throttle)
            else:
                self._throttle.disable()
        self._current_actual_target = float(target)

    def _process_capacity(self) -> float:
        if self.config.kind == "io":
            return float(self._io_capacity_gbps or 0.0)
        if self.config.kind == "sm":
            return 100.0
        return 0.0

    def _process_worker_count_for_target(self, target: float) -> int:
        baseline = self._process_capacity()
        if baseline <= 0 or target <= 0:
            return 0
        if target >= baseline:
            return 0
        total_clients = max(1, int(round(baseline / target)))
        return max(0, total_clients - 1)

    def _max_process_workers_for_pattern(self) -> int:
        return max(
            (self._process_worker_count_for_target(target) for _, target in self._pattern),
            default=0,
        )

    def _set_process_worker_count(self, count: int) -> float:
        if self.config.mode != "process":
            return 0.0
        count = max(0, int(count))
        if count > len(self._processes):
            self._set_process_pool_size(count)
        for idx, active_event in enumerate(self._process_active_events):
            if idx < count:
                active_event.set()
            else:
                active_event.clear()
        self._process_worker_count = count
        baseline = self._process_capacity()
        actual = baseline if count <= 0 else baseline / float(count + 1)
        self._write_pressure_sample(time.perf_counter(), self._current_target, actual)
        return actual

    def _set_process_pool_size(self, count: int) -> None:
        count = max(0, int(count))
        while len(self._processes) > count:
            process = self._processes.pop()
            stop_event = self._process_stop_events.pop()
            active_event = self._process_active_events.pop()
            active_event.clear()
            stop_event.set()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

        while len(self._processes) < count:
            stop_event = self._mp_context.Event()
            active_event = self._mp_context.Event()
            ready_event = self._mp_context.Event()
            if self.config.kind == "io":
                target = _pcie_contention_worker
                kwargs = {
                    "device": str(self._device),
                    "buffer_mb": int(self.config.buffer_mb),
                    "direction": self.config.direction,
                    "stop_event": stop_event,
                    "active_event": active_event,
                    "ready_event": ready_event,
                }
            elif self.config.kind == "sm":
                target = _sm_contention_worker
                kwargs = {
                    "device": str(self._device),
                    "matrix_size": int(self.config.matrix_size),
                    "dtype": self.config.dtype,
                    "stop_event": stop_event,
                    "active_event": active_event,
                    "ready_event": ready_event,
                }
            else:
                raise ValueError(f"Unsupported process pressure kind: {self.config.kind}")
            process = self._mp_context.Process(
                target=target,
                kwargs=kwargs,
                daemon=True,
            )
            process.start()
            if not ready_event.wait(timeout=15.0):
                stop_event.set()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                raise RuntimeError(
                    f"{self.config.kind.upper()} contention worker did not become "
                    "ready before timeout."
                )
            self._processes.append(process)
            self._process_stop_events.append(stop_event)
            self._process_active_events.append(active_event)

    def _get_target(self) -> float:
        with self._target_lock:
            return self._current_target

    def _worker_loop(self) -> None:
        period_s = max(self.config.period_ms / 1000.0, 1e-3)
        while not self._stop_event.is_set():
            target = self._get_target()
            start = time.perf_counter()
            actual = 0.0
            if target > 0:
                if self.config.kind == "io":
                    bytes_target = int(target * period_s * 1e9)
                    bytes_done = self._submit_io_bytes(bytes_target, synchronize=True)
                    elapsed = max(time.perf_counter() - start, 1e-6)
                    actual = bytes_done / elapsed / 1e9
                else:
                    actual = self._burn_sm_for_period(target, period_s)
            self._write_pressure_sample(start, target, actual)
            elapsed = time.perf_counter() - start
            if elapsed < period_s:
                self._stop_event.wait(period_s - elapsed)

    def _submit_io_bytes(self, n_bytes: int, *, synchronize: bool) -> int:
        if n_bytes <= 0 or self._h_cpu is None or self._d_buf is None:
            return 0
        assert self._stream is not None
        chunk = self._h_cpu.numel()
        remaining = int(n_bytes)
        done = 0
        with torch.cuda.stream(self._stream):
            while remaining > 0:
                count = min(remaining, chunk)
                if self.config.direction in ("h2d", "bidirectional"):
                    self._d_buf[:count].copy_(self._h_cpu[:count], non_blocking=True)
                    done += count
                if self.config.direction in ("d2h", "bidirectional"):
                    assert self._d_buf_b is not None
                    self._h_cpu[:count].copy_(self._d_buf_b[:count], non_blocking=True)
                    done += count
                remaining -= count
        if synchronize:
            self._stream.synchronize()
        return done

    def _burn_sm_for_period(self, target_percent: float, period_s: float) -> float:
        cycles_per_ms = self._cycles_per_ms
        if cycles_per_ms is None:
            return 0.0
        active_s = period_s * min(max(target_percent, 0.0), 100.0) / 100.0
        cycles = int(active_s * 1000.0 * cycles_per_ms)
        if cycles > 0:
            assert self._stream is not None
            with torch.cuda.stream(self._stream):
                torch.cuda._sleep(cycles)
            self._stream.synchronize()
        return min(max(target_percent, 0.0), 100.0)

    def _write_pressure_sample(self, start_s: float, target: float, actual: float) -> None:
        if self._csv_writer is None:
            return
        elapsed = 0.0 if self._start_s is None else start_s - self._start_s
        unit = "GB/s" if self.config.kind == "io" else "%"
        self._csv_writer.writerow(
            {
                "elapsed_s": f"{elapsed:.6f}",
                "kind": self.config.kind,
                "mode": self.config.mode,
                "target": f"{target:.6f}",
                "actual": f"{actual:.6f}",
                "unit": unit,
            }
        )
        if self._log_file is not None:
            self._log_file.flush()


__all__ = ["ResourcePressureConfig", "ResourcePressureController"]
