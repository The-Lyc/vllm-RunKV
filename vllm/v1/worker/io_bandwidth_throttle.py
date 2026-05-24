# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream-side IO bandwidth throttle for staged-resource experiments.

Design: docs/design/io_bandwidth_throttle_design.md

Per H2D copy on a throttled stream the throttle enqueues a fixed-length
``torch.cuda._sleep`` on a per-stream auxiliary stream and makes the throttled
stream ``wait_event`` on the sleep's event. The CUDA stream sync primitive
then gives effective wall-time = ``max(copy_end, sleep_end)`` on-GPU with no
host synchronization, no native-bandwidth estimate, and no cross-talk with
the compute stream.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

_GLOBAL_THROTTLE: IOBandwidthThrottle | None = None
_GLOBAL_LOCK = threading.Lock()


def calibrate_cycles_per_ms(
    device: torch.device, samples: int = 5, test_cycles: int = 50_000_000
) -> float:
    """Measure how many GPU cycles ``torch.cuda._sleep`` consumes per ms.

    Property of the GPU clock; independent of the IO path. Reused
    across all throttled streams. Validated by
    ``scripts/benchmark_throttled_io_bandwidth.py``.
    """
    aux = torch.cuda.Stream(device=device)
    measured: list[float] = []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(aux):
            start.record(aux)
            torch.cuda._sleep(test_cycles)
            end.record(aux)
        end.synchronize()
        measured.append(start.elapsed_time(end))
    measured.sort()
    median_ms = measured[len(measured) // 2]
    if median_ms <= 0:
        raise RuntimeError(
            "calibrate_cycles_per_ms got non-positive elapsed time; "
            "torch.cuda._sleep is not behaving as expected."
        )
    return test_cycles / median_ms


class IOBandwidthThrottle:
    """Throttle one or more CUDA streams to a target effective IO bandwidth.

    Hot path per copy:

    * ``aux = self._get_aux_stream(stream)``  (lazy, cached)
    * ``torch.cuda._sleep(cycles)`` on ``aux``
    * record ``e_sleep`` on ``aux``
    * ``stream.wait_event(e_sleep)``

    No host sync, no division on the hot path. ``cycles_per_byte`` is
    re-cached whenever the target changes; ``cycles_per_ms`` is set once at
    ``enable()`` time.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._device: torch.device | None = None
        self._cycles_per_ms: float = 0.0
        self._cycles_per_byte: float = 0.0
        self._target_gbps: float = 0.0
        # id(stream) -> aux stream. Aux streams are created on the same device
        # as the throttled stream and live for the throttle's lifetime.
        self._aux_streams: dict[int, torch.cuda.Stream] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_gbps(self) -> float:
        return self._target_gbps

    def enable(
        self,
        target_gbps: float,
        device: torch.device | str | int,
        *,
        cycles_per_ms: float | None = None,
    ) -> None:
        """Enable the throttle on ``device`` with the given target.

        ``cycles_per_ms`` is calibrated automatically; pass an explicit value
        only for tests where calibration cannot run.
        """
        if target_gbps <= 0:
            raise ValueError(f"target_gbps must be > 0, got {target_gbps}")
        self._device = torch.device(device)
        if cycles_per_ms is None:
            cycles_per_ms = calibrate_cycles_per_ms(self._device)
        self._cycles_per_ms = cycles_per_ms
        self.set_target(target_gbps)
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._target_gbps = 0.0
        self._cycles_per_byte = 0.0
        # Aux streams are kept; cheap to leave around and rebuilt if enable()
        # is called again. They are not freed because torch.cuda.Stream has no
        # explicit free.

    def set_target(self, target_gbps: float) -> None:
        """Cheap stage transition. Recomputes ``cycles_per_byte`` only."""
        if target_gbps <= 0:
            raise ValueError(f"target_gbps must be > 0, got {target_gbps}")
        self._target_gbps = target_gbps
        # target_ms(n_bytes) = n_bytes / (target_gbps * 1e9) * 1e3
        #                    = n_bytes * 1e-6 / target_gbps   [ms]
        # cycles            = cycles_per_ms * target_ms
        # ⇒ cycles_per_byte = cycles_per_ms * 1e-6 / target_gbps
        self._cycles_per_byte = self._cycles_per_ms * 1e-6 / target_gbps

    def throttle_after_copy(
        self, stream: torch.cuda.Stream, n_bytes: int
    ) -> None:
        """Throttle ``stream`` so this copy's effective time ≥ target_ms.

        No-op when the throttle is disabled or ``n_bytes <= 0``. Safe to call
        from any host thread; aux-stream creation is guarded by a lock.
        """
        if not self._enabled or n_bytes <= 0:
            return
        cycles = int(self._cycles_per_byte * n_bytes)
        if cycles <= 0:
            return
        aux = self._get_aux_stream(stream)
        with torch.cuda.stream(aux):
            torch.cuda._sleep(cycles)
            e_sleep = torch.cuda.Event()
            e_sleep.record(aux)
        # The throttled stream's in-stream order already gates subsequent ops
        # behind the real copy_end; wait_event(e_sleep) adds the sleep_end
        # half, giving max(copy_end, sleep_end).
        stream.wait_event(e_sleep)

    def _get_aux_stream(self, stream: torch.cuda.Stream) -> torch.cuda.Stream:
        key = id(stream)
        aux = self._aux_streams.get(key)
        if aux is not None:
            return aux
        with self._lock:
            aux = self._aux_streams.get(key)
            if aux is None:
                aux = torch.cuda.Stream(device=stream.device)
                self._aux_streams[key] = aux
        return aux


def set_throttle(throttle: IOBandwidthThrottle | None) -> None:
    """Install (or remove) the global throttle singleton."""
    global _GLOBAL_THROTTLE
    with _GLOBAL_LOCK:
        _GLOBAL_THROTTLE = throttle


def get_throttle() -> IOBandwidthThrottle | None:
    """Return the global throttle or ``None`` if not installed."""
    return _GLOBAL_THROTTLE


def throttle_after_copy_if_enabled(
    stream: torch.cuda.Stream | None, n_bytes: int
) -> None:
    """Convenience: no-op when no throttle is installed or stream is None.

    Intended to be sprinkled at IO callsites; cost is one attribute load plus
    one branch when the throttle is disabled.
    """
    t = _GLOBAL_THROTTLE
    if t is None or stream is None:
        return
    t.throttle_after_copy(stream, n_bytes)


__all__ = [
    "IOBandwidthThrottle",
    "calibrate_cycles_per_ms",
    "get_throttle",
    "set_throttle",
    "throttle_after_copy_if_enabled",
]
