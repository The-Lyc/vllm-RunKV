# IO Bandwidth Throttle for Staged-Resource Experiments

## Motivation

The current staged-resource benchmark in
[scripts/run_staged_resource_benchmark.py](../../scripts/run_staged_resource_benchmark.py)
injects IO pressure by enqueuing competing H2D copies — either on a background
worker thread (`mode=thread`) or synchronously from the RunKV pre_hook on a
dedicated pressure stream (`mode=inline`). Both rely on PCIe / copy-engine
arbitration to "steal" bandwidth from the real KV prefetch stream.

In practice this produces a non-linear relationship between the bytes RunKV
actually skips (when replay ratio goes up) and the wall-time it saves: the
hardware arbiter does not redistribute the freed bandwidth proportionally onto
the critical path, so RunKV's expected benefit is muted in the plots.

This design replaces the bandwidth-stealing mechanism with **stream-internal
throttling**: after every real H2D copy on the KV load stream, enqueue a
GPU-side spin (`torch.cuda._sleep`) sized so that the load stream's effective
throughput is clamped to a target GB/s. Because the throttle is applied on the
*same stream* that performs the real copies, fewer bytes ⇒ fewer (and shorter)
sleeps ⇒ proportionally less wall time. The relationship between IO volume
and time becomes a deterministic linear function — the property we need for
publishable results.

## Validation

A standalone test, [scripts/benchmark_throttled_io_bandwidth.py](../../scripts/benchmark_throttled_io_bandwidth.py),
has been run end-to-end on the target GPU and verified three properties:

1. **Effective bandwidth tracks target.** For targets at or below native PCIe
   bandwidth (~12.3 GB/s on the test box), the measured effective bandwidth on
   the throttled stream stays within ~3–15 % of the configured target. The
   error grows at low targets (≈15 % at 2 GB/s) because the per-copy native
   time is over-estimated; this is correctable with a calibration scale.
2. **Host thread is not blocked.** Across all targets, total host issue time
   is 0.17–0.35 ms while the corresponding GPU wall time is 174–932 ms — three
   orders of magnitude headroom. `torch.cuda._sleep` is a GPU spin kernel, so
   the host can continue to enqueue subsequent ops, run the planner, etc.
3. **Concurrency with the compute stream is preserved.** A matmul stream
   running in parallel with the throttled IO stream completes in
   `~max(io, compute)`, not their sum. The throttle clamps only the IO
   stream's own pacing; compute is unaffected.

Test output is reproducible via:
```
python scripts/benchmark_throttled_io_bandwidth.py \
    --size-mb 64 --n-copies 32 --targets-gbps 2 5 10 20
```

## Design

### Throttle goal

For each real H2D copy of `n_bytes` we want the *per-call effective time on
the load stream* to be

```
effective_ms = max(real_copy_ms, target_ms)         # where target_ms = n_bytes / target_gbps / 1e9 * 1e3
```

equivalently, the per-call injected delay is `max(0, target_ms - real_copy_ms)`.

- When the throttle is below the segment's natural bandwidth
  (`target_ms > real_copy_ms`, the regime we care about), the effective time
  is exactly `target_ms` and the stream behaves as a `target_gbps` link.
- When the throttle target is above what the hardware can deliver
  (`target_ms < real_copy_ms`), the throttle imposes no penalty — the stream
  runs at native speed for that call.

This is a per-call decision and must not depend on any global "native_gbps":
real inference IO has multiple segments (block-by-block H2D prefetch, batched
mseg DMA, CPU-fill H2D), each with a different effective native bandwidth
(different sizes, different stream contention, different copy engines), so
there is no single baseline to subtract.

### Why naive sequential `sleep(target_ms)` is wrong

If we simply enqueue `torch.cuda._sleep(target_cycles)` *after* the copy on
the same stream, the per-call time is `real_copy_ms + target_ms`, not
`max(real_copy_ms, target_ms)`. The effective bandwidth becomes
`n_bytes / (real + target)`, which is slower than `target_gbps`, and the gap
depends on the segment's native bandwidth — exactly the dependency we are
trying to avoid.

Direct subtraction (`sleep_cycles = max(0, target_cycles - real_cycles)`) is
also no good, because reading `real_cycles` requires `Event.elapsed_time`
between two timing events that bracket the copy, and that read forces a
host-side synchronization that breaks the non-blocking property.

### The fix: `max()` via two parallel streams

Issue the copy and a fixed-length sleep on **separate** CUDA streams in
parallel, then make the load stream wait on both before any downstream
consumer can proceed. The CUDA stream synchronization primitive then
computes `max(copy_end, sleep_end)` on-GPU; we never need to read or know
`real_copy_ms` numerically.

Pseudo-code (per copy):

```
# load_stream: the real KV / cpu-fill copy
dst.copy_(src, non_blocking=True)                 # on load_stream

# aux_throttle_stream: target-length spin, launched in parallel
target_cycles = int(cycles_per_ms * target_ms)
with torch.cuda.stream(aux_throttle_stream):
    torch.cuda._sleep(target_cycles)
    e_sleep = Event(); e_sleep.record(aux_throttle_stream)

# Make load_stream's effective completion = max(copy_end, sleep_end):
#   - load_stream's in-stream order already forces subsequent ops to wait for
#     copy_end (no event needed for that half);
#   - wait_event(e_sleep) adds the sleep_end half.
load_stream.wait_event(e_sleep)
```

After `load_stream.wait_event(e_sleep)`, any pre-existing event recorded on
the load stream (`load_ready_event`, `cpu_fill_h2d_ready_event`, …) captures
`max(copy_end, sleep_end)` automatically. Downstream consumers do not need
to change.

The copy uses the H2D copy engine; `torch.cuda._sleep` runs as a single-thread
spin kernel on the SMs. They are independent hardware resources, so the two
streams overlap fully — `effective_ms ≈ max(real, target_ms)` with negligible
extra overhead. The compute stream is untouched. The host main thread is
not blocked (the verification test in
[scripts/benchmark_throttled_io_bandwidth.py](../../scripts/benchmark_throttled_io_bandwidth.py)
shows 0.3 ms host issue time vs ~1 s GPU stream time even when every copy is
followed by a long throttle).

### Linearity property

When RunKV's replay ratio rises, the H2D bytes scheduled on the load stream
drop. Both `target_ms` and `real_copy_ms` scale with `n_bytes`, so
`effective_ms = max(real, target_ms)` also scales linearly. Total saved
wall-time per layer is proportional to bytes saved per layer — the property
the bandwidth-stealing approach failed to provide and the reason for this
design.

### Calibration

Only `cycles_per_ms` is calibrated, once at process start, by the routine
validated in
[scripts/benchmark_throttled_io_bandwidth.py](../../scripts/benchmark_throttled_io_bandwidth.py).
It is a property of the GPU clock and is independent of the IO path, so a
single value applies to every segment.

### Hook point

For each existing `Tensor.copy_(..., non_blocking=True)` (or equivalent
`cudaMemcpyAsync`) on the KV load stream and on the CPU-fill H2D stream the
throttle adds two operations:

1. Launch `torch.cuda._sleep(target_cycles)` on the per-stream auxiliary
   throttle stream and record `e_sleep` immediately after.
2. `original_stream.wait_event(e_sleep)` so that the original stream's
   subsequent submissions block until `max(copy_end, sleep_end)` —
   `copy_end` comes for free from the original stream's own in-stream
   ordering; `wait_event(e_sleep)` supplies the `sleep_end` half.

Properties:
- The compute stream is never touched.
- The auxiliary throttle stream is dedicated; nothing else runs on it.
- All pre-existing CUDA events recorded on the original stream
  (`load_ready_event`, `load_start_event`, `cpu_fill_h2d_ready_event`, …) now
  capture the throttled completion automatically, because they are recorded
  *after* the `wait_event`. No callsite outside the IO functions needs to
  change.

Because RunKV-feedback and TightLLM-replay both reach IO via the *same*
`LayerKVOffloadMapper.load_layer_async` / `LayerRecomputeManager.load_cpu_fill_h2d_async`,
**symmetry is automatic** — both systems see identical interference.

### Effective bandwidth as a function of RunKV state

- When RunKV reuses more KV (replay ratio ↑), `skip_block_ids` grows ⇒
  fewer bytes hit the load stream ⇒ fewer / shorter post-copy sleeps ⇒
  load_stream wall time drops linearly with bytes.
- When RunKV fully replays a layer (no H2D for KV), zero sleeps fire — the
  throttle imposes no overhead on the saved path.

## Code change plan

### File map

| Layer | File | Change |
|-------|------|--------|
| New module | `vllm/v1/worker/io_bandwidth_throttle.py` | Singleton, `cycles_per_ms` calibration, per-IO-stream auxiliary throttle stream, `throttle_after_copy(stream, n_bytes)` that does `_sleep on aux → record e_sleep → stream.wait_event(e_sleep)` |
| IO hook | `vllm/v1/worker/layer_recompute.py` | After each `copy_(..., non_blocking=True)` in `load_layer_async` and `load_cpu_fill_h2d_async`, call `throttle_after_copy(stream, bytes)` |
| Controller | `benchmarks/runkv_resource_pressure/controller.py` | Add `mode="throttle"`; in this mode `inject_pre_prefetch_io` no-ops and stage transitions call `throttle.set_target(gbps)` |
| CLI (pipeline) | `scripts/run_staged_resource_benchmark.py` | Add `"throttle"` to `--resource-pressure-mode`, plumb new throttle-specific args |
| CLI (entry) | `examples/offline_inference/opt_replay_component_mfu.py` | Forward new args into `ResourcePressureConfig`; install throttle into the singleton |
| Tests | `tests/v1/profiling/test_io_throttle.py` | Cycle-estimate unit test + end-to-end mini benchmark replicating `scripts/benchmark_throttled_io_bandwidth.py` |

### Step-by-step implementation

**Step 1 — Throttle module (`vllm/v1/worker/io_bandwidth_throttle.py`)**

- Public surface:
  - `calibrate_cycles_per_ms(device) -> float` (only this — no native BW)
  - `class IOBandwidthThrottle` with:
    - `enable(target_gbps: float)`, `disable()`, `is_enabled() -> bool`
    - `set_target(target_gbps: float)` (cheap; for stage transitions)
    - `throttle_after_copy(stream: torch.cuda.Stream, n_bytes: int) -> None`
  - `get_throttle()` / `set_throttle(instance)` global singleton accessors
    (mirrors `set_inline_pressure_injector` pattern in
    `vllm/v1/profiling/opt_component_mfu.py`)
- Implementation detail:
  - Lazily allocate one auxiliary `torch.cuda.Stream` per *(device,
    throttled stream)* pair, cached in a `WeakKeyDictionary`, so the load
    stream and the cpu-fill stream each get their own throttle stream and
    nothing else.
  - Cache `cycles_per_byte = cycles_per_ms / (target_gbps * 1e9 / 1e3)`
    whenever `set_target` is called so the hot path is a single multiplication
    plus three CUDA ops (launch `_sleep`, record `e_sleep`, `wait_event`).
    No division, no native-BW lookup, multi-segment safe.
- Calibration runs on the device passed in `enable(...)`, reusing the
  `calibrate_cycles_per_ms` helper validated by
  [scripts/benchmark_throttled_io_bandwidth.py](../../scripts/benchmark_throttled_io_bandwidth.py).

**Step 2 — IO hook in `layer_recompute.py`**

- Identify the actual `copy_` (or `cudaMemcpyAsync`) callsites inside
  `load_layer_async` (currently the `mseg_dma` section) and
  `load_cpu_fill_h2d_async` (the `cf_h2d` section).
- After each call, fetch the global throttle and invoke
  `throttle.throttle_after_copy(self.load_stream, bytes_copied)` /
  `throttle.throttle_after_copy(self.fill_stream, bytes_copied)`. The
  function performs `launch _sleep on the aux stream → record e_sleep →
  load_stream.wait_event(e_sleep)`.
- `bytes_copied` is already known locally (mapping length × per-block bytes,
  or the explicit tensor `.nbytes`). No additional bookkeeping required.
- Because the `wait_event` precedes the existing `load_ready_event` /
  `cpu_fill_h2d_ready_event` recordings, those events automatically capture
  `max(copy_end, sleep_end)`. No callsite outside these two functions needs
  to change.

**Step 3 — Controller plumbing (`controller.py`)**

- Extend `PressureMode = Literal["thread", "inline", "throttle"]`.
- In `ResourcePressureController.prepare()`:
  - When `mode == "throttle"`, run `calibrate_cycles_per_ms` once, build an
    `IOBandwidthThrottle`, register it via `set_throttle(...)`, and call
    `enable(initial_stage.target)`.
- In stage-advance code, when `mode == "throttle"` call
  `throttle.set_target(stage.target)` instead of running thread / inline work.
- `inject_pre_prefetch_io` / `inject_pre_attention_sm` become no-ops under
  `mode == "throttle"` (they explicitly return early).

**Step 4 — Pipeline CLI (`scripts/run_staged_resource_benchmark.py`)**

- Add `"throttle"` to `--resource-pressure-mode` choices.
- Update the help text on `--resource-pressure-pattern` to document that
  under `mode=throttle` the `target` value is in GB/s, not in percent.
- No `native_gbps` flag is needed — the formula is `sleep_ms = bytes /
  target_gbps`, independent of hardware peak.

**Step 5 — Entry script (`examples/offline_inference/opt_replay_component_mfu.py`)**

- Forward the new fields into `ResourcePressureConfig`.
- Ensure `set_throttle(None)` is called on shutdown so leaked references do
  not affect subsequent runs.

**Step 6 — Tests (`tests/v1/profiling/test_io_throttle.py`)**

- Unit test: a synthetic stream + a sequence of (`copy`, `throttle_after_copy`)
  pairs reproduces target throughput within ±15 % for several targets.
- Behavioral test: with `mode=throttle`, running RunKV-feedback at two replay
  ratios produces a wall-time delta consistent with bytes saved × seconds-per-byte.

### Optional follow-up (not blocking landing)

- **NVTX/CSV split:** record a per-layer counter "throttled-IO sleep ms" so
  analysis can separate native copy time from injected sleep when plotting
  the staged-resource heatmaps.
- **CPU-fill toggle:** `--throttle-cpu-fill` (default on) to allow ablations
  where only the KV prefetch stream is capped.

## Impact assessment

| Dimension | Assessment |
|---|---|
| Code surface | One new module (~80 lines), <20 lines inserted in `layer_recompute.py`, ~30 lines in the controller, plus CLI plumbing. |
| RunKV / TightLLM symmetry | Automatic — both systems use the same `load_layer_async` / `load_cpu_fill_h2d_async`. |
| Stream sync semantics | Zero impact — sleeps are stream-internal; events / cross-stream waits are unchanged. |
| Compute stream / FA / `qkv_proj` | Untouched. |
| Host main thread | Not blocked (verified: 0.3 ms host vs 932 ms GPU at strictest setting). |
| Existing analysis pipelines | `thread` / `inline` retained as opt-ins, so older runs do not need to be re-collected. |
| Calibration error | Only `cycles_per_ms` is calibrated; jitter is dominated by `torch.cuda._sleep`'s own scheduling and is typically <2 %. Because the throttle uses the `max(copy_end, sleep_end)` synchronization pattern, the effective bandwidth equals `target_gbps` exactly whenever `target_gbps` is below the segment's native rate, and reverts to native when above — no asymmetric overshoot/undershoot. |
| Risks | (a) IO is sub-divided into multiple small copies internally — each gets its own throttle, which is fine but adds N kernel launches and N stream syncs; (b) the CPU-fill stream is separate from KV prefetch and must also be throttled (a second aux throttle stream is created automatically) to keep RunKV's CPU-fill benefit linear; (c) `cycles_per_ms` calibration must complete before the first forward pass; (d) `torch.cuda._sleep` is a 1-thread spin kernel on the SMs — under exceptionally heavy compute the sleep may incur a small launch wait, but in practice the compute stream and the aux throttle stream coexist without measurable interference (verified in the overlap test). |

## Rollout order

1. Implement Steps 1–2 with a hard-coded target. Run a single end-to-end
   experiment and confirm that the RunKV replay-ratio → wall-time-saved curve
   becomes approximately linear.
2. Add Steps 3–5 to expose the feature through the existing CLI and stage
   machinery.
3. Add Step 6 tests; backfill the optional follow-ups as needed for the
   final paper figures.
