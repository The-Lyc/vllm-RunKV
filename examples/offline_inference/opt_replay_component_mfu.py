#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import contextlib
import gc
import os
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real OPT inference with RunKV replay enabled and summarize "
            "attention/FFN MFU versus replay ratio."
        )
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument(
        "--prefix-blocks",
        default="baseline,4,8,16",
        help=(
            "Comma-separated replay settings. Use 'baseline' for RunKV without "
            "layer recompute, or an integer block count for "
            "layer_recompute_io_prefix_blocks."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--num-device-buffers", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Maximum concurrently active requests. Defaults to --num-prompts. "
            "Set this explicitly when comparing fixed-context memory settings."
        ),
    )
    parser.add_argument(
        "--max-staging-blocks",
        type=int,
        default=None,
        help=(
            "Maximum KV blocks in each GPU staging buffer. When omitted, "
            "--gpu-memory-fraction sizes the buffers automatically."
        ),
    )
    parser.add_argument(
        "--cpu-memory-gb",
        "--cpu-kv-memory-gb",
        dest="cpu_memory_gb",
        type=float,
        default=5e10 / (1024**3),
        help=(
            "Total CPU cache-store budget in GiB. With dynamic replay enabled, "
            "this is shared by full-layer KV and full-layer hidden-state "
            "snapshots. --cpu-kv-memory-gb is accepted as a deprecated alias. "
            "Use 0 to derive the budget from --cpu-memory-fraction. Default "
            "preserves the legacy 5e10-byte budget."
        ),
    )
    parser.add_argument(
        "--cpu-memory-fraction",
        type=float,
        default=0.3,
        help=(
            "Clamp total CPU cache-store budget to this fraction of currently "
            "available system memory; it is also the source of the budget "
            "when --cpu-memory-gb=0."
        ),
    )
    parser.add_argument(
        "--planner",
        choices=["static", "feedback", "tightllm"],
        default="feedback",
        help="Replay planner to use when layer recompute is enabled.",
    )
    parser.add_argument(
        "--planner-dry-run",
        action="store_true",
        help=(
            "Keep the feedback planner in observe-only mode. The runtime still "
            "emits imbalance observations into the profiling trace."
        ),
    )
    parser.add_argument(
        "--use-state-machine",
        action="store_true",
        help=(
            "Route feedback planner through the three-state imbalance "
            "controller (STEADY/TRANSIT/TRACKING) instead of the legacy "
            "Newton secant update. Also enables Delta-budget-driven plan "
            "reuse gating in pre_hook."
        ),
    )
    parser.add_argument(
        "--no-async-plan-build",
        action="store_true",
        help=(
            "Build non-steady replay plans synchronously in pre_hook instead "
            "of consuming speculative background builds."
        ),
    )
    parser.add_argument(
        "--h2d-copy-mode",
        choices=["segment", "gather"],
        default="segment",
        help="KV H2D staging implementation used by RunKV.",
    )
    parser.add_argument(
        "--tightllm-profile-path",
        default=None,
        help="Path to TightLLM offline profile JSON (required for --planner tightllm).",
    )
    parser.add_argument(
        "--tightllm-feedback-correction",
        action="store_true",
        help="Enable additive feedback correction on top of TightLLM ILP prediction.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write per-step JSONL trace files "
            "(opt_component_mfu_<prefix>_<tag>.jsonl and .flat.jsonl). "
            "If omitted, no JSONL is emitted."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Tag appended to JSONL output filenames. Defaults to timestamp.",
    )
    parser.add_argument(
        "--disable-opt-component-mfu-profiling",
        action="store_true",
        help=(
            "Disable OPT component profiling hooks entirely. Useful when only "
            "collecting Nsight Systems traces."
        ),
    )
    parser.add_argument(
        "--disable-nvtx-scopes",
        action="store_true",
        help="Disable RunKV NVTX scope emission for this profiling script.",
    )
    parser.add_argument(
        "--enable-layerwise-nvtx-tracing",
        action="store_true",
        help="Also emit per-module NVTX ranges in addition to RunKV phase ranges.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable CUDA profiler start/stop for nsys --capture-range=cudaProfilerApi."
        ),
    )
    parser.add_argument(
        "--resource-pressure-kind",
        choices=["none", "io", "sm"],
        default="none",
        help="Enable runner-controlled resource pressure during engine.step().",
    )
    parser.add_argument(
        "--resource-pressure-clock",
        choices=["step", "time"],
        default="step",
        help="Use scheduler step index or inference elapsed time for pressure stages.",
    )
    parser.add_argument(
        "--resource-pressure-pattern",
        default="0:0",
        help="Comma-separated start:target schedule. Step clock uses step:target; time clock uses second:target.",
    )
    parser.add_argument(
        "--resource-pressure-log-path",
        default=None,
        help="CSV path for pressure worker actual pressure samples.",
    )
    parser.add_argument(
        "--resource-pressure-step-log-path",
        default=None,
        help="JSONL path for per-step resource stage alignment records.",
    )
    parser.add_argument("--resource-pressure-device", default="cuda:0")
    parser.add_argument("--resource-pressure-buffer-mb", type=int, default=256)
    parser.add_argument(
        "--resource-pressure-direction",
        choices=["h2d", "d2h", "bidirectional"],
        default="h2d",
    )
    parser.add_argument("--resource-pressure-matrix-size", type=int, default=4096)
    parser.add_argument(
        "--resource-pressure-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--resource-pressure-window-s", type=float, default=0.25)
    parser.add_argument("--resource-pressure-period-ms", type=float, default=100.0)
    parser.add_argument(
        "--resource-pressure-max-fraction",
        type=float,
        default=0.5,
        help=(
            "Clamp background pressure target to this fraction of calibrated "
            "IO bandwidth or SM duty-cycle capacity. Use 1.0 to allow full target."
        ),
    )
    parser.add_argument(
        "--resource-pressure-io-calibration-s",
        type=float,
        default=0.5,
        help=(
            "Seconds spent calibrating the standalone IO copy path before "
            "inference starts. Set 0 to disable auto calibration."
        ),
    )
    parser.add_argument(
        "--resource-pressure-io-max-gbps",
        type=float,
        default=None,
        help=(
            "Manual IO capacity in GB/s for target clamping. Overrides "
            "auto calibration when positive."
        ),
    )
    parser.add_argument(
        "--resource-pressure-mode",
        choices=["thread", "inline", "throttle"],
        default="thread",
        help=(
            "thread: legacy background-worker pressure (races with model "
            "execution at host-sync boundaries). inline: pressure injected "
            "synchronously from RunKV pre_hook on each layer (IO before "
            "prefetch on a dedicated stream, SM before FA on the compute "
            "stream). throttle: IO-only; cap the KV / cpu-fill load streams "
            "to --resource-pressure-pattern target_GBps via a stream-side "
            "torch.cuda._sleep paired with each copy. Throttle produces a "
            "deterministic linear relationship between H2D bytes and wall "
            "time (RunKV's replay savings become exactly proportional) and "
            "is the recommended mode for staged-resource RunKV vs TightLLM "
            "experiments."
        ),
    )
    parser.add_argument(
        "--resource-pressure-inline-layer-period-ms",
        type=float,
        default=5.0,
        help=(
            "Estimated per-layer wall time used to size each inline burst. "
            "IO bytes/layer = target_GBps * this_ms * 1e6; SM ms/layer = "
            "this_ms * target_percent / 100. Set to a representative steady-"
            "state layer time observed in a baseline run."
        ),
    )
    return parser.parse_args()


@contextlib.contextmanager
def nvtx_range(name: str, color: str = "blue"):
    if os.environ.get("VLLM_NVTX_SCOPES_FOR_PROFILING", "0") != "1":
        yield
        return

    cm = None
    try:
        import nvtx

        cm = nvtx.annotate(name, color=color)
    except ImportError:
        try:
            import torch.cuda.nvtx as torch_nvtx

            cm = torch_nvtx.range(name)
        except Exception:
            cm = None

    if cm is None:
        yield
    else:
        with cm:
            yield


def cuda_profiler_start() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.cudart().cudaProfilerStop()


def build_prompts(num_prompts: int, prompt_words: int) -> list[str]:
    repeated = " ".join(["replay"] * prompt_words)
    return [
        f"Request {idx}: summarize the pattern and continue briefly. {repeated}"
        for idx in range(num_prompts)
    ]


def parse_prefix_settings(raw: str) -> list[str | int]:
    settings: list[str | int] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        if value.lower() == "baseline":
            settings.append("baseline")
        else:
            settings.append(int(value))
    if not settings:
        raise ValueError("No valid --prefix-blocks settings were provided.")
    return settings


def make_kv_offload_config(
    setting: str | int,
    *,
    gpu_memory_fraction: float,
    num_device_buffers: int,
    max_staging_blocks: int | None,
    cpu_memory_gb: float,
    cpu_memory_fraction: float,
    planner: str,
    planner_dry_run: bool,
    async_plan_build: bool = True,
    h2d_copy_mode: str = "segment",
    use_state_machine: bool = False,
    tightllm_profile_path: str | None = None,
    tightllm_feedback_correction: bool = False,
) -> dict:
    cpu_memory_limit = (
        int(cpu_memory_gb * 1024**3) if cpu_memory_gb > 0 else None
    )
    config = {
        "enabled": True,
        "num_device_buffers": num_device_buffers,
        "max_staging_blocks": max_staging_blocks,
        "gpu_memory_fraction": gpu_memory_fraction,
        "enable_async_prefetch": True,
        "enable_async_offload": True,
        "cpu_memory_limit": cpu_memory_limit,
        "cpu_memory_fraction": cpu_memory_fraction,
    }
    if setting == "baseline":
        config["enable_layer_recompute"] = False
    else:
        config["enable_layer_recompute"] = True
        config["layer_recompute_mode"] = "prev_layer_output_dynamic"
        config["layer_recompute_io_prefix_blocks"] = [int(setting)]
        config["layer_recompute_planner"] = planner
        config["layer_recompute_planner_dry_run"] = planner_dry_run
        config["layer_recompute_async_plan_build"] = async_plan_build
        config["h2d_copy_mode"] = h2d_copy_mode
        config["layer_recompute_use_state_machine"] = use_state_machine
        if planner == "tightllm":
            if not tightllm_profile_path:
                raise ValueError(
                    "--tightllm-profile-path is required when --planner=tightllm"
                )
            config["tightllm_profile_path"] = tightllm_profile_path
            config["tightllm_enable_feedback_correction"] = tightllm_feedback_correction
    return config


def build_engine(
    *,
    model: str,
    gpu_memory_utilization: float,
    kv_offload_config: dict,
    enable_layerwise_nvtx_tracing: bool,
    profiler_config: dict | None,
    enable_opt_component_mfu_profiling: bool,
    opt_component_mfu_output_path: str | None,
    opt_component_mfu_peak_tflops: float | None,
    max_num_seqs: int,
):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    engine_args = EngineArgs(
        model=model,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_cascade_attn=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        enable_layerwise_nvtx_tracing=enable_layerwise_nvtx_tracing,
        profiler_config=profiler_config,
        kv_offload_config=kv_offload_config,
        enable_opt_component_mfu_profiling=enable_opt_component_mfu_profiling,
        opt_component_mfu_output_path=opt_component_mfu_output_path,
        opt_component_mfu_peak_tflops=opt_component_mfu_peak_tflops,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    executor_class = Executor.get_class(vllm_config)
    return LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
        usage_context=UsageContext.ENGINE_CONTEXT,
        multiprocess_mode=False,
    )


def run_prompts_with_engine(
    engine,
    prompts: list[str],
    *,
    max_tokens: int,
    enable_profiling: bool,
    resource_controller: Any | None = None,
) -> None:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.profiling.opt_component_mfu import (
        set_inline_pressure_injector,
        set_opt_component_mfu_resource_context,
    )

    pending_requests: set[str] = set()

    with nvtx_range("add_requests", color="green"):
        for idx, prompt in enumerate(prompts):
            request_id = f"req_{idx}"
            engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
            )
            pending_requests.add(request_id)

    if resource_controller is not None:
        resource_controller.start()
        # Register the controller as the per-layer inline injector when it's
        # configured for inline mode. The pre_hook in gpu_model_runner picks
        # it up via vllm.v1.profiling.opt_component_mfu.get_inline_pressure_
        # injector(); a None registration keeps thread-mode behavior intact.
        if getattr(resource_controller, "inline_mode", False):
            set_inline_pressure_injector(resource_controller)

    if enable_profiling:
        cuda_profiler_start()

    step = 0
    try:
        with nvtx_range("inference_loop", color="blue"):
            while pending_requests:
                resource_context = None
                step_start_s = None
                if resource_controller is not None:
                    resource_context = resource_controller.before_step(step)
                    step_start_s = resource_context.get("step_start_s")
                set_opt_component_mfu_resource_context(resource_context)

                with nvtx_range(f"step_{step}", color="yellow"):
                    step_outputs = engine.step()

                for out in step_outputs:
                    request_id = getattr(out, "request_id", None)
                    if request_id is not None and getattr(out, "finished", False):
                        pending_requests.discard(request_id)

                if resource_controller is not None:
                    resource_controller.after_step(
                        step_id=step,
                        step_start_s=step_start_s,
                        step_end_s=resource_controller.elapsed_s(),
                        output_count=len(step_outputs),
                        pending_count=len(pending_requests),
                    )

                step += 1
    finally:
        set_opt_component_mfu_resource_context(None)
        set_inline_pressure_injector(None)
        if enable_profiling:
            cuda_profiler_stop()
        if resource_controller is not None:
            resource_controller.stop()


def _derive_setting_path(
    raw_path: str | None,
    *,
    setting: str | int,
    settings_count: int,
) -> str | None:
    if raw_path is None:
        return None
    if settings_count <= 1:
        return raw_path
    path = Path(raw_path)
    return str(path.with_name(f"{path.stem}_{setting}{path.suffix}"))


def _make_resource_controller(
    args: argparse.Namespace,
    *,
    setting: str | int,
    settings_count: int,
    run_tag: str,
):
    if args.resource_pressure_kind == "none":
        return None

    from benchmarks.runkv_resource_pressure.controller import (
        ResourcePressureConfig,
        ResourcePressureController,
    )

    log_path = _derive_setting_path(
        args.resource_pressure_log_path,
        setting=setting,
        settings_count=settings_count,
    )
    step_log_path = _derive_setting_path(
        args.resource_pressure_step_log_path,
        setting=setting,
        settings_count=settings_count,
    )
    if log_path is None and args.output_dir:
        log_path = str(Path(args.output_dir) / f"pressure_{setting}_{run_tag}.csv")
    if step_log_path is None and args.output_dir:
        step_log_path = str(
            Path(args.output_dir) / f"resource_steps_{setting}_{run_tag}.jsonl"
        )

    config = ResourcePressureConfig(
        kind=args.resource_pressure_kind,
        clock=args.resource_pressure_clock,
        pattern=args.resource_pressure_pattern,
        log_path=log_path,
        step_log_path=step_log_path,
        device=args.resource_pressure_device,
        buffer_mb=args.resource_pressure_buffer_mb,
        direction=args.resource_pressure_direction,
        matrix_size=args.resource_pressure_matrix_size,
        dtype=args.resource_pressure_dtype,
        window_s=args.resource_pressure_window_s,
        period_ms=args.resource_pressure_period_ms,
        max_fraction=args.resource_pressure_max_fraction,
        io_calibration_s=args.resource_pressure_io_calibration_s,
        io_max_gbps=args.resource_pressure_io_max_gbps,
        mode=getattr(args, "resource_pressure_mode", "thread"),
        inline_layer_period_ms=getattr(
            args, "resource_pressure_inline_layer_period_ms", 5.0
        ),
    )
    return ResourcePressureController(config)


def main() -> None:
    args = parse_args()
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be > 0 when set")
    if args.max_staging_blocks is not None and args.max_staging_blocks <= 0:
        raise ValueError("--max-staging-blocks must be > 0 when set")
    if args.cpu_memory_gb < 0:
        raise ValueError("--cpu-memory-gb must be >= 0")
    if not (0.0 < args.cpu_memory_fraction <= 1.0):
        raise ValueError("--cpu-memory-fraction must be in (0, 1]")
    if not args.disable_nvtx_scopes:
        os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    mfu_profiler_enabled = not args.disable_opt_component_mfu_profiling

    # Resolve run tag once for the whole invocation (consistent across settings)
    if args.run_tag:
        _run_tag = args.run_tag
    else:
        from datetime import datetime

        _run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix_settings = parse_prefix_settings(args.prefix_blocks)

    for setting in prefix_settings:
        prompts = build_prompts(args.num_prompts, args.prompt_words)

        # Build per-setting JSONL output path if --output-dir was given
        if mfu_profiler_enabled and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            _mfu_out = os.path.join(
                args.output_dir,
                f"opt_component_mfu_{setting}_{_run_tag}.jsonl",
            )
        else:
            _mfu_out = None

        with nvtx_range("build_engine", color="purple"):
            engine = build_engine(
                model=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx_tracing,
                profiler_config=(
                    {"profiler": "cuda"} if not args.disable_nvtx_scopes else None
                ),
                kv_offload_config=make_kv_offload_config(
                    setting,
                    gpu_memory_fraction=args.gpu_memory_fraction,
                    num_device_buffers=args.num_device_buffers,
                    max_staging_blocks=args.max_staging_blocks,
                    cpu_memory_gb=args.cpu_memory_gb,
                    cpu_memory_fraction=args.cpu_memory_fraction,
                    planner=args.planner,
                    planner_dry_run=args.planner_dry_run,
                    async_plan_build=not args.no_async_plan_build,
                    h2d_copy_mode=args.h2d_copy_mode,
                    use_state_machine=args.use_state_machine,
                    tightllm_profile_path=args.tightllm_profile_path,
                    tightllm_feedback_correction=args.tightllm_feedback_correction,
                ),
                enable_opt_component_mfu_profiling=mfu_profiler_enabled,
                opt_component_mfu_output_path=_mfu_out,
                opt_component_mfu_peak_tflops=args.peak_tflops,
                max_num_seqs=args.max_num_seqs or max(args.num_prompts, 1),
            )

        resource_controller = _make_resource_controller(
            args,
            setting=setting,
            settings_count=len(prefix_settings),
            run_tag=_run_tag,
        )
        if resource_controller is not None:
            with nvtx_range("resource_pressure_prepare", color="red"):
                resource_controller.prepare()

        run_prompts_with_engine(
            engine,
            prompts,
            max_tokens=args.max_tokens,
            enable_profiling=args.profile,
            resource_controller=resource_controller,
        )

        # ---- Collect imbalance statistics from the replay plan provider ----
        try:
            model_runner = engine.model_executor.driver_worker.worker.model_runner
            provider = getattr(model_runner, "replay_plan_provider", None)
            if provider is not None and hasattr(provider, "get_imbalance_stats"):
                stats = provider.get_imbalance_stats()
                if stats and stats.get("count", 0) > 0:
                    print(f"\n  Imbalance stats ({stats.get('provider', '?')}):")
                    for k, v in stats.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.4f}")
                        else:
                            print(f"    {k}: {v}")
        except Exception as e:
            print(f"  Warning: could not collect imbalance stats: {e}")

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nOPT replay run finished.")
    if mfu_profiler_enabled and args.output_dir:
        print(
            f"JSONL traces written to: {args.output_dir}/opt_component_mfu_*_{_run_tag}.jsonl"
        )
    else:
        print("JSONL trace emission disabled (pass --output-dir to enable).")


if __name__ == "__main__":
    main()
