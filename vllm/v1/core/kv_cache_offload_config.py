# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RunKVOffloadConfig:
    """RunKV Layer-wise KV cache offload configuration."""

    enabled: bool = False

    # num of buffers reserved on GPU (ring buffer count)
    num_device_buffers: int = 3

    # Maximum number of KV blocks that can be staged in each GPU buffer.
    # This determines how many blocks can be processed in a single forward step.
    # If None, will be computed based on available GPU memory.
    # Example: with block_size=16 and max_staging_blocks=256,
    # each buffer can hold KV for 256*16=4096 tokens.
    max_staging_blocks: int | None = None

    # Fraction of available GPU memory to use for staging buffers (0.0-1.0)
    # Only used if max_staging_blocks is None.
    gpu_memory_fraction: float = 0.9

    # whether turn on async prefetch
    enable_async_prefetch: bool = True

    # whether turn on async offload
    enable_async_offload: bool = True

    # CPU memory limit for KV cache offload in bytes
    cpu_memory_limit: int | None = None

    # If cpu_memory_limit is not set, cap CPU backing store budget to
    # (available_system_memory * cpu_memory_fraction).
    # This is a safety knob to avoid consuming all host memory by default.
    # When `enable_layer_recompute=True`, this budget is shared by:
    # - CPU KV cache
    # - CPU hidden-state snapshots for recompute
    cpu_memory_fraction: float = 0.3

    # Enable layer-wise KV recompute + RunKV IO hybrid path.
    enable_layer_recompute: bool = False

    # Number of IO prefix blocks per layer. Empty list means disabled/unset.
    # A single value is allowed and expanded to all layers by the runner.
    layer_recompute_io_prefix_blocks: list[int] = field(default_factory=list)

    # Emit additional recompute vs IO overlap timing metrics.
    layer_recompute_measure_overhead: bool = False

    # Replay input source mode for layer recompute.
    layer_recompute_mode: Literal["io_hidden_states", "prev_layer_output_dynamic"] = (
        "io_hidden_states"
    )

    # Planner mode for dynamic replay. "static" preserves the current behavior.
    # "tightllm" uses an offline-profiled ILP solver (TightLLM, Hu et al. 2025).
    layer_recompute_planner: Literal["static", "feedback", "tightllm"] = "static"

    # When enabled, planner state may update but must not change execution.
    layer_recompute_planner_dry_run: bool = False

    # --- TightLLM planner settings ---
    # Path to offline profile JSON produced by tightllm_offline_profiler.
    # Required when layer_recompute_planner == "tightllm".
    tightllm_profile_path: str | None = None

    # When True, the ILP prediction is refined by a small additive correction
    # derived from the runtime imbalance signal (compute vs DMA).
    tightllm_enable_feedback_correction: bool = False

    # --- Feedback planner: state-machine controller ---
    # When True, FeedbackReplayPlanProvider uses the three-state
    # imbalance controller (STEADY / TRANSIT / TRACKING) described in
    # docs/design/imbalance_state_machine_controller.md instead of the
    # legacy per-layer Newton secant update.  Plan-reuse gating on the
    # pre_hook switches from the `last_observed_stable()` heuristic to a
    # Δbudget-driven hint (unchanged / small_delta / significant_delta).
    layer_recompute_use_state_machine: bool = False
