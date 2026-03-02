# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field


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
    gpu_memory_fraction: float = 0.1

    # whether turn on async prefetch
    enable_async_prefetch: bool = True

    # whether turn on async offload
    enable_async_offload: bool = True

    # CPU memory limit for KV cache offload in bytes
    cpu_memory_limit: int | None = None

    # If cpu_memory_limit is not set, cap CPU KV cache backing store to
    # (available_system_memory * cpu_memory_fraction).
    # This is a safety knob to avoid consuming all host memory by default.
    cpu_memory_fraction: float = 0.7

    # Enable layer-wise KV recompute + RunKV IO hybrid path.
    enable_layer_recompute: bool = False

    # Number of IO prefix blocks per layer. Empty list means disabled/unset.
    # A single value is allowed and expanded to all layers by the runner.
    layer_recompute_io_prefix_blocks: list[int] = field(default_factory=list)

    # Emit additional recompute vs IO overlap timing metrics.
    layer_recompute_measure_overhead: bool = False
