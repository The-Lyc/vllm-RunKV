#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the Llama benchmark batch with a Flux contender around every case."""

from __future__ import annotations

import sys

from run_llama_benchmark_batch import main


if __name__ == "__main__":
    sys.exit(
        main(
            pipeline_args=("--flux-contender",),
            description=__doc__,
        )
    )
