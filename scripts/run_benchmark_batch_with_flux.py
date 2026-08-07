#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run benchmark batch settings with a fresh Flux service per subtest.

The configuration and benchmark parameters are identical to
``run_benchmark_batch.py``. Before each RunKV or TightLLM subtest, this runner
starts the Flux contender on GPU 0 from its diffusion virtual environment.
The service is stopped after that subtest before the next one is started.
"""

from __future__ import annotations

import sys

from run_benchmark_batch import main


if __name__ == "__main__":
    sys.exit(main(pipeline_args=("--flux-contender",), description=__doc__))
