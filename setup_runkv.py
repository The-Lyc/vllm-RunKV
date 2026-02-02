# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Setup script to build RunKV batch copy CUDA extension (standalone module).

This builds the kernel as a separate module that can be used with existing
vLLM installations without rebuilding vLLM.

Usage:
    python setup_runkv.py install
    # or
    python setup_runkv.py build_ext --inplace
    # build, then copy the built .so into the current interpreter's site-packages
    python setup_runkv.py build_ext --inplace  # copies by default
    # disable the auto-copy behavior
    python setup_runkv.py build_ext --inplace --no-copy-to-site
"""

import os
import shutil
import sysconfig

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# NOTE: This is a standalone module; users often want "build_ext --inplace"
# but still be able to import it from anywhere in the same Python env.
class _BuildExtAndCopyToSite(BuildExtension):
    user_options = BuildExtension.user_options + [
        (
            "install-dir=",
            None,
            "Directory to copy the built"
            " extension into (defaults to sysconfig platlib)",
        ),
        ("no-copy-to-site", None, "Do not copy the built extension into site-packages"),
    ]
    boolean_options = getattr(BuildExtension, "boolean_options", []) + [
        "no-copy-to-site",
    ]

    def initialize_options(self):
        super().initialize_options()
        self.install_dir = None
        self.no_copy_to_site = False

    def run(self):
        super().run()
        if self.no_copy_to_site:
            return

        target_dir = self.install_dir or sysconfig.get_path("platlib")
        if not target_dir:
            raise RuntimeError("Could not determine site-packages (sysconfig platlib).")

        os.makedirs(target_dir, exist_ok=True)

        # Copy each built extension artifact into site-packages so it is importable
        # from anywhere in this Python environment.
        for ext in self.extensions:
            built_path = os.path.abspath(self.get_ext_fullpath(ext.name))
            target_path = os.path.join(target_dir, os.path.basename(built_path))
            if os.path.samefile(os.path.dirname(built_path), target_dir):
                continue
            shutil.copy2(built_path, target_path)


# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get torch lib path for RPATH
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

# Get CUDA lib path
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cuda_lib_path = os.path.join(cuda_home, "lib64")

# Set RPATH so the .so can find dependencies without LD_LIBRARY_PATH
rpath_flags = [
    f"-Wl,-rpath,{torch_lib_path}",
    f"-Wl,-rpath,{cuda_lib_path}",
]

setup(
    name="runkv_kernels",
    version="0.1.0",
    packages=[],  # Don't include any Python packages, only the extension
    ext_modules=[
        CUDAExtension(
            name="runkv_kernels",
            sources=[
                os.path.join(script_dir, "csrc/runkv_copy_kernels.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
            extra_link_args=rpath_flags,
            # Define RUNKV_STANDALONE_MODULE to enable PYBIND11 module
            define_macros=[("RUNKV_STANDALONE_MODULE", "1")],
        ),
    ],
    cmdclass={"build_ext": _BuildExtAndCopyToSite},
)
