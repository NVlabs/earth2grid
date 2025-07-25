# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
import warnings
from typing import List

import torch
from setuptools import setup
from torch.utils import cpp_extension

VERSION = "2025.7.1"


def get_compatible_torch_version():
    # the built wheel will only be compatible with the currently installed torch
    # minor version
    major, minor = torch.__version__.split(".")[:2]
    major = int(major)
    minor = int(minor)
    torch_version_str = f"torch>={major}.{minor},<{major}.{minor + 1}"
    return torch_version_str


def get_version_str():
    major, minor = torch.__version__.split(".")[:2]
    major = int(major)
    minor = int(minor)
    return VERSION + f"+torch{major}{minor}"


def get_compiler():
    try:
        # Try to get the compiler path from the CC environment variable
        # If not set, it will default to gcc (which could be symlinked to clang or g++)
        compiler = subprocess.check_output(["gcc", "--version"], universal_newlines=True)  # noqa: S603, S607

        if "clang" in compiler:
            return "clang"
        elif "g++" in compiler or "gcc" in compiler:
            return "gnu"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error detecting compiler: {e}")
        return "unknown"


compiler_type = get_compiler()
extra_compile_args: List[str] = ["-std=c++20"]

if compiler_type == "clang":
    print("Detected Clang compiler.")
    # Additional settings or flags specific to Clang can be added here
    extra_compile_args += ["-Wno-error=c++11-narrowing", "-Wno-c++11-narrowing"]
elif compiler_type == "gnu":
    print("Detected GNU compiler.")
    # Additional settings or flags specific to G++ can be added here
else:
    print("Could not detect compiler or unknown compiler detected.")


src_files = [
    "earth2grid/csrc/healpix_bare_wrapper.cpp",
]
cuda_src_files = [
    "earth2grid/csrc/healpixpad/healpixpad_cuda.cpp",
    "earth2grid/csrc/healpixpad/healpixpad_cuda_fwd.cu",
    "earth2grid/csrc/healpixpad/healpixpad_cuda_bwd.cu",
]

ext_modules = [
    cpp_extension.CppExtension(
        "earth2grid._healpix_bare",
        src_files,
        extra_compile_args=extra_compile_args,
        include_dirs=[os.path.abspath("earth2grid/csrc"), os.path.abspath("earth2grid/third_party/healpix_bare")],
    ),
]

try:
    from torch.utils.cpp_extension import CUDAExtension

except ImportError:
    warnings.warn("Cuda extensions for torch not found, skipping cuda healpix padding module")

    CUDAExtension = None

if CUDAExtension is not None:
    try:
        ext_modules.append(
            CUDAExtension(
                name="healpixpad_cuda",
                sources=cuda_src_files,
                extra_compile_args={"nvcc": ["-O2"]},
            ),
        )
    except OSError:
        warnings.warn(
            "CUDA extension raised an OSError. Will not build the cuda extension. Do you have the CUDA compilers installed?"
        )


dependencies = ["einops>=0.7.0", "numpy>=1.23.3", get_compatible_torch_version(), "scipy"]

setup(
    name="earth2grid",
    version=get_version_str(),
    ext_modules=ext_modules,
    install_requires=dependencies,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
