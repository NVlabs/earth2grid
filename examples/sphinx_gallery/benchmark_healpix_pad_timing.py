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
# limitations under the License
"""
Benchmark different HEALPix padding implementations
---------------------------------------------------
"""
import time

import torch

from earth2grid import healpix
from earth2grid.healpix import pad_backend

# Print GPU information
if torch.cuda.is_available():
    print("CUDA available: Yes")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device capability: {torch.cuda.get_device_capability()}")
else:
    print("CUDA not available")
print("\n")

nside = 128
padding = nside // 2
channels = 384
dtype = torch.float32

neval = 10


def test_func(label, pad, compile=False):
    # Reset memory stats and clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # warm up
    if compile:
        pad = torch.compile(pad)
    out = pad(p, padding=padding)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(neval):
        out = pad(p, padding=padding)
    torch.cuda.synchronize()
    stop = time.time()
    gb_per_sec = out.nbytes * neval / (stop - start) / 1e9
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    label = label + ":"
    label = label + max(30 - len(label), 0) * " "
    print(f"{label} {gb_per_sec=:.2f} peak_memory={peak_memory:.2f}MB")


for batch_size in [1, 2]:
    p = torch.randn(size=(batch_size, 12, channels, nside, nside), dtype=dtype)
    print(f"Benchmarking results {neval=} {p.size()=} {padding=} {dtype=}")

    p = p.cuda()

    with pad_backend(healpix.PaddingBackends.indexing):
        test_func("Python", healpix.pad)
        test_func("Python + compile", healpix.pad, compile=True)

    with pad_backend(healpix.PaddingBackends.cuda):
        test_func("HEALPix Pad", healpix.pad)

    with pad_backend(healpix.PaddingBackends.zephyr):
        test_func("Zephyr pad", healpix.pad)
        print("Zephyr pad doesn't work well with torch.compile. Doesn't finish compiling.")

    p = torch.randn(size=(batch_size, 12 * nside * nside, channels), dtype=dtype).cuda()
    test_func("Python: channels dim last*", lambda x, padding: healpix.pad_with_dim(x, padding, dim=1), compile=False)
    test_func(
        "Python + torch.compile: channels dim last*",
        lambda x, padding: healpix.pad_with_dim(x, padding, dim=1),
        compile=True,
    )
    p_python_shape = p.shape

    p = p.view(batch_size, 12, nside, nside, channels).permute(0, 1, 4, 2, 3)
    with pad_backend(healpix.PaddingBackends.cuda):
        test_func("HEALPix Pad: channels dim last", healpix.pad)

    print("")


print(f"* shape for Python channels dim last: {p_python_shape}")
print(f"* shape for HEALPix Pad channels dim last: {p.shape}")
