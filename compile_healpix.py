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
import time

import torch

from earth2grid.healpix import padding
from earth2grid.healpix.core import pad as healpixpad
from earth2grid.third_party.zephyr.healpix import healpix_pad as zephyr_pad

nside = 128


neval = 10


def test_func(label, pad):
    # warm up
    out = pad(p, padding=nside // 2)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(neval):
        out = pad(p, padding=nside // 2)
    torch.cuda.synchronize()
    stop = time.time()
    gb_per_sec = out.nbytes * neval / (stop - start) / 1e9
    label = label + ":"
    label = label + max(30 - len(label), 0) * " "
    print(f"{label} {gb_per_sec=:.2f}")


for batch_size in [1, 2]:
    p = torch.randn(batch_size, 12, 384, nside, nside)
    print(f"Benchmarking results {neval=} {p.size()=}")

    p = p.cuda()
    test_func("Python", padding.pad_compatible)
    pad = torch.compile(padding.pad_compatible)
    # pad = padding.pad_compatible
    out = pad(p, padding=nside // 2)
    test_func("Python + torch.compile", pad)
    test_func("HEALPix Pad", healpixpad)

    test_func("Zephyr pad", zephyr_pad)
    print("Zephyr pad doesn't work well with torch.compile. Doesn't finish compiling.")
    # test_func("Zephyr pad (torch.compile)", torch.compile(zephyr_pad))

    pad = torch.compile(padding.pad)
    # pad = padding.pad
    p = torch.randn(batch_size, 12 * nside * nside, 384).cuda()
    pad(p, padding=nside // 2)
    test_func("Python + torch.compile: channels dim last*", pad)
    print("")


print(f"* shape for channel dim last: {p.shape}")
