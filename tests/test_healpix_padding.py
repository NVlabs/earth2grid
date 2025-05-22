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
import numpy
import torch

from earth2grid.healpix import XY, Grid, pad_with_dim
from earth2grid.third_party.zephyr.healpix import healpix_pad


def test_hpx_pad(regtest):
    order = 2
    nside = 2**order
    face = 5

    grid = Grid(order, pixel_order=XY())
    lat = torch.from_numpy(grid.lat)
    padded = pad_with_dim(lat, padding=nside, dim=-1)
    m = nside + 2 * nside
    padded = padded.reshape(12, m, m)

    numpy.savetxt(regtest, padded[face].cpu(), fmt="%.2f")


def test_healpix_pad():
    ntile = 12
    nside = 32
    padding = 1
    n = 3
    x = torch.ones([n, ntile, nside, nside])
    out = healpix_pad(x, padding=padding)
    assert out.shape == (n, ntile, nside + padding * 2, nside + padding * 2)
