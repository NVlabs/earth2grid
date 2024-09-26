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
import numpy as np
import pytest
import torch

import earth2grid.healpix_bare


def test_ring2nest():
    n = 8
    i = torch.arange(n * n * 12)

    j = earth2grid.healpix_bare.ring2nest(n, i)
    i_round = earth2grid.healpix_bare.nest2ring(n, j)
    numpy.testing.assert_array_equal(i, i_round)


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("lonlat", [True, False])
def test_pix2ang(nest, lonlat, regtest):
    n = 2
    i = torch.arange(n * n * 12)

    x, y = earth2grid.healpix_bare.pix2ang(n, i, nest=nest, lonlat=lonlat)
    print("x", file=regtest)
    np.savetxt(regtest, x, fmt="%.5e")

    print("y", file=regtest)
    np.savetxt(regtest, y, fmt="%.5e")


def savetxt(file, array):
    np.savetxt(file, array, fmt="%.5e")


def test_hpc2loc(regtest):
    x = torch.tensor([0.0]).double()
    y = torch.tensor([0.0]).double()
    f = torch.tensor([0])

    loc = earth2grid.healpix_bare.hpc2loc(x, y, f)
    vec = earth2grid.healpix_bare.loc2vec(loc)
    for array in vec:
        savetxt(regtest, array)


def test_boundaries(regtest):
    ipix = torch.tensor([0])
    boundaries = earth2grid.healpix_bare.corners(1, ipix, False)
    assert not torch.any(torch.isnan(boundaries)), boundaries
    assert boundaries.shape == (1, 3, 4)
    savetxt(regtest, boundaries.flatten())


def test_get_interp_weights_vector():
    lon = torch.tensor([23, 84, -23]).float()
    lat = torch.tensor([0, 12, 67]).float()
    pix, weights = earth2grid.healpix_bare.get_interp_weights(8, lon, lat)
    assert pix.device == lon.device
    assert pix.shape == (4, 3)
    assert weights.shape == (4, 3)


def test_get_interp_weights_vector_interp_y():
    nside = 16
    inpix = torch.tensor([0, 1, 5, 6])

    lon, lat = earth2grid.healpix_bare.pix2ang(nside, inpix, lonlat=True)
    ay = 0.8

    lonc = (lon[0] + lon[1]) / 2
    latc = lat[0] * ay + lat[3] * (1 - ay)

    pix, weights = earth2grid.healpix_bare.get_interp_weights(nside, lonc.unsqueeze(0), latc.unsqueeze(0))

    assert torch.all(pix == inpix[:, None])
    expected_weights = torch.tensor([ay / 2, ay / 2, (1 - ay) / 2, (1 - ay) / 2]).double()[:, None]
    assert torch.allclose(weights, expected_weights)


@pytest.mark.parametrize("lonlat", [True, False])
def test_ang2pix(lonlat):
    if lonlat:
        lon = torch.tensor([32.0])
        lat = torch.tensor([45.0])
    else:
        lon = torch.tensor([1.0])
        lat = torch.tensor([2.0])

    n = 2**16

    pix = earth2grid.healpix_bare.ang2pix(n, lon, lat, lonlat=lonlat)
    lon_out, lat_out = earth2grid.healpix_bare.pix2ang(n, pix, lonlat=lonlat)

    assert lon.item() == pytest.approx(lon_out.item(), rel=1e-4)
    assert lat.item() == pytest.approx(lat_out.item(), rel=1e-4)
