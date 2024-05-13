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
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from earth2grid import get_regridder, healpix


@pytest.mark.xfail
def test_grid_visualize():
    grid = healpix.Grid(level=4, pixel_order=healpix.XY())
    z = np.cos(10 * np.deg2rad(grid.lat))
    grid.visualize(z)
    plt.savefig("test_grid_visualize.png")


@pytest.mark.parametrize("origin", list(healpix.Compass))
def test_grid_healpix_orientations(tmp_path, origin):

    nest_grid = healpix.Grid(level=4, pixel_order=healpix.PixelOrder.NEST)
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin))

    nest_lat = nest_grid.lat.reshape([12, -1])
    lat = grid.lat.reshape([12, -1])

    for i in range(12):
        assert set(nest_lat[i]) == set(lat[i])


@pytest.mark.parametrize("rot", range(4))
def test_rotate_index_same_values(tmp_path, rot):
    n = 8
    i = np.arange(12 * n * n)
    i_rot = healpix._rotate_index(n, rot, i=i)
    i = i.reshape(12, -1)
    i_rot = i_rot.reshape(12, -1)
    for f in range(12):
        assert set(i[f]) == set(i_rot[f])


@pytest.mark.parametrize("rot", range(4))
def test_rotate_index(rot):
    n = 32
    i = np.arange(12 * n * n)
    i_rot = healpix._rotate_index(n, rot, i=i)
    i_back = healpix._rotate_index(n, 4 - rot, i=i_rot)
    np.testing.assert_array_equal(i_back, i)


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
def test_reorder(tmp_path, origin, clockwise):
    src_grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    dest_grid = healpix.Grid(level=4, pixel_order=healpix.PixelOrder.NEST)

    z = np.cos(np.deg2rad(src_grid.lat)) * np.cos(np.deg2rad(src_grid.lon))
    z = torch.from_numpy(z)
    z_reorder = src_grid.reorder(dest_grid.pixel_order, z)
    z_roundtrip = dest_grid.reorder(src_grid.pixel_order, z_reorder)
    np.testing.assert_array_equal(z, z_roundtrip)


def get_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices += [torch.device("cuda")]
    return devices


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("device", get_devices())
def test_grid_healpix_pad(tmp_path, origin, clockwise, padding, device):
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    hpx_pad_grid = healpix.Grid(level=4, pixel_order=healpix.HEALPIX_PAD_XY)
    z = np.cos(np.deg2rad(grid.lat)) * np.cos(np.deg2rad(grid.lon))
    z = torch.from_numpy(z)
    regrid = get_regridder(grid, hpx_pad_grid)
    z_hpx_pad = regrid(z)

    n = grid._nside()
    z = z.view(-1, 12, n, n)
    z_hpx_pad = z_hpx_pad.view(-1, 12, n, n)

    padded = healpix.pad(z_hpx_pad.to(device), padding).cpu()

    def grad_abs(z):
        fx, fy = np.gradient(z, axis=(-1, -2))
        return np.mean(np.abs(fx)) + np.mean(np.abs(fy))

    # the padded dtile should not vary much more than the non-padded tile
    sigma_padded = grad_abs(padded)
    sigma = grad_abs(z)

    if sigma_padded > sigma * 1.1:

        fig, axs = plt.subplots(3, 4)
        axs = axs.ravel()
        for i in range(12):
            ax = axs[i]
            ax.pcolormesh(padded[0, i])
        output_path = tmp_path / "output.png"
        fig.savefig(output_path.as_posix())

        raise ValueError(
            f"The gradient of the padded data {sigma_padded} is too large. "
            f"Examine the padding in the image at {output_path}."
        )


def test_to_image():
    grid = healpix.Grid(level=4)
    lat = torch.tensor(grid.lat)
    lat_img = grid.to_image(lat)
    n = 2**grid.level
    assert lat_img.shape == (5 * n, 5 * n)


def test_conv2d():
    f = 12
    nside = 16
    npix = f * nside * nside
    cin = 3
    cout = 4
    n = 1

    x = torch.ones(n, cin, 1, npix)
    weight = torch.zeros(cout, cin, 3, 3)
    out = healpix.conv2d(x, weight, padding=(1, 1))
    assert out.shape == (n, cout, 1, npix)
