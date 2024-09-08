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
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import earth2grid
from earth2grid import BilinearInterpolator


@pytest.mark.parametrize("with_channels", [True, False])
def test_latlon_regridder(with_channels, tmp_path):
    nlat = 30
    nlon = 60

    src = earth2grid.healpix.Grid(level=6, pixel_order=earth2grid.healpix.XY())

    lat = np.linspace(-90, 90, nlat + 2)[1:-1]
    lon = np.linspace(0, 360, nlon)
    dest = earth2grid.latlon.LatLonGrid(lat, lon)
    regridder = earth2grid.get_regridder(src, dest)

    z = np.cos(10 * np.deg2rad(src.lat))
    z = torch.from_numpy(z)
    if with_channels:
        z = torch.stack([z, 0 * z])

    out = regridder(z)

    if out.ndim == 3:
        out = out[0]

    assert out.shape[-2:] == (nlat, nlon)

    expected = np.cos(10 * np.deg2rad(lat))[:, None]
    diff = np.mean(np.abs(out.numpy() - expected))
    if diff > 1e-3 * 90 / nlat:
        plt.figure()
        if with_channels:
            out = out[0]
        plt.pcolormesh(lon, lat, out - expected)
        plt.title("regridded - expected")
        plt.colorbar()
        image_path = tmp_path / "test_latlon_regridder.png"
        plt.savefig(image_path)
        raise ValueError(f"{diff} too big. See {image_path}.")


@pytest.mark.parametrize("with_channels", [True, False])
def test_healpix_to_lat_lon(with_channels):
    dest = earth2grid.healpix.Grid(level=6, pixel_order=earth2grid.healpix.XY())
    src = earth2grid.latlon.equiangular_lat_lon_grid(33, 64)
    regrid = earth2grid.get_regridder(src, dest)

    def f(lat, lon):
        lat = torch.from_numpy(lat)
        lon = torch.from_numpy(lon)
        return torch.cos(torch.deg2rad(lat)) * torch.sin(2 * torch.deg2rad(lon))

    z = f(src.lat, src.lon)
    if with_channels:
        z = z[None]
    z_regridded = regrid(z)
    expected = f(dest.lat, dest.lon)
    assert torch.allclose(z_regridded, expected, rtol=0.01)


@pytest.mark.parametrize("with_channels", [True, False])
@pytest.mark.parametrize("level", [4, 5])
def test_healpix_to_healpix(with_channels, level):
    """Test finer healpix interpolation"""
    src = earth2grid.healpix.Grid(level=4)
    dest = earth2grid.healpix.Grid(level=level)
    regrid = earth2grid.get_regridder(src, dest)

    def f(lat, lon):
        lat = torch.from_numpy(lat)
        lon = torch.from_numpy(lon)
        return torch.cos(torch.deg2rad(lat)) * torch.sin(2 * torch.deg2rad(lon))

    z = f(src.lat, src.lon)
    if with_channels:
        z = z[None]
    z_regridded = regrid(z)
    expected = f(dest.lat, dest.lon)
    max_diff = torch.max(z_regridded - expected)
    print(max_diff)
    assert torch.allclose(z_regridded, expected, atol=0.03)


@pytest.mark.parametrize("reverse", [True, False])
def test_latlon_to_latlon(reverse):
    nlat = 30
    nlon = 60
    lon = np.linspace(0, 360, nlon)
    lat = np.linspace(-90, 90, nlat)
    src = earth2grid.latlon.LatLonGrid(lat[::-1] if reverse else lat, lon)
    dest = earth2grid.latlon.LatLonGrid(lat, lon)
    regrid = earth2grid.get_regridder(src, dest)
    regrid.float()

    z = torch.zeros(src.shape)
    regrid(z)


class TestBilinearInterpolateNonUniform(unittest.TestCase):
    def test_interpolation(self):
        # Setup
        H, W = 5, 5  # Input tensor height and width
        input_tensor = torch.arange(1.0, H * W + 1).view(H, W)  # Example 2D tensor
        x_coords = torch.linspace(-1, 1, steps=W)  # Example non-uniform x-coordinates
        y_coords = torch.linspace(-1, 1, steps=H)  # Example non-uniform y-coordinates
        x_query = torch.tensor([0.0])  # Query x-coordinates at the center
        y_query = torch.tensor([0.0])  # Query y-coordinates at the center

        # Expected value at the center of a linearly spaced grid
        expected = torch.tensor([(H * W + 1) / 2])

        # Execute
        interpolator = BilinearInterpolator(x_coords, y_coords, x_query, y_query)
        interpolator.to(input_tensor)
        result = interpolator(input_tensor)

        # Verify
        self.assertTrue(torch.allclose(result, expected), "The interpolated value does not match the expected value.")

    def test_raises_error_when_coordinates_not_increasing_x(self):
        x_coords = torch.linspace(1, -1, steps=32)  # Example non-uniform x-coordinates
        y_coords = torch.linspace(-1, 1, steps=32)  # Example non-uniform y-coordinates
        with self.assertRaises(ValueError):
            BilinearInterpolator(x_coords, y_coords, torch.tensor([0]), torch.tensor([0]))

    def test_raises_error_when_coordinates_not_increasing_y(self):
        x_coords = torch.linspace(-1, 1, steps=32)  # Example non-uniform x-coordinates
        y_coords = torch.linspace(1, -1, steps=32)  # Example non-uniform y-coordinates
        with self.assertRaises(ValueError):
            BilinearInterpolator(x_coords, y_coords, torch.tensor([0]), torch.tensor([0]))

    def test_interpolation_func(self):
        # Setup
        H, W = 32, 32  # Input tensor height and width

        def func(x, y):

            return 10 * x + 5 * y**2 + 4

        x_coords = torch.linspace(-1, 1, steps=W)  # Example non-uniform x-coordinates
        y_coords = torch.linspace(-1, 1, steps=H)  # Example non-uniform y-coordinates
        x_query = torch.tensor([0.0, 0.5, 0.25])  # Query x-coordinates at the center
        y_query = torch.tensor([0.0, 0.0, -0.4])  # Query y-coordinates at the center

        input_tensor = func(x_coords, y_coords[:, None])

        # Expected value at the center of a linearly spaced grid
        expected = func(x_query, y_query)

        # Execute
        interpolator = BilinearInterpolator(x_coords, y_coords, x_query, y_query)
        interpolator.to(input_tensor)
        result = interpolator(input_tensor)

        # Verify
        self.assertTrue(torch.allclose(result, expected, rtol=0.01))

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            interpolator.cuda()
            interpolator(input_tensor.cuda())


def test_out_of_bounds():
    x_coords = torch.tensor([0, 1, 2]).float()
    y_coords = torch.tensor([0, 1, 2]).float()

    x_query = torch.tensor([-1, 3]).float()
    y_query = torch.tensor([-1, 3]).float()
    regrid = BilinearInterpolator(x_coords, y_coords, x_query, y_query)

    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
    regrid.to(data)
    output = regrid(data)

    assert torch.all(torch.isnan(output))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_NearestNeighborInterpolator(k):
    n = 10000
    torch.manual_seed(0)
    lon = torch.rand(n) * 360
    lat = torch.rand(n) * 180 - 90

    lond = torch.rand(n) * 360
    latd = torch.rand(n) * 180 - 90

    interpolate = earth2grid.S2NearestNeighborInterpolator(lon, lat, lond, latd, k=k)
    out = interpolate(torch.cos(torch.deg2rad(lon)))
    expected = torch.cos(torch.deg2rad(lond))
    mae = torch.mean(torch.abs(out - expected))
    assert mae.item() < 0.02

    # load-reload
    earth2grid.Regridder.from_state_dict(interpolate.state_dict())
