import unittest

import torch

import earth2grid
from earth2grid._regrid import BilinearInterpolator


def test_regridder_healpix():
    dest = earth2grid.healpix.Grid(level=6, pixel_order=earth2grid.healpix.PixelOrder.XY)
    src = earth2grid.latlon.equiangular_lat_lon_grid(33, 64)
    regrid = earth2grid.get_regridder(src, dest)

    def f(lat, lon):
        lat = torch.from_numpy(lat)
        lon = torch.from_numpy(lon)
        return torch.cos(torch.deg2rad(lat)) * torch.sin(2 * torch.deg2rad(lon))

    z = f(src.lat[:, None], src.lon)
    z_regridded = regrid(z)
    expected = f(dest.lat, dest.lon)
    assert torch.allclose(z_regridded, expected, rtol=0.01)


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
        result = interpolator(input_tensor)

        # Verify
        self.assertTrue(torch.allclose(result, expected), "The interpolated value does not match the expected value.")
