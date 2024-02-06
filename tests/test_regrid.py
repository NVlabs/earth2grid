import torch

import earth2grid


def test_regridder_healpix():
    dest = earth2grid.healpix.Grid(level=6, pixel_order=earth2grid.healpix.PixelOrder.XY)
    src = earth2grid.latlon.equiangular_lat_lon_grid(33, 64)
    regrid = earth2grid.get_regridder(src, dest)
    x = torch.zeros(src.shape)

    assert regrid(x).shape == dest.shape
