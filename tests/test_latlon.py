import torch

from earth2grid.latlon import equiangular_lat_lon_grid


def test_lat_lon_bilinear_regrid_to():
    src = equiangular_lat_lon_grid(15, 30)
    dest = equiangular_lat_lon_grid(30, 60)
    regrid = src.get_bilinear_regridder_to(dest.lat, dest.lon)

    regrid.float()
    lat = torch.broadcast_to(torch.tensor(src.lat), src.shape)
    z = torch.tensor(lat).float()

    out = regrid(z)
    assert out.shape == dest.shape
