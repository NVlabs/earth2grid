import numpy
import torch

from earth2grid.healpix import XY, Grid
from earth2grid.healpix.padding import _xy_with_filled_tile, local2xy


def test_hpx_pad(regtest):
    order = 2
    nside = 2**order
    pad = pad_x = nside
    face = 5

    grid = Grid(order, pixel_order=XY())
    lat = torch.from_numpy(grid.lat)

    x = torch.arange(-pad_x, nside + pad_x)
    y = torch.arange(-pad, nside + pad)
    f = torch.tensor([face])

    f, y, x = torch.meshgrid(f, y, x, indexing="ij")

    x1, y1, f1 = local2xy(nside, x, y, f)

    def _to_pix(xy):
        x, y, f = xy
        return torch.where(f < 12, nside**2 * f + nside * y + x, -1)

    xy_east, xy_west = _xy_with_filled_tile(nside, x1, y1, f1)

    xy_east = _to_pix(xy_east)
    xy_west = _to_pix(xy_west)

    padded_from_west = torch.where(xy_west >= 0, lat[xy_west], 0)
    padded_from_east = torch.where(xy_east >= 0, lat[xy_east], 0)
    denom = (xy_west >= 0).int() + (xy_east >= 0).int()

    padded = (padded_from_east + padded_from_west) / denom
    numpy.savetxt(regtest, padded[0].cpu(), fmt="%.2f")
