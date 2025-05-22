import numpy
import torch

from earth2grid.healpix import XY, Grid, padding


def test_hpx_pad(regtest):
    order = 2
    nside = 2**order
    face = 5

    grid = Grid(order, pixel_order=XY())
    lat = torch.from_numpy(grid.lat)
    padded = padding.pad(lat, padding=nside, dim=-1)
    m = nside + 2 * nside
    padded = padded.reshape(12, m, m)

    numpy.savetxt(regtest, padded[face].cpu(), fmt="%.2f")
