import torch

from earth2grid.third_party.zephyr.healpix import healpix_pad


def test_healpix_pad():
    ntile = 12
    nside = 32
    padding = 1
    n = 3
    x = torch.ones([n, ntile, nside, nside])
    out = healpix_pad(x, padding=padding)
    assert out.shape == (n, ntile, nside + padding * 2, nside + padding * 2)
