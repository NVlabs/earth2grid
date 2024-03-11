import healpy
import numpy
import numpy as np
import pytest
import torch

import earth2grid.healpix_bare


@pytest.mark.parametrize("func", ["ring2nest", "nest2ring"])
def test_ring2nest(func):
    n = 8
    i = torch.arange(n * n * 12)

    bare_func = getattr(earth2grid.healpix_bare, func)
    healpy_func = getattr(healpy, func)

    answer = bare_func(n, i)
    expected = healpy_func(n, i)
    numpy.testing.assert_array_equal(answer, expected)


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("lonlat", [True, False])
def test_pix2ang(nest, lonlat, tmp_path):
    n = 32
    i = torch.arange(n * n * 12)

    x, y = earth2grid.healpix_bare.pix2ang(n, i, nest=nest, lonlat=lonlat)
    xe, ye = healpy.pix2ang(n, i, nest=nest, lonlat=lonlat)
    import matplotlib.pyplot as plt

    try:
        numpy.testing.assert_allclose(x, xe)
        numpy.testing.assert_allclose(y, ye)
    except AssertionError as e:
        healpy.cartview(x, sub=(2, 2, 1), nest=nest)
        healpy.cartview(xe, sub=(2, 2, 2), nest=nest)
        healpy.cartview(y, sub=(2, 2, 3), nest=nest)
        healpy.cartview(ye, sub=(2, 2, 4), nest=nest)

        plt.tight_layout()
        path = tmp_path / "image.png"
        plt.savefig(path, bbox_inches="tight")
        e.add_note(str(path))
        raise e


def test_hpc2loc():
    x = torch.tensor([0.0]).double()
    y = torch.tensor([0.0]).double()
    f = torch.tensor([0])

    loc = earth2grid.healpix_bare.hpc2loc(x, y, f)
    vec = earth2grid.healpix_bare.loc2vec(loc)
    print(vec)
    print(healpy.boundaries(1, 1, 1, nest=True))


def test_boundaries():
    boundaries = earth2grid.healpix_bare.boundaries(1, 0, 1)
    assert not torch.any(torch.isnan(boundaries)), boundaries

    expected = healpy.boundaries(1, 0, 1)
    np.testing.assert_allclose(boundaries, expected)
