import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2grid import healpix
from earth2grid.third_party.zephyr.healpix import healpix_pad

import pytest


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
    i = np.arange(12*n*n)
    i_rot = healpix._rotate_index(n, rot, i=i)
    i = i.reshape(12, -1)
    i_rot = i_rot.reshape(12, -1)
    for f in range(12):
        assert set(i[f]) == set(i_rot[f])


@pytest.mark.parametrize("rot", range(4))
def test_rotate_index(rot):
    n = 32
    i = np.arange(12*n*n)
    i_rot = healpix._rotate_index(n, rot, i=i)
    i_back = healpix._rotate_index(n, 4 - rot, i=i_rot)
    np.testing.assert_array_equal(i_back, i)


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
def test_to_from_faces(tmp_path, origin, clockwise):
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    z = np.cos(np.deg2rad(grid.lat)) * np.cos(np.deg2rad(grid.lon))
    z = torch.from_numpy(z)
    # padded = healpix_pad(z.clone(), 1)
    faces = grid.to_faces(z)
    z_roundtrip = grid.from_faces(faces)
    np.testing.assert_array_equal(z, z_roundtrip)


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
def test_grid_healpix_pad(tmp_path, origin, clockwise):
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    z = np.cos(np.deg2rad(grid.lat)) * np.cos(np.deg2rad(grid.lon))
    z = torch.from_numpy(z)
    # add singleton dimension for compatibility with hpx pad
    z = z[None]
    z = grid.to_faces(z)
    padded = healpix_pad(z, 1)

    def grad_abs(z):
        fx, fy = np.gradient(z, axis=(-1, -2))
        return np.mean(np.abs(fx)) + np.mean(np.abs(fy))

    # the padded dtile should not vary much more than the non-padded tile
    sigma_padded = grad_abs(padded)
    sigma = grad_abs(z)

    if  sigma_padded > sigma * 1.1:

        fig, axs = plt.subplots(3, 4)
        axs = axs.ravel()
        for i in range(12):
            ax = axs[i]
            ax.pcolormesh(z[0, i])
        output_path = tmp_path / "output.png"
        fig.savefig(output_path.as_posix())

        raise ValueError(f"The gradient of the padded data {sigma_padded} is too large. "
                         f"Examine the boundaries in {output_path}.")
