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
def test_grid_healpix_plot_orientation(tmp_path, rot):
    level = 3
    grid = healpix.Grid(level=level, pixel_order=healpix.XY())
    z = np.cos(np.deg2rad(grid.lat))
    z = torch.from_numpy(z)
    n = 2**level
    i = np.arange(12*n*n)

    i_rot = healpix._rotate_index(n, rot, flip=False, i=i)
    z_rot = z[i_rot]
    z_rot = z_rot.reshape([12, grid._nside(), grid._nside()])


    i = i.reshape(12, -1)
    i_rot = i_rot.reshape(12, -1)

    for f in range(12):
        assert set(i[f]) == set(i_rot[f])


def test_grid_healpix_pad(tmp_path):
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=healpix.Compass.N, clockwise=True))
    z = np.cos(np.deg2rad(grid.lat)) * np.cos(np.deg2rad(grid.lon))
    z = z.reshape([1, 12, grid._nside(), grid._nside()])
    z = torch.from_numpy(z)
    padded = healpix_pad(z.clone(), 1)


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
            ax.pcolormesh(padded[0, i])
        output_path = tmp_path / "output.png"
        fig.savefig(output_path.as_posix())

        raise ValueError(f"The gradient of the padded data {sigma_padded} is too large. "
                         f"Examine the boundaries in {output_path}.")
