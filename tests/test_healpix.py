from earth2grid import healpix
import numpy as np
import matplotlib.pyplot as plt


def test_grid_visualize():
    grid = healpix.Grid(level=4, pixel_order=healpix.PixelOrder.XY)
    z = np.cos(10 * np.deg2rad(grid.lat))
    grid.visualize(z)
    plt.savefig("test_grid_visualize.png")
