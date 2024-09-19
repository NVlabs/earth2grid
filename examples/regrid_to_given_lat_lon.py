import numpy as np
import torch

import earth2grid

device = "cuda"


# the source grid (from 90N to 90S and 0E to 360E)
ll = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)

# a 2d grid of target lat lons
target_lat = np.linspace(30, 50, 32)
target_lon = np.linspace(100, 120, 64)
target_lat, target_lon = np.meshgrid(target_lat, target_lon)

# Some source data on the original grid
data = torch.ones([721, 1440]).to(device)

# Create a bilinear regridding object earth2grid
regrid = ll.get_bilinear_regridder_to(target_lat, target_lon)

# need to move the weights to same device and dtype as data
regrid.to(data)

# perform the regridding
out = regrid(data)
assert out.shape == target_lat.shape  # noqa
print("data shape", out.shape)
