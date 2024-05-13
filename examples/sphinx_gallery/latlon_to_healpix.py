# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HealPIX regridding
------------------

In this notebook, I demonstrate bilinear regridding onto healpy grids in O(10)
ms. this is a 3 order of magnitude speed-up compared to what Dale has reported.

Now, lets define a healpix grid with indexing in the XY convention. we convert
to NEST indexing in order to use the `healpy.pix2ang` to get the lat lon
coordinates. This operation is near instant.

"""
# %%

import matplotlib.pyplot as plt
import numpy as np
import torch

import earth2grid

# level is the resolution
level = 6
hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
src = earth2grid.latlon.equiangular_lat_lon_grid(32, 64)
regrid = earth2grid.get_regridder(src, hpx)


z = np.cos(np.deg2rad(src.lat[:, None])) * np.cos(np.deg2rad(src.lon))


z_torch = torch.as_tensor(z)
z_hpx = regrid(z_torch)

fig, (a, b) = plt.subplots(2, 1)
a.pcolormesh(src.lon, src.lat, z)
a.set_title("Lat Lon Grid")

b.scatter(hpx.lon, hpx.lat, c=z_hpx, s=0.1)
b.set_title("Healpix")

# %% one tile
nside = 2**level
reshaped = z_hpx.reshape(12, nside, nside)
lat_r = hpx.lat.reshape(12, nside, nside)
lon_r = hpx.lon.reshape(12, nside, nside)

tile = 11
fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
axs = axs.ravel()

for tile in range(12):
    axs[tile].pcolormesh(lon_r[tile], lat_r[tile], reshaped[tile])
