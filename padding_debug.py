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
# ruff: noqa
# %%
import matplotlib.pyplot as plt
import torch

from earth2grid.healpix import XY, Grid
from earth2grid import healpix
from earth2grid.healpix.padding import local2xy, _xy_with_filled_tile
from earth2grid.healpix import padding

order = 2
nside = 2**order
pad = pad_x = nside
face = 5

pix = torch.arange(12 * nside**2)
grid = Grid(order, pixel_order=XY())
lat = torch.from_numpy(grid.lat)

x = torch.arange(-pad_x, nside + pad_x)
y = torch.arange(-pad, nside + pad)
# y = torch.arange(0, nside)
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


z = padded[0]
# z = torch.where(xy_west >= 0, xy_west % nside, torch.nan)[0]

plt.pcolormesh((x - y)[0], (y + x)[0], z)
# plt.xlabel("x")
# plt.ylabel("y")
plt.colorbar()

# %%


# %%
x = torch.from_numpy(grid.lat).reshape(1, 12, nside, nside)
padding.pad(x, 3)

# %%

padgrid = Grid(order, healpix.HEALPIX_PAD_XY)
lat = torch.from_numpy(padgrid.lat).reshape([1, 12, nside, nside])
pix = torch.arange(padgrid.shape[0]).float().reshape([1, 12, nside, nside])
padded = healpix.pad(lat, pad)
padded = torch.flip(padded[0, face], (0, 1))
plt.imshow(padded)
plt.title("Healpix PAD")
plt.colorbar()

# %%
