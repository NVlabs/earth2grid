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
from earth2grid.healpix.padding import local2xy

order = 4
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

xy_loc = local2xy(nside, x, y, f)
padded0 = lat[xy_loc]

xy_loc = local2xy(nside, x, y, f, right_first=False)
padded1 = lat[xy_loc]

dist_x = torch.where(x < 0, -x, torch.where(x < nside, 0, x - nside))
dist_y = torch.where(y < 0, -y, torch.where(y < nside, 0, y - nside))
padded = torch.where(dist_x > dist_y, padded0, torch.where(dist_x == dist_y, (padded0 + padded1) / 2, padded1))
# padded = (padded0 + padded1) / 2


plt.imshow(padded0[0])
plt.colorbar()

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
