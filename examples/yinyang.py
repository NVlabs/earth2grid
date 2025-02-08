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
# limitations under the License.from earth2grid.yinyang import Ying, Yang, YangProjection
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from earth2grid.yinyang import Yang, Ying

nlat = 721
nlon = 1440
delta = 64

nlat = 37
nlon = 72
delta = 0

ying = Ying(nlat, nlon, delta)
yang = Yang(nlat, nlon, delta)


def structured_grid(lon, lat):
    y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    z = np.sin(np.deg2rad(lat))
    grid = pv.StructuredGrid(x, y, z)
    return grid


lon, lat = np.meshgrid(ying.lon, ying.lat)
ying_g = structured_grid(lon, lat)
yang_g = structured_grid(yang.lon, yang.lat)

pl = pv.Plotter()
pl.add_mesh(ying_g, show_edges=True)
# scale slightly so yang is on top
pl.add_mesh(yang_g.scale(1.002), show_edges=True, color="red", opacity=0.5)
pl.show()


y2y = ying.get_bilinear_regridder_to(yang.lat, yang.lon)
y2y.float()

x = torch.ones(ying.shape)
y = y2y(x)
y = y.reshape(yang.shape)
print("mask", torch.isnan(y).sum() / y.numel())

plt.figure()
# TODO fix yang.shape, it is the opposite it should be
plt.imshow(y.reshape(*ying.shape))
plt.colorbar()
plt.show()
