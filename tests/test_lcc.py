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

#%%
from earth2grid.lcc import HRRR_CONUS_GRID
import numpy as np
import torch
from pytest import approx

def test_grid_shape():
    assert HRRR_CONUS_GRID.lat.shape == HRRR_CONUS_GRID.shape 
    assert HRRR_CONUS_GRID.lon.shape == HRRR_CONUS_GRID.shape

lats = np.array([
        [21.138123, 21.801926, 22.393631, 22.911015],
        [23.636763, 24.328228, 24.944668, 25.48374 ],
        [26.155672, 26.875362, 27.517046, 28.078257],
        [28.69017 , 29.438608, 30.106009, 30.68978 ]])

lons = np.array([
        [-122.71953 , -120.03195 , -117.304596, -114.54146 ],
        [-123.491356, -120.72898 , -117.92319 , -115.07828 ],
        [-124.310524, -121.469505, -118.58098 , -115.649574],
        [-125.181404, -122.25762 , -119.28173 , -116.25871 ]])


def test_grid_vals():
    assert HRRR_CONUS_GRID.lat[0:400:100,0:400:100] == approx(lats)
    assert HRRR_CONUS_GRID.lon[0:400:100,0:400:100] == approx(lons)

def test_grid_slice():
    slice_grid = HRRR_CONUS_GRID[0:400:100,0:400:100]
    assert slice_grid.lat == approx(lats)
    assert slice_grid.lon == approx(lons)

def test_grid_slice():

    src = HRRR_CONUS_GRID
    dest = HRRR_CONUS_GRID[1:-1, 1:-1]
    regrid = src.get_bilinear_regridder_to(dest.lat, dest.lon)
    lat = torch.broadcast_to(torch.tensor(src.lat), src.shape)
    out = regrid(lat)

    assert torch.allclose(out, lat[1:-1, 1:-1])
