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
"""Yin Yang

the ying yang grid is an overset grid for the sphere containing two faces
- Yin: a normal lat lon grid for 2/3 of lon, and 2/3 of lat
- Yang: Yin but with pole along x


Key facts

ying
lon: [-3 pi /4  - delta, 3 pi / 4 + delta ]
lat: [-pi / 4 - delta, pi / 4 + delta]

ying to yang transformation: alpha = 0, beta = 90, gamma = 180

(x, y, z) - > (-x, z, y)

"""
import math

import numpy as np
import torch

from earth2grid import latlon, projections, spatial


def Ying(nlat: int, nlon: int, delta: int):
    """The ying grid

    nlat, and nlon are as in the latlon.equiangular_latlon_grid and
    refer to full sphere.

    ``nlat`` includes the poles [90, -90], and ``nlon`` is [0, 2 pi).

    ``delta`` is the amount of overlap in terms of number of grid points.

    """
    # TODO test that min(lat) = -max(lat), and for lon too

    dlat = 180 / (nlat - 1)
    dlon = 360 / nlon

    n = math.ceil(3 * nlon / 8)
    lon = np.arange(-n - delta, n + delta + 1) * dlon
    lat = np.arange(-(nlat - 1) // 4 - delta, (nlat + 1) // 4 + delta + 1) * dlat

    return latlon.LatLonGrid(lat.tolist(), lon.tolist(), cylinder=False)


class YangProjection(projections.Projection):
    def project(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the projected x,y from lat,lon.
        """
        lat = torch.from_numpy(lat)
        lon = torch.from_numpy(lon)

        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)

        x, y, z = spatial.ang2vec(lat=lat, lon=lon)
        x, y, z = -x, z, y
        lon, lat = spatial.vec2ang(x, y, z)

        lat = torch.rad2deg(lat)
        lon = torch.rad2deg(lon)

        return lat.numpy(), lon.numpy()

    def inverse_project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the lat,lon from the projected x,y.
        """
        # ying-yang is its own inverse
        return self.project(x, y)


def Yang(nlat, nlon, delta):
    ying = Ying(nlat, nlon, delta)
    return projections.Grid(YangProjection(), ying.lat, ying.lon)
