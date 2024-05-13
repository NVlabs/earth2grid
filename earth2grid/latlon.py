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
import numpy as np

from earth2grid import base

try:
    import pyvista as pv
except ImportError:
    pv = None


class LatLonGrid(base.Grid):
    def __init__(self, lat: list[float], lon: list[float]):
        """
        Args:
            lat: center of lat cells
            lon: center of lon cells
        """
        self._lat = lat
        self._lon = lon

    @property
    def lat(self):
        return np.array(self._lat)

    @property
    def lon(self):
        return np.array(self._lon)

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))

    def _lonb(self):
        edges = (self.lon[1:] + self.lon[:-1]) / 2
        d_left = self.lon[1] - self.lon[0]
        d_right = self.lon[-1] - self.lon[-2]
        return np.concatenate([self.lon[0:1] - d_left / 2, edges, self.lon[-1:] + d_right / 2])

    def visualize(self, data):
        raise NotImplementedError()

    def to_pyvista(self):
        # TODO need to make lat the cell centers rather than boundaries

        if pv is None:
            raise ImportError("Need to install pyvista")

        print(self._lonb())

        lon, lat = np.meshgrid(self._lonb(), self.lat)
        y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
        x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
        z = np.sin(np.deg2rad(lat))
        grid = pv.StructuredGrid(x, y, z)
        return grid


def equiangular_lat_lon_grid(nlat: int, nlon: int, includes_south_pole: bool = True) -> LatLonGrid:
    """Return a regular lat-lon grid

    Lat is ordered from 90 to -90. Includes -90 and only if if includes_south_pole is True.
    Lon is ordered from 0 to 360. includes 0, but not 360.

    Args:
        nlat: number of latitude points
        nlon: number of longtidue points
        includes_south_pole: if true the final ``nlat`` includes the south pole

    """  # noqa
    lat = np.linspace(90, -90, nlat, endpoint=includes_south_pole)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return LatLonGrid(lat.tolist(), lon.tolist())
