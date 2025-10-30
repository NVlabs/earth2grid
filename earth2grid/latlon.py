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
import torch

from earth2grid import base
from earth2grid._regrid import BilinearInterpolator

try:
    import pyvista as pv
except ImportError:
    pv = None


class LatLonGrid(base.Grid):
    def __init__(self, lat: list[float], lon: list[float], cylinder: bool = True):
        """
        Args:
            lat: center of lat cells
            lon: center of lon cells
            cylinder: if true, then lon is considered a periodic coordinate
                on cylinder so that interpolation wraps around the edge.
                Otherwise, it is assumed to be a finite plane.
        """
        self._lat = lat
        self._lon = lon
        self.cylinder = cylinder

    @property
    def lat(self):
        return np.array(self._lat)[:, None]

    @property
    def lon(self):
        return np.array(self._lon)

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Get regridder to the specified lat and lon points"""
        return _RegridFromLatLon(self, lat, lon, cylinder=self.cylinder)

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


class _RegridFromLatLon(torch.nn.Module):
    """Regrid from lat-lon to unstructured grid with bilinear interpolation"""

    def __init__(self, src: LatLonGrid, lat: np.ndarray, lon: np.ndarray, cylinder: bool = True):
        """
        Args:
            cylinder: if True than lon is assumed to be periodic
        """
        super().__init__()
        self.cylinder = cylinder

        lat, lon = np.broadcast_arrays(lat, lon)
        self.shape = lat.shape

        # TODO add device switching logic (maybe use torch registers for this
        # info)
        long = src.lon.ravel()
        long_min = long[0]  # lon should be non-descending, so first is min

        if self.cylinder:
            long = np.concatenate([long, [long_min + 360]], axis=-1)
        long_t = torch.from_numpy(long)

        # make sure lon_query is in the same range as long
        lon_query = torch.from_numpy(lon.ravel())
        lon_query = (lon_query - long_min) % 360 + long_min

        # flip the order latg since bilinear only works with increasing coordinate values
        lat_increasing = src.lat[1] > src.lat[0]
        latg_t = torch.from_numpy(src.lat.ravel())
        lat_query = torch.from_numpy(lat.ravel())

        if not lat_increasing:
            lat_query = -lat_query
            latg_t = -latg_t

        self._bilinear = BilinearInterpolator(long_t, latg_t, y_query=lat_query, x_query=lon_query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad z in lon direction
        # only works for a global grid
        # TODO generalize this to local grids and add options for padding
        if self.cylinder:
            x = torch.cat([x, x[..., 0:1]], axis=-1)
        out = self._bilinear(x)
        return out.view(out.shape[:-1] + self.shape)


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
