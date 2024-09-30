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

__all__ = [
    "LambertConformalConicProjection",
    "LambertConformalConicGrid",
    "HRRR_CONUS_PROJECTION",
    "HRRR_CONUS_GRID",
]


class LambertConformalConicProjection:
    def __init__(self, lat0: float, lon0: float, lat1: float, lat2: float, radius: float):
        """

        Args:
            lat0: latitude of origin (degrees)
            lon0: longitude of origin (degrees)
            lat1: first standard parallel (degrees)
            lat2: second standard parallel (degrees)
            radius: radius of sphere (m)

        """

        self.lon0 = lon0
        self.lat0 = lat0
        self.lat1 = lat1
        self.lat2 = lat2
        self.radius = radius

        c1 = np.cos(np.deg2rad(lat1))
        c2 = np.cos(np.deg2rad(lat2))
        t1 = np.tan(np.pi / 4 + np.deg2rad(lat1) / 2)
        t2 = np.tan(np.pi / 4 + np.deg2rad(lat2) / 2)

        if np.abs(lat1 - lat2) < 1e-8:
            self.n = np.sin(np.deg2rad(lat1))
        else:
            self.n = np.log(c1 / c2) / np.log(t2 / t1)

        self.RF = radius * c1 * np.power(t1, self.n) / self.n
        self.rho0 = self._rho(lat0)

    def _rho(self, lat):
        return self.RF / np.power(np.tan(np.pi / 4 + np.deg2rad(lat) / 2), self.n)

    def _theta(self, lon):
        """
        Angle of deviation (in radians) of the projected grid from the regular grid,
        for a given longitude (in degrees).

        To convert to U and V on the projected grid to easterly / northerly components:
            UN =   cos(theta) * U + sin(theta) * V
            VN = - sin(theta) * U + cos(theta) * V
        """
        # center about reference longitude
        delta_lon = lon - self.lon0
        delta_lon = delta_lon - np.round(delta_lon / 360) * 360  # convert to [-180, 180]
        return self.n * np.deg2rad(delta_lon)

    def project(self, lat, lon):
        """
        Compute the projected x,y from lat,lon.
        """
        rho = self._rho(lat)
        theta = self._theta(lon)

        x = rho * np.sin(theta)
        y = self.rho0 - rho * np.cos(theta)
        return x, y

    def inverse_project(self, x, y):
        """
        Compute the lat,lon from the projected x,y.
        """
        rho = np.hypot(x, self.rho0 - y)
        theta = np.arctan2(x, self.rho0 - y)

        lat = np.rad2deg(2 * np.arctan(np.power(self.RF / rho, 1 / self.n))) - 90
        lon = self.lon0 + np.rad2deg(theta / self.n)
        return lat, lon


# Projection used by HRRR CONUS (Continental US) data
# https://rapidrefresh.noaa.gov/hrrr/HRRR_conus.domain.txt
HRRR_CONUS_PROJECTION = LambertConformalConicProjection(lon0=-97.5, lat0=38.5, lat1=38.5, lat2=38.5, radius=6371229.0)


class LambertConformalConicGrid(base.Grid):
    # nothing here is specific to the projection, so could be shared by any projected rectilinear grid
    def __init__(self, projection: LambertConformalConicProjection, x, y):
        """
        Args:
            projection: LambertConformalConicProjection object
            x: range of x values
            y: range of y values

        """
        self.projection = projection

        self.x = np.array(x)
        self.y = np.array(y)

    @property
    def lat_lon(self):
        mesh_x, mesh_y = np.meshgrid(self.x, self.y)
        return self.projection.inverse_project(mesh_x, mesh_y)

    @property
    def lat(self):
        return self.lat_lon[0]

    @property
    def lon(self):
        return self.lat_lon[1]

    @property
    def shape(self):
        return (len(self.y), len(self.x))

    def __getitem__(self, idxs):
        yidxs, xidxs = idxs
        return LambertConformalConicGrid(self.projection, x=self.x[xidxs], y=self.y[yidxs])

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Get regridder to the specified lat and lon points"""

        x, y = self.projection.project(lat, lon)

        return BilinearInterpolator(
            x_coords=torch.from_numpy(self.x),
            y_coords=torch.from_numpy(self.y),
            x_query=torch.from_numpy(x),
            y_query=torch.from_numpy(y),
        )

    def visualize(self, data):
        raise NotImplementedError()

    def to_pyvista(self):
        if pv is None:
            raise ImportError("Need to install pyvista")

        lat, lon = self.lat_lon
        y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
        x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
        z = np.sin(np.deg2rad(lat))
        grid = pv.StructuredGrid(x, y, z)
        return grid


def hrrr_conus_grid(ix0=0, iy0=0, nx=1799, ny=1059):
    # coordinates of point in top-left corner
    lat0 = 21.138123
    lon0 = 237.280472
    # grid length (m)
    scale = 3000.0
    # coordinates on projected space
    x0, y0 = HRRR_CONUS_PROJECTION.project(lat0, lon0)

    x = [x0 + i * scale for i in range(ix0, ix0 + nx)]
    y = [y0 + i * scale for i in range(iy0, iy0 + ny)]

    return LambertConformalConicGrid(HRRR_CONUS_PROJECTION, x, y)


# Grid used by HRRR CONUS (Continental US) data
HRRR_CONUS_GRID = hrrr_conus_grid()
