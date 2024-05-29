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

try:
    import pyvista as pv
except ImportError:
    pv = None


class BilinearInterpolator(torch.nn.Module):
    """Bilinear interpolation for a non-uniform grid"""

    def __init__(
        self, x_coords: torch.Tensor, y_coords: torch.Tensor, x_query: torch.Tensor, y_query: torch.Tensor
    ) -> None:
        """

        Args:
            x_coords (Tensor): X-coordinates of the input grid, shape [W]. Must be in increasing sorted order.
            y_coords (Tensor): Y-coordinates of the input grid, shape [H]. Must be in increasing sorted order.
            x_query (Tensor): X-coordinates for query points, shape [N].
            y_query (Tensor): Y-coordinates for query points, shape [N].
        """
        super().__init__()

        # Ensure input coordinates are float for interpolation
        x_coords, y_coords = x_coords.float(), y_coords.float()

        if torch.any(x_coords[1:] <= x_coords[:-1]):
            raise ValueError("x_coords must be in increasing order.")

        if torch.any(y_coords[1:] <= y_coords[:-1]):
            raise ValueError("y_coords must be in increasing order.")

        # Find indices for the closest lower and upper bounds in x and y directions
        x_l_idx = torch.searchsorted(x_coords, x_query, right=True) - 1
        x_u_idx = x_l_idx + 1
        y_l_idx = torch.searchsorted(y_coords, y_query, right=True) - 1
        y_u_idx = y_l_idx + 1

        # Clip indices to ensure they are within the bounds of the input grid
        x_l_idx = x_l_idx.clamp(0, x_coords.size(0) - 2)
        x_u_idx = x_u_idx.clamp(1, x_coords.size(0) - 1)
        y_l_idx = y_l_idx.clamp(0, y_coords.size(0) - 2)
        y_u_idx = y_u_idx.clamp(1, y_coords.size(0) - 1)

        # Compute weights
        x_l_weight = (x_coords[x_u_idx] - x_query) / (x_coords[x_u_idx] - x_coords[x_l_idx])
        x_u_weight = (x_query - x_coords[x_l_idx]) / (x_coords[x_u_idx] - x_coords[x_l_idx])
        y_l_weight = (y_coords[y_u_idx] - y_query) / (y_coords[y_u_idx] - y_coords[y_l_idx])
        y_u_weight = (y_query - y_coords[y_l_idx]) / (y_coords[y_u_idx] - y_coords[y_l_idx])
        weights = torch.stack(
            [x_l_weight * y_l_weight, x_u_weight * y_l_weight, x_l_weight * y_u_weight, x_u_weight * y_u_weight], dim=-1
        )

        self.register_buffer("weights", weights)

        stride = x_coords.size(-1)
        index = torch.stack(
            [
                x_l_idx + stride * y_l_idx,
                x_u_idx + stride * y_l_idx,
                x_l_idx + stride * y_u_idx,
                x_u_idx + stride * y_u_idx,
            ],
            dim=-1,
        )
        self.register_buffer("index", index)

    def forward(self, z: torch.Tensor):
        """
        Interpolate the field

        Args:
            z: shape [Y, X]
        """
        *shape, y, x = z.shape
        zrs = z.view(-1, y * x).T
        # using embedding bag is 2x faster on cpu and 4x on gpu.
        interpolated = torch.nn.functional.embedding_bag(self.index, zrs, per_sample_weights=self.weights, mode='sum')
        interpolated = interpolated.T.view(*shape, self.weights.size(0))
        return interpolated


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
        return np.array(self._lat)[:, None]

    @property
    def lon(self):
        return np.array(self._lon)

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Get regridder to the specified lat and lon points"""
        return _RegridFromLatLon(self, lat, lon)

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

    def __init__(self, src: LatLonGrid, lat: np.ndarray, lon: np.ndarray):
        super().__init__()

        lat, lon = np.broadcast_arrays(lat, lon)
        self.shape = lat.shape

        # TODO add device switching logic (maybe use torch registers for this
        # info)
        long = np.concatenate([src.lon.ravel(), [360]], axis=-1)
        long_t = torch.from_numpy(long)

        # flip the order latg since bilinear only works with increasing coordinate values
        lat_increasing = src.lat[1] > src.lat[0]
        latg_t = torch.from_numpy(src.lat.ravel())
        lat_query = torch.from_numpy(lat.ravel())

        if not lat_increasing:
            lat_query = -lat_query
            latg_t = -latg_t

        self._bilinear = BilinearInterpolator(long_t, latg_t, y_query=lat_query, x_query=torch.from_numpy(lon.ravel()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad z in lon direction
        # only works for a global grid
        # TODO generalize this to local grids and add options for padding
        x = torch.cat([x, x[..., 0:1]], axis=-1)
        return self._bilinear(x)


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
