# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

import einops
import netCDF4 as nc
import numpy as np
import pandas
import torch

from earth2grid import base, healpix
from earth2grid.latlon import LatLonGrid


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

        self.x_l_idx = x_l_idx
        self.x_u_idx = x_u_idx
        self.y_l_idx = y_l_idx
        self.y_u_idx = y_u_idx

        self.register_buffer("x_l_weight", x_l_weight)
        self.register_buffer("x_u_weight", x_u_weight)
        self.register_buffer("y_l_weight", y_l_weight)
        self.register_buffer("y_u_weight", y_u_weight)

    def forward(self, z: torch.Tensor):
        """
        Interpolate the field

        Args:
            z: shape [Y, X]
        """

        # Perform bilinear interpolation
        interpolated_values = (
            z[..., self.y_l_idx, self.x_l_idx] * self.x_l_weight * self.y_l_weight
            + z[..., self.y_l_idx, self.x_u_idx] * self.x_u_weight * self.y_l_weight
            + z[..., self.y_u_idx, self.x_l_idx] * self.x_l_weight * self.y_u_weight
            + z[..., self.y_u_idx, self.x_u_idx] * self.x_u_weight * self.y_u_weight
        )
        return interpolated_values


class TempestRegridder(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        dataset = nc.Dataset(file_path)
        self.lat = dataset["latc_b"][:]
        self.lon = dataset["lonc_b"][:]

        i = dataset["row"][:] - 1
        j = dataset["col"][:] - 1
        M = dataset["S"][:]

        i = i.data
        j = j.data
        M = M.data

        self.M = torch.sparse_coo_tensor((i, j), M, [max(i) + 1, max(j) + 1]).float()

    def to(self, device):
        self.M = self.M.to(device)
        return self

    def forward(self, x):
        xr = einops.rearrange(x, "b c x y -> b c (x y)")
        yr = xr @ self.M.T
        y = einops.rearrange(yr, "b c (x y) -> b c x y", x=self.lat.size, y=self.lon.size)
        return y


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class RegridLatLon(torch.nn.Module):
    def __init__(self, src_grid: LatLonGrid, dest_grid: LatLonGrid):
        super().__init__()
        self._src_grid = src_grid
        self._dest_grid = dest_grid
        self._lat_index = pandas.Index(src_grid.lat).get_indexer(dest_grid.lat)
        assert not np.any(self._lat_index == -1)  # noqa

        self._lon_index = pandas.Index(src_grid.lon).get_indexer(dest_grid.lon)
        assert not np.any(self._lon_index == -1)  # noqa

    def forward(self, x):
        if x.shape[-2:] != self._src_grid.shape:
            raise ValueError(f"Input shape {x.shape} does not match grid shape" f"{self._src_grid.shape}")

        return x[..., self._lat_index, :][..., self._lon_index]


class _RegridFromLatLon(torch.nn.Module):
    """Regrid from lat-lon to unstructured grid with bilinear interpolation"""

    def __init__(self, src: LatLonGrid, dest: base.Grid):
        super().__init__()

        # potentially relax this
        assert len(dest.shape) == 1

        # TODO add device switching logic (maybe use torch registers for this
        # info)
        long = np.concatenate([src.lon.ravel(), [360]], axis=-1)
        long_t = torch.from_numpy(long)

        # flip the order latg since bilinear only works with increasing coordinate values
        lat_increasing = src.lat[1] > src.lat[0]
        latg_t = torch.from_numpy(src.lat.ravel())
        lat_query = torch.from_numpy(dest.lat.ravel())

        if not lat_increasing:
            lat_query = -lat_query
            latg_t = -latg_t

        self._bilinear = BilinearInterpolator(
            long_t, latg_t, y_query=lat_query, x_query=torch.from_numpy(dest.lon.ravel())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad z in lon direction
        # only works for a global grid
        # TODO generalize this to local grids and add options for padding
        x = torch.cat([x, x[..., 0:1]], axis=-1)
        return self._bilinear(x)


def get_regridder(src: base.Grid, dest: base.Grid) -> torch.nn.Module:
    """Get a regridder from `src` to `dest`"""
    if src == dest:
        return Identity()
    elif isinstance(src, LatLonGrid) and isinstance(dest, LatLonGrid):
        return RegridLatLon(src, dest)
    elif isinstance(src, LatLonGrid) and isinstance(dest, healpix.Grid):
        return _RegridFromLatLon(src, dest)
    elif isinstance(src, healpix.Grid) and isinstance(dest, LatLonGrid):
        return src.get_latlon_regridder(dest.lat, dest.lon)
    raise ValueError(src, dest, "not supported.")
