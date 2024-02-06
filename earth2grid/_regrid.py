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
from scipy.interpolate import interpn

from earth2grid import base, healpix
from earth2grid.latlon import LatLonGrid


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
        xi = np.stack([dest.lon, dest.lat], axis=1)

        # TODO add device switching logic (maybe use torch registers for this
        # info)
        self.long = src.lon.ravel()
        self.latg = src.lat.ravel()
        self.xi = xi
        self.src = src
        self.dest = dest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.numpy()
        # implement this in pytorch for differentiability
        # pad z in lon direction
        # only works for a global grid
        # TODO generalize this to local grids and add options for padding
        z = np.concatenate([z, z[..., 0:1]], axis=-1)
        long = np.concatenate([self.long, [360]], axis=-1)

        out = interpn([long, self.latg], z.T, self.xi, method="linear")
        return torch.as_tensor(out)


def get_regridder(src: base.Grid, dest: base.Grid) -> torch.nn.Module:
    if src == dest:
        return Identity()
    elif isinstance(src, LatLonGrid) and isinstance(dest, LatLonGrid):
        return RegridLatLon(src, dest)
    elif isinstance(src, LatLonGrid) and isinstance(dest, healpix.Grid):
        return _RegridFromLatLon(src, dest)
    raise ValueError(src, dest, "not supported.")
