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
import math
from typing import Dict, Sequence

import einops
import netCDF4 as nc
import torch
from scipy import spatial

from earth2grid.spatial import ang2vec, haversine_distance


class Regridder(torch.nn.Module):
    """Regridder to n points, with p nonzero maps weights

    Forward:
        (*, m) -> (*,) + shape
    """

    def __init__(self, shape: Sequence[int], p: int):
        super().__init__()
        self.register_buffer("index", torch.empty(*shape, p, dtype=torch.long))
        self.register_buffer("weight", torch.ones(*shape, p))

    def forward(self, z):
        *shape, x = z.shape
        zrs = z.view(-1, x).T

        *output_shape, p = self.index.shape
        index = self.index.view(-1, p)
        weight = self.weight.view(-1, p)

        # using embedding bag is 2x faster on cpu and 4x on gpu.
        output = torch.nn.functional.embedding_bag(index, zrs, per_sample_weights=weight, mode='sum')
        output = output.T.view(*shape, -1)
        return output.reshape(list(shape) + output_shape)

    @staticmethod
    def from_state_dict(d: Dict[str, torch.Tensor]) -> "Regridder":
        n, p = d["index"].shape
        regridder = Regridder((n,), p)
        regridder.load_state_dict(d)
        return regridder


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


class BilinearInterpolator(torch.nn.Module):
    """Bilinear interpolation for a non-uniform grid"""

    def __init__(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        x_query: torch.Tensor,
        y_query: torch.Tensor,
        fill_value=math.nan,
    ) -> None:
        """

        Args:
            x_coords (Tensor): X-coordinates of the input grid, shape [W]. Must be in increasing sorted order.
            y_coords (Tensor): Y-coordinates of the input grid, shape [H]. Must be in increasing sorted order.
            x_query (Tensor): X-coordinates for query points, shape [N].
            y_query (Tensor): Y-coordinates for query points, shape [N].
        """
        super().__init__()
        self.fill_value = fill_value

        # Ensure input coordinates are float for interpolation
        x_coords, y_coords = x_coords.double(), y_coords.double()
        x_query = x_query.double()
        y_query = y_query.double()

        if torch.any(x_coords[1:] < x_coords[:-1]):
            raise ValueError("x_coords must be in non-decreasing order.")

        if torch.any(y_coords[1:] < y_coords[:-1]):
            raise ValueError("y_coords must be in non-decreasing order.")

        # Find indices for the closest lower and upper bounds in x and y directions
        x_l_idx = torch.searchsorted(x_coords, x_query, right=True) - 1
        x_u_idx = x_l_idx + 1
        y_l_idx = torch.searchsorted(y_coords, y_query, right=True) - 1
        y_u_idx = y_l_idx + 1

        # fill in nan outside mask
        def isin(x, a, b):
            return (x <= b) & (x >= a)

        mask = (
            isin(x_l_idx, 0, x_coords.size(0) - 2)
            & isin(x_u_idx, 1, x_coords.size(0) - 1)
            & isin(y_l_idx, 0, y_coords.size(0) - 2)
            & isin(y_u_idx, 1, y_coords.size(0) - 1)
        )
        x_u_idx = x_u_idx[mask]
        x_l_idx = x_l_idx[mask]
        y_u_idx = y_u_idx[mask]
        y_l_idx = y_l_idx[mask]
        x_query = x_query[mask]
        y_query = y_query[mask]

        # Compute weights
        x_l_weight = (x_coords[x_u_idx] - x_query) / (x_coords[x_u_idx] - x_coords[x_l_idx])
        x_u_weight = (x_query - x_coords[x_l_idx]) / (x_coords[x_u_idx] - x_coords[x_l_idx])
        y_l_weight = (y_coords[y_u_idx] - y_query) / (y_coords[y_u_idx] - y_coords[y_l_idx])
        y_u_weight = (y_query - y_coords[y_l_idx]) / (y_coords[y_u_idx] - y_coords[y_l_idx])
        weights = torch.stack(
            [x_l_weight * y_l_weight, x_u_weight * y_l_weight, x_l_weight * y_u_weight, x_u_weight * y_u_weight], dim=-1
        )

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
        self.register_buffer("weights", weights)
        self.register_buffer("mask", mask)
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
        output = torch.nn.functional.embedding_bag(self.index, zrs, per_sample_weights=self.weights, mode='sum')
        interpolated = torch.full(
            [self.mask.numel(), zrs.shape[1]], fill_value=self.fill_value, dtype=z.dtype, device=z.device
        )
        interpolated.masked_scatter_(self.mask.unsqueeze(-1), output)
        interpolated = interpolated.T.view(*shape, self.mask.numel())
        return interpolated


def KNNS2Interpolator(
    src_lon: torch.Tensor,
    src_lat: torch.Tensor,
    dest_lon: torch.Tensor,
    dest_lat: torch.Tensor,
    k: int = 1,
    eps=1e-7,
) -> Regridder:
    """K-nearest neighbor interpolator with inverse distance weighting

    Args:
        src_lon: (m,) source longitude in degrees E
        src_lat: (m,) source latitude in degrees N
        dest_lon: (n,) output longitude in degrees E
        dest_lat: (n,) output latitude in degrees N
        k: number of neighbors, default: 1
        eps: regularization factor for inverse distance weighting. Only used if
            k > 1.

    """
    if (src_lat.ndim != 1) or (src_lon.ndim != 1) or (dest_lat.ndim != 1) or (dest_lon.ndim != 1):
        raise ValueError("All input coordinates must be 1 dimensional.")

    src_lon = torch.deg2rad(src_lon.cpu())
    src_lat = torch.deg2rad(src_lat.cpu())

    dest_lon = torch.deg2rad(dest_lon.cpu())
    dest_lat = torch.deg2rad(dest_lat.cpu())

    vec = torch.stack(ang2vec(src_lon, src_lat), -1)

    # havesign distance and euclidean are monotone for points on S2 so can use 3d lookups.
    tree = spatial.KDTree(vec)
    vec = torch.stack(ang2vec(dest_lon.cpu(), dest_lat.cpu()), -1)
    _, neighbors = tree.query(vec, k=k)
    regridder = Regridder(dest_lon.shape, k)
    regridder.index.copy_(torch.as_tensor(neighbors).view(-1, k))
    if k > 1:
        d = haversine_distance(dest_lon[:, None], dest_lat[:, None], src_lon[neighbors], src_lat[neighbors])
        lam = 1 / (d + eps)
        lam = lam / lam.sum(-1, keepdim=True)
        regridder.weight.copy_(lam)

    return regridder


class Identity(torch.nn.Module):
    def forward(self, x):
        return x
