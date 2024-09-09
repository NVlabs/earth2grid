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
import einops
import torch


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Haversine distance between two points on unit sphere

    Args:
        lon1 (float): Longitude of the first point in radians.
        lat1 (float): Latitude of the first point in radians.
        lon2 (float): Longitude of the second point in radians.
        lat2 (float): Latitude of the second point in radians.

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return c


def barycentric_coords_with_origin(points, simplices):
    """

        P = a A + b B + c C + d O
          = a A + b B + c C + (1 - a -b - c) 0

          a + b + c + d = 0


    points: (n, d)
    simplices: (m, 3, 3)

    Returns
        (m n v) shaped boolean
    """
    m, v, d = simplices.shape
    assert d == v == 3
    L = simplices
    # L.shape == (m d v), v is vertex (A,B)
    inv = torch.linalg.inv(L)
    b = points
    coords = einops.einsum(inv, b, "m v d, n d -> n m v")
    return coords


def select_simplex(bary_coords, tol=1e-5):
    """Select the simplex given the barycentric coordinates

    A point (aA + bB + cC) is in the spherical triangle defined by (A,B,C) when a,b,c>=0

    Args:

        bary_coords: (m, n, 3) m is the number of simplices, n the number of points
        tol: the tolerance to use for membership tests. Allows for numerical errors
            in the computation of barycentric coordinates.
    Returns:
        index between 0 and m - 1 (inclusive). shaped (n,). If no containing
        simplex is found, then contains -1. In this case you may want to change
        tolerance or ensure that your simplices cover all of S2.
    """
    in_simplex = torch.all(bary_coords > -tol, dim=-1)  # (m, n)

    # handle case when point in multiple simplices
    num_simplices = in_simplex.sum(dim=0)  # (n,)

    # For points in multiple simplices, select the one with the highest minimum coordinate
    min_coords = bary_coords.min(dim=-1).values  # (m, n)
    best_simplex = min_coords.argmax(dim=0)  # (n,)

    # Create a mask for points that are in at least one simplex
    valid_points = num_simplices == 1  # (n,)

    # Initialize the result tensor with -1 (invalid index)
    asdf
    return torch.where(valid_points, best_simplex, -1)


def ang2vec(lon, lat):
    """convert lon,lat in radians to cartesian coordinates"""
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return (x, y, z)
