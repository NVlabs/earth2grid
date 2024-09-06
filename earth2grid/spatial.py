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


def ang2vec(lon, lat):
    """convert lon,lat in radians to cartesian coordinates"""
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return (x, y, z)
