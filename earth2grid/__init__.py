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

from earth2grid import base, healpix, latlon
from earth2grid._regrid import (
    BilinearInterpolator,
    Identity,
    S2NearestNeighborInterpolator,
)

__all__ = [
    "base",
    "healpix",
    "latlon",
    "get_regridder",
    "BilinearInterpolator",
    "S2NearestNeighborInterpolator",
    "S2LinearBarycentricInterpolator",
]


def get_regridder(src: base.Grid, dest: base.Grid) -> torch.nn.Module:
    """Get a regridder from `src` to `dest`"""
    if src == dest:
        return Identity()
    elif isinstance(src, latlon.LatLonGrid) and isinstance(dest, latlon.LatLonGrid):
        return src.get_bilinear_regridder_to(dest.lat, dest.lon)
    elif isinstance(src, latlon.LatLonGrid) and isinstance(dest, healpix.Grid):
        return src.get_bilinear_regridder_to(dest.lat, dest.lon)
    elif isinstance(src, healpix.Grid):
        return src.get_bilinear_regridder_to(dest.lat, dest.lon)
    elif isinstance(dest, healpix.Grid):
        return src.get_healpix_regridder(dest)  # type: ignore

    raise ValueError(src, dest, "not supported.")
