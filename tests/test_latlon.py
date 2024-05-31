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

from earth2grid.latlon import equiangular_lat_lon_grid


def test_lat_lon_bilinear_regrid_to():
    src = equiangular_lat_lon_grid(15, 30)
    dest = equiangular_lat_lon_grid(30, 60)
    regrid = src.get_bilinear_regridder_to(dest.lat, dest.lon)

    regrid.float()
    lat = torch.broadcast_to(torch.tensor(src.lat), src.shape)
    z = torch.tensor(lat).float()

    out = regrid(z)
    assert out.shape == dest.shape
