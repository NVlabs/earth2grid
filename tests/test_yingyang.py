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
# limitations under the License.from earth2grid.yinyang import Ying, Yang, YangProjection
import numpy as np
import pytest
import torch

from earth2grid.yinyang import Yang, Ying


def test_yingyang():
    nlat = 721
    nlon = 1440
    delta = 64

    nlat = 37
    nlon = 72
    delta = 0

    ying = Ying(nlat, nlon, delta)
    yang = Yang(nlat, nlon, delta)

    assert ying.lat.min() == pytest.approx(-45)
    assert ying.lat.max() == pytest.approx(45)
    assert ying.lat.min() == -ying.lat.max()
    assert ying.lon.min() == -ying.lon.max()
    y2y = ying.get_bilinear_regridder_to(yang.lat, yang.lon)
    y2y.float()

    x = torch.ones(ying.shape)
    y = y2y(x)
    mask = ~torch.isnan(y)
    # this is a regression check. will need to verify and change for different res
    fraction_missing = 1 - mask.sum().item() / mask.numel()
    assert fraction_missing == pytest.approx(0.8038, abs=0.01)
    assert torch.allclose(y[mask], torch.tensor(1).float())

    # more complex check
    lat, lon = np.meshgrid(ying.lat, ying.lon, indexing='ij')
    x = torch.as_tensor(lat, dtype=torch.float).deg2rad().cos()
    y = y2y(x)
    expected = torch.as_tensor(yang.lat).deg2rad().cos().float()
    assert torch.allclose(y[mask], expected[mask], atol=0.01)
