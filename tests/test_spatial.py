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

import pytest
import torch

from earth2grid import spatial


def test_vec2ang2vec():
    vec = torch.randn(3)
    vec /= torch.norm(vec)
    x, y, z = vec

    lon, lat = spatial.vec2ang(x, y, z)
    x1, y1, z1 = spatial.ang2vec(lon, lat)
    assert torch.allclose(torch.stack([x1, y1, z1]), torch.stack([x, y, z]))


def test_vec2ang():
    lon, lat = spatial.vec2ang(torch.tensor(0), torch.tensor(0), torch.tensor(1))
    assert lat == pytest.approx(math.pi / 2)

    lon, _ = spatial.vec2ang(torch.tensor(1), torch.tensor(0), torch.tensor(0))
    assert lon == pytest.approx(0)

    lon, _ = spatial.vec2ang(torch.tensor(0), torch.tensor(1), torch.tensor(0))
    assert lon == pytest.approx(math.pi / 2)
