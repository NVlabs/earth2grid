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
from typing import Protocol

import numpy as np


class Grid(Protocol):
    """lat and lon should be broadcastable arrays"""

    @property
    def lat(self):
        pass

    @property
    def lon(self):
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        pass

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Return a regridder from `self` to lat/lon.

        Args:
            lat, lon: broadcastable arrays for the lat/lon
        """
        raise NotImplementedError()

    def visualize(self, data):
        pass

    def to_pyvista(self):
        pass
