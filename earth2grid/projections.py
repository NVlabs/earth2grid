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
import abc

import numpy as np
import torch

from earth2grid import base
from earth2grid._regrid import BilinearInterpolator

try:
    import pyvista as pv
except ImportError:
    pv = None


class Projection(abc.ABC):
    @abc.abstractmethod
    def project(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the projected x,y from lat,lon.
        """
        pass

    @abc.abstractmethod
    def inverse_project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the lat,lon from the projected x,y.
        """
        pass


class Grid(base.Grid):
    # nothing here is specific to the projection, so could be shared by any projected rectilinear grid
    def __init__(self, projection: Projection, x, y):
        """
        Args:
            x: range of x values
            y: range of y values

        """
        self.projection = projection

        self.x = np.array(x)
        self.y = np.array(y)

    @property
    def lat_lon(self):
        mesh_x, mesh_y = np.meshgrid(self.x, self.y, indexing='ij')
        return self.projection.inverse_project(mesh_x, mesh_y)

    @property
    def lat(self):
        return self.lat_lon[0]

    @property
    def lon(self):
        return self.lat_lon[1]

    @property
    def shape(self):
        return (len(self.x), len(self.y))

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Get regridder to the specified lat and lon points"""

        x, y = self.projection.project(lat, lon)

        return BilinearInterpolator(
            x_coords=torch.from_numpy(self.x),
            y_coords=torch.from_numpy(self.y),
            x_query=torch.from_numpy(x),
            y_query=torch.from_numpy(y),
        )

    def visualize(self, data):
        raise NotImplementedError()

    def to_pyvista(self):
        if pv is None:
            raise ImportError("Need to install pyvista")

        lat, lon = self.lat_lon
        y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
        x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
        z = np.sin(np.deg2rad(lat))
        grid = pv.StructuredGrid(x, y, z)
        return grid
