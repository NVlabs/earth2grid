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
"""
Cubesphere Grid
===============

The cubesphere grid divides the sphere into 6 faces, each a square grid.
This module provides utilities for working with cubesphere data, including:

- Padding faces with data from neighboring faces
- Converting between different pixel orderings (E3SM element order, 2D meshgrid)

The face layout is:

    | 5 |           (North cap)
    | 0 | 1 | 2 | 3 |  (Equatorial band)
    | 4 |           (South cap)

Pixel Orderings
---------------

**E3SMpgOrder**: Element-major ordering used by E3SM's pg2 physgrid.
    Each face has ne×ne spectral elements, each with npg×npg physics cells.

**XY**: Standard 2D meshgrid format for image-like operations.
    Each face is a 2D array of shape (face_size, face_size).

Example:
    Convert E3SM data to XY format for padding:

    >>> from earth2grid import cubesphere
    >>> e3sm = cubesphere.E3SMpgOrder(ne=1024, npg=2)
    >>> xy = cubesphere.XY(face_size=2048)
    >>> data = torch.randn(3, 6 * 1024 * 1024 * 2 * 2)
    >>> faces = cubesphere.reorder(data, src=e3sm, dest=xy)
    >>> padded = cubesphere.pad(faces, padding=64)
"""

from earth2grid.cubesphere._padding import get_cubesphere_neighbors, pad
from earth2grid.cubesphere.core import (
    XY,
    E3SMpgOrder,
    reorder,
)

__all__ = [
    # Padding
    "pad",
    "get_cubesphere_neighbors",
    # Pixel orderings
    "E3SMpgOrder",
    "XY",
    "reorder",
]
