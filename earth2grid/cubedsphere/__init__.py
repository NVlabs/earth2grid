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
Cubed-sphere Grid
=================

The cubed-sphere grid divides the sphere into 6 faces, each a square grid.
This module provides entry points for cubed-sphere variants.

The face layout is:

    | 5 |           (North cap)
    | 0 | 1 | 2 | 3 |  (Equatorial band)
    | 4 |           (South cap)

Example:
    Convert E3SM data to XY format for padding:

    >>> from earth2grid.cubedsphere import e3sm
    >>> grid = e3sm.E3SMpgOrder(num_elements=1024, num_pg_cells=2)
    >>> xy = e3sm.XY(face_size=2048)
    >>> data = torch.randn(3, 6 * 1024 * 1024 * 2 * 2)
    >>> faces = e3sm.reorder(data, src=grid, dest=xy)
    >>> padded = e3sm.pad(faces, padding=64)
"""

from earth2grid.cubedsphere import e3sm

__all__ = [
    # Variants
    "e3sm",
]
