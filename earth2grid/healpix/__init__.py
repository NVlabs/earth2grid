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

HEALPix
=======

From this notebook: https://colab.research.google.com/drive/1MzTyeNFiy-7RNY6UtGKsmDavX5dk6epU



"""


from earth2grid.healpix._padding import PaddingBackends, pad, pad_backend, pad_with_dim
from earth2grid.healpix.coordinates import (
    angular_to_global,
    face_to_global,
    global_to_angular,
    global_to_face,
)
from earth2grid.healpix.core import (
    HEALPIX_PAD_XY,
    NEST,
    RING,
    XY,
    Compass,
    Grid,
    PixelOrder,
    local2xy,
    nest2xy,
    npix2level,
    npix2nside,
    nside2level,
    reorder,
    ring2double,
    ring2xy,
    to_double_pixelization,
    to_rotated_pixelization,
    xy2local,
    xy2nest,
    xy2xy,
    zonal_average,
)
from earth2grid.healpix.nn import conv2d
from earth2grid.healpix.visualization import pcolormesh

__all__ = [
    "angular_to_global",
    "Compass",
    "conv2d",
    "face_to_global",
    "global_to_angular",
    "global_to_face",
    "Grid",
    "HEALPIX_PAD_XY",
    "NEST",
    "RING",
    "local2xy",
    "nest2xy",
    "npix2level",
    "npix2nside",
    "nside2level",
    "pad_backend",
    "pad_with_dim",
    "pad",
    "PaddingBackends",
    "pcolormesh",
    "PixelOrder",
    "reorder",
    "ring2double",
    "ring2xy",
    "to_double_pixelization",
    "to_rotated_pixelization",
    "XY",
    "xy2xy",
    "xy2local",
    "xy2nest",
    "zonal_average",
]
