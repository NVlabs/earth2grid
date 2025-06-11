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
# limitations under the License.""
"""
Coordinate transformations for HEALPix

This module contains transformations between these coordinate systems:
- angular coordinates (lon, lat)
- face coordinates (x, y, f)
- global HEALPix projection (xₛ, yₛ)

Implementation notes
--------------------

From Gorski (2005), the forward projection is given by::

    xs = φ                                             (26)
    ys = (3π / 8) * z                                  (27)

    and in the HEALPix polar caps (|z| > 2/3):

    xs = φ - (|σ(z)| - 1) * (φ_t - π/4)                 (28)
    ys = (π / 4) * σ(z)                                (29)

    where:
    z = cos(θ)
    φ_t = φ mod (π/2)
    σ(z) = 2 - sqrt(3 * (1 - z))   for z > 0
    σ(-z) = -σ(z)

And, the inverse projection in the polar cap is given by::

    Inverse mapping from the (xs, ys) plane to the sphere (θ, φ).

    In the HEALPix Equatorial zone (|ys| < π/4):
        φ = xs                                      (31)
        cos(θ) = (8 / (3 * π)) * ys                 (32)

    In the HEALPix polar caps (|ys| > π/4):
        φ = xs - (|ys| - π/4) / (|ys| - π/2) * (xt - π/4)    (33)

        cos(θ) = [1 - (1/3) * (2 - (4 * |ys|) / π)^2] * (ys / |ys|)    (34)

Note:
- xt = xs mod (π/2)
- |ys| denotes the absolute value of ys
- The term (ys / |ys|) ensures the correct sign for cos(θ) in the polar region

"""
import math

import torch


def angular_to_global(lon: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert angular coordinates (lon, lat) to global HEALPix coordinates (xₛ, yₛ).

    Args:
        lon: Longitude in degrees East
        lat: Latitude in degrees North
    Returns:
        tuple of (xs, ys) coordinates
    """
    # Convert to radians
    phi = torch.deg2rad(lon % 360)
    theta = torch.deg2rad(90 - lat)  # Convert to colatitude
    z = torch.cos(theta)

    # Calculate φ_t = φ mod (π/2)
    phi_t = phi % (math.pi / 2)

    # Calculate σ(z)
    # σ(z) = 2 - sqrt(3 * (1 - z))   for z > 0
    sigma_z = 2 - torch.sqrt(3 * (1 - torch.abs(z)))
    sigma_z = torch.where(z < 0, -sigma_z, sigma_z)

    # Determine if we're in polar caps (|z| > 2/3)
    in_polar_cap = torch.abs(z) > 2 / 3

    # Calculate ys
    ys = torch.where(
        in_polar_cap,
        # ys = (π / 4) * σ(z)                                (29)
        (math.pi / 4) * sigma_z,  # Polar cap formula
        (3 * math.pi / 8) * z,  # Equatorial formula
    )

    # Calculate xs
    xs = torch.where(
        in_polar_cap,
        # xs = φ - (|σ(z)| - 1) * (φ_t - π/4)                 (28)
        phi - (torch.abs(sigma_z) - 1) * (phi_t - math.pi / 4),  # Polar cap formula
        phi,  # Equatorial formula
    )

    return xs, ys


def global_to_angular(xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse HEALPix projection equations.

    Returns:
        tuple of (longitude, latitude) in degrees
    """
    # Determine if we're in polar caps (|ys| > π/4)
    abs_ys = torch.abs(ys)
    in_polar_cap = abs_ys > (math.pi / 4)

    # xt = xs mod (π/2)
    xt = xs % (math.pi / 2)

    # φ (phi)
    phi_equator = xs
    phi_polar = xs - ((abs_ys - (math.pi / 4)) / (abs_ys - (math.pi / 2))) * (xt - (math.pi / 4))
    phi = torch.where(in_polar_cap, phi_polar, phi_equator)

    # cos(θ) (z)
    z_equator = (8 / (3 * math.pi)) * ys
    # [1 - (1/3) * (2 - (4 * |ys|) / π)^2] * (ys / |ys|)
    sign_ys = torch.sign(ys)
    term = 2 - (4 * abs_ys) / math.pi
    z_polar = (1 - (1 / 3) * term**2) * sign_ys
    z = torch.where(in_polar_cap, z_polar, z_equator)

    # Clamp z to [-1, 1] to avoid NaNs from arccos
    z = torch.clamp(z, -1.0, 1.0)

    # θ = arccos(z)
    theta = torch.arccos(z)
    # latitude = 90 - θ (in degrees)
    lat = 90 - torch.rad2deg(theta)

    lon = torch.rad2deg(phi) % 360
    return lon, lat


def global_to_face(xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (xₛ, yₛ) to (x, y, f)

    Continuous version of ``earth2grid.healpix.double2xy``

    Args:
        xs: xs coordinate [0, 2 pi)
        ys: ys coordinate [-pi / 2, pi / 2]

    Returns:
        tuple of (x, y, f)
    """
    # map to [0, 8] x [-2, 2]
    xs, ys = xs / (math.pi / 4), ys / (math.pi / 4)

    # pivot clockwise 45 deg around lower left corner
    # (-1, 0) -> (0, 0)
    x = (xs + ys + 1.0) / 2.0
    y = (ys - xs - 1.0) / 2.0

    x_block = x.floor().int()
    # faces are ordered N to S
    y_block = (-y).floor().int()

    # north
    face = torch.where(x_block > y_block, y_block, 0)
    # equator
    face = torch.where(x_block == y_block, x_block % 4 + 4, face)
    # south
    face = torch.where(x_block < y_block, x_block % 4 + 8, face)

    return x % 1.0, y % 1.0, face


def face_to_global(x: torch.Tensor, y: torch.Tensor, face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (x, y, f) to (xₛ, yₛ)

    Inverse of ``global_to_face``

    Args:
        x: x coordinate [0, 1) within face
        y: y coordinate [0, 1) within face
        face: face index [0, 11]

    Returns:
        tuple of (xs, ys) coordinates
    """
    # Create lookup table mapping face -> (x_block, y_block)
    face_to_x_origin = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3], dtype=x.dtype, device=x.device)
    face_to_y_origin = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5], dtype=x.dtype, device=x.device)

    # Use tensor indexing to get block coordinates
    x_origin = face_to_x_origin[face.long()]
    y_origin = face_to_y_origin[face.long()]

    # in rotated coordinates
    x_rot = x_origin + x
    y_rot = -y_origin + y

    xs = x_rot - y_rot - 1.0
    ys = x_rot + y_rot

    # Scale back to original coordinate system
    xs = xs * (math.pi / 4)
    ys = ys * (math.pi / 4)

    return xs, ys
