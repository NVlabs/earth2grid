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

from earth2grid import _healpix_bare
from earth2grid._healpix_bare import (
    ang2ring,
    corners,
    hpc2loc,
    hpd2loc,
    nest2hpd,
    nest2ring,
    ring2hpd,
    ring2nest,
)

__all__ = [
    "pix2ang",
    "ang2pix",
    "ring2nest",
    "nest2ring",
    "hpc2loc",
    "corners",
]


def pix2ang(nside, i, nest=False, lonlat=False):
    """
    Returns:
        theta, phi: (lon, lat) in degrees if lonlat=True else (colat, lon) in
            radians

    """
    if nest:
        hpd = nest2hpd(nside, i)
    else:
        hpd = ring2hpd(nside, i)
    loc = hpd2loc(nside, hpd)
    lon, lat = _loc2ang(loc)

    if lonlat:
        return torch.rad2deg(lon), 90 - torch.rad2deg(lat)
    else:
        return lat, lon


def ang2pix(nside, theta, phi, nest=False, lonlat=False):
    """Find the pixel containing a given angular coordinate

    Args:
        theta, phi: (lon, lat) in degrees if lonlat=True else (colat, lon) in
            radians

    """
    if lonlat:
        lon = theta
        lat = phi

        theta = torch.deg2rad(90 - lat)
        phi = torch.deg2rad(lon)

    ang = torch.stack([theta, phi], -1)
    pix = ang2ring(nside, ang.double())
    if nest:
        pix = ring2nest(nside, pix)

    return pix


def _loc2ang(loc):
    """
    static t_ang loc2ang(tloc loc)
    { return (t_ang){atan2(loc.s,loc.z), loc.phi}; }
    """
    z = loc[..., 0]
    s = loc[..., 1]
    phi = loc[..., 2]
    return phi % (2 * torch.pi), torch.atan2(s, z)


def _ang2loc(lat, lon):
    pass


def loc2vec(loc):
    z = loc[..., 0]
    s = loc[..., 1]
    phi = loc[..., 2]
    x = (s * torch.cos(phi),)
    y = (s * torch.sin(phi),)
    return x, y, z


def get_interp_weights(nside: int, lon: torch.Tensor, lat: torch.Tensor):
    """

    Args:
        lon: longtiude in deg E. Shape (*)
        lat: latitdue in deg E. Shape (*)

    Returns:
        pix, weights: both shaped (4, *). pix is given in RING convention.

    """
    shape = lon.shape
    lon = lon.double().cpu().flatten()
    lat = lat.double().cpu().flatten()
    pix, weights = _healpix_bare.get_interp_weights(nside, lon, lat)

    return pix.view(4, *shape), weights.view(4, *shape)
