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

from earth2grid.healpix import core as healpix

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _to_mesh(z):
    npix = z.shape[0]
    i_ring = torch.arange(npix)
    nside = healpix.npix2nside(npix)
    y, x = healpix.ring2double(nside, i_ring)

    # rotate clockwise by 45 degrees
    x, y = (x + y) // 2, (x - y) // 2

    y -= y.min()
    x -= x.min()

    out = torch.full((x.max() + 1, y.max() + 1), torch.nan)
    out[x, y] = z.float()

    x, y = (x + y) / 2, (x - y) / 2
    xx, yy = torch.meshgrid(torch.arange(out.shape[0] + 1), torch.arange(out.shape[1] + 1), indexing="ij")
    return (xx + yy) / 2, -(xx - yy) / 2, out


def pcolormesh(z, ax=None, show_axes: bool = False, **kwargs):
    """
    Plot a RING ordered HEALPix map as a pcolormesh.

    Args:
        z: 1D
        ax: An optional matplotlib axis object.
    """

    z = torch.as_tensor(z)
    xx, yy, out = _to_mesh(z)
    nside = healpix.npix2nside(z.shape[0])

    if ax is None:
        ax = plt.gca()

    im = ax.pcolormesh(xx, yy, out, **kwargs)
    ax.set_ylim(-nside, nside)
    ax.set_xlim((nside - 1) / 2, nside * 9 / 2)
    ax.set_aspect('equal')
    if not show_axes:
        ax.set_axis_off()
    return im
