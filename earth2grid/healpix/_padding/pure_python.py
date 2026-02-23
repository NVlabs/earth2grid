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

from earth2grid.healpix import core

# most routines in this module use the standard origin=S convention
PIXEL_ORDER = core.XY()


def _xy_with_filled_tile(nside, x1, y1, f1):
    """Handles an points with missing tile information following the HPXPAD strategy

    Missing tiles are defined for face >= 12. 12-16 are the N missing tiles, and
    16-20 the south missing tiles (from W to east).

    Since there is an ambiguity return both x and y.
    """

    # handle missing tiles
    # for N tiles
    # f(x, y) is filled by shuffling from the left
    # case x > y: (x, y) -> (y, )
    # examples  (for nside = 4)
    #   (3, 1)-> (0, 1)
    #   (3, 2) -> (1, 2)
    #   (3, 3) -> (2, 3)
    # generalize
    #   (i, j)-> (i + j, j)  in the missing face
    #   (i' - j, j) -> (i', j)

    is_missing_n_pole_tile = (f1 >= 12) & (f1 < 16)
    west_face = torch.where(is_missing_n_pole_tile, f1 - 13, 0) % 4
    east_face = (west_face + 1) % 4

    # two sets of indices
    def _pad_from_west(x1, y1, west_face):
        f_west = torch.where(is_missing_n_pole_tile & (x1 <= y1), west_face, f1)
        x_west = torch.where(is_missing_n_pole_tile & (x1 < y1), (x1 - y1) % nside, x1)
        x_west = torch.where(is_missing_n_pole_tile & (x1 == y1), nside - 1, x_west)
        y_west = y1
        return x_west, y_west, f_west

    x_west, y_west, f_west = _pad_from_west(x1, y1, west_face)
    y_east, x_east, f_east = _pad_from_west(y1, x1, east_face)

    # S pole
    is_missing_s_pole_tile = (f1 >= 16) & (f1 < 20)
    east_face = (f1 - 16) % 4 + 8
    west_face = (east_face - 9) % 4 + 8

    # two sets of indices
    def _pad_from_east(x1, y1, east_face, f1):
        """Test cases

        (1, 0) -> (0, 0)
        (3, 2) -> (0, 2)
        """
        f_west = torch.where(is_missing_s_pole_tile & (x1 >= y1), east_face, f1)
        # x_west = torch.where(is_missing_s_pole_tile & (x1 > y1), 1(x1-y1) %nside, x1)
        x_west = torch.where(is_missing_s_pole_tile & (x1 > y1), (x1 - y1 - 1) % nside, x1)
        x_west = torch.where(is_missing_s_pole_tile & (x1 == y1), 0, x_west)
        y_west = y1
        return x_west, y_west, f_west

    x_west, y_west, f_west = _pad_from_east(x_west, y_west, east_face, f_west)
    y_east, x_east, f_east = _pad_from_east(y_east, x_east, west_face, f_east)
    return (x_west, y_west, f_west), (x_east, y_east, f_east)


def _get_indices(device, nside, padding, pixel_order=core.XY()):
    """
    Returns flat indices (xy_west, xy_east) for padding a healpix grid.
    """
    pad = padding

    # setup padded grid
    i = torch.arange(-pad, nside + pad, device=device)
    j = torch.arange(-pad, nside + pad, device=device)
    f = torch.arange(12, device=device)
    f, j, i = torch.meshgrid(f, j, i, indexing="ij")

    # convert these ponints to origin=S, clockwise=False order
    # (this is the order expected by local2xy and _xy_with_filled_tile)
    i, j = core.local2local(nside, pixel_order, PIXEL_ORDER, i, j)

    # get indices in source data for target points
    i1, j1, f1 = core.local2xy(nside, i, j, f)

    (i1, j1, f1), (i2, j2, f2) = _xy_with_filled_tile(nside, i1, j1, f1)
    # convert these back to ``pixel_order`` since we will be grabbing
    # data from ``x`` in this order
    i1, j1 = core.local2local(nside, PIXEL_ORDER, pixel_order, i1, j1)
    i2, j2 = core.local2local(nside, PIXEL_ORDER, pixel_order, i2, j2)

    # prepare final flat indexes
    f1 = f1.where(f1 < 12, -1)
    f2 = f2.where(f2 < 12, -1)
    xy_west = torch.flatten(f1 * (nside * nside) + j1 * nside + i1)
    xy_east = torch.flatten(f2 * (nside * nside) + j2 * nside + i2)
    index = torch.stack([xy_west, xy_east], dim=0)
    return index


def pad_with_dim(x, padding, dim=1, pixel_order=core.XY()):
    """Pad a HEALPix grid along a spatial dimension.

    x[dim] is the spatial dim (size 12 * nside**2).

    Implementation notes:
        The core operation is two index_select gathers weighted by a validity
        mask, averaged where both sources are valid (ambiguous corner pixels).

        Alternatives that were benchmarked and rejected:
        - Sparse matmul (W @ x): ~4x slower uncompiled; cuSPARSE is optimized
          for dense sparsity patterns, not 2-nonzeros-per-row matrices.
        - Precomputed nn.Module (Pad): same speed uncompiled, slower compiled.
          torch.compile constant-folds _get_indices (indices depend only on
          nside/padding), so recomputing each call is free when compiled and
          avoids loading index buffers from GPU memory.
        - torch.where with zero-fill: replaced by index_select * weight, which
          avoids materializing zero-filled intermediates and gives the compiler
          a cleaner fuse target (~40% faster compiled).
    """
    dim = dim % x.ndim
    nside = core.npix2nside(x.shape[dim])
    index = _get_indices(x.device, nside, padding, pixel_order=pixel_order)

    shape = [1] * x.ndim
    shape[dim] = index.shape[1]

    valid = index >= 0  # [2, N]
    index = index.clamp_min(0)
    out = x.index_select(dim, index[0]) * valid[0].reshape(shape)
    out = out + x.index_select(dim, index[1]) * valid[1].reshape(shape)
    denom = (valid[0].int() + valid[1].int()).to(x.dtype).reshape(shape)
    return out / denom


def pad(x, padding):
    """A padding function compatible with healpixpad inputs

    Args:
        x: (n, f, x, y) or (n, f, c, h, w)
        padding: int
    """
    ndim = 5
    if x.ndim == 4:
        x = x.unsqueeze(2)
        ndim = 4

    # x - (n, f, c, x, y) in origin=N hpx pad order
    n, f, c, nside, _ = x.shape
    x = torch.movedim(x, 1, 2).reshape(n, c, f * nside**2)
    x = pad_with_dim(x, padding, dim=-1, pixel_order=core.HEALPIX_PAD_XY)
    x = x.reshape(n, c, f, nside + 2 * padding, nside + 2 * padding).movedim(2, 1)

    if ndim == 4:
        x = x.squeeze(2)

    return x
