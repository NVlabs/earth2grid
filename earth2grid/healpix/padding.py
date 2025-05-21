# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License.import torch
import torch


def local2xy(
    nside: int, x: torch.Tensor, y: torch.Tensor, face: torch.Tensor, right_first: bool = True
) -> torch.Tensor:
    """Convert a local x, y coordinate in a given face (S origin) to a global pixel index

    The local coordinates can be < 0 or > nside

    This can be used to implement padding

    Args:
        nside: int
        x: local coordinate [-nside, 2 * nside), origin=S
        y: local coordinate [-nside, 2 * nside), origin=S
        face: index of the face [0, 12)
        right_first: if True then traverse to the face in the x-direction first

    Returns:
        global XY pixel index
    """
    # adjacency graph (8 neighbors, counter-clockwise from S)
    # Any faces > 11 are missing
    neighbors = torch.tensor(
        [
            # pole
            [8, 5, 1, 1, 2, 3, 3, 4],
            [9, 6, 2, 2, 3, 0, 0, 5],
            [10, 7, 3, 3, 0, 1, 1, 6],
            [11, 4, 0, 0, 1, 2, 2, 7],
            # equator
            [16, 8, 5, 0, 12, 3, 7, 11],
            [17, 9, 6, 1, 13, 0, 4, 8],
            [18, 10, 7, 2, 14, 1, 5, 9],
            [19, 11, 4, 3, 15, 2, 6, 10],
            # south pole
            [10, 9, 9, 5, 0, 4, 11, 11],
            [11, 10, 10, 6, 1, 5, 8, 8],
            [8, 11, 11, 7, 2, 6, 9, 9],
            [9, 8, 8, 4, 3, 7, 10, 10],
        ],
        device=x.device,
    )

    # number of left turns the path takes while traversing from face i to j
    turns = torch.tensor(
        [
            # pole
            [0, 0, 0, 3, 2, 1, 0, 0],
            # equator
            [0, 0, 0, 0, 0, 0, 0, 0],
            # south pole
            [2, 1, 0, 0, 0, 0, 0, 3],
        ],
        device=x.device,
    )
    # x direction
    face_shift_x = x // nside
    face_shift_y = y // nside

    # TODO what if more face_shift_x, face_shift_y = 2, 1 or similar?
    # which direction should we traverse faces in?
    direction_lookup = torch.tensor([[0, 7, 6], [1, -1, 5], [2, 3, 4]], device=x.device)

    direction = direction_lookup[face_shift_x + 1, face_shift_y + 1]
    new_face = torch.where(direction != -1, neighbors[face, direction], face)
    origin = torch.where(direction != -1, turns[face // 4, direction], 0)

    # rotate back to origin = S convection
    for i in range(1, 4):
        nx, ny = _rotate(i, x, y)
        x = torch.where(origin == i, nx, x)
        y = torch.where(origin == i, ny, y)

    face = new_face
    return x % nside, y % nside, face
    return torch.where(face != -1, face * (nside * nside) + (y % nside) * nside + (x % nside), -1)


def _rotate(rotations: int, x, y):
    """rotate (x,y) counter clockwise"""
    k = rotations % 4
    # Apply the rotation based on k
    if k == 1:  # 90 degrees counterclockwise
        return -y - 1, x
    elif k == 2:  # 180 degrees
        return -x - 1, -y - 1
    elif k == 3:  # 270 degrees counterclockwise
        return y, -x - 1
    else:  # k == 0, no change
        return x, y


def _get_xy(nside, f, x, y):
    return torch.where(f < 12, f * nside**2 + y * nside + x, -1)


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

    xy_west = _get_xy(nside, f_west, x_west, y_west)
    xy_east = _get_xy(nside, f_east, x_east, y_east)

    return xy_west, xy_east


def pad(x, pad):
    # x - (n, c, f, x, y) in origin=S order
    n, c = x.shape[:2]
    nside = x.shape[-1]
    x = x.reshape((n, c, -1))

    # setup padded grid
    i = torch.arange(-pad, nside + pad, device=x.device)
    j = torch.arange(-pad, nside + pad, device=x.device)
    f = torch.arange(12, device=x.device)

    # get indices in source data for target points
    f, j, i = torch.meshgrid(f, j, i, indexing="ij")
    i1, j1, f1 = local2xy(nside, i, j, f)
    xy_east, xy_west = _xy_with_filled_tile(nside, i1, j1, f1)

    # average the potential ambiguous regions
    padded_from_west = torch.where(xy_west >= 0, x[..., xy_west], 0)
    padded_from_east = torch.where(xy_east >= 0, x[..., xy_east], 0)
    denom = (xy_west >= 0).int() + (xy_east >= 0).int()

    return (padded_from_east + padded_from_west) / denom
