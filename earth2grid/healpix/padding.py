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
    # adjacency graph (right, up, left, down)
    neighbors = torch.tensor(
        [
            # pole
            [1, 3, 4, 5],
            [2, 0, 5, 6],
            [3, 1, 6, 7],
            [0, 2, 7, 4],
            # equator
            [0, 3, 11, 8],
            [1, 0, 8, 9],
            [2, 1, 9, 10],
            [3, 2, 10, 11],
            # south pole
            [5, 4, 11, 9],
            [6, 5, 8, 10],
            [7, 6, 9, 11],
            [4, 7, 10, 8],
        ],
        device=x.device,
    )

    # number of left turns the path takes while traversing from face i to j
    turns = torch.tensor(
        [
            # pole
            [-1, 1, 0, 0],
            # equator
            [0, 0, 0, 0],
            # south pole
            [0, 0, -1, 1],
        ],
        device=x.device,
    )

    # x direction
    face_shift_x = x // nside
    face_shift_y = y // nside

    # there is a conflic when x_shift > 1 and y_shift > 1
    # there are three ways to resolve
    # 1. always shift x first
    # 2. always shift y first
    # 3. do both and average (hpxpad approach)

    # state of face traversal is (f, origin), we first need to find the new face and the origin of travel
    origin = 0

    # step x
    def _step(shift: int, x: int, origin, face):
        """origin: 0 == x, 1 == y"""
        new_face = face
        new_face = torch.where(shift >= 1, neighbors[face, (x + origin) % 4], new_face)
        new_face = torch.where(shift <= -1, neighbors[face, (x + origin + 2) % 4], new_face)

        new_origin = origin
        new_origin = torch.where(shift >= 1, origin + turns[face // 4, (x + origin) % 4], new_origin)
        new_origin = torch.where(shift <= -1, origin + turns[face // 4, (x + origin + 2) % 4], new_origin)
        return new_origin % 4, new_face

    if right_first:
        origin, face = _step(face_shift_x, 0, origin, face)
        origin, face = _step(face_shift_y, 1, origin, face)
    else:
        origin, face = _step(face_shift_y, 1, origin, face)
        origin, face = _step(face_shift_x, 0, origin, face)

    for i in range(4):
        nx, ny = _rotate(i, x, y)
        x = torch.where(origin == i, nx, x)
        y = torch.where(origin == i, ny, y)

    return face * (nside * nside) + (y % nside) * nside + (x % nside)


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
