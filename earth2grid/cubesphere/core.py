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
Cubesphere Core Utilities
=========================

Functions for converting between different cubesphere data representations.

Pixel Orderings
---------------

Cubesphere data can be stored in different orderings:

**E3SMpgOrder**: Element-major ordering used by E3SM's physgrid (https://docs.e3sm.org/E3SM/EAM/tech-guide/atmosphere-grid-overview/).
    - Each face has num_elements×num_elements spectral elements (blocks)
    - Each element contains num_pg_cells×num_pg_cells physics cells
    - Elements are in row-major order (x varies fastest, then y)
    - Physics cells within each element are also in row-major order
    - Shape: (..., 6 * num_elements² * num_pg_cells²) for global, or (..., num_elements² * num_pg_cells²) per face

    Element layout on each face (num_elements=4 example)::

        y
        ↑  ...  ...  ...  ...
        |   9   10   11   12
        |   5    6    7    8
        |   1    2    3    4
        +-------------------→ x

    Physics cell layout within each element (num_pg_cells=2)::

        y
        ↑   3    4
        |   1    2
        +----------→ x

**XY**: Standard 2D meshgrid ordering.
    - Each face is a 2D array of shape (face_size, face_size) where face_size = num_elements * num_pg_cells
    - Shape: (..., 6, face_size, face_size) for global, or (..., face_size, face_size) per face

    XY layout on each face (face_size=8 example)::

        y
        ↑  ...  ...  ...  ...  ...  ...  ...  ...
        |   9   10   11   12   13   14   15   16
        |   1    2    3    4    5    6    7    8
        +----------------------------------------→ x

Use the ``reorder`` function to convert between orderings.
"""

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class E3SMpgOrder:
    """E3SM spectral element ordering for pg2 physgrid.

    The pg2 "physgrid" divides each spectral element into a 2×2 grid of
    physics cells. Data is stored with elements in row-major order within
    each face, and physics cells in row-major order within each element.

    See: Hannah et al. 2021 for details on the physgrid.

    Index Ordering
    --------------

    **Element (block) indexing within each face:**

    Elements are numbered in row-major order, starting from the bottom-left::

        y
        ↑
        |  ...  ...  ...  ...
        |   9   10   11   12
        |   5    6    7    8
        |   1    2    3    4
        +-------------------→ x

    For num_elements=4, element indices go from 1 to 16 (shown as 1-indexed for clarity,
    but stored as 0-indexed in the data).

    **Physics cell indexing within each 2×2 element (num_pg_cells=2):**

    Within each spectral element, the 4 physics cells are ordered::

        y
        ↑
        |   3    4
        |   1    2
        +----------→ x

    So cell 1 is bottom-left, cell 2 is bottom-right,
    cell 3 is top-left, cell 4 is top-right.

    **Combined flat index:**

    For a point at element (ex, ey) and physics cell (px, py):
        flat_index = (ey * ne + ex) * (npg * npg) + (py * npg + px)

    Attributes:
        ne: Number of spectral elements per cube face edge (num_elements)
        npg: Physics grid points per element dimension (num_pg_cells, default 2 for pg2)

    Example:
        For ne=1024, npg=2:
        - Each face has 1024×1024 = 1,048,576 elements
        - Each element has 2×2 = 4 physics cells
        - Total columns per face: 1024*1024*2*2 = 4,194,304
        - Total global columns: 6 * 4,194,304 = 25,165,824
    """

    num_elements: int
    num_pg_cells: int = 2

    @property
    def face_size(self) -> int:
        """Size of each face in the XY representation."""
        return self.num_elements * self.num_pg_cells

    @property
    def npix_per_face(self) -> int:
        """Number of points per face."""
        return self.num_elements * self.num_elements * self.num_pg_cells * self.num_pg_cells

    @property
    def total_pix(self) -> int:
        """Total number of points across all 6 faces."""
        return 6 * self.npix_per_face


@dataclass(frozen=True)
class XY:
    """Standard 2D meshgrid ordering for cubesphere.

    Data is arranged as a 2D grid on each face with shape (face_size, face_size).
    This format is suitable for image-like operations such as convolutions and padding.

    Attributes:
        face_size: Size of each face (typically num_elements * num_pg_cells from E3SM ordering)
    """

    face_size: int

    @property
    def npix_per_face(self) -> int:
        """Number of points per face."""
        return self.face_size * self.face_size

    @property
    def total_pix(self) -> int:
        """Total number of points across all 6 faces."""
        return 6 * self.npix_per_face


# Type alias for any cubesphere ordering
CubesphereOrderT = Union[E3SMpgOrder, XY]


def _e3sm_to_xy_single_face(array: torch.Tensor, order: E3SMpgOrder) -> torch.Tensor:
    """Convert single-face E3SM element order to 2D meshgrid.

    Args:
        array: Tensor with shape (..., num_elements*num_elements*num_pg_cells*num_pg_cells)
        order: E3SMpgOrder specifying num_elements and num_pg_cells

    Returns:
        Tensor with shape (..., num_elements*num_pg_cells, num_elements*num_pg_cells)
    """
    num_elements, num_pg_cells = order.num_elements, order.num_pg_cells
    side = num_elements * num_pg_cells
    expected = num_elements * num_elements * num_pg_cells * num_pg_cells

    if array.size(-1) != expected:
        raise ValueError(f"Expected last dim = {expected} (=num_elements²×num_pg_cells²), got {array.size(-1)}")

    other_dims = array.shape[:-1]
    x = array.reshape((*other_dims, num_elements, num_elements, num_pg_cells, num_pg_cells))
    nd = len(other_dims)
    perm = list(range(nd)) + [nd + 0, nd + 2, nd + 1, nd + 3]
    x = x.permute(perm).reshape((*other_dims, side, side))
    return x


def _xy_to_e3sm_single_face(array: torch.Tensor, order: E3SMpgOrder) -> torch.Tensor:
    """Convert single-face 2D meshgrid to E3SM element order.

    Args:
        array: Tensor with shape (..., num_elements*num_pg_cells, num_elements*num_pg_cells)
        order: E3SMpgOrder specifying num_elements and num_pg_cells

    Returns:
        Tensor with shape (..., num_elements*num_elements*num_pg_cells*num_pg_cells)
    """
    num_elements, num_pg_cells = order.num_elements, order.num_pg_cells
    side = num_elements * num_pg_cells
    npix = num_elements * num_elements * num_pg_cells * num_pg_cells

    if array.shape[-1] != side or array.shape[-2] != side:
        raise ValueError(f"Expected last two dims = ({side}, {side}), got {array.shape[-2:]}")

    other_dims = array.shape[:-2]
    x = array.reshape((*other_dims, num_elements, num_pg_cells, num_elements, num_pg_cells))
    nd = len(other_dims)
    # Permutation [0, 2, 1, 3] swaps positions 1 and 2 (self-inverse)
    perm = list(range(nd)) + [nd + 0, nd + 2, nd + 1, nd + 3]
    x = x.permute(perm).reshape((*other_dims, npix))
    return x


def reorder(x: torch.Tensor, src: CubesphereOrderT, dest: CubesphereOrderT) -> torch.Tensor:
    """Reorder cubesphere data between different pixel orderings.

    Args:
        x: Input tensor. Shape depends on source ordering:
           - E3SMpgOrder: (..., 6 * num_elements² * num_pg_cells²) for global, or (..., num_elements² * num_pg_cells²) per face
           - XY: (..., 6, face_size, face_size) for global
        src: Source pixel ordering
        dest: Destination pixel ordering

    Returns:
        Tensor reordered to destination format

    Examples:
        Convert E3SM global data to XY format for padding:

        >>> e3sm = E3SMpgOrder(num_elements=1024, num_pg_cells=2)
        >>> xy = XY(face_size=2048)
        >>> data = torch.randn(3, 6 * 1024 * 1024 * 2 * 2)  # 3 channels, global
        >>> faces = reorder(data, src=e3sm, dest=xy)
        >>> faces.shape
        torch.Size([3, 6, 2048, 2048])

        Convert back after processing:

        >>> result = reorder(faces, src=xy, dest=e3sm)
        >>> result.shape
        torch.Size([3, 25165824])
    """
    # Same ordering - no conversion needed
    if src == dest:
        return x

    # E3SM -> XY
    if isinstance(src, E3SMpgOrder) and isinstance(dest, XY):
        if src.face_size != dest.face_size:
            raise ValueError(
                f"Face size mismatch: E3SMpgOrder has face_size={src.face_size}, " f"XY has face_size={dest.face_size}"
            )

        # Check if global (6 faces) or single face
        if x.shape[-1] == src.total_pix:
            # Global: (..., 6 * npix_per_face) -> (..., 6, face_size, face_size)
            leading_dims = x.shape[:-1]
            x_6faces = x.reshape(*leading_dims, 6, src.npix_per_face)
            faces_2d = _e3sm_to_xy_single_face(x_6faces, src)
            return faces_2d
        elif x.shape[-1] == src.npix_per_face:
            # Single face: (..., npix_per_face) -> (..., face_size, face_size)
            return _e3sm_to_xy_single_face(x, src)
        else:
            raise ValueError(
                f"Last dim {x.shape[-1]} doesn't match E3SMpgOrder: "
                f"expected {src.total_pix} (global) or {src.npix_per_face} (per face)"
            )

    # XY -> E3SM
    if isinstance(src, XY) and isinstance(dest, E3SMpgOrder):
        if src.face_size != dest.face_size:
            raise ValueError(
                f"Face size mismatch: XY has face_size={src.face_size}, " f"E3SMpgOrder has face_size={dest.face_size}"
            )

        # Check if global (6 faces) or single face
        if x.ndim >= 3 and x.shape[-3] == 6:
            # Global: (..., 6, face_size, face_size) -> (..., 6 * npix_per_face)
            leading_dims = x.shape[:-3]
            data_flat_faces = _xy_to_e3sm_single_face(x, dest)
            result = data_flat_faces.reshape(*leading_dims, dest.total_pix)
            return result
        elif x.shape[-1] == src.face_size and x.shape[-2] == src.face_size:
            # Single face: (..., face_size, face_size) -> (..., npix_per_face)
            return _xy_to_e3sm_single_face(x, dest)
        else:
            raise ValueError(
                f"Shape {x.shape} doesn't match XY ordering: "
                f"expected (..., 6, {src.face_size}, {src.face_size}) or (..., {src.face_size}, {src.face_size})"
            )

    raise ValueError(f"Unsupported conversion from {type(src).__name__} to {type(dest).__name__}")


def _rotate_tensor(
    n: int, rotations: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized rotation for tensor rotations in {0,1,2,3}."""
    x0, y0 = x, y
    x1, y1 = n - 1 - y, x
    x2, y2 = n - 1 - x, n - 1 - y
    x3, y3 = y, n - 1 - x
    r = rotations % 4
    x_out = torch.where(r == 0, x0, torch.where(r == 1, x1, torch.where(r == 2, x2, x3)))
    y_out = torch.where(r == 0, y0, torch.where(r == 1, y1, torch.where(r == 2, y2, y3)))
    return x_out, y_out


def local2xy(
    face_size: int,
    x: torch.Tensor,
    y: torch.Tensor,
    face: torch.Tensor,
    *,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map local (x, y, face) coordinates to source coordinates for padding.

    This follows the same logic structure as healpix.local2xy:
    - determine direction from (x // n, y // n)
    - lookup neighbor face
    - rotate coordinates by a precomputed number of 90° CCW turns
    - wrap into [0, n)
    """
    n = face_size
    device = x.device
    neighbor_face = torch.tensor(
        [
            [5, 1, 4, 3],
            [5, 2, 4, 0],
            [5, 3, 4, 1],
            [5, 0, 4, 2],
            [0, 1, 2, 3],
            [2, 1, 0, 3],
        ],
        dtype=torch.long,
        device=device,
    )
    turns = torch.tensor(
        [
            [0, 0, 0, 0],
            [3, 0, 1, 0],
            [2, 0, 2, 0],
            [1, 0, 3, 0],
            [0, 3, 2, 1],
            [2, 1, 0, 3],
        ],
        dtype=torch.long,
        device=device,
    )

    # Determine which face shift occurred (like healpix local2xy)
    face_shift_x = torch.div(x, n, rounding_mode="floor")
    face_shift_y = torch.div(y, n, rounding_mode="floor")

    # direction lookup: (-1,0,1) shifts map to top/right/bottom/left, corners invalid
    # order: 0=top, 1=right, 2=bottom, 3=left, -1=inside, -2=corner
    direction_lookup = torch.tensor(
        [
            [-2, 2, -2],
            [3, -1, 1],
            [-2, 0, -2],
        ],
        device=device,
    )
    direction = direction_lookup[face_shift_x + 1, face_shift_y + 1]
    inside = direction == -1
    corner = direction == -2

    # Neighbor face and rotation for each point
    nbr_face = neighbor_face[face, direction.clamp(min=0)]
    rot = turns[face, direction.clamp(min=0)]

    # Rotate and wrap into [0, n)
    x_rot, y_rot = _rotate_tensor(n, rot, x, y)
    x_src = x_rot % n
    y_src = y_rot % n

    # Fill inside points
    x_src = torch.where(inside, x, x_src)
    y_src = torch.where(inside, y, y_src)
    face_src = torch.where(inside, face, nbr_face)

    # Mark corners invalid (filled later)
    face_src = torch.where(
        corner,
        torch.tensor(6, device=device, dtype=face_src.dtype),
        face_src,
    )
    x_src = torch.where(corner, x, x_src)
    y_src = torch.where(corner, y, y_src)

    return x_src, y_src, face_src
