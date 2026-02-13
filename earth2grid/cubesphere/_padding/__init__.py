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
Cubesphere Padding
==================

This module provides padding functionality for cubesphere grids (specifically for E3SM physicsgrid).

The cubesphere grid consists of 6 faces arranged as:

    | 5 |
    | 0 | 1 | 2 | 3 |
    | 4 |

Each face is connected to its 4 neighbors (top, bottom, left, right).
The padding operation fills halo regions from neighboring faces with
proper orientation handling.

Corner filling follows Appendix A2 of https://arxiv.org/abs/2311.06253
"""

import torch

__all__ = ["pad"]

def _fill_corners(padded: torch.Tensor, pad_width: int, face_size: int) -> None:
    """
    Fill the 4 corner regions of padded faces in-place.

    The filling procedure follows Appendix A2 of https://arxiv.org/abs/2311.06253

    For each corner:
    - v_edge: (pad_width, 1) from adjacent horizontal halo, varies by row
    - h_edge: (1, pad_width) from adjacent vertical halo, varies by column
    - One side of diagonal uses v_edge broadcast, other uses h_edge broadcast
    - On diagonal: average of both

    Args:
        padded: Tensor of shape (..., 6, padded_size, padded_size) to modify in-place
        pad_width: Width of padding
        face_size: Original face size (before padding)
    """
    padded_size = face_size + 2 * pad_width
    device = padded.device
    pw, fs = pad_width, face_size

    # Create index grids (pad_width, pad_width)
    i_idx = torch.arange(pw, device=device).view(pw, 1)
    j_idx = torch.arange(pw, device=device).view(1, pw)

    # Main diagonal masks (for bottom-left and top-right)
    main_below = i_idx > j_idx  # i > j
    main_above = i_idx < j_idx  # i < j

    # Anti-diagonal masks (for bottom-right and top-left)
    anti_below = (i_idx + j_idx) > (pw - 1)  # i + j > pw - 1
    anti_above = (i_idx + j_idx) < (pw - 1)  # i + j < pw - 1

    # Corner specs: (row_slice, col_slice, v_col, h_row, diag_type, v_side)
    # diag_type: 'main' or 'anti'
    # v_side: which side of diagonal uses v_edge ('above' or 'below')
    corners = [
        # bottom-left: main diagonal, v_edge (right) used when j > i (above)
        (slice(0, pw), slice(0, pw), pw, pw, "main", "above"),
        # bottom-right: anti-diagonal, v_edge (left) when i+j < pw-1 (above)
        (slice(0, pw), slice(pw + fs, padded_size), pw + fs - 1, pw, "anti", "above"),
        # top-left: anti-diagonal, v_edge (right) when i+j > pw-1 (below)
        (slice(pw + fs, padded_size), slice(0, pw), pw, pw + fs - 1, "anti", "below"),
        # top-right: main diagonal, v_edge (left) when i < j (above)
        (
            slice(pw + fs, padded_size),
            slice(pw + fs, padded_size),
            pw + fs - 1,
            pw + fs - 1,
            "main",
            "below",
        ),
    ]

    for row_slice, col_slice, v_col, h_row, diag_type, v_side in corners:
        # v_edge: (..., pad_width, 1) - column slice, values vary by row
        v_edge = padded[..., row_slice, v_col : v_col + 1]
        # h_edge: (..., 1, pad_width) - row slice, values vary by column
        h_edge = padded[..., h_row : h_row + 1, col_slice]

        # Broadcast to corner shape (..., pad_width, pad_width)
        corner_shape = padded.shape[:-2] + (pw, pw)
        v_broadcast = v_edge.expand(corner_shape)
        h_broadcast = h_edge.expand(corner_shape)

        # Select appropriate masks based on diagonal type
        if diag_type == "main":
            above_mask, below_mask = main_above, main_below
        else:  # anti
            above_mask, below_mask = anti_above, anti_below

        # Assign v_edge and h_edge to correct sides
        if v_side == "above":
            v_mask, h_mask = above_mask, below_mask
        else:  # v_side == 'below'
            v_mask, h_mask = below_mask, above_mask

        # Fill corner
        corner = torch.where(
            v_mask,
            v_broadcast,
            torch.where(h_mask, h_broadcast, (v_broadcast + h_broadcast) / 2),
        )

        padded[..., row_slice, col_slice] = corner


def pad(data: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Pad each face consistently with its neighboring faces in the cubesphere grid.

    Args:
        data: Input tensor of shape (..., 6, H, W) where the 6 faces are ordered as:
              Face 0-3: equatorial band, Face 4: south cap, Face 5: north cap.
              H and W must be equal (square faces). Data must be ordered in XY format.
        padding: Width of padding/halo region to add

    Returns:
        Padded tensor of shape (..., 6, H + 2*padding, W + 2*padding)

    Examples:
        >>> data = torch.randn(1, 6, 64, 64)
        >>> padded = pad(data, padding=4)
        >>> padded.shape
        torch.Size([1, 6, 72, 72])

    Notes:
        The cubesphere layout is:

            | 5 |
            | 0 | 1 | 2 | 3 |
            | 4 |

        Corner filling follows Appendix A2 of https://arxiv.org/abs/2311.06253
    """
    if padding == 0:
        return data

    # Get dimensions
    *leading_dims, num_faces, face_size_x, face_size_y = data.shape
    if num_faces != 6:
        raise ValueError(f"Expected 6 faces, got {num_faces}")
    if face_size_x != face_size_y:
        raise ValueError(f"Face must be square, got {face_size_x}x{face_size_y}")

    face_size = face_size_x
    padded_size = face_size + 2 * padding

    # Build padded coordinate grid
    xs = torch.arange(-padding, face_size + padding, device=data.device)
    ys = torch.arange(-padding, face_size + padding, device=data.device)
    fs = torch.arange(6, device=data.device)
    f_grid, x_grid, y_grid = torch.meshgrid(fs, xs, ys, indexing="ij")

    # Map to source coordinates using local2xy
    x_src, y_src, f_src = local2xy(
        face_size,
        x_grid.reshape(-1),
        y_grid.reshape(-1),
        f_grid.reshape(-1),
        padding=padding,
    )

    # Compute flat indices into source data
    pix = f_src * (face_size * face_size) + x_src * face_size + y_src

    # Handle invalid indices (corners)
    invalid = f_src >= 6
    pix = torch.where(invalid, torch.tensor(0, device=data.device, dtype=pix.dtype), pix)

    # Gather data
    data_flat = data.reshape(*leading_dims, 6 * face_size * face_size)
    padded_flat = data_flat[..., pix]
    padded = padded_flat.reshape(*leading_dims, 6, padded_size, padded_size)

    # Fill corners
    _fill_corners(padded, padding, face_size)

    return padded
