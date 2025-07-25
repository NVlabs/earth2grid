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
import contextlib
from enum import Enum, auto

import torch

from earth2grid.healpix._padding import cuda
from earth2grid.healpix._padding.pure_python import pad as pad_python
from earth2grid.healpix._padding.pure_python import pad_with_dim
from earth2grid.third_party.zephyr.healpix import healpix_pad as python_legacy

__all__ = ["pad_backend", "pad", "PaddingBackends", "pad_with_dim"]


class PaddingBackends(Enum):
    indexing = auto()
    zephyr = auto()
    cuda = auto()


# default backend is cuda
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    _backend = PaddingBackends.cuda
else:
    _backend = PaddingBackends.indexing


@contextlib.contextmanager
def pad_backend(backend: PaddingBackends):
    """Select the backend for padding"""
    global _backend

    if not isinstance(backend, PaddingBackends):
        raise ValueError()

    old_backend = _backend
    _backend = backend
    yield
    _backend = old_backend


def pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Pad each face consistently with its according neighbors in the HEALPix

    Args:
        x: The input tensor of shape [N, F, H, W] or [N, F, C, H, W]. Must be
            ordered in HEALPIX_PAD_XY pixel ordering.
        padding: the amount of padding

    Returns:
        The tensor padded along the spatial dimensions. For 4D input [N, F, H, W], returns shape [N, F, H+2*padding, W+2*padding].
        For 5D input, returns [N, F, C, H+2*padding, W+2*padding],

    Examples:

        Ths example show to pad data described by a :py:class:`Grid` object.

        >>> grid = Grid(level=4, pixel_order=PixelOrder.RING)
        >>> lon = torch.from_numpy(grid.lon)
        >>> faces = grid.reorder(HEALPIX_PAD_XY, lon)
        >>> faces = faces.view(1, 12, grid._nside(), grid._nside())
        >>> faces.shape
        torch.Size([1, 12, 16, 16])
        >>> padded = pad(faces, padding=1)
        >>> padded.shape
        torch.Size([1, 12, 18, 18])

    """
    backend = _backend
    if x.device.type != "cuda" and _backend == PaddingBackends.cuda:
        backend = PaddingBackends.indexing

    if backend == PaddingBackends.zephyr:
        return python_legacy(x, padding)
    elif backend == PaddingBackends.indexing:
        return pad_python(x, padding)
    elif backend == PaddingBackends.cuda:
        if x.ndim == 5:
            channels_last = x.stride(2) == 1
            if channels_last:
                x = x.permute(0, 1, 3, 4, 2)  # (B, F, C, H, W) -> (B, F, H, W, C) contiguous
            out = cuda._HEALPixPadFunction.apply(x, padding, channels_last)
            if channels_last:
                out = out.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)
            return out
        else:
            return cuda._HEALPixPadFunction.apply(x.unsqueeze(2), padding, False).squeeze(2)
