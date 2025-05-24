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
import math

import einops
import torch

from earth2grid.healpix._padding import pad


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

    Applies a 2D convolution over an input image composed of several input
    planes.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    See :class:`~torch.nn.Conv2d` for details and output shape.

    """
    px, py = padding
    if px != py:
        raise ValueError(f"Padding should be equal in x and y, got px={px}, py={py}")

    n, c, x, y = input.shape
    npix = input.size(-1)
    nside2 = npix // 12
    nside = int(math.sqrt(nside2))
    if nside**2 * 12 != npix:
        raise ValueError(f"Incompatible npix ({npix}) and nside ({nside})")

    input = einops.rearrange(input, "n c () (f x y) -> (n c) f x y", f=12, x=nside)
    input = pad(input, px)
    input = einops.rearrange(input, "(n c) f x y -> n c f x y", c=c)
    padding = (0, 0, 0)
    padding = 'valid'

    if not isinstance(stride, int):
        stride = stride + (1,)

    if not isinstance(dilation, int):
        dilation = (1,) + dilation

    weight = weight.unsqueeze(-3)
    out = torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)
    return einops.rearrange(out, "n c f x y -> n c () (f x y)")
