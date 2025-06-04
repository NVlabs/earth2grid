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

try:
    import healpixpad_cuda
except ImportError:
    healpixpad_cuda = None


class _HEALPixPadFunction(torch.autograd.Function):
    """
    A torch autograd class that pads a healpixpad xy tensor
    """

    @staticmethod
    def forward(ctx, input, pad):
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) format
            where F == 12 and H == W
        pad: int
            The amount to pad each face of the tensor

        Returns
        -------
        torch.tensor: The padded tensor
        """
        ctx.pad = pad
        if input.ndim != 5:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), got {len(input.shape)} dimensions instead"
            )
        if input.shape[1] != 12:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), with F == 12, got {input.shape[1]}"
            )
        if input.shape[3] != input.shape[4]:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), with H == @, got {input.shape[3]},  {input.shape[4]}"
            )
        # make contiguous
        input = input.contiguous()
        out = healpixpad_cuda.forward(input, pad)[0]
        return out

    @staticmethod
    def backward(ctx, grad):
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) format
            where F == 12 and H == W
        pad: int
            The amount to pad each face of the tensor

        Returns
        -------
        torch.tensor: The padded tensor
        """
        pad = ctx.pad
        grad = grad.contiguous()
        out = healpixpad_cuda.backward(grad, pad)[0]
        return out, None
