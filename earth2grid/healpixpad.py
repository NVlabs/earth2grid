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
# limitations under the License
#
# Written by Mauro Bisson <maurob@nvidia.com> and Thorsten Kurth <tkurth@nvidia.com>.


import healpixpad_cuda
import torch


class HEALPixPadFunction(torch.autograd.Function):
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
        if len(input.shape) != 5:
            raise ValueError(f"Input tensor must be have 4 dimensions (B, F, C, H, W), got {len(input.shape)} dimensions instead")
        # make contiguous
        input = input.contiguous()
        out = healpixpad_cuda.forward(input, pad)[0]
        return out

    @staticmethod
    def backward(ctx, grad):
        pad = ctx.pad
        grad = grad.contiguous()
        out = healpixpad_cuda.backward(grad, pad)[0]
        return out, None

class HEALPixPad(torch.nn.Module):
    """
    A torch module that handles padding of healpixpad xy tensors

    Paramaeters
    -----------
    padding: int
        The amount to pad the tensors
    """
    def __init__(self, padding: int):
        super(HEALPixPad, self).__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) format
            where F == 12 and H == W

        Returns
        -------
        torch.tensor: The padded tensor
        """
        return HEALPixPadFunction.apply(input, self.padding)

