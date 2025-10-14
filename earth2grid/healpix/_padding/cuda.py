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

def validate_shape(
    x: torch.Tensor,  # (B,F,C,H,W) or (B,F,H,W,C)
    channels_last: bool,
):
    if channels_last:
        b, f, h, w, c = x.shape # will errror if ndim is not 5...no need to raise error your self IMO since it should be pretty clear
        dim_str = "(B, F, H, W, C)"
    else:
        b, f, c, h, w = x.shape
        dim_str = "(B, F, C, H, W)"
        

    if h != w:
        raise ValueError(
            f"Input tensor must be have 5 dimensions {dim_str}, with H == W, got {h},  {w}"
        )

    if f != 12:
        raise ValueError(f"Input tensor must be have 5 dimensions {dim_str}, with F == 12, got {f}")
    
@torch.library.custom_op("earth2grid::healpixpad_fprop", mutates_args=())
def healpixpad_fprop(
    x: torch.Tensor,  # (B,F,C,H,W) or (B,F,H,W,C)
    pad: int,
    channels_last: bool,
) -> torch.Tensor:

    validate_shape(x, channels_last)
    # make contiguous
    x = x.contiguous()
    if pad == 0:
        return x.clone()
    out = healpixpad_cuda.forward(x, pad, channels_last)[0]
    return out


@healpixpad_fprop.register_fake
def fake_healpixpad_fprop(x, pad, channels_last):
    """
    The forward pass of the padding class

    Parameters
    ----------
    input: torch.tensor
        The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) or (B, F, H, W, C) for channels last format
        where F == 12 and H == W
    pad: int
        The amount to pad each face of the tensor

    Returns
    -------
    torch.tensor: The padded tensor
    """
    validate_shape(x, channels_last)
    ndim = 5

    # x - (n, f, c, x, y) in origin=N hpx pad order
    if channels_last:
        if x.ndim == 4:
            x = x.unsqueeze(-1)
            ndim = 4
        n, f, nside, _, c = x.shape
        H_out = nside + 2 * pad
        out = x.new_empty((n, f, H_out, H_out, c))
        if ndim == 4:
            out = out.squeeze(-1)
    else:
        if x.ndim == 4:
            x = x.unsqueeze(2)
            ndim = 4
        n, f, c, nside, _ = x.shape
        H_out = nside + 2 * pad
        out = x.new_empty((n, f, c, H_out, H_out))
        if ndim == 4:
            out = out.squeeze(2)

    return out


@torch.library.custom_op("earth2grid::healpixpad_bprop", mutates_args=())
def healpixpad_bprop(
    grad: torch.Tensor,
    pad: int,
    channels_last: bool,
) -> torch.Tensor:
    """
    The back pass of the padding class

    Parameters
    ----------
    grad: torch.tensor
        Result of backward


    Returns
    -------
    torch.tensor: The bw result
    """
    grad = grad.contiguous()
    out = healpixpad_cuda.backward(grad, pad, channels_last)[0]
    return out


@healpixpad_bprop.register_fake
def fake_healpixpad_bprop(grad, pad, channels_last):
    """
    The back pass of the padding class

    Parameters
    ----------
    grad: torch.tensor
        Result of backward


    Returns
    -------
    torch.tensor: The bw result
    """
    ndim = 5
    if channels_last:
        if grad.ndim == 4:
            grad = grad.unsqueeze(-1)
            ndim = 4
        n, f, H_out, _, c = grad.shape
        nside = H_out - 2 * pad
        out = grad.new_empty((n, f, nside, nside, c))
        if ndim == 4:
            out = out.squeeze(-1)
    else:
        if grad.ndim == 4:
            grad = grad.unsqueeze(2)
            ndim = 4
        n, f, c, H_out, _ = grad.shape
        nside = H_out - 2 * pad
        out = grad.new_empty((n, f, c, nside, nside))
        if ndim == 4:
            out = out.squeeze(2)
    return out


def backward(ctx, grad_output):
    # retrive saved info
    pad = ctx.pad
    channels_last = ctx.channels_last

    out = healpixpad_bprop(grad_output, pad, channels_last)
    return out, None, None


def setup_context(ctx, inputs, output):
    x, pad, channels_last = inputs
    ctx.pad = pad
    ctx.channels_last = channels_last


healpixpad_fprop.register_autograd(backward, setup_context=setup_context)


class _HEALPixPadFunction(torch.nn.Module):
    """
    nn.Module wrapper over the custom op `earth2grid::healpixpad_fprop` that pads a healpixpad xy tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, pad, channels_last):
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) or (B, F, H, W, C) for channels last format
            where F == 12 and H == W
        pad: int
            The amount to pad each face of the tensor

        Returns
        -------
        torch.tensor: The padded tensor
        """
        return healpixpad_fprop(input, pad, channels_last)
