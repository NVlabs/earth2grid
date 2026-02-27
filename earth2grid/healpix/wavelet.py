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

from typing import Any, Tuple

import einops
import numpy as np
import ptwt
import torch

from earth2grid.healpix import pad as healpix_pad


class WaveletDecomposer:
    """
    Efficient wavelet decomposition with caching of zero detail coefficients.
    Avoids recomputing zeros when input size is fixed.
    """

    def __init__(self, wavelet: str = "db4", level: int = 4):
        self.wavelet = wavelet
        self.level = level
        self._cached_shape = None
        self._cached_zero_details: Tuple[Any, ...] = tuple()

    def _create_zero_details(self, x: torch.Tensor) -> Tuple[Any, ...]:
        """Create zero detail coefficients for the given input tensor."""
        # Perform decomposition once to get the structure
        _, *details = ptwt.wavedec2(x, wavelet=self.wavelet, level=self.level, mode="symmetric")

        # Create zeroed-out detail coefficients
        zero_details = tuple(
            ptwt.WaveletDetailTuple2d(
                horizontal=torch.zeros_like(d.horizontal),
                vertical=torch.zeros_like(d.vertical),
                diagonal=torch.zeros_like(d.diagonal),
            )
            for d in details
        )

        return zero_details

    def decompose_lf_hf(self, x: torch.Tensor):
        """
        Decompose input 2D tensor into low-frequency (LF) and high-frequency (HF) components
        using wavelet decomposition.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            lf (torch.Tensor): Low-frequency (approximation) component
            hf (torch.Tensor): High-frequency (detail) component
        """
        # Check if we need to update the cache
        current_shape = x.shape
        if (
            self._cached_shape != current_shape
            or len(self._cached_zero_details) == 0
            or self._cached_zero_details[0].horizontal.device != x.device
        ):
            self._cached_shape = current_shape
            self._cached_zero_details = self._create_zero_details(x)

        # Perform multilevel wavelet decomposition
        approx, *_ = ptwt.wavedec2(x, wavelet=self.wavelet, level=self.level, mode="symmetric")

        # Reconstruct LF using only the approximation coefficients and cached zeros
        lf = ptwt.waverec2((approx, *self._cached_zero_details), wavelet=self.wavelet)

        # Compute HF as the residual
        hf = x - lf

        return lf, hf

    def clear_cache(self):
        """Clear the cached zero details (useful for memory cleanup)."""
        self._cached_shape = None
        self._cached_zero_details = tuple()


class HEALPixWaveletDecomposer(WaveletDecomposer):
    """
    Wavelet decomposition of global HEALPix data.
    Avoids recomputing zeros when input size is fixed.
    """

    def __init__(self, wavelet: str = "db4", level: int = 4):
        self.wavelet = wavelet
        self.level = level
        self._cached_shape = None
        self._cached_zero_details: Tuple[Any, ...] = tuple()

    def coarsen(self, x: torch.Tensor, compute_hf=False):
        """
        Pass inputs as [B, X] in HEALPIX_PAD_XY format
        """
        F = 12
        B = x.shape[0]
        nx = x.shape[-1]
        nside = int(np.sqrt(nx / 12))
        if nside * nside * F != nx:
            raise RuntimeError(f"x is not a HEALPix map, x.shape[-1] ({nx}) should be 12 * 4^l")

        # Get required padding
        padding = max(ptwt._util._get_pad(nside, ptwt._util._get_len(self.wavelet)))

        # Fold all batch, channel and time dimensions
        x = einops.rearrange(x, "b (f x y) -> b f x y", y=nside, x=nside, f=F)
        x_pad = healpix_pad(x, padding)

        current_shape = x_pad.shape
        if (
            self._cached_shape != current_shape
            or len(self._cached_zero_details) == 0
            or self._cached_zero_details[0].horizontal.device != x_pad.device
        ):
            self._cached_shape = current_shape
            self._cached_zero_details = self._create_zero_details(x_pad)

        # Perform multilevel wavelet decomposition on padded faces
        approx, *_ = ptwt.wavedec2(x_pad, wavelet=self.wavelet, level=self.level, mode="symmetric")

        # Reconstruct LF using only the approximation coefficients and cached zeros
        lf = ptwt.waverec2((approx, *self._cached_zero_details), wavelet=self.wavelet)

        # Crop to get back to nside x nside
        lf = lf[:, :, padding:-padding, padding:-padding]

        # Unfold batch, channel, time dims
        x = einops.rearrange(x, "b f x y -> b (f x y)", b=B, y=nside, x=nside, f=F)
        lf = einops.rearrange(lf, "b f x y -> b (f x y)", b=B, y=nside, x=nside, f=F)

        if compute_hf:
            # Compute HF as the residual
            hf = x - lf
            return lf, hf
        else:
            return lf
