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
import matplotlib.pyplot as plt
import numpy
import pytest
import torch

from earth2grid.healpix import XY, Grid, PaddingBackends, pad, pad_backend, pad_with_dim


def test_hpx_pad(regtest):
    order = 2
    nside = 2**order
    face = 5

    grid = Grid(order, pixel_order=XY())
    lat = torch.from_numpy(grid.lat)
    padded = pad_with_dim(lat, padding=nside, dim=-1)
    m = nside + 2 * nside
    padded = padded.reshape(12, m, m)

    numpy.savetxt(regtest, padded[face].cpu(), fmt="%.2f")


def test_hpx_pad_versus_zephyr(tmp_path):
    order = 3
    nside = 2**order

    grid = Grid(order, pixel_order=XY())
    lat = torch.from_numpy(grid.lat)
    lon = torch.from_numpy(grid.lon)
    z = lon + 3 * lat
    z = torch.arange(grid.shape[0])

    z = z.reshape(1, 12, nside, nside).float()

    with pad_backend(PaddingBackends.zephyr):
        expected = pad(z, padding=nside)

    with pad_backend(PaddingBackends.indexing):
        ans = pad(z, padding=nside)

    if not torch.allclose(expected, ans):
        fig, axs = plt.subplots(3, 12, figsize=(20, 5))
        for i in range(12):
            # Plot expected values
            axs[0, i].imshow(expected[0, i].cpu(), cmap='viridis')
            axs[0, i].set_title(f'Expected Face {i}')
            axs[0, i].axis('off')

            # Plot actual values
            axs[1, i].imshow(ans[0, i].cpu(), cmap='viridis')
            axs[1, i].set_title(f'Actual Face {i}')
            axs[1, i].axis('off')

            # Plot difference
            diff = expected[0, i].cpu() - ans[0, i].cpu()
            axs[2, i].imshow(diff, cmap='RdBu')
            axs[2, i].set_title(f'Diff Face {i}')
            axs[2, i].axis('off')

        plt.tight_layout(pad=1.0)
        fig_path = tmp_path / 'healpix_padding_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        raise AssertionError(f"Padding results differ between backends. Check the saved visualization {fig_path}")


@pytest.mark.parametrize("backend", list(PaddingBackends))
def test_healpix_pad(backend):
    ntile = 12
    nside = 32
    padding = 1
    n = 3
    x = torch.ones([n, ntile, nside, nside])
    with pad_backend(backend):
        out = pad(x, padding=padding)
    assert out.shape == (n, ntile, nside + padding * 2, nside + padding * 2)
