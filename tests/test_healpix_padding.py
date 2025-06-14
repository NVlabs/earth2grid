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
    lon = torch.from_numpy(grid.lon)
    z = lon + 3 * lat
    padded = pad_with_dim(z, padding=nside, dim=-1)
    m = nside + 2 * nside
    padded = padded.reshape(12, m, m)

    for face in range(padded.shape[0]):
        print(f"{face=}", file=regtest)
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
            axs[0, i].imshow(expected[0, i].cpu(), cmap="viridis")
            axs[0, i].set_title(f"Expected Face {i}")
            axs[0, i].axis("off")

            # Plot actual values
            axs[1, i].imshow(ans[0, i].cpu(), cmap="viridis")
            axs[1, i].set_title(f"Actual Face {i}")
            axs[1, i].axis("off")

            # Plot difference
            diff = expected[0, i].cpu() - ans[0, i].cpu()
            axs[2, i].imshow(diff, cmap="RdBu")
            axs[2, i].set_title(f"Diff Face {i}")
            axs[2, i].axis("off")

        plt.tight_layout(pad=1.0)
        fig_path = tmp_path / "healpix_padding_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        raise AssertionError(f"Padding results differ between backends. Check the saved visualization {fig_path}")


@pytest.mark.parametrize("backend", list(PaddingBackends))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_healpix_pad(backend, device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available.")
    if backend == PaddingBackends.cuda and device == "cpu":
        pytest.skip("CUDA padding backend not supported on CPU")

    ntile = 12
    nside = 32
    padding = 1
    n = 3
    x = torch.ones([n, ntile, nside, nside], device=device, requires_grad=True)
    with pad_backend(backend):
        out = pad(x, padding=padding)
    out.mean().backward()
    assert out.shape == (n, ntile, nside + padding * 2, nside + padding * 2)
    assert x.grad.shape == (n, ntile, nside, nside)


# Validate CUDA routine against pure python reference
@pytest.mark.parametrize("nchannels", [1, 2, 4, 8, 12, 32, 33])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("nside", [4, 8, 16])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_healpix_pad_cuda_channels_last(nchannels, padding, nside, offset, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    # Setup input tensors
    ntile = 12
    n = 3
    x = torch.randn([n * ntile * nchannels * nside * nside + offset], device="cuda", dtype=dtype)
    x = x[offset:].reshape([n, ntile, nchannels, nside, nside])

    inputs = {
        'cuda_standard': x.clone().requires_grad_(True),
        'cuda_channels_last': x.permute(0, 1, 3, 4, 2).contiguous().permute(0, 1, 4, 2, 3).requires_grad_(True),
        'python_ref': x.clone().requires_grad_(True),
    }

    # Verify forward pass outputs match
    outputs = {}
    with pad_backend(PaddingBackends.cuda):
        outputs['cuda_standard'] = pad(inputs['cuda_standard'], padding=padding)
        outputs['cuda_channels_last'] = pad(inputs['cuda_channels_last'], padding=padding)
    with pad_backend(PaddingBackends.indexing):
        outputs['python_ref'] = pad(inputs['python_ref'], padding=padding)

    assert torch.allclose(outputs['cuda_standard'], outputs['python_ref'])
    assert torch.allclose(outputs['cuda_channels_last'], outputs['python_ref'])

    # Verify backward pass gradients match
    for out in outputs.values():
        out.sum().backward()

    assert torch.allclose(inputs['cuda_standard'].grad, inputs['python_ref'].grad)
    assert torch.allclose(inputs['cuda_channels_last'].grad, inputs['python_ref'].grad)
