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
Tests for cubedsphere padding functionality.

These tests mirror the structure of test_healpix_padding.py.
"""
import pytest
import torch

from earth2grid.cubedsphere import e3sm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cubedsphere_pad(device):
    """Test cubedsphere padding with shape and gradient validation.

    Mirrors test_healpix_pad from test_healpix_padding.py.
    """
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available.")

    nface = 6
    face_size = 32
    padding = 1
    n = 3
    x = torch.ones([n, nface, face_size, face_size], device=device, requires_grad=True)
    out = e3sm.pad(x, padding=padding)
    out.mean().backward()
    assert out.shape == (n, nface, face_size + padding * 2, face_size + padding * 2)
    assert x.grad.shape == (n, nface, face_size, face_size)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cubedsphere_reorder(device):
    """Test cubedsphere reorder roundtrip.

    Mirrors test_reorder from test_healpix.py.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    ne, npg = 8, 2
    e3sm_grid = e3sm.E3SMpgOrder(num_elements=ne, num_pg_cells=npg)
    xy = e3sm.XY(face_size=ne * npg)

    data = torch.randn(1, 2, e3sm_grid.total_pix, device=device)
    out = e3sm.reorder(data, src=e3sm_grid, dest=xy)
    out = e3sm.reorder(out, src=xy, dest=e3sm_grid)
    assert torch.all(data == out), data - out
