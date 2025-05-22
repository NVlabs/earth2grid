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
import numpy
import torch

from earth2grid.healpix import XY, Grid, pad_with_dim

# Print GPU information
if torch.cuda.is_available():
    print("CUDA available: Yes")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device capability: {torch.cuda.get_device_capability()}")
    print(f"Device memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Device memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("CUDA not available")


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
