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
HealPIX Pixelization
--------------------

HealPIX maps can be viewed as a 2D image rotated by 45 deg or alternatively with
double pixelization that is not rotated.  This is useful for quick visualization
with image viewers without distorting the native pixels of the image.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np

from earth2grid import healpix

n = 8
npix = 12 * n * n
ncap = 2 * n * (n - 1)
p = np.arange(npix)

grid = healpix.Grid(healpix.nside2level(n))
i, jp = healpix.ring2double(n, p)
plt.figure(figsize=(10, 3))
# plt.scatter(jp, i, c=grid.lon[p])
plt.scatter(jp + 1, i, c=grid.lon[p])
plt.grid()

# %%
n = 4
npix = 12 * n * n
ncap = 2 * n * (n - 1)
p = np.arange(npix)

grid = healpix.Grid(healpix.nside2level(n))
out = healpix.to_double_pixelization(grid.lon)
plt.imshow(out)

# %%
out = healpix.to_rotated_pixelization(grid.lon)
plt.imshow(out)
