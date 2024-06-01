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
HealPIX Image visualization
---------------------------

HealPIX maps can be viewed as a 2D image rotated by 45 deg. This is useful for
quick visualization with image viewers without distorting the native pixels of
the image.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
from matplotlib.colors import Normalize
from PIL import Image

from earth2grid.healpix import Grid

grid = Grid(level=8)
lat = torch.tensor(grid.lat)
lat_img = grid.to_image(lat)

# Use Image to save at full resolution
normalizer = Normalize(vmin=np.nanmin(lat_img), vmax=np.nanmax(lat_img))
array = normalizer(lat_img)
array = plt.cm.viridis(array)
array = (256 * array).astype("uint8")
# set transparency for nans
array[..., -1] = np.where(np.isnan(lat_img), 0, 255)
image = Image.fromarray(array)
image.save("hpx_grid.png")
