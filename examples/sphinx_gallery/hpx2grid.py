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
