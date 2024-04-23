from earth2grid.healpix import Grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


grid = Grid(level=8)
lat = torch.tensor(grid.lat)
lat_img = grid.to_image(lat)

plt.pcolormesh(lat_img)
plt.show()
