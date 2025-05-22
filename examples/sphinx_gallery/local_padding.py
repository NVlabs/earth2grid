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
# ruff: noqa
# %%
"""
Working with HEALPix Grids and Local Patches
--------------------------------------------

This example demonstrates how to work with HEALPix grids and extract local patches
around specific points on the sphere. We'll show:

1. Creating a HEALPix grid
2. Generating example data on the sphere
3. Extracting a local patch around a point
4. Visualizing both global and local data

Note that the patch can cover multiple HEALPix faces.

The example uses a combination of HEALPix operations and Cartopy for visualization.
"""

import matplotlib.pyplot as plt
import torch
import cartopy.crs

from earth2grid.healpix import XY, Grid
from earth2grid import healpix
from earth2grid.healpix_bare import ang2pix
from earth2grid.spatial import ang2vec
from earth2grid.healpix import local2xy, ring2xy


def create_healpix_grid(order):
    """Create a HEALPix grid of specified order."""
    nside = 2**order
    grid = Grid(order, pixel_order=XY())
    return grid, torch.from_numpy(grid.lat), torch.from_numpy(grid.lon)


def generate_example_data(lon, lat):
    """Generate example data on the sphere using spherical harmonics."""
    x, y, z = ang2vec(lon.deg2rad(), lat.deg2rad())
    return torch.cos(x + 3 * y) + torch.sin(z)


def extract_local_patch(nside, lon0, lat0, pad):
    """Extract a local patch of pixels around the specified point."""
    # Convert target point to pixel index
    i = ang2pix(nside, torch.tensor([lon0]), torch.tensor([lat0]), lonlat=True)
    origin = ring2xy(nside, i)
    x0 = origin % nside
    y0 = (origin % nside**2) // nside
    face0 = origin // (nside**2)

    # Create offset grid
    dx = torch.arange(-pad, pad)
    dy = torch.arange(-pad, pad)
    dy, dx = torch.meshgrid(dy, dx, indexing="ij")

    # Get local coordinates
    x, y, f = local2xy(nside, x0 + dx, y0 + dy, face0)

    # Convert to pixel indices
    pix = torch.where(f < 12, nside**2 * f + nside * y + x, -1)
    return pix


def plot_global_data(z):
    """Plot the global data in HEALPix RING ordering."""
    plt.figure(figsize=(10, 5))
    plt.imshow(healpix.to_double_pixelization(healpix.reorder(z, healpix.XY(), healpix.PixelOrder.RING)))
    plt.colorbar(label='Value')
    plt.title('Global HEALPix Data (RING ordering)')
    plt.show()


def plot_local_patch(lon, lat, z, pix, lon0, lat0):
    """Plot a local patch of data using Cartopy."""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=cartopy.crs.Orthographic(central_latitude=lat0, central_longitude=lon0))

    plt.pcolormesh(lon[pix], lat[pix], z[pix], transform=cartopy.crs.PlateCarree(), cmap='RdBu_r')

    # Mark the center point
    ax.plot(lon0, lat0, marker="*", markersize=10.0, color='yellow')
    ax.coastlines()
    plt.colorbar(label='Value')
    plt.title(f'Local Patch around ({lon0}°, {lat0}°)')
    plt.show()


def main():
    """Run the example demonstrating HEALPix grid operations."""
    # Set up grid parameters
    order = 4
    nside = 2**order
    pad = nside // 2  # Size of local patch

    # Create grid and get coordinates
    grid, lat, lon = create_healpix_grid(order)

    # Generate example data
    z = generate_example_data(lon, lat)

    # Choose a point of interest
    lon0, lat0 = 10, 85  # Example point in northern hemisphere

    # Extract local patch
    pix = extract_local_patch(nside, lon0, lat0, pad)

    # Visualize results
    plot_global_data(z)
    plot_local_patch(lon, lat, z, pix, lon0, lat0)


if __name__ == "__main__":
    main()
