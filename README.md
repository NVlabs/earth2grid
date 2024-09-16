# Earth2 Grid Utilities
[![img](https://github.com/nvlabs/earth2grid/actions/workflows/ci.yml/badge.svg)](https://github.com/nvlabs/earth2grid/actions/workflows/ci.yml)

<img src="docs/img/image.jpg" width="800px"/>


Utilities for working geographic data defined on various grids.

Features:
- regridding
- Permissively licensed python healpix utilities

Grids currently supported:
- regular lat lon
- HealPIX

* Documentation: <https://nvlabs.github.io/earth2grid/>
* GitHub: <https://github.com/NVlabs/earth2grid>

## Install

```
git clone https://github.com/NVlabs/earth2grid.git
pip install --no-build-isolation ./earth2grid
```

## Example

```
>>> import earth2grid
>>> import torch
... # level is the resolution
... level = 6
... hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
... src = earth2grid.latlon.equiangular_lat_lon_grid(32, 64)
... z_torch = torch.cos(torch.deg2rad(torch.tensor(src.lat)))
... z_torch = z_torch.broadcast_to(src.shape)
>>> regrid = earth2grid.get_regridder(src, hpx)
>>> z_hpx = regrid(z_torch)
>>> z_hpx.shape
torch.Size([49152])
>>> nside = 2**level
... reshaped = z_hpx.reshape(12, nside, nside)
... lat_r = hpx.lat.reshape(12, nside, nside)
... lon_r = hpx.lon.reshape(12, nside, nside)
>>> reshaped.shape
torch.Size([12, 64, 64])
```
