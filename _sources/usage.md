# Usage

To use Earth2 Grid Utilities in a project

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
