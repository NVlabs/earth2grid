# Usage

To use Earth2 Grid Utilities in a project

```
import earth2grid
# level is the resolution
level = 6
hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.PixelOrder.XY)
src = earth2grid.latlon.equiangular_lat_lon_grid(32, 64)
z_torch = torch.as_tensor(z)
z_torch = torch.as_tensor(z)
regrid = earth2grid.get_regridder(src, hpx)
z_hpx = regrid(z_torch)
z_hpx.shape
nside = 2**level
reshaped = z_hpx.reshape(12, nside, nside)
```
