# Changelog

## Latest

New APIs
- earth2grid.healpix
  - zonal_average
  - ring2double
  - to_rotated_pixelization
  - to_double_pixelization
  - pcolormesh
  - pad_with_dim
  - pad_context - switches between padding backends
- earth2grid.projections. Grids in arbitrary projections
- earth2grid.yingyang

Enhancements:
  - pure python implementations of healpix padding and reordering operations,
    that are efficient on GPU when used with torch.compile
  - more performant CUDA healpix padding and channels last padding support

Breaking changes:

- change coordinate transform and shape of lcc grid

## 2025.4.1

Breaking changes:

- renamed package name from "earth2-grid" to "earth2grid"
- `earth2grid.latlon.BilinearInterpolator` moved to `earth2grid.BilinearInterpolator`

New features:
- added `earth2grid.healpix.reorder`
- added Lambert Conformal conic grid (for use with HRRR)

## 2024.8.1

- made visualization dependencies optional
- `healpix.pad` now supports 5d [n, c, f, x, y] shaped arrays.
- Add CUDA implementation of healpix padding (Thorsten Kurth, Mauro Bisson, David Pruitt)

## 2024.5.2

- First publicly available release
