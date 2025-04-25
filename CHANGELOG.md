# Changelog

## Latest

New APIs
- earth2grid.healpix.{zonal_average,pixels_to_ring}

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
