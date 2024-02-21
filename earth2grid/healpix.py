"""

From this notebook: https://colab.research.google.com/drive/1MzTyeNFiy-7RNY6UtGKsmDavX5dk6epU


Healpy has two indexing conventions NEST and RING. But for convolutions we want
2D array indexing in row or column major order. Here are some vectorized
routines `nest2xy` and `x2nest` for going in between these conventions. The
previous code shared by Dale used string computations to handle these
operations, which was probably quite slow. Here we use vectorized bit-shifting.
"""
from dataclasses import dataclass
from enum import Enum

import einops
import healpy
import numpy as np
import torch

try:
    import pyvista as pv
except ImportError:
    pv = None

from earth2grid import base


class PixelOrder(Enum):
    RING = 0
    NEST = 1
    XY = 2


class ApplyWeights(torch.nn.Module):
    def __init__(self, pix: torch.Tensor, weight: torch.Tensor):
        super().__init__()
        self._pix = pix
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pix = self._pix
        weight = self.weight
        selected = x[..., pix.ravel()].reshape(*x.shape[:-1], *pix.shape)
        return torch.sum(selected * weight, axis=0)


@dataclass
class Grid(base.Grid):
    """A Healpix Grid"""

    level: int
    pixel_order: PixelOrder = PixelOrder.RING

    def _nside(self):
        return 2**self.level

    def _npix(self):
        return self._nside() ** 2 * 12

    def _nest_ipix(self):
        """convert to nested index number"""
        i = np.arange(self._npix())
        if self.pixel_order == PixelOrder.RING:
            return healpy.ring2nest(self._nside(), i)
        elif self.pixel_order == PixelOrder.NEST:
            return i
        else:
            return xy2nest(self._nside(), i)

    def _nest2me(self, ipix: np.ndarray) -> np.ndarray:
        """return the index in my PIXELORDER corresponding to ipix in NEST ordering"""
        if self.pixel_order == PixelOrder.RING:
            i_me = healpy.nest2ring(self._nside(), ipix)
        elif self.pixel_order == PixelOrder.NEST:
            i_me = ipix
        elif self.pixel_order == PixelOrder.XY:
            i_me = nest2xy(self._nside(), ipix)

        return i_me

    @property
    def lat(self):
        _, lat = healpy.pix2ang(self._nside(), self._nest_ipix(), lonlat=True, nest=True)
        return lat

    @property
    def lon(self):
        lon, _ = healpy.pix2ang(self._nside(), self._nest_ipix(), lonlat=True, nest=True)
        return lon

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._npix(),)

    def visualize(self, map):
        i = np.arange(self._npix())
        j = self._nest2me(i)
        healpy.mollview(map[j], nest=True)

    def to_pyvista(self):
        if pv is None:
            raise ImportError("Need to install pyvista")

        # Make grid
        nside = 2**self.level
        pix = self._nest_ipix()
        points = healpy.boundaries(nside, pix, step=1, nest=True)
        out = einops.rearrange(points, "n d s -> (n s) d")
        unique_points, inverse = np.unique(out, return_inverse=True, axis=0)
        assert unique_points.ndim == 2
        assert unique_points.shape[1] == 3
        inverse = einops.rearrange(inverse, "(n s) -> n s", n=pix.size)
        n, s = inverse.shape
        cells = np.ones_like(inverse, shape=(n, s + 1))
        cells[:, 0] = s
        cells[:, 1:] = inverse
        celltypes = np.full(shape=(n,), fill_value=pv.CellType.QUAD)
        grid = pv.UnstructuredGrid(cells, celltypes, unique_points)
        return grid

    def get_latlon_regridder(self, lat: np.ndarray, lon: np.ndarray):
        latg, long = np.meshgrid(lat, lon, indexing="ij")
        i_nest, weights = healpy.get_interp_weights(self._nside(), long, latg, nest=True, lonlat=True)
        i_me = self._nest2me(i_nest)
        return ApplyWeights(i_me, torch.from_numpy(weights))


# nside = 2^ZOOM_LEVELS
ZOOM_LEVELS = 12


def _extract_every_other_bit(binary_number):
    result = 0
    shift_count = 0

    for i in range(ZOOM_LEVELS):
        # Check if the least significant bit is 1
        # Set the corresponding bit in the result
        result |= (binary_number & 1) << shift_count

        # Shift to the next bit to check
        binary_number = binary_number >> 2
        shift_count += 1

    return result


def nest2xy(nside, i):
    """convert NEST to XY index"""
    tile = i // nside**2
    j = i % (nside**2)
    x = _extract_every_other_bit(j)
    y = _extract_every_other_bit(j >> 1)
    return tile * nside**2 + y * nside + x


def xy2nest(nside, i):
    """convert XY index to NEST"""
    tile = i // (nside**2)
    y = (i % (nside**2)) // nside
    x = i % nside

    result = 0
    for i in range(ZOOM_LEVELS):
        # Extract the ith bit from the number
        extracted_bit = (x >> i) & 1
        result |= extracted_bit << (2 * i)

        extracted_bit = (y >> i) & 1
        result |= extracted_bit << (2 * i + 1)
    return result | (tile * nside**2)
