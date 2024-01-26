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

import healpy
import numpy as np

from earth2grid import base


class PixelOrder(Enum):
    RING = 0
    NEST = 1
    XY = 2


@dataclass
class Grid(base.Grid):
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

    def _reorder_to_nest(self, map):
        """convert to nested index number"""
        i = np.arange(self._npix())
        if self.pixel_order == PixelOrder.RING:
            j = healpy.nest2ring(self._nside(), i)
            return map[j]
        elif self.pixel_order == PixelOrder.NEST:
            return map
        else:
            j = nest2xy(self._nside(), i)
            return map[j]

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
        healpy.mollview(self._reorder_to_nest(map), nest=True)


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
