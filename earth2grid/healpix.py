"""

From this notebook: https://colab.research.google.com/drive/1MzTyeNFiy-7RNY6UtGKsmDavX5dk6epU


Healpy has two indexing conventions NEST and RING. But for convolutions we want
2D array indexing in row or column major order. Here are some vectorized
routines `nest2xy` and `x2nest` for going in between these conventions. The
previous code shared by Dale used string computations to handle these
operations, which was probably quite slow. Here we use vectorized bit-shifting.

## XY orientation

For array-like indexing can have a different origin and orientation.  For
example, the default is the origin is S and the data arr[f, y, x] follows the
right hand rule.  In other words, (x + 1, y) being counterclockwise from (x, y)
when looking down on the face.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

import einops
import healpy
import numpy as np
import torch

try:
    import pyvista as pv
except ImportError:
    pv = None

from earth2grid import base
from earth2grid.third_party.zephyr.healpix import healpix_pad as pad

__all__ = ["pad", "PixelOrder", "XY", "Compass", "Grid"]


class PixelOrder(Enum):
    RING = 0
    NEST = 1


class Compass(Enum):
    """Cardinal directions in counter clockwise order"""

    S = 0
    E = 1
    N = 2
    W = 3


@dataclass(frozen=True)
class XY:
    """
    Assumes
        - i = n * n * f + n * y + x
        - the origin (x,y)=(0,0) is South
        - if clockwise=False follows the hand rule:

        Space
          |
          |
          |  / y
          | /
          |/______ x

        (Thumb points towards Space, index finger towards x, middle finger towards y)
    """

    origin: Compass = Compass.S
    clockwise: bool = False


PixelOrderT = Union[PixelOrder, XY]

HEALPIX_PAD_XY = XY(origin=Compass.N, clockwise=True)


def _convert_xyindex(nside: int, src: XY, dest: XY, i):
    if src.clockwise != dest.clockwise:
        i = _flip_xy(nside, i)

    rotations = dest.origin.value - src.origin.value
    i = _rotate_index(nside=nside, rotations=-rotations if dest.clockwise else rotations, i=i)
    return i


class ApplyWeights(torch.nn.Module):
    def __init__(self, pix: torch.Tensor, weight: torch.Tensor):
        super().__init__()
        self._pix = pix
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pix = self._pix
        weight = self.weight
        selected = x[..., pix.ravel()].reshape(*x.shape[:-1], *pix.shape)
        non_spatial_dims = x.ndim - 1
        return torch.sum(selected * weight, axis=non_spatial_dims)


@dataclass
class Grid(base.Grid):
    """A Healpix Grid

    Attrs:
        level: 2^level = nside
        pixel_order: the ordering convection of the data
    """

    level: int
    pixel_order: PixelOrderT = PixelOrder.RING

    def __post_init__(self):
        if self.level > ZOOM_LEVELS:
            raise ValueError(f"`level` must be less than or equal to {ZOOM_LEVELS}")

    def _nside(self):
        return 2**self.level

    def _npix(self):
        return self._nside() ** 2 * 12

    def _nest_ipix(self):
        """convert to nested index number"""
        i = np.arange(self._npix())
        if isinstance(self.pixel_order, XY):
            i_xy = _convert_xyindex(nside=self._nside(), src=self.pixel_order, dest=XY(), i=i)
            return xy2nest(self._nside(), i_xy)
        elif self.pixel_order == PixelOrder.RING:
            return healpy.ring2nest(self._nside(), i)
        elif self.pixel_order == PixelOrder.NEST:
            return i
        else:
            raise ValueError(self.pixel_order)

    def _nest2me(self, ipix: np.ndarray) -> np.ndarray:
        """return the index in my PIXELORDER corresponding to ipix in NEST ordering"""
        if isinstance(self.pixel_order, XY):
            i_xy = nest2xy(self._nside(), ipix)
            i_me = _convert_xyindex(nside=self._nside(), src=XY(), dest=self.pixel_order, i=i_xy)
        elif self.pixel_order == PixelOrder.RING:
            i_me = healpy.nest2ring(self._nside(), ipix)
        elif self.pixel_order == PixelOrder.NEST:
            i_me = ipix
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

    def approximate_grid_length_meters(self):
        return approx_grid_length_meters(self._nside())

    def reorder(self, order: PixelOrderT, x: torch.Tensor) -> torch.Tensor:
        """Rorder the pixels of ``x`` to have ``order``"""
        output_grid = Grid(level=self.level, pixel_order=order)
        i_nest = output_grid._nest_ipix()
        i_me = self._nest2me(i_nest)
        return x[..., i_me]

    def get_healpix_regridder(self, dest: "Grid"):
        if self.level != dest.level:
            raise NotImplementedError(f"{self} and {dest} must have the same level.")

        def regridder(x: torch.Tensor) -> torch.Tensor:
            return self.reorder(dest.pixel_order, x)

        return regridder


# nside = 2^ZOOM_LEVELS
ZOOM_LEVELS = 20


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


def _flip_xy(nside: int, i):
    n2 = nside * nside
    f = i // n2
    y = (i % n2) // nside
    x = i % nside
    return n2 * f + nside * x + y


def _rotate_index(nside: int, rotations: int, i):
    # Extract f, x, and y from i
    # convention is arr[f, y, x] ... x is the fastest changing index
    n2 = nside * nside
    f = i // n2
    y = (i % n2) // nside
    x = i % nside

    # Reduce k to its equivalent in the range [0, 3]
    k = rotations % 4

    assert 0 <= k < 4

    # Apply the rotation based on k
    if k == 1:  # 90 degrees counterclockwise
        new_x, new_y = -y - 1, x
    elif k == 2:  # 180 degrees
        new_x, new_y = -x - 1, -y - 1
    elif k == 3:  # 270 degrees counterclockwise
        new_x, new_y = y, -x - 1
    else:  # k == 0, no change
        new_x, new_y = x, y

    # Adjust for negative indices
    new_x = new_x % nside
    new_y = new_y % nside

    # Recalculate the linear index with the rotated x and y
    return n2 * f + nside * new_y + new_x


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


def approx_grid_length_meters(nside):
    r_m = 6378140
    area = 4 * np.pi * r_m**2 / (12 * nside**2)
    return np.sqrt(area)
