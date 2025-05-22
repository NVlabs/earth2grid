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

## HPXPad

For the missing diamonds, N and S of the equatorial faces, the method is not
what I expected. Instead of traversing the edge graph as I would expect, it
fills in the missing diamond first, and then treats this as if it were an actual
tile.  For, the NW tile, the rows (along x) are simply shifted along x into the
missing face until they hit the diagonal until they intersect the diagonal.

"""

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TypeVar, Union

import einops
import numpy as np
import torch

from earth2grid import _bit_ops, healpix_bare
from earth2grid._regrid import Regridder

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import healpixpad_cuda

    healpixpad_cuda_avail = True
except ImportError:
    healpixpad_cuda_avail = False
    warnings.warn("healpixpad_cuda module not available, reverting to CPU for all padding routines")


from earth2grid import base
from earth2grid.third_party.zephyr.healpix import healpix_pad as heapixpad_cpu

try:
    import cuhpx
except ImportError:
    cuhpx = None


def _get_array_library(x):
    if isinstance(x, np.ndarray):
        return np
    else:
        return torch


ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Pad each face consistently with its according neighbors in the HEALPix

    Args:
        x: The input tensor of shape [N, F, H, W] or [N, F, C, H, W]. Must be
            ordered in HEALPIX_PAD_XY pixel ordering.
        padding: the amount of padding

    Returns:
        The padded tensor with shape [N, F, H+2*padding, W+2*padding]

    Examples:

        Ths example show to pad data described by a :py:class:`Grid` object.

        >>> grid = Grid(level=4, pixel_order=PixelOrder.RING)
        >>> lon = torch.from_numpy(grid.lon)
        >>> faces = grid.reorder(HEALPIX_PAD_XY, lon)
        >>> faces = faces.view(1, 12, grid._nside(), grid._nside())
        >>> faces.shape
        torch.Size([1, 12, 16, 16])
        >>> padded = pad(faces, padding=1)
        >>> padded.shape
        torch.Size([1, 12, 18, 18])

    """
    if x.device.type != 'cuda' or not healpixpad_cuda_avail:
        return heapixpad_cpu(x, padding)
    elif x.ndim == 5:
        return _HEALPixPadFunction.apply(x, padding)
    else:
        return _HEALPixPadFunction.apply(x.unsqueeze(2), padding).squeeze(2)


def _apply_cuhpx_remap(func, x, **kwargs):
    shape = x.shape
    x = x.view(-1, 1, shape[-1])
    nside = npix2nside(x.shape[-1])
    x = func(x.contiguous(), **kwargs, nside=nside)
    x = x.contiguous()
    x = x.view(shape[:-1] + (-1,))
    return x


def npix2nside(npix: int):
    nside = math.sqrt(npix // 12)
    return int(nside)


def npix2level(npix: int):
    return nside2level(npix2nside(npix))


def nside2level(nside: int):
    return int(math.log2(nside))


class PixelOrder(Enum):
    RING = 0
    NEST = 1

    def reorder_from_cuda(self, x, src: "PixelOrderT"):
        if self == PixelOrder.RING:
            return src.to_ring_cuda(x)
        elif self == PixelOrder.NEST:
            return src.to_nest_cuda(x)

    def to_ring_cuda(self, x: torch.Tensor):
        if self == PixelOrder.RING:
            return x
        elif self == PixelOrder.NEST:
            if cuhpx is None:
                pix = _arange_like(x, -1)
                n = npix2nside(pix.numel())
                pix_nest = ring2nest(n, pix)
                return x[..., pix_nest]
            else:
                return _apply_cuhpx_remap(cuhpx.nest2ring, x)

    def to_nest_cuda(self, x: torch.Tensor):
        if self == PixelOrder.RING:
            if cuhpx is None:
                pix = _arange_like(x, -1)
                n = npix2nside(pix.numel())
                pix_nest = ring2nest(n, pix)
                return torch.empty_like(x).scatter_(-1, pix_nest.broadcast_to(x.shape), x)
            else:
                return _apply_cuhpx_remap(cuhpx.ring2nest, x)
        elif self == PixelOrder.NEST:
            return x

    def to_xy_cuda(self, x: torch.Tensor, dest: "XY"):
        if self == PixelOrder.RING:
            if cuhpx is None:
                return _reorder_via_permutation(x, self, dest)
            else:
                return _apply_cuhpx_remap(cuhpx.ring2flat, x, clockwise=dest.clockwise, origin=dest.origin.name)
        elif self == PixelOrder.NEST:
            nside = npix2nside(x.size(-1))
            i_dest = torch.arange(x.shape[-1], dtype=torch.int64, device=x.device)
            i = xy2nest(nside, i_dest, dest)
            return x[..., i]


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

    def reorder_from_cuda(self, x, src: "PixelOrderT"):
        return src.to_xy_cuda(x, self)

    def to_xy_cuda(self, x: torch.Tensor, dest: "XY"):
        nside = npix2nside(x.size(-1))
        i_dest = torch.arange(x.shape[-1], dtype=torch.int64, device=x.device)
        i = xy2xy(nside, src=dest, dest=self, i=i_dest)
        return x[..., i]

    def to_ring_cuda(self, x: torch.Tensor):
        if cuhpx is None:
            return _reorder_via_permutation(x, self, PixelOrder.RING)
        else:
            return _apply_cuhpx_remap(
                cuhpx.flat2ring,
                x,
                origin=self.origin.name,
                clockwise=self.clockwise,
            )

    def to_nest_cuda(self, x: torch.Tensor):
        nside = npix2nside(x.size(-1))
        i_dest = torch.arange(x.shape[-1], dtype=torch.int64, device=x.device)
        i = nest2xy(nside, i_dest, self)
        return x[..., i]


PixelOrderT = Union[PixelOrder, XY]

HEALPIX_PAD_XY = XY(origin=Compass.N, clockwise=True)


def reorder(x: torch.Tensor, src_pixel_order: PixelOrderT, dest_pixel_order: PixelOrderT):
    """Reorder x from one pixel order to another"""
    grid = Grid(level=npix2level(x.size(-1)), pixel_order=src_pixel_order)
    return grid.reorder(dest_pixel_order, x)


@lru_cache()
def _get_permutation_from_src_to_dest(level: int, src: PixelOrder, dest: PixelOrder, device):
    """Return a index that can be used to reorder src to dest

    >>> i_me = _get_permutation_from_src_to_dest(6, ...)
    >>> x_in_dest_order = x_in_src_order[..., i_me]

    """
    src_grid = Grid(level, pixel_order=src)
    dest_grid = Grid(level, pixel_order=dest)
    i_nest = dest_grid._nest_ipix()
    i_me = src_grid._nest2me(i_nest)
    return i_me.to(device)


def _reorder_via_permutation(x: torch.Tensor, src: PixelOrderT, dest: PixelOrderT):
    """Reorder by indexing using the permutation map from src to dest

    This is a fallback that works when cuhpx is not installed. The
    permutation is constructed on the cpu, but can be applied on the GPU.
    An LRU cache is used to avoid recomputing the permutation map."""
    level = npix2level(x.size(-1))
    i_me = _get_permutation_from_src_to_dest(level, src=src, dest=dest, device=x.device)
    return x[..., i_me]


def xy2xy(nside: int, src: XY, dest: XY, i: torch.Tensor):
    """Convert flat index between pixel ordering conventions`

    Args:
        i: int64

    """
    if src == dest:
        return i

    if src.clockwise != dest.clockwise:
        i = _flip_xy(nside, i)

    rotations = dest.origin.value - src.origin.value
    i = _rotate_index(nside=nside, rotations=-rotations if dest.clockwise else rotations, i=i)
    return i


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
        i = torch.arange(self._npix(), device="cpu")
        if isinstance(self.pixel_order, XY):
            i_xy = xy2xy(nside=self._nside(), src=self.pixel_order, dest=XY(), i=i)
            i = xy2nest(self._nside(), i_xy)
        elif self.pixel_order == PixelOrder.RING:
            i = healpix_bare.ring2nest(self._nside(), i)
        elif self.pixel_order == PixelOrder.NEST:
            pass
        else:
            raise ValueError(self.pixel_order)
        return i

    def _nest2me(self, ipix: torch.Tensor) -> torch.Tensor:
        """return the index in my PIXELORDER corresponding to ipix in NEST ordering"""
        if isinstance(self.pixel_order, XY):
            i_xy = nest2xy(self._nside(), ipix)
            i_me = xy2xy(nside=self._nside(), src=XY(), dest=self.pixel_order, i=i_xy)
        elif self.pixel_order == PixelOrder.RING:
            i_me = healpix_bare.nest2ring(self._nside(), ipix)
        elif self.pixel_order == PixelOrder.NEST:
            i_me = ipix
        return i_me

    @property
    def lat(self):
        ipix = self._nest_ipix()
        _, lat = healpix_bare.pix2ang(self._nside(), ipix, lonlat=True, nest=True)
        return lat.numpy()

    @property
    def lon(self):
        ipix = self._nest_ipix()
        lon, _ = healpix_bare.pix2ang(self._nside(), ipix, lonlat=True, nest=True)
        return lon.numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._npix(),)

    def visualize(self, map):
        raise NotImplementedError()

    def to_pyvista(self):
        if pv is None:
            raise ImportError("Need to install pyvista")

        # Make grid
        nside = 2**self.level
        pix = self._nest_ipix()
        points = healpix_bare.corners(nside, pix, True).numpy()
        out = einops.rearrange(points, "n d s -> (n s) d")
        unique_points, inverse = np.unique(out, return_inverse=True, axis=0)
        if unique_points.ndim != 2:
            raise ValueError(f"unique_points.ndim should be 2, got {unique_points.ndim}.")
        if unique_points.shape[1] != 3:
            raise ValueError(f"unique_points.shape[1] should be 3, got {unique_points.shape[1]}.")
        inverse = einops.rearrange(inverse, "(n s) -> n s", n=pix.numel())
        n, s = inverse.shape
        cells = np.ones_like(inverse, shape=(n, s + 1))
        cells[:, 0] = s
        cells[:, 1:] = inverse
        celltypes = np.full(shape=(n,), fill_value=pv.CellType.QUAD)
        grid = pv.UnstructuredGrid(cells, celltypes, unique_points)
        return grid

    def get_bilinear_regridder_to(self, lat: np.ndarray, lon: np.ndarray):
        """Get regridder to the specified lat and lon points"""
        lat, lon = np.broadcast_arrays(lat, lon)
        i_ring, weights = healpix_bare.get_interp_weights(self._nside(), torch.tensor(lon), torch.tensor(lat))
        i_nest = healpix_bare.ring2nest(self._nside(), i_ring.ravel())
        i_me = self._nest2me(i_nest).reshape(i_ring.shape)

        # reshape to (*, p)
        weights = weights.movedim(0, -1)
        index = i_me.movedim(0, -1)

        regridder = Regridder(weights.shape[:-1], p=weights.shape[-1])
        regridder.to(weights)
        regridder.index.copy_(index)
        regridder.weight.copy_(weights)
        return regridder

    def approximate_grid_length_meters(self):
        return approx_grid_length_meters(self._nside())

    def reorder(self, order: PixelOrderT, x: torch.Tensor) -> torch.Tensor:
        """Rorder the pixels of ``x`` to have ``order``"""
        if x.device.type == "cuda":
            return order.reorder_from_cuda(x, self.pixel_order)
        else:
            return _reorder_via_permutation(x, self.pixel_order, order)

    def get_healpix_regridder(self, dest: "Grid"):
        if self.level != dest.level:
            return self.get_bilinear_regridder_to(dest.lat, dest.lon)

        def regridder(x: torch.Tensor) -> torch.Tensor:
            return self.reorder(dest.pixel_order, x)

        return regridder

    def to_image(self, x: torch.Tensor, fill_value=torch.nan) -> torch.Tensor:
        """Use the 45 degree rotated grid pixelation
        i points to SE, j point to NE
        """
        return to_rotated_pixelization(x, fill_value)


class _HEALPixPadFunction(torch.autograd.Function):
    """
    A torch autograd class that pads a healpixpad xy tensor
    """

    @staticmethod
    def forward(ctx, input, pad):
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) format
            where F == 12 and H == W
        pad: int
            The amount to pad each face of the tensor

        Returns
        -------
        torch.tensor: The padded tensor
        """
        ctx.pad = pad
        if input.ndim != 5:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), got {len(input.shape)} dimensions instead"
            )
        if input.shape[1] != 12:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), with F == 12, got {input.shape[1]}"
            )
        if input.shape[3] != input.shape[4]:
            raise ValueError(
                f"Input tensor must be have 5 dimensions (B, F, C, H, W), with H == @, got {input.shape[3]},  {input.shape[4]}"
            )
        # make contiguous
        input = input.contiguous()
        out = healpixpad_cuda.forward(input, pad)[0]
        return out

    @staticmethod
    def backward(ctx, grad):
        """
        The forward pass of the padding class

        Parameters
        ----------
        input: torch.tensor
            The tensor to pad, must have 5 dimensions and be in (B, F, C, H, W) format
            where F == 12 and H == W
        pad: int
            The amount to pad each face of the tensor

        Returns
        -------
        torch.tensor: The padded tensor
        """
        pad = ctx.pad
        grad = grad.contiguous()
        out = healpixpad_cuda.backward(grad, pad)[0]
        return out, None


# nside = 2^ZOOM_LEVELS
ZOOM_LEVELS = 20


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

    if k < 0 or k >= 4:
        raise ValueError(f"k not in [0, 3], got {k}")

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


def nest2xy(nside, i, pixel_order: XY = XY()):
    """convert NEST to XY index"""
    tile = i // nside**2
    j = i % (nside**2)
    x = _bit_ops.compact_bits(j)
    y = _bit_ops.compact_bits(j >> 1)
    xy = tile * nside**2 + y * nside + x
    xy = xy2xy(nside, XY(), pixel_order, xy)
    return xy


def xy2nest(nside, i, pixel_order: XY = XY()):
    """convert XY index to NEST"""
    i = xy2xy(nside, pixel_order, XY(), i)
    tile = i // (nside**2)
    y = (i % (nside**2)) // nside
    x = i % nside

    result = 0
    result |= _bit_ops.spread_bits(x)
    result |= _bit_ops.spread_bits(y) << 1
    return result | (tile * nside**2)


def approx_grid_length_meters(nside):
    r_m = 6378140
    area = 4 * np.pi * r_m**2 / (12 * nside**2)
    return np.sqrt(area)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

    Applies a 2D convolution over an input image composed of several input
    planes.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    See :class:`~torch.nn.Conv2d` for details and output shape.

    """
    px, py = padding
    if px != py:
        raise ValueError(f"Padding should be equal in x and y, got px={px}, py={py}")

    n, c, x, y = input.shape
    npix = input.size(-1)
    nside2 = npix // 12
    nside = int(math.sqrt(nside2))
    if nside**2 * 12 != npix:
        raise ValueError(f"Incompatible npix ({npix}) and nside ({nside})")

    input = einops.rearrange(input, "n c () (f x y) -> (n c) f x y", f=12, x=nside)
    input = pad(input, px)
    input = einops.rearrange(input, "(n c) f x y -> n c f x y", c=c)
    padding = (0, 0, 0)
    padding = 'valid'

    if not isinstance(stride, int):
        stride = stride + (1,)

    if not isinstance(dilation, int):
        dilation = (1,) + dilation

    weight = weight.unsqueeze(-3)
    out = torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)
    return einops.rearrange(out, "n c f x y -> n c () (f x y)")


def _isqrt(x):
    return x.sqrt().to(x.dtype)


def _pixels_to_rings(nside: int, p: torch.Tensor) -> torch.Tensor:
    """Get the ring number of a pixel ``i`` in RING order"""
    # See eq (2-5) of Gorski
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)

    i_north = (1 + _isqrt(1 + 2 * p)) >> 1
    j_north = p - 2 * (i_north - 1) * i_north

    p_eq = p - ncap
    i_eq = p_eq // (4 * nside) + nside - 1
    j_eq = p_eq % (4 * nside)

    p_south = npix - p - 1
    i_south = (1 + _isqrt(1 + 2 * p_south)) >> 1
    j_south = p_south - 2 * (i_south - 1) * i_south
    length_south = i_south * 4

    i = i_north - 1
    i = torch.where(p >= ncap, i_eq, i)
    i = torch.where(p >= (npix - ncap), 4 * nside - i_south - 1, i)

    j = j_north
    j = torch.where(p >= ncap, j_eq, j)
    j = torch.where(p >= (npix - ncap), length_south - 1 - j_south, j)

    return i, j


def ring_length(nside: int, i: ArrayT) -> ArrayT:
    """The number of pixels in ring 0 <= i < 4 * nside - 2"""
    xp = _get_array_library(i)

    length_north = 4 * (i + 1)
    length_eq = 4 * nside
    length_south = (4 * nside - i - 1) * 4

    length = length_north
    # test i =1, nside = 1
    length = xp.where(i >= nside, length_eq, length)
    # test: i = 2, nside=1, should have len 4
    length = xp.where(i >= nside * 3 - 1, length_south, length)
    return length


def double2xy(nside, i, j):
    xp = _get_array_library(i)
    # make upper left corner j=-nside, i = 2 * nside
    # y counts down from top left
    #        x
    #     |------
    #   y |
    #     |
    #     ↓
    # the diagonal blocks in this coordinate system are the equator-only tiles
    y = (i + j - nside) >> 1
    x = (j - i + 3 * nside - 1) >> 1

    # local coordinates (w/ origin in S corner)
    fx = x % nside
    fy = (-y - 1) % nside

    x_block = x // nside
    y_block = y // nside

    # north
    face = xp.where(x_block > y_block, y_block, 0)
    # equator
    face = xp.where(x_block == y_block, x_block % 4 + 4, face)
    # south
    face = xp.where(x_block < y_block, x_block % 4 + 8, face)
    return face * nside**2 + fy * nside + fx


def ring2double(nside: int, p: ArrayT):
    """Compute the (i,j) index in the double pixelization scheme of Calabretta (2007)

    This is a visually appealing way to visualize healpix data without any
    interpolation.

    See Fig 5

    Calabretta, M. R., & Roukema, B. F. (2007). Mapping on the HEALPix grid. Monthly Notices of the Royal Astronomical Society, 381(2), 865–872. https://doi.org/10.1111/j.1365-2966.2007.12297.x

    """
    numpy = False
    if isinstance(p, np.ndarray):
        numpy = True
        p = torch.from_numpy(p)

    n = nside
    i, j = _pixels_to_rings(n, p)
    n_per_pyramid = ring_length(n, i) // 4

    pyramid = j // n_per_pyramid
    left = n - i
    jp_north = 2 * pyramid * n + left + 2 * (j % n_per_pyramid)
    jp_eq = (n - i) % 2 + 2 * j

    left = i - 3 * n + 2
    jp_south = 2 * pyramid * n + left + 2 * (j % n_per_pyramid)

    jp = torch.where(i >= n, jp_eq, jp_north)
    jp = torch.where(i >= 3 * n, jp_south, jp)

    if numpy:
        i = i.numpy()
        jp = jp.numpy()

    return i + 1, jp


def ring2xy(nside, pix):
    i, j = ring2double(nside, pix)
    return double2xy(nside, i, j)


def ring2nest(nside, pix):
    return xy2nest(nside, ring2xy(nside, pix))


def to_rotated_pixelization(x, fill_value=math.nan):
    """Convert an array to a 2D-iamge w/ the rotated pixelization"""

    numpy_out = False
    if isinstance(x, np.ndarray):
        numpy_out = True
        x = torch.from_numpy(x)

    grid = [[6, 9, -1, -1, -1], [1, 5, 8, -1, -1], [-1, 0, 4, 11, -1], [-1, -1, 3, 7, 10], [-1, -1, -1, 2, 6]]
    pixel_order = XY(origin=Compass.W, clockwise=True)
    self = Grid(npix2level(x.shape[-1]))
    x = self.reorder(pixel_order, x)
    nside = self._nside()
    *shape, _ = x.shape
    x = x.reshape((*shape, 12, nside, nside))
    output = torch.full((*shape, 5 * nside, 5 * nside), device=x.device, dtype=x.dtype, fill_value=fill_value)

    for j in range(len(grid)):
        for i in range(len(grid[0])):
            face = grid[j][i]
            if face != -1:
                output[j * nside : (j + 1) * nside, i * nside : (i + 1) * nside] = x[face]

    if numpy_out:
        return output.numpy()
    else:
        return output


def _arange_like(x, dim, dtype=torch.int64):
    n = x.shape[dim]
    if isinstance(x, np.ndarray):
        batch = np.arange(n)
    else:
        batch = torch.arange(n, device=x.device, dtype=dtype)
    return batch


def _zeros_like(x, shape=None, dtype=None):
    if isinstance(x, np.ndarray):
        return np.zeros_like(x, shape=shape, dtype=dtype)
    else:
        return torch.zeros(shape or x.shape, dtype=dtype, device=x.device)


def to_double_pixelization(x: ArrayT, fill_value=0) -> ArrayT:
    """Convert the array x to 2D-image w/ the double pixelization

    ``x`` must be in RING pixel order

    """
    numpy = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        numpy = True

    dtype = torch.float32

    n = npix2nside(x.shape[-1])
    i, jp = ring2double(n, _arange_like(x, dim=-1))
    out = _zeros_like(x, shape=x.shape[:-1] + (4 * n, 8 * n + 1), dtype=dtype)
    num = _zeros_like(out, dtype=torch.int32)

    x = x.to(out)

    out[i, jp] = x
    num[i, jp] += 1

    out[i, jp + 1] = x
    num[i, jp + 1] += 1

    out[i, jp - 1] += x
    num[i, jp - 1] += 1
    out[num == 0] = fill_value
    num[num == 0] = 1
    out /= num

    if numpy:
        out = out.numpy()

    return out


def zonal_average(x: ArrayT, dim=-1) -> ArrayT:
    """Compute the zonal average of a map in ring format"""
    numpy = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        numpy = True

    dim = dim % x.ndim
    shape = [x.shape[i] for i in range(x.ndim) if i != dim]
    x = torch.moveaxis(x, dim, -1)
    x = x.reshape([-1, x.shape[-1]])

    npix = x.shape[-1]
    nside = npix2nside(npix)

    iring, _ = _pixels_to_rings(nside, _arange_like(x, dim=-1))
    nring = iring.max() + 1
    batch = _arange_like(x, dim=0)

    i_flat = batch[:, None] * nring + iring
    i_flat = i_flat.ravel()
    num = torch.bincount(i_flat, weights=x.ravel(), minlength=nring * x.shape[0])
    denom = torch.bincount(i_flat, minlength=nring * x.shape[0])
    average = num / denom
    average = average.reshape((*shape, nring))  # type: ignore
    out = torch.moveaxis(average, -1, dim)

    if numpy:
        out = out.numpy()
    return out
