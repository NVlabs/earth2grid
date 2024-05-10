import torch

from earth2grid import _healpix_bare
from earth2grid._healpix_bare import corners, hpc2loc, hpd2loc, nest2hpd, nest2ring, ring2hpd, ring2nest

__all__ = [
    "pix2ang",
    "ring2nest",
    "nest2ring",
    "hpc2loc",
    "corners",
]


def pix2ang(nside, i, nest=False, lonlat=False):
    if nest:
        hpd = nest2hpd(nside, i)
    else:
        hpd = ring2hpd(nside, i)
    loc = hpd2loc(nside, hpd)
    lon, lat = _loc2ang(loc)

    if lonlat:
        return torch.rad2deg(lon), 90 - torch.rad2deg(lat)
    else:
        return lat, lon


def _loc2ang(loc):
    """
    static t_ang loc2ang(tloc loc)
    { return (t_ang){atan2(loc.s,loc.z), loc.phi}; }
    """
    z = loc[..., 0]
    s = loc[..., 1]
    phi = loc[..., 2]
    return phi % (2 * torch.pi), torch.atan2(s, z)


def loc2vec(loc):
    z = loc[..., 0]
    s = loc[..., 1]
    phi = loc[..., 2]
    x = (s * torch.cos(phi),)
    y = (s * torch.sin(phi),)
    return x, y, z


def get_interp_weights(nside: int, lon: torch.Tensor, lat: torch.Tensor):
    """

    Args:
        lon: longtiude in deg E. Shape (*)
        lat: latitdue in deg E. Shape (*)

    Returns:
        pix, weights: both shaped (4, *). pix is given in RING convention.

    """
    shape = lon.shape
    lon = lon.double().cpu().flatten()
    lat = lat.double().cpu().flatten()
    pix, weights = _healpix_bare.get_interp_weights(nside, lon, lat)

    return pix.view(4, *shape), weights.view(4, *shape)
