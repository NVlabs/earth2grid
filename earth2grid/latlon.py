import numpy as np

from earth2grid import base


class LatLonGrid(base.Grid):
    def __init__(self, lat: list[float], lon: list[float]):
        self._lat = lat
        self._lon = lon

    @property
    def lat(self):
        return np.array(self._lat)[None, :]

    @property
    def lon(self):
        return np.array(self._lat)[:, None]

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))

    def visualize(self, data):
        raise NotImplementedError()


def equiangular_lat_lon_grid(nlat: int, nlon: int, includes_south_pole: bool = True) -> LatLonGrid:
    """A regular lat-lon grid

    Lat is ordered from 90 to -90. Includes -90 and only if if includes_south_pole is True.
    Lon is ordered from 0 to 360. includes 0, but not 360.

    """  # noqa
    lat = np.linspace(90, -90, nlat, endpoint=includes_south_pole)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return LatLonGrid(lat.tolist(), lon.tolist())
