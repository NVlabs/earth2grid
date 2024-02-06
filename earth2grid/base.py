from typing import Protocol


class Grid(Protocol):
    """lat and lon should be broadcastable arrays"""

    @property
    def lat(self):
        pass

    @property
    def lon(self):
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        pass

    def visualize(self, data):
        pass
