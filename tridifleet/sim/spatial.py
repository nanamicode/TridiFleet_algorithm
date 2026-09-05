from __future__ import annotations

import math
from collections import defaultdict

from .domain import Person


class SpatialIndex:
    def __init__(self, people: dict[int, Person], cell_km: float = 0.08):
        self.cell_km = cell_km
        self.cells: dict[tuple[int, int], list[Person]] = defaultdict(list)
        for p in people.values():
            self.cells[self._cell(p.x, p.y)].append(p)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self.cell_km), math.floor(y / self.cell_km))

    def query(self, x: float, y: float, radius_km: float) -> list[tuple[Person, float]]:
        cx, cy = self._cell(x, y)
        reach = math.ceil(radius_km / self.cell_km) + 1
        out: list[tuple[Person, float]] = []
        r2 = radius_km * radius_km
        for ix in range(cx - reach, cx + reach + 1):
            for iy in range(cy - reach, cy + reach + 1):
                for p in self.cells.get((ix, iy), ()):
                    dx, dy = p.x - x, p.y - y
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        out.append((p, math.sqrt(d2)))
        return out
