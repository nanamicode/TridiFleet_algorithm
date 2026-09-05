from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .domain import Road, Zone, TotemState


@dataclass
class CityMap:
    radius_km: float
    roads: list[Road]
    zones: list[Zone]
    intersections: list[tuple[float, float]]
    totems: list[TotemState]

    @staticmethod
    def generate(radius_km: float, n_totems: int, seed: int) -> "CityMap":
        rng = random.Random(seed)
        line_count = max(8, min(40, math.ceil(math.sqrt(n_totems * 1.7)) + 5))

        def make_coords() -> list[float]:
            coords = {0.0}
            while len(coords) < line_count:
                # Triangular distribution creates a denser downtown core.
                v = rng.triangular(-0.92 * radius_km, 0.92 * radius_km, 0.0)
                coords.add(round(v, 3))
            return sorted(coords)

        xs = make_coords()
        ys = make_coords()
        roads: list[Road] = []

        for i, x in enumerate(xs):
            extent = math.sqrt(max(0.0, radius_km * radius_km - x * x))
            kind = (
                "arterial"
                if abs(x) < radius_km * 0.13 or i % 7 == 0
                else "collector" if i % 3 == 0 else "local"
            )
            roads.append(Road(f"V{i}", x, -extent, x, extent, kind))

        for i, y in enumerate(ys):
            extent = math.sqrt(max(0.0, radius_km * radius_km - y * y))
            kind = (
                "arterial"
                if abs(y) < radius_km * 0.13 or i % 7 == 0
                else "collector" if i % 3 == 0 else "local"
            )
            roads.append(Road(f"H{i}", -extent, y, extent, y, kind))

        intersections = [
            (x, y)
            for x in xs
            for y in ys
            if x * x + y * y <= (radius_km * 0.94) ** 2
        ]

        zone_kinds = [
            "downtown",
            "commercial",
            "residential",
            "office",
            "leisure",
            "transit",
            "school",
            "health",
        ]
        zones: list[Zone] = []
        zone_count = max(8, min(28, int(radius_km * 2.5) + 7))
        for i in range(zone_count):
            x, y = rng.choice(intersections)
            if i == 0:
                x, y, kind, intensity = 0.0, 0.0, "downtown", 1.65
            else:
                kind = rng.choices(
                    zone_kinds,
                    weights=[8, 18, 25, 12, 10, 10, 8, 9],
                    k=1,
                )[0]
                intensity = {
                    "downtown": 1.5,
                    "commercial": 1.35,
                    "residential": 0.8,
                    "office": 1.15,
                    "leisure": 1.0,
                    "transit": 1.45,
                    "school": 0.9,
                    "health": 0.95,
                }[kind] * rng.uniform(0.85, 1.15)
            zones.append(
                Zone(
                    zone_id=f"zone-{i+1:02d}",
                    kind=kind,
                    x=x,
                    y=y,
                    radius_km=rng.uniform(0.25, min(1.1, radius_km * 0.35)),
                    intensity=intensity,
                )
            )

        def point_score(p: tuple[float, float]) -> float:
            x, y = p
            center = math.exp(-math.hypot(x, y) / max(0.5, radius_km * 0.55))
            zone_score = 0.0
            for z in zones:
                d = math.hypot(x - z.x, y - z.y)
                zone_score += z.intensity * math.exp(-d / max(0.15, z.radius_km))
            return 0.2 + center + zone_score

        available = list(intersections)
        selected: list[tuple[float, float]] = []
        for _ in range(min(n_totems, len(available))):
            weights = [point_score(p) for p in available]
            p = rng.choices(available, weights=weights, k=1)[0]
            selected.append(p)
            available.remove(p)

        # In the unlikely case of more requested totens than intersections,
        # place additional units with tiny offsets near high-value intersections.
        while len(selected) < n_totems:
            bx, by = rng.choice(intersections)
            selected.append(
                (
                    max(-radius_km, min(radius_km, bx + rng.uniform(-0.015, 0.015))),
                    max(-radius_km, min(radius_km, by + rng.uniform(-0.015, 0.015))),
                )
            )

        def region_for(x: float, y: float) -> str:
            if math.hypot(x, y) < radius_km * 0.24:
                return "centro"
            if abs(y) >= abs(x):
                return "norte" if y >= 0 else "sul"
            return "leste" if x >= 0 else "oeste"

        totems = [
            TotemState(
                totem_id=f"T{i+1:03d}",
                x=x,
                y=y,
                region=region_for(x, y),
                location_id=f"loc-{i+1:03d}",
            )
            for i, (x, y) in enumerate(selected)
        ]
        return CityMap(radius_km, roads, zones, intersections, totems)

    def weighted_intersection(self, rng: random.Random, hour: float) -> tuple[float, float]:
        def zone_time_multiplier(kind: str) -> float:
            if kind == "office":
                return 1.7 if 7 <= hour <= 10 or 16 <= hour <= 19 else 0.65
            if kind == "residential":
                return 1.25 if hour < 8 or hour >= 18 else 0.75
            if kind == "commercial":
                return 1.4 if 10 <= hour <= 20 else 0.55
            if kind == "leisure":
                return 1.55 if 17 <= hour <= 23 else 0.6
            if kind == "school":
                return 1.55 if 6.5 <= hour <= 8.5 or 11 <= hour <= 13.5 or 16 <= hour <= 18 else 0.45
            if kind == "transit":
                return 1.8 if 6.5 <= hour <= 9.5 or 16 <= hour <= 20 else 0.8
            return 1.0

        sample = rng.sample(self.intersections, k=min(48, len(self.intersections)))
        weights = []
        for x, y in sample:
            w = 0.2 + math.exp(-math.hypot(x, y) / max(0.4, self.radius_km * 0.6))
            for z in self.zones:
                d = math.hypot(x - z.x, y - z.y)
                w += (
                    z.intensity
                    * zone_time_multiplier(z.kind)
                    * math.exp(-d / max(0.12, z.radius_km))
                )
            weights.append(w)
        return rng.choices(sample, weights=weights, k=1)[0]

    def serialize(self) -> dict:
        return {
            "radius_km": self.radius_km,
            "roads": [
                {
                    "id": r.road_id,
                    "x1": r.x1,
                    "y1": r.y1,
                    "x2": r.x2,
                    "y2": r.y2,
                    "kind": r.kind,
                }
                for r in self.roads
            ],
            "zones": [
                {
                    "id": z.zone_id,
                    "kind": z.kind,
                    "x": z.x,
                    "y": z.y,
                    "radius_km": z.radius_km,
                    "intensity": z.intensity,
                }
                for z in self.zones
            ],
        }
