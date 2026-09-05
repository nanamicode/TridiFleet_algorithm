from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Road:
    road_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    kind: Literal["arterial", "collector", "local"]


@dataclass(frozen=True)
class Zone:
    zone_id: str
    kind: str
    x: float
    y: float
    radius_km: float
    intensity: float


@dataclass
class Person:
    person_id: int
    x: float
    y: float
    age: int
    female: bool
    speed_kmh: float
    interests: set[str]
    attention_propensity: float
    route: list[tuple[float, float]]
    route_index: int = 1
    ttl_seconds: float = 1800.0
    prev_x: float = 0.0
    prev_y: float = 0.0

    def advance(self, distance_km: float) -> bool:
        self.prev_x, self.prev_y = self.x, self.y
        remaining = max(0.0, distance_km)

        while remaining > 0 and self.route_index < len(self.route):
            tx, ty = self.route[self.route_index]
            dx, dy = tx - self.x, ty - self.y
            segment = (dx * dx + dy * dy) ** 0.5
            if segment <= 1e-9:
                self.route_index += 1
                continue
            if remaining >= segment:
                self.x, self.y = tx, ty
                remaining -= segment
                self.route_index += 1
            else:
                ratio = remaining / segment
                self.x += dx * ratio
                self.y += dy * ratio
                remaining = 0.0
        return self.route_index >= len(self.route)


@dataclass
class TotemState:
    totem_id: str
    x: float
    y: float
    region: str
    location_id: str
    current_ad_id: str | None = None
    current_decision_id: str | None = None
    last_decision_at: datetime | None = None
    last_reach: int = 0
    last_impressions: int = 0
    last_avg_view_seconds: float = 0.0
    last_completion_rate: float = 0.0
    last_reward: float = 0.0
    female_share: float = 0.5
    mean_age: float = 38.0
    flow_per_minute: float = 0.0
    plays: int = 0


@dataclass(frozen=True)
class SimConfig:
    n_totems: int = 50
    radius_km: float = 4.0
    seed: int = 42
    base_sim_seconds_per_real_second: float = 120.0
    decision_interval_sim_seconds: float = 60.0
    detection_radius_km: float = 0.025

    def validate(self) -> "SimConfig":
        if not 1 <= self.n_totems <= 500:
            raise ValueError("n_totems must be between 1 and 500")
        if not 0.5 <= self.radius_km <= 25:
            raise ValueError("radius_km must be between 0.5 and 25")
        return self


@dataclass
class MetricPoint:
    timestamp: datetime
    observed_reward: float
    random_baseline: float
    oracle_ceiling: float
    uplift_vs_random: float
    model_uncertainty: float
    people: int
    decisions: int
