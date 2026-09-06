from __future__ import annotations

import math
import random
from datetime import datetime

from ..models import ContextEvent, Feedback
from .domain import Person, TotemState


class SimulatedTotemSensor:
    """Converts hidden pedestrians into noisy telemetry like an edge device would."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed + 707)

    def context(
        self,
        totem: TotemState,
        audience: list[tuple[Person, float]],
        timestamp: datetime,
        previous: Feedback | None,
        detection_radius_km: float,
        decision_interval_seconds: float,
    ) -> ContextEvent:
        detected: list[Person] = []
        for person, _ in audience:
            # Reach detection is high but not perfect.
            if self.rng.random() < 0.965:
                detected.append(person)

        ages: list[float] = []
        female_count = 0
        for person in detected:
            # Approximate CV measurement error, independent of the decision engine.
            ages.append(max(8.0, min(90.0, self.rng.gauss(person.age, 4.3))))
            observed_female = person.female
            if self.rng.random() < 0.035:
                observed_female = not observed_female
            female_count += int(observed_female)

        reach = len(detected)
        mean_age = sum(ages) / len(ages) if ages else None
        age_std = None
        if len(ages) >= 2 and mean_age is not None:
            age_std = math.sqrt(sum((a - mean_age) ** 2 for a in ages) / len(ages))

        female_share = female_count / reach if reach else None
        age_distribution: dict[str, float] = {}
        if ages:
            counts = {"u18": 0, "18-24": 0, "25-34": 0, "35-44": 0, "45-59": 0, "60+": 0}
            for age in ages:
                if age < 18:
                    key = "u18"
                elif age < 25:
                    key = "18-24"
                elif age < 35:
                    key = "25-34"
                elif age < 45:
                    key = "35-44"
                elif age < 60:
                    key = "45-59"
                else:
                    key = "60+"
                counts[key] += 1
            age_distribution = {k: v / len(ages) for k, v in counts.items()}

        previous_impressions = previous.impressions if previous else 0
        flow_per_minute = reach / max(1e-6, decision_interval_seconds / 60.0)
        area_m2 = math.pi * (detection_radius_km * 1000.0) ** 2
        crowd_density = reach / max(1.0, area_m2 / 100.0)

        return ContextEvent(
            totem_id=totem.totem_id,
            timestamp=timestamp,
            location_id=totem.location_id,
            region=totem.region,
            x_km=totem.x,
            y_km=totem.y,
            reach_window=reach,
            impressions_window=previous_impressions,
            female_share=female_share,
            mean_age=mean_age,
            age_std=age_std,
            age_distribution=age_distribution,
            flow_per_minute=flow_per_minute,
            crowd_density=crowd_density,
        )
