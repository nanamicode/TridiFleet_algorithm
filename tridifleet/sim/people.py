from __future__ import annotations

import math
import random
from datetime import datetime

from .city import CityMap
from .domain import Person


INTERESTS = [
    "food",
    "coffee",
    "beauty",
    "health",
    "fitness",
    "pets",
    "family",
    "fashion",
    "technology",
    "cars",
    "home",
    "education",
    "finance",
    "travel",
    "entertainment",
    "services",
]


class PopulationEngine:
    def __init__(self, city: CityMap, seed: int):
        self.city = city
        self.rng = random.Random(seed + 101)
        self.people: dict[int, Person] = {}
        self.next_id = 1

    def _day_activity(self, timestamp: datetime) -> float:
        h = timestamp.hour + timestamp.minute / 60.0
        # Smooth urban pedestrian profile with morning/lunch/evening peaks.
        morning = math.exp(-((h - 8.0) / 1.6) ** 2) * 0.70
        lunch = math.exp(-((h - 12.7) / 2.1) ** 2) * 0.75
        evening = math.exp(-((h - 18.2) / 2.2) ** 2) * 1.00
        night = math.exp(-((h - 22.0) / 2.8) ** 2) * 0.35
        floor = 0.08
        weekend = 0.88 if timestamp.weekday() >= 5 else 1.0
        return max(0.06, (floor + morning + lunch + evening + night) * weekend)

    def target_population(self, timestamp: datetime) -> int:
        area = math.pi * self.city.radius_km * self.city.radius_km
        nominal = min(6500, max(180, area * 24.0 + len(self.city.totems) * 5.0))
        return int(max(60, nominal * self._day_activity(timestamp)))

    def _sample_age(self) -> int:
        r = self.rng.random()
        if r < 0.08:
            return self.rng.randint(12, 17)
        if r < 0.45:
            return self.rng.randint(18, 34)
        if r < 0.78:
            return self.rng.randint(35, 54)
        if r < 0.94:
            return self.rng.randint(55, 69)
        return self.rng.randint(70, 84)

    def _sample_interests(self, age: int, female: bool) -> set[str]:
        weights = {k: 1.0 for k in INTERESTS}
        if age < 25:
            weights["technology"] += 1.0
            weights["fashion"] += 0.8
            weights["entertainment"] += 1.2
            weights["education"] += 0.9
        elif age < 45:
            weights["family"] += 1.0
            weights["food"] += 0.7
            weights["finance"] += 0.6
            weights["home"] += 0.7
        else:
            weights["health"] += 1.2
            weights["home"] += 0.8
            weights["finance"] += 0.8
        if female:
            weights["beauty"] += 0.65
            weights["fashion"] += 0.35
        else:
            weights["cars"] += 0.45
            weights["technology"] += 0.25

        pool = list(weights)
        chosen: set[str] = set()
        for _ in range(self.rng.randint(2, 5)):
            chosen.add(self.rng.choices(pool, weights=[weights[k] for k in pool], k=1)[0])
        return chosen

    def _route(self, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
        sx, sy = start
        ex, ey = end
        if self.rng.random() < 0.5:
            bend = (ex, sy)
        else:
            bend = (sx, ey)
        return [start, bend, end]

    def _spawn_person(self, timestamp: datetime) -> Person:
        hour = timestamp.hour + timestamp.minute / 60.0
        start = self.city.weighted_intersection(self.rng, hour)
        end = self.city.weighted_intersection(self.rng, hour)
        tries = 0
        while end == start and tries < 8:
            end = self.city.weighted_intersection(self.rng, hour)
            tries += 1

        age = self._sample_age()
        female = self.rng.random() < 0.505
        person = Person(
            person_id=self.next_id,
            x=start[0],
            y=start[1],
            prev_x=start[0],
            prev_y=start[1],
            age=age,
            female=female,
            speed_kmh=max(2.4, min(7.2, self.rng.gauss(4.8, 0.8))),
            interests=self._sample_interests(age, female),
            attention_propensity=max(-1.5, min(1.5, self.rng.gauss(0.0, 0.55))),
            route=self._route(start, end),
            ttl_seconds=self.rng.uniform(900, 4200),
        )
        self.next_id += 1
        return person

    def _reroute(self, person: Person, timestamp: datetime) -> None:
        hour = timestamp.hour + timestamp.minute / 60.0
        start = (person.x, person.y)
        end = self.city.weighted_intersection(self.rng, hour)
        person.route = self._route(start, end)
        person.route_index = 1
        person.ttl_seconds += self.rng.uniform(300, 1200)

    def update(self, dt_seconds: float, timestamp: datetime) -> None:
        target = self.target_population(timestamp)
        current = len(self.people)
        max_adjust = max(4, int(max(target, current) * 0.035))

        if current < target:
            for _ in range(min(target - current, max_adjust)):
                p = self._spawn_person(timestamp)
                self.people[p.person_id] = p
        elif current > target:
            remove_n = min(current - target, max_adjust)
            for pid in self.rng.sample(list(self.people), k=remove_n):
                self.people.pop(pid, None)

        to_remove: list[int] = []
        for person in list(self.people.values()):
            person.ttl_seconds -= dt_seconds
            arrived = person.advance(person.speed_kmh * dt_seconds / 3600.0)
            if arrived:
                if person.ttl_seconds <= 0 or self.rng.random() < 0.28:
                    to_remove.append(person.person_id)
                else:
                    self._reroute(person, timestamp)
            elif person.ttl_seconds <= -600:
                to_remove.append(person.person_id)

        for pid in to_remove:
            self.people.pop(pid, None)

    def serialized_points(self, limit: int = 4000) -> list[dict]:
        values = list(self.people.values())
        if len(values) > limit:
            values = self.rng.sample(values, k=limit)
        return [
            {
                "id": p.person_id,
                "x": round(p.x, 4),
                "y": round(p.y, 4),
                "age": p.age,
                "gender": "F" if p.female else "M",
            }
            for p in values
        ]
