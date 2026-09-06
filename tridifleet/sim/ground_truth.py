from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime

from ..models import Ad, Feedback
from ..taxonomy import canonical_tag, canonical_tags
from .domain import Person


TAG_INTERESTS = {
    "food": ("food",),
    "grocery": ("food", "home", "family"),
    "coffee": ("coffee", "food"),
    "personal_care": ("health", "home"),
    "beauty": ("beauty", "fashion"),
    "health": ("health",),
    "fitness": ("fitness", "health"),
    "pets": ("pets", "family"),
    "family": ("family",),
    "children": ("family", "education"),
    "fashion": ("fashion",),
    "technology": ("technology",),
    "cars": ("cars",),
    "home": ("home",),
    "education": ("education",),
    "finance": ("finance",),
    "travel": ("travel",),
    "entertainment": ("entertainment",),
    "services": ("services",),
}


@dataclass(frozen=True)
class HiddenCreativeProfile:
    quality: float
    target_age: float
    age_width: float
    female_bias: float
    peak_hour: float
    interest_weights: dict[str, float]


def _sigmoid(v: float) -> float:
    if v >= 0:
        z = math.exp(-v)
        return 1.0 / (1.0 + z)
    z = math.exp(v)
    return z / (1.0 + z)


class GroundTruthModel:
    """Private simulator physics.

    IMPORTANT: the intelligence module never imports or receives this object.
    It is the hidden world that produces outcomes after an ad decision.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._profiles: dict[str, HiddenCreativeProfile] = {}

    def _rng_for_ad(self, ad: Ad) -> random.Random:
        raw = f"{self.seed}|{ad.ad_id}|{ad.name}|{'|'.join(sorted(ad.tags))}"
        digest = hashlib.blake2b(raw.encode(), digest_size=8).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def profile(self, ad: Ad) -> HiddenCreativeProfile:
        cached = self._profiles.get(ad.ad_id)
        if cached:
            return cached

        rng = self._rng_for_ad(ad)
        tags = canonical_tags(ad.tags)
        if ad.category:
            tags.add(canonical_tag(ad.category))

        all_interests = {
            "food", "coffee", "beauty", "health", "fitness", "pets", "family",
            "fashion", "technology", "cars", "home", "education", "finance",
            "travel", "entertainment", "services",
        }
        interest_weights = {k: rng.uniform(-0.12, 0.18) for k in all_interests}
        mapped = {interest for tag in tags for interest in TAG_INTERESTS.get(tag, ())}
        for interest in mapped:
            interest_weights[interest] = rng.uniform(0.65, 1.20)

        female_bias = rng.uniform(-0.18, 0.18)
        if "women" in tags or "beauty" in mapped:
            female_bias += rng.uniform(0.22, 0.52)
        if "men" in tags or "cars" in mapped:
            female_bias -= rng.uniform(0.15, 0.40)

        target_age = rng.uniform(22, 58)
        if "young" in tags:
            target_age = rng.uniform(19, 31)
        elif "children" in tags:
            target_age = rng.uniform(28, 40)
        elif "finance" in mapped or "home" in mapped:
            target_age = rng.uniform(32, 52)

        peak_hour = rng.uniform(9, 21)
        if "morning" in tags or "breakfast" in tags or "coffee" in mapped:
            peak_hour = rng.uniform(7, 10)
        elif "evening" in tags:
            peak_hour = rng.uniform(18, 21.5)
        elif "lunch" in tags:
            peak_hour = rng.uniform(11.5, 14)

        profile = HiddenCreativeProfile(
            quality=rng.gauss(0.0, 0.32),
            target_age=target_age,
            age_width=rng.uniform(12, 25),
            female_bias=female_bias,
            peak_hour=peak_hour,
            interest_weights=interest_weights,
        )
        self._profiles[ad.ad_id] = profile
        return profile

    def _affinity(self, person: Person, ad: Ad, timestamp: datetime) -> float:
        p = self.profile(ad)
        interest = sum(p.interest_weights.get(i, 0.0) for i in person.interests)
        interest /= max(1.6, len(person.interests) ** 0.5)
        gender_term = p.female_bias * (1.0 if person.female else -1.0)
        age_term = math.exp(-abs(person.age - p.target_age) / p.age_width) - 0.45

        hour = timestamp.hour + timestamp.minute / 60.0
        circular = abs(hour - p.peak_hour)
        circular = min(circular, 24.0 - circular)
        time_term = math.exp(-(circular / 4.0) ** 2) - 0.35

        duration_penalty = max(0.0, ad.duration_seconds - 12.0) * 0.025
        return (
            p.quality
            + interest * 0.95
            + gender_term
            + age_term * 0.75
            + time_term * 0.48
            - duration_penalty
        )

    def person_expectation(
        self,
        person: Person,
        distance_km: float,
        detection_radius_km: float,
        ad: Ad,
        timestamp: datetime,
        crowd_size: int,
    ) -> tuple[float, float, float]:
        affinity = self._affinity(person, ad, timestamp)
        proximity = 1.0 - min(1.0, distance_km / max(1e-6, detection_radius_km))
        crowd_penalty = min(0.55, max(0, crowd_size - 5) * 0.018)

        gaze_logit = (
            -1.85
            + 1.05 * affinity
            + 0.85 * proximity
            + 0.55 * person.attention_propensity
            - crowd_penalty
        )
        gaze_probability = max(0.01, min(0.93, _sigmoid(gaze_logit)))

        retention_logit = (
            -0.55
            + 1.20 * affinity
            + 0.42 * person.attention_propensity
            + 0.30 * proximity
        )
        retention_ratio = max(0.04, min(0.98, _sigmoid(retention_logit)))
        completion_probability = max(
            0.0, min(1.0, _sigmoid((retention_ratio - 0.74) * 9.0))
        )
        return gaze_probability, retention_ratio, completion_probability

    def expected_reward(
        self,
        audience: list[tuple[Person, float]],
        ad: Ad,
        timestamp: datetime,
        detection_radius_km: float,
    ) -> float:
        if not audience:
            return 0.0
        gaze_sum = 0.0
        dwell_weighted = 0.0
        completion_weighted = 0.0
        for person, distance in audience:
            gaze, retention, completion = self.person_expectation(
                person, distance, detection_radius_km, ad, timestamp, len(audience)
            )
            gaze_sum += gaze
            dwell_weighted += gaze * retention
            completion_weighted += gaze * completion
        capture = gaze_sum / len(audience)
        if gaze_sum <= 1e-9:
            return 0.0
        dwell = dwell_weighted / gaze_sum
        completion = completion_weighted / gaze_sum
        return float(
            (max(capture, 1e-6) ** 0.35)
            * (max(dwell, 1e-6) ** 0.50)
            * (max(completion, 1e-6) ** 0.15)
        )

    def simulate_feedback(
        self,
        decision_id: str,
        audience: list[tuple[Person, float]],
        ad: Ad,
        timestamp: datetime,
        detection_radius_km: float,
        rng: random.Random,
    ) -> Feedback:
        reach = len(audience)
        dwell_seconds: list[float] = []
        completions = 0

        for person, distance in audience:
            gaze_p, retention_ratio, completion_p = self.person_expectation(
                person, distance, detection_radius_km, ad, timestamp, reach
            )
            if rng.random() > gaze_p:
                continue

            # Individual dwell is noisy, but centered on the hidden expected response.
            ratio = max(0.02, min(1.0, rng.gauss(retention_ratio, 0.13)))
            dwell = ratio * ad.duration_seconds
            dwell_seconds.append(dwell)
            if ratio >= 0.85 or rng.random() < completion_p * 0.18:
                completions += 1

        impressions = len(dwell_seconds)
        avg_view = sum(dwell_seconds) / impressions if impressions else 0.0
        completion_rate = completions / impressions if impressions else 0.0
        return Feedback(
            decision_id=decision_id,
            reach=reach,
            impressions=impressions,
            avg_view_seconds=avg_view,
            ad_duration_seconds=ad.duration_seconds,
            completion_rate=completion_rate,
        )
