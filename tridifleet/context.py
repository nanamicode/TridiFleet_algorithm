from __future__ import annotations

from datetime import datetime

from .models import ContextEvent


def _daypart(ts: datetime) -> str:
    h = ts.hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "afternoon"
    if 17 <= h < 23:
        return "evening"
    return "night"


def _age_bucket(age: float | None) -> str:
    if age is None:
        return "age_unknown"
    if age < 18:
        return "age_u18"
    if age < 25:
        return "age_18_24"
    if age < 35:
        return "age_25_34"
    if age < 45:
        return "age_35_44"
    if age < 60:
        return "age_45_59"
    return "age_60_plus"


def _gender_bucket(female_share: float | None) -> str:
    if female_share is None:
        return "gender_unknown"
    if female_share >= 0.60:
        return "female_skew"
    if female_share <= 0.40:
        return "male_skew"
    return "gender_balanced"


def context_keys(event: ContextEvent) -> list[str]:
    """Hierarchical contexts ordered from broadest to most specific."""
    daypart = _daypart(event.timestamp)
    gender = _gender_bucket(event.female_share)
    if event.age_distribution:
        dominant = max(event.age_distribution.items(), key=lambda item: item[1])[0]
        age = {
            "u18": "age_u18",
            "18-24": "age_18_24",
            "25-34": "age_25_34",
            "35-44": "age_35_44",
            "45-59": "age_45_59",
            "60+": "age_60_plus",
        }.get(dominant, _age_bucket(event.mean_age))
    else:
        age = _age_bucket(event.mean_age)

    keys = [
        "global",
        f"time:{daypart}",
        f"demo:{gender}:{age}",
        f"time_demo:{daypart}:{gender}:{age}",
    ]
    if event.region:
        keys.append(f"region:{event.region}")
        keys.append(f"region_time:{event.region}:{daypart}")
    if event.location_id:
        keys.append(f"location:{event.location_id}")
    keys.append(f"totem:{event.totem_id}")
    return keys
