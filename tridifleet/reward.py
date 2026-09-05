from __future__ import annotations

import math

from .config import settings
from .models import Feedback


def retention_reward(feedback: Feedback) -> float:
    """Return a bounded [0,1] attention reward.

    We combine:
    - look-through rate: impressions / reach
    - normalized continuous viewing time: avg_view_seconds / ad duration

    The weighted geometric form prevents a great dwell time on a tiny number of
    viewers from hiding poor attention capture, while not crushing the signal as
    aggressively as a raw multiplication.
    """
    if feedback.reach <= 0:
        return 0.0

    impression_rate = min(1.0, feedback.impressions / feedback.reach)
    dwell_ratio = min(1.0, feedback.avg_view_seconds / feedback.ad_duration_seconds)

    if impression_rate <= 0 or dwell_ratio <= 0:
        return 0.0

    reward = (
        impression_rate ** settings.impression_exponent
        * dwell_ratio ** settings.dwell_exponent
    )
    return float(max(0.0, min(1.0, reward)))


def evidence_weight(feedback: Feedback) -> float:
    """Bound how much one aggregate window is allowed to move a posterior."""
    if feedback.reach <= 0:
        return 0.0
    return min(settings.max_evidence_weight, max(1.0, math.sqrt(feedback.reach)))
