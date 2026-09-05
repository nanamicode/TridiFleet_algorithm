from __future__ import annotations

import math

from .config import settings
from .models import Feedback


def retention_reward(feedback: Feedback) -> float:
    """Bounded attention reward built only from observed telemetry.

    Capture (looking at all), depth (continuous viewing) and completion are kept
    separate so an ad cannot look excellent merely because a tiny audience
    watched for a long time.
    """
    if feedback.reach <= 0:
        return 0.0

    capture = min(1.0, feedback.impressions / feedback.reach)
    dwell = min(1.0, feedback.avg_view_seconds / feedback.ad_duration_seconds)
    completion = feedback.completion_rate
    if completion is None:
        completion = max(0.0, min(1.0, (dwell - 0.55) / 0.45))

    if capture <= 0 or dwell <= 0:
        return 0.0

    # Weighted geometric objective. Sustained attention is the primary goal.
    reward = (capture ** 0.35) * (dwell ** 0.50) * (max(completion, 1e-6) ** 0.15)
    return float(max(0.0, min(1.0, reward)))


def evidence_weight(feedback: Feedback) -> float:
    """Bound the statistical influence of one aggregate observation window."""
    if feedback.reach <= 0:
        return 0.0
    return min(settings.max_evidence_weight, max(1.0, math.sqrt(feedback.reach)))
