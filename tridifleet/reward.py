from __future__ import annotations

import math

from .config import settings
from .models import Feedback


REWARD_VERSION = "visual_attention_v2"


def retention_reward(feedback: Feedback) -> float:
    """Bounded visual-attention reward built only from observed telemetry.

    The primary product objective is:
      1) capture a passerby's direct gaze;
      2) sustain that gaze for a large fraction of the creative.

    Completion remains an audited diagnostic but is not part of the optimizer:
    on low-footfall physical media it is too sparse/noisy and would make a
    one-person slot swing the posterior excessively.
    """
    if feedback.reach <= 0:
        return 0.0

    capture = min(1.0, feedback.impressions / feedback.reach)
    dwell = min(1.0, feedback.avg_view_seconds / feedback.ad_duration_seconds)
    if capture <= 0 or dwell <= 0:
        return 0.0

    reward = (
        capture ** settings.impression_exponent
        * dwell ** settings.dwell_exponent
    )
    return float(max(0.0, min(1.0, reward)))


def evidence_weight(feedback: Feedback) -> float:
    """Bound the statistical influence of one aggregate observation window."""
    if feedback.reach <= 0:
        return 0.0
    return min(settings.max_evidence_weight, max(1.0, math.sqrt(feedback.reach)))
