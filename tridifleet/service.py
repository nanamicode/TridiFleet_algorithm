from __future__ import annotations

from .models import Feedback


class RetentionService:
    def __init__(self, store, intelligence):
        self.store = store
        self.intelligence = intelligence

    def apply_feedback(self, feedback: Feedback) -> dict[str, float | str]:
        return self.intelligence.apply_feedback(feedback)
