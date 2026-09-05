from __future__ import annotations

from .bandit import HierarchicalThompsonBandit
from .models import Feedback
from .reward import evidence_weight, retention_reward
from .store import MemoryStore


class RetentionService:
    def __init__(self, store: MemoryStore, bandit: HierarchicalThompsonBandit):
        self.store = store
        self.bandit = bandit

    def apply_feedback(self, feedback: Feedback) -> dict[str, float | str]:
        decision = self.store.get_decision(feedback.decision_id)
        if decision is None:
            raise KeyError("unknown decision_id")

        reward = retention_reward(feedback)
        weight = evidence_weight(feedback)

        # A no-traffic window has no evidence and therefore must not punish an ad.
        if weight > 0:
            self.store.update(
                ad_id=decision.ad_id,
                keys=decision.context_keys,
                reward=reward,
                weight=weight,
            )

        return {
            "decision_id": decision.decision_id,
            "ad_id": decision.ad_id,
            "reward": reward,
            "evidence_weight": weight,
        }
