from __future__ import annotations

import uuid
import numpy as np

from .config import settings
from .context import context_keys
from .models import Decision
from .store import MemoryStore


class HierarchicalThompsonBandit:
    """Contextual Thompson Sampling with hierarchical shrinkage/backoff."""

    def __init__(self, store: MemoryStore, seed: int | None = None):
        self.store = store
        self.rng = np.random.default_rng(seed)

    def sample_ad(self, ad_id: str, keys: list[str]) -> float:
        blended = 0.0
        residual = 1.0

        specific_keys = [key for key in keys if key != "global"]
        for key in reversed(specific_keys):
            p = self.store.posterior(ad_id, key)
            n = p.observations
            confidence = n / (n + settings.shrinkage_k) if n > 0 else 0.0
            if confidence <= 0:
                continue
            sample = float(self.rng.beta(p.alpha, p.beta))
            take = residual * confidence
            blended += take * sample
            residual -= take
            if residual <= 1e-9:
                break

        gp = self.store.posterior(ad_id, "global")
        global_sample = float(self.rng.beta(gp.alpha, gp.beta))
        blended += residual * global_sample
        return blended

    def _sample_ad(self, ad_id: str, keys: list[str]) -> float:
        # Compatibility alias for tests/older callers.
        return self.sample_ad(ad_id, keys)

    def choose(self, totem_id: str) -> Decision:
        event = self.store.get_context(totem_id)
        if event is None:
            raise KeyError(f"no context registered for totem {totem_id}")

        ads = self.store.active_ads()
        if not ads:
            raise RuntimeError("no active ads")

        keys = context_keys(event)
        scores = [(self.sample_ad(ad.ad_id, keys), ad) for ad in ads]
        score, winner = max(scores, key=lambda pair: pair[0])

        decision = Decision(
            decision_id=str(uuid.uuid4()),
            ad_id=winner.ad_id,
            totem_id=totem_id,
            sampled_score=score,
            context_keys=keys,
            policy="hierarchical_thompson",
            residual_score=score,
        )
        self.store.put_decision(decision)
        return decision
