from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .config import settings
from .models import Ad, ContextEvent, Decision


@dataclass
class Posterior:
    alpha: float = settings.prior_alpha
    beta: float = settings.prior_beta

    @property
    def observations(self) -> float:
        return max(0.0, self.alpha + self.beta - settings.prior_alpha - settings.prior_beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class MemoryStore:
    """Thread-safe local store used for development and tests."""

    def __init__(self):
        self._lock = RLock()
        self.ads: dict[str, Ad] = {}
        self.contexts: dict[str, ContextEvent] = {}
        self.posteriors: dict[tuple[str, str], Posterior] = {}
        self.decisions: dict[str, Decision] = {}

    def put_ad(self, ad: Ad) -> None:
        with self._lock:
            self.ads[ad.ad_id] = ad

    def active_ads(self) -> list[Ad]:
        with self._lock:
            return [a for a in self.ads.values() if a.active]

    def put_context(self, event: ContextEvent) -> None:
        with self._lock:
            self.contexts[event.totem_id] = event

    def get_context(self, totem_id: str) -> ContextEvent | None:
        with self._lock:
            return self.contexts.get(totem_id)

    def posterior(self, ad_id: str, key: str) -> Posterior:
        with self._lock:
            return self.posteriors.setdefault((ad_id, key), Posterior())

    def update(self, ad_id: str, keys: list[str], reward: float, weight: float) -> None:
        with self._lock:
            for key in keys:
                p = self.posteriors.setdefault((ad_id, key), Posterior())
                p.alpha += reward * weight
                p.beta += (1.0 - reward) * weight

    def decay_posteriors(self, factor: float) -> None:
        factor = max(0.0, min(1.0, factor))
        with self._lock:
            for p in self.posteriors.values():
                p.alpha = settings.prior_alpha + (p.alpha - settings.prior_alpha) * factor
                p.beta = settings.prior_beta + (p.beta - settings.prior_beta) * factor

    def put_decision(self, decision: Decision) -> None:
        with self._lock:
            self.decisions[decision.decision_id] = decision

    def get_decision(self, decision_id: str) -> Decision | None:
        with self._lock:
            return self.decisions.get(decision_id)

    def delete_decision(self, decision_id: str) -> None:
        with self._lock:
            self.decisions.pop(decision_id, None)

    def list_posteriors(self, ad_id: str) -> list[tuple[str, Posterior]]:
        with self._lock:
            return sorted(
                [(key, p) for (stored_ad_id, key), p in self.posteriors.items() if stored_ad_id == ad_id],
                key=lambda pair: pair[0],
            )
