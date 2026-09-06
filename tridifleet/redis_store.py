from __future__ import annotations

import json
from redis import Redis

from .config import settings
from .models import Ad, ContextEvent, Decision
from .store import Posterior


class RedisStore:
    """Redis-backed state store for multi-process / restart-safe deployments."""

    def __init__(self, url: str):
        self.redis = Redis.from_url(url, decode_responses=True)

    def put_ad(self, ad: Ad) -> None:
        self.redis.hset("tridifleet:ads", ad.ad_id, ad.model_dump_json())

    def active_ads(self) -> list[Ad]:
        values = self.redis.hvals("tridifleet:ads")
        return [a for raw in values if (a := Ad.model_validate_json(raw)).active]

    def put_context(self, event: ContextEvent) -> None:
        self.redis.set(
            f"tridifleet:context:{event.totem_id}",
            event.model_dump_json(),
            ex=3600,
        )

    def get_context(self, totem_id: str) -> ContextEvent | None:
        raw = self.redis.get(f"tridifleet:context:{totem_id}")
        return ContextEvent.model_validate_json(raw) if raw else None

    def posterior(self, ad_id: str, key: str) -> Posterior:
        redis_key = f"tridifleet:posterior:{ad_id}:{key}"
        raw = self.redis.hgetall(redis_key)
        if not raw:
            return Posterior()
        return Posterior(alpha=float(raw["alpha"]), beta=float(raw["beta"]))

    def update(self, ad_id: str, keys: list[str], reward: float, weight: float) -> None:
        if weight <= 0:
            return
        alpha_delta = reward * weight
        beta_delta = (1.0 - reward) * weight
        pipe = self.redis.pipeline(transaction=True)
        for key in keys:
            redis_key = f"tridifleet:posterior:{ad_id}:{key}"
            pipe.hsetnx(redis_key, "alpha", settings.prior_alpha)
            pipe.hsetnx(redis_key, "beta", settings.prior_beta)
            pipe.hincrbyfloat(redis_key, "alpha", alpha_delta)
            pipe.hincrbyfloat(redis_key, "beta", beta_delta)
            pipe.sadd(f"tridifleet:posterior_keys:{ad_id}", key)
        pipe.execute()

    def decay_posteriors(self, factor: float) -> None:
        factor = max(0.0, min(1.0, factor))
        for raw_ad in self.redis.hvals("tridifleet:ads"):
            ad = Ad.model_validate_json(raw_ad)
            keys = self.redis.smembers(f"tridifleet:posterior_keys:{ad.ad_id}")
            pipe = self.redis.pipeline(transaction=True)
            for key in keys:
                p = self.posterior(ad.ad_id, key)
                alpha = settings.prior_alpha + (p.alpha - settings.prior_alpha) * factor
                beta = settings.prior_beta + (p.beta - settings.prior_beta) * factor
                redis_key = f"tridifleet:posterior:{ad.ad_id}:{key}"
                pipe.hset(redis_key, mapping={"alpha": alpha, "beta": beta})
            pipe.execute()

    def put_decision(self, decision: Decision) -> None:
        self.redis.set(
            f"tridifleet:decision:{decision.decision_id}",
            decision.model_dump_json(),
            ex=86400 * 7,
        )

    def get_decision(self, decision_id: str) -> Decision | None:
        raw = self.redis.get(f"tridifleet:decision:{decision_id}")
        return Decision.model_validate_json(raw) if raw else None

    def delete_decision(self, decision_id: str) -> None:
        self.redis.delete(f"tridifleet:decision:{decision_id}")

    def list_posteriors(self, ad_id: str) -> list[tuple[str, Posterior]]:
        keys = sorted(self.redis.smembers(f"tridifleet:posterior_keys:{ad_id}"))
        return [(key, self.posterior(ad_id, key)) for key in keys]
