from __future__ import annotations

import hashlib
import math
import threading
import uuid
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass

import numpy as np

from .bandit import HierarchicalThompsonBandit
from .context import context_keys
from .models import Ad, ContextEvent, Decision, Feedback
from .reward import evidence_weight, retention_reward
from .taxonomy import canonical_tag, canonical_tags


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


def _signed_hash(text: str, buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % buckets, 1.0 if (value >> 8) & 1 else -1.0


class FeatureEncoder:
    """Fixed-size feature hashing for arbitrary creative tags + real-valued context."""

    dim = 160
    tag_start = 24
    tag_buckets = 32
    interaction_start = 56
    interaction_buckets = 104

    def encode(self, ad: Ad, ctx: ContextEvent) -> np.ndarray:
        x = np.zeros(self.dim, dtype=np.float64)
        hour = ctx.timestamp.hour + ctx.timestamp.minute / 60.0
        phase = 2.0 * math.pi * hour / 24.0
        dow_phase = 2.0 * math.pi * ctx.timestamp.weekday() / 7.0

        x[0] = 1.0
        x[1] = ((ctx.female_share if ctx.female_share is not None else 0.5) - 0.5) * 2.0
        x[2] = ((ctx.mean_age if ctx.mean_age is not None else 38.0) - 38.0) / 28.0
        x[3] = min(2.0, (ctx.age_std if ctx.age_std is not None else 15.0) / 15.0)
        x[4] = min(2.0, math.log1p(ctx.flow_per_minute) / 3.0)
        x[5] = min(2.0, math.log1p(ctx.crowd_density) / 2.0)
        x[6] = math.sin(phase)
        x[7] = math.cos(phase)
        x[8] = math.sin(dow_phase)
        x[9] = math.cos(dow_phase)
        x[10] = math.tanh((ctx.x_km or 0.0) / 5.0)
        x[11] = math.tanh((ctx.y_km or 0.0) / 5.0)
        x[12] = min(2.0, math.log1p(ctx.reach_window) / 3.0)
        # Deliberately do NOT feed the previous ad's impression rate back into
        # the next decision. It is policy-dependent and would create an
        # endogenous self-reinforcing feature. Keep it in telemetry for audit,
        # not as a decision input.
        x[13] = 1.0 if ctx.female_share is not None and ctx.mean_age is not None else 0.0
        duration_norm = min(2.0, ad.duration_seconds / 15.0)
        x[14] = duration_norm
        # Daily budget is deliberately excluded from retention features. Money
        # constrains eligibility/pacing; it must not teach the model that a
        # higher budget makes humans like a creative more.
        x[15] = duration_norm * duration_norm

        age_dist = ctx.age_distribution or {}
        x[16] = age_dist.get("u18", 0.0)
        x[17] = age_dist.get("18-24", 0.0)
        x[18] = age_dist.get("25-34", 0.0)
        x[19] = age_dist.get("35-44", 0.0)
        x[20] = age_dist.get("45-59", 0.0)
        x[21] = age_dist.get("60+", 0.0)
        x[22] = max(age_dist.values()) if age_dist else 0.0
        x[23] = 1.0 if age_dist else 0.0

        tags = canonical_tags(ad.tags)
        if ad.category:
            category = canonical_tag(ad.category)
            tags.add(category)
            tags.add(f"category:{category}")

        female = x[1]
        age = x[2]
        time_sin = x[6]
        flow = x[4]

        for tag in sorted(tags):
            bucket, sign = _signed_hash(f"tag:{tag}", self.tag_buckets)
            x[self.tag_start + bucket] += sign

            for suffix, value in (
                ("gender", female),
                ("age", age),
                ("time", time_sin),
                ("flow", flow),
            ):
                ib, isign = _signed_hash(
                    f"interaction:{tag}:{suffix}", self.interaction_buckets
                )
                x[self.interaction_start + ib] += isign * value

        norm = np.linalg.norm(x)
        if norm > 6.0:
            x *= 6.0 / norm
        return x


@dataclass
class DiagonalBayesianRegressor:
    """Online Bayesian linear approximation with diagonal covariance.

    The model is intentionally cheap enough for many local decisions while
    retaining uncertainty for Thompson sampling.
    """

    dim: int
    prior_variance: float = 0.20
    observation_variance: float = 0.035

    def __post_init__(self):
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.mean[0] = 0.35
        self.var = np.full(self.dim, self.prior_variance, dtype=np.float64)
        self.var[0] = 0.08

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, np.sqrt(np.maximum(self.var, 1e-8)))

    def predict_mean(self, x: np.ndarray) -> float:
        return float(np.clip(np.dot(self.mean, x), 0.0, 1.0))

    def forget(self, factor: float) -> None:
        factor = max(0.0, min(1.0, factor))
        prior_mean = np.zeros(self.dim, dtype=np.float64)
        prior_mean[0] = 0.35
        prior_var = np.full(self.dim, self.prior_variance, dtype=np.float64)
        prior_var[0] = 0.08
        self.mean = prior_mean + (self.mean - prior_mean) * factor
        # Forgetting widens uncertainty back toward the prior.
        self.var = np.minimum(
            prior_var,
            prior_var - (prior_var - self.var) * factor,
        )

    def update(self, x: np.ndarray, y: float, weight: float) -> None:
        if weight <= 0:
            return
        effective_noise = self.observation_variance / max(0.25, weight)
        pred = float(np.dot(self.mean, x))
        denom = effective_noise + float(np.dot(self.var, x * x))
        if denom <= 1e-12:
            return
        gain = (self.var * x) / denom
        error = float(y - pred)
        self.mean += gain * error
        self.var = np.maximum(1e-6, self.var * (1.0 - gain * x))


class HybridRetentionIntelligence:
    """Shared contextual Bayesian model + per-creative hierarchical Thompson residual.

    The shared model generalizes through context and creative tags. The
    hierarchical Beta component captures creative-specific evidence and preserves
    strong cold-start exploration.
    """

    policy_name = "hybrid_contextual_thompson_v2"

    def __init__(self, store, seed: int | None = None):
        self.store = store
        self.encoder = FeatureEncoder()
        self.shared = DiagonalBayesianRegressor(self.encoder.dim)
        self.rng = np.random.default_rng(seed)
        self.residual = HierarchicalThompsonBandit(store, seed=seed)
        self._completed = OrderedDict()
        self.max_pending = 10000
        self._decision_features: dict[str, tuple[np.ndarray, str, list[str]]] = {}
        self._decision_traces: dict[str, dict] = {}
        self._trace_order: list[str] = []
        self._lock = threading.RLock()
        self.total_updates = 0
        self.total_reward = 0.0

    @synchronized
    def choose(self, totem_id: str, candidates: list[Ad] | None = None) -> Decision:
        if len(self._decision_features) >= self.max_pending:
            raise RuntimeError("pending feedback capacity reached; submit feedback before requesting more decisions")
        ctx = self.store.get_context(totem_id)
        if ctx is None:
            raise KeyError(f"no context registered for totem {totem_id}")

        ads = candidates if candidates is not None else self.store.active_ads()
        ads = [ad for ad in ads if ad.active]
        if not ads:
            raise RuntimeError("no active ads")

        keys = context_keys(ctx)
        theta = self.shared.sample(self.rng)
        scored: list[tuple[float, float, float, float, Ad, np.ndarray]] = []

        for ad in ads:
            features = self.encoder.encode(ad, ctx)
            model_sample = float(np.clip(np.dot(theta, features), 0.0, 1.0))
            residual_sample = self.residual.sample_ad(ad.ad_id, keys)

            # Early in the system, tag/context generalization matters more. The
            # creative-specific posterior grows in influence as it collects data.
            ad_n = self.store.posterior(ad.ad_id, "global").observations
            residual_weight = min(0.45, 0.18 + ad_n / (ad_n + 80.0) * 0.27)
            score = (1.0 - residual_weight) * model_sample + residual_weight * residual_sample
            uncertainty = float(
                np.sqrt(np.maximum(0.0, np.dot(self.shared.var, features * features)))
            )
            scored.append((score, model_sample, residual_sample, uncertainty, ad, features))

        score, model_score, residual_score, uncertainty, winner, features = max(
            scored, key=lambda row: row[0]
        )

        decision = Decision(
            decision_id=str(uuid.uuid4()),
            ad_id=winner.ad_id,
            totem_id=totem_id,
            sampled_score=float(score),
            context_keys=keys,
            policy=self.policy_name,
            ad_duration_seconds=winner.duration_seconds,
            model_score=float(model_score),
            residual_score=float(residual_score),
        )
        self.store.put_decision(decision)
        with self._lock:
            self._decision_features[decision.decision_id] = (
                features.copy(),
                winner.ad_id,
                list(keys),
            )
            ranked = sorted(scored, key=lambda row: row[0], reverse=True)[:5]
            self._decision_traces[decision.decision_id] = {
                "decision_id": decision.decision_id,
                "totem_id": totem_id,
                "winner": winner.ad_id,
                "policy": self.policy_name,
                "context_keys": list(keys),
                "top_candidates": [
                    {
                        "ad_id": row[4].ad_id,
                        "name": row[4].name,
                        "score": float(row[0]),
                        "shared_model": float(row[1]),
                        "creative_residual": float(row[2]),
                        "uncertainty": float(row[3]),
                        "global_observations": float(
                            self.store.posterior(row[4].ad_id, "global").observations
                        ),
                    }
                    for row in ranked
                ],
            }
            self._trace_order.append(decision.decision_id)
            if len(self._trace_order) > 2500:
                old = self._trace_order.pop(0)
                self._decision_traces.pop(old, None)
        return decision

    @synchronized
    def apply_feedback(self, feedback: Feedback) -> dict:
        payload = feedback.model_dump(mode="json")
        previous = self._completed.get(feedback.decision_id)
        if previous is not None:
            if previous[0] != payload:
                raise ValueError("conflicting feedback for an already completed decision")
            return {**previous[1], "duplicate": True}
        decision = self.store.get_decision(feedback.decision_id)
        if decision is None:
            raise KeyError("unknown or expired decision_id")
        if decision.ad_duration_seconds is not None and not math.isclose(
            feedback.ad_duration_seconds, decision.ad_duration_seconds, rel_tol=1e-6
        ):
            raise ValueError("feedback duration differs from the selected creative")
        cached = self._decision_features.get(feedback.decision_id)
        if cached is None:
            raise ValueError("decision belongs to another learner session; shared features unavailable")
        reward = retention_reward(feedback)
        weight = evidence_weight(feedback)
        if weight > 0:
            self.store.update(decision.ad_id, decision.context_keys, reward, weight)
            self.shared.update(cached[0], reward, weight)
            self.total_updates += 1
            self.total_reward += reward
        result = {
            "decision_id": decision.decision_id, "ad_id": decision.ad_id,
            "reward": reward, "evidence_weight": weight,
            "policy": self.policy_name, "duplicate": False,
        }
        self._decision_features.pop(feedback.decision_id, None)
        self.store.delete_decision(feedback.decision_id)
        self._completed[feedback.decision_id] = (payload, result)
        if len(self._completed) > 10000:
            self._completed.popitem(last=False)
        return result

    def mean_prediction(self, ad: Ad, ctx: ContextEvent) -> float:
        return self.shared.predict_mean(self.encoder.encode(ad, ctx))

    @synchronized
    def advance_day(self, factor: float) -> None:
        factor = max(0.0, min(1.0, factor))
        if hasattr(self.store, "decay_posteriors"):
            self.store.decay_posteriors(factor)
        with self._lock:
            self.shared.forget(factor)

    def trace(self, decision_id: str | None) -> dict | None:
        if not decision_id:
            return None
        with self._lock:
            trace = self._decision_traces.get(decision_id)
            return dict(trace) if trace else None

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy_name,
            "updates": self.total_updates,
            "mean_observed_reward": (
                self.total_reward / self.total_updates if self.total_updates else 0.0
            ),
            "mean_parameter_uncertainty": float(np.mean(self.shared.var)),
        }
