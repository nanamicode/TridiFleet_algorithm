from __future__ import annotations

import math
import copy
import os
import random
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from ..config import settings
from ..intelligence import HybridRetentionIntelligence
from ..models import Ad
from ..reward import retention_reward
from ..store import MemoryStore
from .evaluation import outcomes, PairedEvidence, VERSION
from .audit import AuditLog
from .catalog import default_creatives
from .city import CityMap
from .domain import MetricPoint, SimConfig
from .ground_truth import GroundTruthModel
from .people import PopulationEngine
from .sensor import SimulatedTotemSensor
from .spatial import SpatialIndex


BRAZIL_TZ = timezone(timedelta(hours=-3))


class SimulationEngine:
    """Local digital twin.

    The background thread owns world evolution. Browser tabs are observers only,
    so closing the UI never pauses learning.
    """

    def __init__(self, config: SimConfig):
        self.config = config.validate()
        self.lock = threading.RLock()
        self.rng = random.Random(config.seed + 909)
        self.city = CityMap.generate(config.radius_km, config.n_totems, config.seed)
        self.population = PopulationEngine(self.city, config.seed)
        self.truth = GroundTruthModel(config.seed)
        self.evidence = PairedEvidence()
        self.sensor = SimulatedTotemSensor(config.seed)

        self.store = MemoryStore()
        self.intelligence = HybridRetentionIntelligence(self.store, seed=config.seed)
        for ad in default_creatives(config.seed):
            self.store.put_ad(ad)

        self.sim_time = datetime(2026, 9, 1, 6, 0, tzinfo=BRAZIL_TZ)
        audit_path = os.getenv("TRIDIFLEET_AUDIT_DB", "data/tridifleet_lab.sqlite3")
        self.audit = AuditLog(audit_path)
        self.run_id = self.audit.start_run(self.sim_time.isoformat(), self.config)

        self._step_remainder = 0.0
        self.exposure_seconds = {t.totem_id: {} for t in self.city.totems}
        self.exposure_streak = {t.totem_id: {} for t in self.city.totems}
        self._last_checkpoint_wall = time.monotonic()
        self.last_error = None
        self.running = False
        self.paused = False
        self.speed = 1.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.spend_today: dict[str, float] = {}
        self.plays_by_ad: dict[str, int] = {}
        self.rewards_by_ad: dict[str, deque[float]] = {}
        self.total_decisions = 0
        self.total_feedback = 0
        self.total_events = 0
        self.no_fill_slots = 0
        self.current_day = self.sim_time.date()

        self.recent_observed = deque(maxlen=1200)
        self.recent_policy_expected = deque(maxlen=1200)
        self.recent_random = deque(maxlen=1200)
        self.recent_oracle = deque(maxlen=1200)
        self.recent_regret = deque(maxlen=1200)
        self.recent_exploration = deque(maxlen=1200)
        self.recent_ad_choices = deque(maxlen=1200)
        self.cumulative_regret = 0.0
        self.metric_history: deque[MetricPoint] = deque(maxlen=360)
        self._last_metric_at = self.sim_time
        # Each active slot accumulates unique pedestrians across the whole
        # interval. This avoids counting only whoever happens to be present at
        # the exact decision instant and prevents double-counting the same
        # person on every physics tick.
        self.exposure_seen: dict[str, dict[int, tuple[object, float]]] = {
            t.totem_id: {} for t in self.city.totems
        }
        # Counterfactual evaluation is delayed until the slot finishes, so the
        # chosen policy, random baseline and oracle are all scored on the exact
        # same pedestrians that were actually exposed.
        self.pending_evaluation: dict[str, dict] = {}

    def start_background(self) -> None:
        with self.lock:
            if self._thread and self._thread.is_alive():
                self.running = True
                self.paused = False
                return
            self.running = True
            self.paused = False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="tridifleet-digital-twin",
                daemon=True,
            )
            self._thread.start()

    def checkpoint_path(self):
        if self.audit.path == ":memory:": return None
        return os.getenv("TRIDIFLEET_CHECKPOINT", "data/lab-checkpoint.json.gz")

    def save_checkpoint(self):
        path = self.checkpoint_path()
        if path:
            from .checkpoint import save
            save(self, path)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        self.save_checkpoint()
        self.running = False
        try:
            self.audit.append("run_stop", self.sim_time.isoformat(), {"reason": "stopped"})
        except Exception:
            pass
        try:
            self.audit.close()
        except Exception:
            pass

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = bool(paused)

    def set_speed(self, speed: float) -> float:
        # 1.0 is the laboratory's nominal clock. We intentionally never allow
        # >1x: "accelerate" only means returning from a slowed state to nominal.
        allowed = (0.25, 0.5, 1.0)
        nearest = min(allowed, key=lambda v: abs(v - speed))
        with self.lock:
            self.speed = nearest
        return nearest

    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            wall_dt = min(0.5, max(0.0, now - last))
            last = now

            with self.lock:
                active = self.running and not self.paused
                speed = self.speed
            if active and wall_dt > 0:
                sim_dt = wall_dt * self.config.base_sim_seconds_per_real_second * speed
                try:
                    self.step(sim_dt)
                    if time.monotonic() - self._last_checkpoint_wall >= 30:
                        self.save_checkpoint()
                        self._last_checkpoint_wall = time.monotonic()
                except Exception as exc:
                    import logging
                    logging.exception("Simulation stopped after processing error")
                    with self.lock:
                        self.last_error = str(exc)
                        self.running = False
            time.sleep(0.08)

    def _eligible_ads(self) -> list[Ad]:
        ads = self.store.active_ads()
        return [
            ad
            for ad in ads
            if self.spend_today.get(ad.ad_id, 0.0) + ad.cost_per_play <= ad.daily_budget + 1e-9
        ]

    def _mean(self, values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _creative_diversity(self) -> float:
        if not self.recent_ad_choices:
            return 0.0
        counts: dict[str, int] = {}
        for ad_id in self.recent_ad_choices:
            counts[ad_id] = counts.get(ad_id, 0) + 1
        total = len(self.recent_ad_choices)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(max(p, 1e-12))
        denom = math.log(max(2, len(self.store.active_ads())))
        return min(1.0, entropy / denom) if denom > 0 else 0.0

    def _record_metric_if_due(self) -> None:
        if (self.sim_time - self._last_metric_at).total_seconds() < 15 * 60:
            return
        observed = self._mean(self.recent_observed)
        policy_expected = self._mean(self.recent_policy_expected)
        random_baseline = self._mean(self.recent_random)
        oracle = self._mean(self.recent_oracle)
        uplift = ((policy_expected / random_baseline) - 1.0) if random_baseline > 1e-9 else 0.0
        mean_regret = self._mean(self.recent_regret)
        exploration_rate = self._mean(self.recent_exploration)
        diversity = self._creative_diversity()
        diag = self.intelligence.diagnostics()
        self.metric_history.append(
            MetricPoint(
                timestamp=self.sim_time,
                observed_reward=observed,
                policy_expected=policy_expected,
                random_baseline=random_baseline,
                oracle_ceiling=oracle,
                uplift_vs_random=uplift,
                mean_regret=mean_regret,
                exploration_rate=exploration_rate,
                creative_diversity=diversity,
                model_uncertainty=float(diag["mean_parameter_uncertainty"]),
                people=len(self.population.people),
                decisions=self.total_decisions,
            )
        )
        self._last_metric_at = self.sim_time

    def _new_day_if_needed(self) -> None:
        if self.sim_time.date() != self.current_day:
            days = max(1, (self.sim_time.date() - self.current_day).days)
            self.current_day = self.sim_time.date()
            self.spend_today.clear()
            factor = settings.daily_memory_decay ** days
            self.intelligence.advance_day(factor)
            self.audit.append(
                "model_decay",
                self.sim_time.isoformat(),
                {"days": days, "factor": factor},
            )

    def _evaluate_completed_slot(self, totem, audience) -> dict | None:
        pending = self.pending_evaluation.pop(totem.totem_id, None)
        if pending is None:
            return None

        candidate_ads = [
            self.store.ads[ad_id]
            for ad_id in pending["candidate_ids"]
            if ad_id in self.store.ads
        ]
        if not candidate_ads:
            return None

        decision_time = pending["decision_time"]
        elapsed = self.sim_time - decision_time
        midpoint = decision_time + elapsed / 2

        batch = outcomes(self.truth, audience, candidate_ads, midpoint,
            self.config.detection_radius_km, self.exposure_seconds[totem.totem_id],
            key=f"{totem.totem_id}|{decision_time.isoformat()}")
        expected = batch.expected.tolist()
        chosen_index = next(i for i, ad in enumerate(candidate_ads) if ad.ad_id == pending["chosen_ad_id"])
        paired_random = float(batch.realized.mean())
        chosen_observed = float(batch.realized[chosen_index])
        self.evidence.add(midpoint, totem.totem_id, chosen_observed, paired_random, len(audience))
        completed_feedback = batch.feedback(chosen_index, pending["decision_id"], candidate_ads[chosen_index], len(audience))
        expected_by_ad = {
            ad.ad_id: value for ad, value in zip(candidate_ads, expected)
        }
        random_expected = sum(expected) / len(expected) if expected else 0.0
        oracle_expected = max(expected) if expected else 0.0
        chosen_expected = expected_by_ad.get(pending["chosen_ad_id"], 0.0)
        regret = max(0.0, oracle_expected - chosen_expected)

        # Empty streets contain no information about policy quality, so they
        # are audited but excluded from retention/uplift averages.
        if audience:
            self.recent_policy_expected.append(chosen_expected)
            self.recent_random.append(random_expected)
            self.recent_oracle.append(oracle_expected)
            self.recent_regret.append(regret)
            self.recent_exploration.append(float(pending["exploring"]))
            self.cumulative_regret += regret

        return {
            "feedback": completed_feedback,
            "evaluation_version": VERSION,
            "paired_random_observed": paired_random,
            "chosen_observed": chosen_observed,
            "monte_carlo_draws": 64,
            "random_expected": random_expected,
            "oracle_expected": oracle_expected,
            "chosen_expected": chosen_expected,
            "regret": regret,
            "exploration": bool(pending["exploring"]),
            "audience_size": len(audience),
            "evaluated_at": midpoint.isoformat(),
        }

    def _serve_totem(self, totem, current_audience) -> None:
        completed = list(self.exposure_seen.get(totem.totem_id, {}).values())
        # The completed slot is the correct population for its delayed outcome.
        # The next decision can use that rolling one-minute audience as context;
        # if the lab has just started, fall back to the people present now.
        audience = completed or current_audience

        previous_feedback = None
        if totem.current_ad_id and totem.current_decision_id:
            ad = self.store.ads.get(totem.current_ad_id)
            if ad:
                evaluation = self._evaluate_completed_slot(totem, completed)
                slot_start = totem.last_decision_at or self.sim_time
                slot_midpoint = slot_start + (self.sim_time - slot_start) / 2
                previous_feedback = evaluation.pop("feedback")
                for person, _ in completed:
                    live = self.population.people.get(person.person_id)
                    if live:
                        live.ad_exposures[ad.ad_id] = live.ad_exposures.get(ad.ad_id, 0) + 1
                outcome = self.intelligence.apply_feedback(previous_feedback)
                reward = float(outcome["reward"])
                self.audit.append(
                    "feedback",
                    self.sim_time.isoformat(),
                    {
                        "totem_id": totem.totem_id,
                        "ad_id": ad.ad_id,
                        "feedback": previous_feedback,
                        "reward": reward,
                        "evaluation_only": evaluation,
                    },
                )
                totem.last_reach = previous_feedback.reach
                totem.last_impressions = previous_feedback.impressions
                totem.last_avg_view_seconds = previous_feedback.avg_view_seconds
                totem.last_completion_rate = previous_feedback.completion_rate or 0.0
                totem.last_reward = reward
                if float(outcome["evidence_weight"]) > 0:
                    self.recent_observed.append(reward)
                    self.rewards_by_ad.setdefault(ad.ad_id, deque(maxlen=300)).append(reward)
                self.total_feedback += 1


        ctx = self.sensor.context(
            totem=totem,
            audience=audience,
            timestamp=self.sim_time,
            previous=previous_feedback,
            detection_radius_km=self.config.detection_radius_km,
            decision_interval_seconds=self.config.decision_interval_sim_seconds,
        )
        self.store.put_context(ctx)
        totem.female_share = ctx.female_share if ctx.female_share is not None else 0.5
        totem.mean_age = ctx.mean_age if ctx.mean_age is not None else 38.0
        totem.age_distribution = dict(ctx.age_distribution)
        totem.flow_per_minute = ctx.flow_per_minute

        candidates = self._eligible_ads()
        if not candidates:
            self.pending_evaluation.pop(totem.totem_id, None)
            self.exposure_seen[totem.totem_id] = {}
            totem.current_ad_id = None
            totem.current_decision_id = None
            totem.last_decision_at = self.sim_time
            self.no_fill_slots += 1
            self.audit.append(
                "no_fill",
                self.sim_time.isoformat(),
                {
                    "totem_id": totem.totem_id,
                    "reason": "all_daily_budgets_exhausted",
                },
            )
            return

        decision = self.intelligence.choose(totem.totem_id, candidates=candidates)
        chosen_ad = self.store.ads[decision.ad_id]
        self.spend_today[decision.ad_id] = self.spend_today.get(decision.ad_id, 0.0) + chosen_ad.cost_per_play
        global_n = self.store.posterior(decision.ad_id, "global").observations
        exploring = global_n < 8.0
        self.recent_ad_choices.append(decision.ad_id)
        self.pending_evaluation[totem.totem_id] = {
            "decision_id": decision.decision_id,
            "chosen_ad_id": decision.ad_id,
            "candidate_ids": [ad.ad_id for ad in candidates],
            "decision_time": self.sim_time,
            "exploring": exploring,
        }

        self.audit.append(
            "decision",
            self.sim_time.isoformat(),
            {
                "context": ctx,
                "decision": decision,
                "trace": self.intelligence.trace(decision.decision_id),
                "candidate_ids": [ad.ad_id for ad in candidates],
                "exploration": exploring,
                "evaluation_only": "deferred_until_slot_completion",
            },
        )
        self.exposure_seen[totem.totem_id] = {}
        self.exposure_seconds[totem.totem_id] = {}
        self.exposure_streak[totem.totem_id] = {}
        totem.current_ad_id = decision.ad_id
        totem.current_decision_id = decision.decision_id
        totem.last_decision_at = self.sim_time
        totem.plays += 1
        self.plays_by_ad[decision.ad_id] = self.plays_by_ad.get(decision.ad_id, 0) + 1
        self.total_decisions += 1
        self.total_events += 1

    def step(self, sim_dt_seconds: float) -> None:
        if not math.isfinite(sim_dt_seconds) or sim_dt_seconds < 0:
            raise ValueError("step must be finite and nonnegative")
        with self.lock:
            self._step_remainder += sim_dt_seconds
            while self._step_remainder >= 1.0:
                self._tick(1.0)
                self._step_remainder -= 1.0

    def _tick(self, sim_dt_seconds: float) -> None:
        with self.lock:
            self.sim_time += timedelta(seconds=sim_dt_seconds)
            self._new_day_if_needed()
            self.population.update(sim_dt_seconds, self.sim_time)
            spatial = SpatialIndex(self.population.people)

            for totem in self.city.totems:
                current_audience = spatial.query(
                    totem.x,
                    totem.y,
                    self.config.detection_radius_km,
                )

                # Accumulate the minimum observed distance for every pedestrian
                # exposed while the current creative is on screen.
                if totem.current_ad_id:
                    bucket = self.exposure_seen.setdefault(totem.totem_id, {})
                    previous_streak = self.exposure_streak[totem.totem_id]
                    streak = {}
                    self.exposure_streak[totem.totem_id] = streak
                    for person, distance in current_audience:
                        durations = self.exposure_seconds[totem.totem_id]
                        streak[person.person_id] = previous_streak.get(person.person_id, 0.0) + sim_dt_seconds
                        durations[person.person_id] = max(durations.get(person.person_id, 0.0), streak[person.person_id])
                        prev = bucket.get(person.person_id)
                        if prev is None or distance < prev[1]:
                            if prev is None:
                                frozen = copy.copy(person)
                                frozen.ad_exposures = dict(person.ad_exposures)
                                bucket[person.person_id] = (frozen, distance)
                            else:
                                bucket[person.person_id] = (prev[0], distance)

                due = (
                    totem.last_decision_at is None
                    or (self.sim_time - totem.last_decision_at).total_seconds()
                    >= self.config.decision_interval_sim_seconds
                )
                if due:
                    self._serve_totem(totem, current_audience)

            self._record_metric_if_due()

    def add_creative(self, ad: Ad) -> Ad:
        with self.lock:
            self.store.put_ad(ad)
            self.audit.append(
                "creative_added",
                self.sim_time.isoformat(),
                {"creative": ad},
            )
            return ad

    def audit_status(self, full: bool = False) -> dict:
        with self.lock:
            return self.audit.full_verify_status() if full else self.audit.status()

    def creative_stats(self) -> list[dict]:
        with self.lock:
            result = []
            for ad in sorted(self.store.active_ads(), key=lambda a: a.ad_id):
                rewards = self.rewards_by_ad.get(ad.ad_id, ())
                posterior = self.store.posterior(ad.ad_id, "global")
                result.append(
                    {
                        **ad.model_dump(),
                        "spent_today": round(self.spend_today.get(ad.ad_id, 0.0), 2),
                        "plays": self.plays_by_ad.get(ad.ad_id, 0),
                        "observed_reward": round(
                            sum(rewards) / len(rewards) if rewards else 0.0, 4
                        ),
                        "posterior_mean": round(posterior.mean, 4),
                        "observations": round(posterior.observations, 1),
                    }
                )
            return result

    def totem_detail(self, totem_id: str) -> dict | None:
        with self.lock:
            totem = next((t for t in self.city.totems if t.totem_id == totem_id), None)
            if not totem:
                return None
            ad = self.store.ads.get(totem.current_ad_id) if totem.current_ad_id else None
            return {
                "id": totem.totem_id,
                "x": totem.x,
                "y": totem.y,
                "region": totem.region,
                "current_ad": ad.model_dump() if ad else None,
                "reach": totem.last_reach,
                "impressions": totem.last_impressions,
                "capture_rate": (
                    totem.last_impressions / totem.last_reach if totem.last_reach else 0.0
                ),
                "avg_view_seconds": totem.last_avg_view_seconds,
                "completion_rate": totem.last_completion_rate,
                "reward": totem.last_reward,
                "female_share": totem.female_share,
                "mean_age": totem.mean_age,
                "age_distribution": dict(totem.age_distribution),
                "flow_per_minute": totem.flow_per_minute,
                "plays": totem.plays,
                "decision_trace": self.intelligence.trace(totem.current_decision_id),
            }

    def metrics(self) -> dict:
        with self.lock:
            observed = self._mean(self.recent_observed)
            policy_expected = self._mean(self.recent_policy_expected)
            random_baseline = self._mean(self.recent_random)
            oracle = self._mean(self.recent_oracle)
            uplift = ((policy_expected / random_baseline) - 1.0) if random_baseline > 1e-9 else 0.0
            return {
                "config": asdict(self.config),
                "evaluation_version": VERSION,
                "paired_evidence": self.evidence.summary(),
                "observed_reward": observed,
                "policy_expected": policy_expected,
                "random_baseline": random_baseline,
                "oracle_ceiling": oracle,
                "uplift_vs_random": uplift,
                "mean_regret": self._mean(self.recent_regret),
                "cumulative_regret": self.cumulative_regret,
                "exploration_rate": self._mean(self.recent_exploration),
                "creative_diversity": self._creative_diversity(),
                "people": len(self.population.people),
                "decisions": self.total_decisions,
                "feedback_events": self.total_feedback,
                "no_fill_slots": self.no_fill_slots,
                "simulation_time": self.sim_time.isoformat(),
                "day": self.sim_time.date().isoformat(),
                "speed": self.speed,
                "paused": self.paused,
                "intelligence": self.intelligence.diagnostics(),
                "audit": self.audit.status(),
                "history": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "observed_reward": p.observed_reward,
                        "policy_expected": p.policy_expected,
                        "random_baseline": p.random_baseline,
                        "oracle_ceiling": p.oracle_ceiling,
                        "uplift_vs_random": p.uplift_vs_random,
                        "mean_regret": p.mean_regret,
                        "exploration_rate": p.exploration_rate,
                        "creative_diversity": p.creative_diversity,
                        "model_uncertainty": p.model_uncertainty,
                        "people": p.people,
                        "decisions": p.decisions,
                    }
                    for p in self.metric_history
                ],
            }

    def snapshot(self) -> dict:
        with self.lock:
            ads = {ad.ad_id: ad.name for ad in self.store.active_ads()}
            return {
                "configured": True,
                "last_error": self.last_error,
                "running": self.running,
                "paused": self.paused,
                "speed": self.speed,
                "sim_time": self.sim_time.isoformat(),
                "people_count": len(self.population.people),
                "people": self.population.serialized_points(),
                "totems": [
                    {
                        "id": t.totem_id,
                        "x": round(t.x, 4),
                        "y": round(t.y, 4),
                        "region": t.region,
                        "ad_id": t.current_ad_id,
                        "ad_name": ads.get(t.current_ad_id or "", "—"),
                        "reach": t.last_reach,
                        "impressions": t.last_impressions,
                        "retention": round(t.last_avg_view_seconds, 2),
                        "reward": round(t.last_reward, 4),
                        "female_share": round(t.female_share, 3),
                        "mean_age": round(t.mean_age, 1),
                        "flow": round(t.flow_per_minute, 1),
                    }
                    for t in self.city.totems
                ],
                "metrics": {
                    "paired_evidence": self.evidence.summary(),
                    "observed_reward": self._mean(self.recent_observed),
                    "policy_expected": self._mean(self.recent_policy_expected),
                    "random_baseline": self._mean(self.recent_random),
                    "oracle_ceiling": self._mean(self.recent_oracle),
                    "uplift_vs_random": (
                        self._mean(self.recent_policy_expected) / self._mean(self.recent_random) - 1.0
                        if self._mean(self.recent_random) > 1e-9 else 0.0
                    ),
                    "mean_regret": self._mean(self.recent_regret),
                    "exploration_rate": self._mean(self.recent_exploration),
                    "creative_diversity": self._creative_diversity(),
                    "decisions": self.total_decisions,
                    "feedback_events": self.total_feedback,
                    "no_fill_slots": self.no_fill_slots,
                    "uncertainty": self.intelligence.diagnostics()[
                        "mean_parameter_uncertainty"
                    ],
                },
            }


class SimulationManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.engine: SimulationEngine | None = None

    def configure(self, config: SimConfig) -> SimulationEngine:
        config.validate()
        with self.lock:
            if self.engine:
                self.engine.stop()
            self.engine = SimulationEngine(config)
            return self.engine

    def get(self) -> SimulationEngine | None:
        with self.lock:
            return self.engine
