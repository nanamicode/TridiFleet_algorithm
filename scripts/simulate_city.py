"""Small synthetic city simulation for validating learning behavior.

Run:
    python scripts/simulate_city.py
"""
from datetime import datetime, timezone
import random

from tridifleet.bandit import HierarchicalThompsonBandit
from tridifleet.context import context_keys
from tridifleet.models import Ad, ContextEvent
from tridifleet.store import MemoryStore


TRUE_REWARD = {
    ("morning", "coffee"): 0.82,
    ("morning", "makeup"): 0.30,
    ("morning", "bakery"): 0.70,
    ("afternoon", "coffee"): 0.42,
    ("afternoon", "makeup"): 0.74,
    ("afternoon", "bakery"): 0.48,
    ("evening", "coffee"): 0.28,
    ("evening", "makeup"): 0.55,
    ("evening", "bakery"): 0.86,
}


def main(rounds: int = 5000) -> None:
    rng = random.Random(42)
    store = MemoryStore()
    ads = ["coffee", "makeup", "bakery"]
    for ad_id in ads:
        store.put_ad(Ad(ad_id=ad_id, name=ad_id.title(), duration_seconds=10))

    bandit = HierarchicalThompsonBandit(store, seed=42)
    selected = {k: {a: 0 for a in ads} for k in ("morning", "afternoon", "evening")}

    hours = {"morning": 8, "afternoon": 14, "evening": 19}
    for i in range(rounds):
        period = rng.choice(list(hours))
        event = ContextEvent(
            totem_id="sim-01",
            timestamp=datetime(2026, 9, 5, hours[period], i % 60, tzinfo=timezone.utc),
            region="center",
            location_id="sim-square",
            female_share=0.65 if period == "afternoon" else 0.48,
            mean_age=27 if period != "evening" else 38,
        )
        store.put_context(event)
        decision = bandit.choose(event.totem_id)
        selected[period][decision.ad_id] += 1

        # Noisy bounded reward around the hidden ground truth.
        truth = TRUE_REWARD[(period, decision.ad_id)]
        observed = max(0.0, min(1.0, rng.gauss(truth, 0.12)))
        store.update(decision.ad_id, context_keys(event), observed, weight=1.0)

    print("Selections by daypart after learning:")
    for period, counts in selected.items():
        best = max(counts, key=counts.get)
        print(f"{period:10s} -> {counts} | learned favorite={best}")


if __name__ == "__main__":
    main()
