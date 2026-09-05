from datetime import datetime, timezone

from tridifleet.bandit import HierarchicalThompsonBandit
from tridifleet.context import context_keys
from tridifleet.models import Ad, ContextEvent
from tridifleet.store import MemoryStore


def test_bandit_learns_better_ad():
    store = MemoryStore()
    store.put_ad(Ad(ad_id="good", name="Good", duration_seconds=10))
    store.put_ad(Ad(ad_id="bad", name="Bad", duration_seconds=10))
    event = ContextEvent(
        totem_id="t1",
        timestamp=datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc),
        region="center",
        female_share=0.65,
        mean_age=28,
    )
    store.put_context(event)
    keys = context_keys(event)

    # Synthetic historical evidence: same context, clearly different retention.
    for _ in range(100):
        store.update("good", keys, reward=0.80, weight=1.0)
        store.update("bad", keys, reward=0.20, weight=1.0)

    bandit = HierarchicalThompsonBandit(store, seed=7)
    wins = {"good": 0, "bad": 0}
    for _ in range(300):
        wins[bandit.choose("t1").ad_id] += 1

    assert wins["good"] > 285
