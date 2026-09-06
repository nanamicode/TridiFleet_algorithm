from datetime import datetime, timezone

from tridifleet.intelligence import HybridRetentionIntelligence
from tridifleet.models import Ad, ContextEvent
from tridifleet.store import MemoryStore


def test_daily_forgetting_reduces_old_evidence_and_restores_uncertainty():
    store = MemoryStore()
    ad = Ad(ad_id="a", name="A", duration_seconds=10, tags=["food"])
    store.put_ad(ad)
    ctx = ContextEvent(
        totem_id="T",
        timestamp=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
        female_share=0.5,
        mean_age=35,
    )
    store.put_context(ctx)
    intel = HybridRetentionIntelligence(store, seed=1)
    x = intel.encoder.encode(ad, ctx)

    for _ in range(40):
        intel.shared.update(x, 0.8, 1.0)
        store.update("a", ["global"], 0.8, 1.0)

    n_before = store.posterior("a", "global").observations
    var_before = float(intel.shared.var.mean())
    intel.advance_day(0.90)
    n_after = store.posterior("a", "global").observations
    var_after = float(intel.shared.var.mean())

    assert n_after < n_before
    assert var_after > var_before
