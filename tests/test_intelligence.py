from datetime import datetime, timezone
import inspect

from tridifleet import intelligence as intelligence_module
from tridifleet.intelligence import HybridRetentionIntelligence
from tridifleet.models import Ad, ContextEvent
from tridifleet.store import MemoryStore


def make_ctx():
    return ContextEvent(
        totem_id="T1",
        timestamp=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        region="centro",
        x_km=0.2,
        y_km=-0.1,
        reach_window=30,
        impressions_window=8,
        female_share=0.58,
        mean_age=32,
        age_std=11,
        flow_per_minute=30,
        crowd_density=0.7,
    )


def test_shared_model_learns_tag_context_signal():
    store = MemoryStore()
    intel = HybridRetentionIntelligence(store, seed=11)
    ctx = make_ctx()
    food = Ad(ad_id="food", name="Food", duration_seconds=10, tags=["food", "family"])
    tech = Ad(ad_id="tech", name="Tech", duration_seconds=10, tags=["technology"])

    xf = intel.encoder.encode(food, ctx)
    xt = intel.encoder.encode(tech, ctx)

    for _ in range(80):
        intel.shared.update(xf, 0.85, 1.0)
        intel.shared.update(xt, 0.18, 1.0)

    assert intel.shared.predict_mean(xf) > intel.shared.predict_mean(xt)


def test_intelligence_has_no_ground_truth_dependency():
    source = inspect.getsource(intelligence_module)
    assert "sim.ground_truth" not in source
    assert "GroundTruthModel" not in source
