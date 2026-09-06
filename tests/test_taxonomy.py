from datetime import datetime, timezone
import numpy as np

from tridifleet.intelligence import FeatureEncoder
from tridifleet.models import Ad, ContextEvent
from tridifleet.taxonomy import canonical_tag


def test_portuguese_and_english_tags_share_canonical_semantics():
    assert canonical_tag("supermercado") == canonical_tag("supermarket") == "grocery"
    assert canonical_tag("sabonete") == canonical_tag("hygiene") == "personal_care"

    ctx = ContextEvent(
        totem_id="T",
        timestamp=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
        female_share=0.5,
        mean_age=35,
    )
    a = Ad(
        ad_id="a", name="A", category="higiene",
        tags=["sabonete", "supermercado", "familia"], duration_seconds=10,
    )
    b = Ad(
        ad_id="b", name="B", category="hygiene",
        tags=["hygiene", "supermarket", "family"], duration_seconds=10,
    )
    enc = FeatureEncoder()
    assert np.allclose(enc.encode(a, ctx), enc.encode(b, ctx))


def test_daily_budget_is_not_a_retention_feature():
    ctx = ContextEvent(
        totem_id="T",
        timestamp=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
        female_share=0.5,
        mean_age=35,
    )
    low = Ad(ad_id="a", name="A", tags=["food"], duration_seconds=10, daily_budget=10)
    high = Ad(ad_id="b", name="B", tags=["food"], duration_seconds=10, daily_budget=10000)
    enc = FeatureEncoder()
    assert np.allclose(enc.encode(low, ctx), enc.encode(high, ctx))


def test_ground_truth_accepts_canonical_tags():
    from tridifleet.sim.ground_truth import GroundTruthModel
    ad = Ad(
        ad_id="soap", name="Sabonete", category="higiene",
        tags=["sabonete", "supermercado"], duration_seconds=10,
    )
    profile = GroundTruthModel(3).profile(ad)
    assert "health" in profile.interest_weights
    assert "home" in profile.interest_weights
