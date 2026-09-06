from datetime import datetime, timezone

from tridifleet.context import context_keys
from tridifleet.intelligence import FeatureEncoder
from tridifleet.models import Ad, ContextEvent


def test_dominant_age_distribution_drives_hierarchical_context():
    event = ContextEvent(
        totem_id="T",
        timestamp=datetime(2026, 9, 5, 14, tzinfo=timezone.utc),
        female_share=0.55,
        mean_age=39,
        age_distribution={"18-24": 0.62, "25-34": 0.18, "45-59": 0.20},
    )
    keys = context_keys(event)
    assert any("age_18_24" in key for key in keys)


def test_age_distribution_changes_dense_feature_vector():
    ad = Ad(ad_id="a", name="A", duration_seconds=10, tags=["food"])
    base = dict(
        totem_id="T",
        timestamp=datetime(2026, 9, 5, 14, tzinfo=timezone.utc),
        female_share=0.5,
        mean_age=40,
    )
    young = ContextEvent(**base, age_distribution={"18-24": 1.0})
    older = ContextEvent(**base, age_distribution={"60+": 1.0})
    enc = FeatureEncoder()
    assert not (enc.encode(ad, young) == enc.encode(ad, older)).all()
