from datetime import datetime, timezone

from tridifleet.context import context_keys
from tridifleet.models import ContextEvent


def test_context_builds_hierarchy():
    event = ContextEvent(
        totem_id="55",
        timestamp=datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc),
        region="north",
        location_id="mall-a",
        female_share=0.65,
        mean_age=22,
    )
    keys = context_keys(event)
    assert "global" in keys
    assert "time:afternoon" in keys
    assert "demo:female_skew:age_18_24" in keys
    assert "region:north" in keys
    assert "location:mall-a" in keys
    assert "totem:55" in keys
