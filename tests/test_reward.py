from tridifleet.models import Feedback
from tridifleet.reward import evidence_weight, retention_reward


def test_reward_is_bounded_and_attention_sensitive():
    low = Feedback(
        decision_id="x", reach=100, impressions=20,
        avg_view_seconds=2, ad_duration_seconds=10,
    )
    high = Feedback(
        decision_id="x", reach=100, impressions=70,
        avg_view_seconds=8, ad_duration_seconds=10,
    )
    assert 0 <= retention_reward(low) < retention_reward(high) <= 1


def test_zero_reach_has_zero_evidence():
    fb = Feedback(
        decision_id="x", reach=0, impressions=0,
        avg_view_seconds=0, ad_duration_seconds=10,
    )
    assert evidence_weight(fb) == 0
