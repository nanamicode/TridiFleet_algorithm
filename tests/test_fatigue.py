from datetime import datetime, timezone

from tridifleet.models import Ad
from tridifleet.sim.domain import Person
from tridifleet.sim.ground_truth import GroundTruthModel


def test_repeated_exposure_creates_hidden_frequency_fatigue():
    truth = GroundTruthModel(9)
    ad = Ad(ad_id="A", name="A", duration_seconds=10, tags=["food"])
    person = Person(
        person_id=1, x=0, y=0, age=30, female=False, speed_kmh=4.5,
        interests={"food"}, attention_propensity=0.2, route=[(0,0),(1,0)],
    )
    ts = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)

    fresh = truth.person_expectation(person, 0.005, 0.025, ad, ts, 2)[0]
    person.ad_exposures[ad.ad_id] = 5
    fatigued = truth.person_expectation(person, 0.005, 0.025, ad, ts, 2)[0]

    assert fatigued < fresh
