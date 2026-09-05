from tridifleet.models import Ad
from tridifleet.sim.domain import SimConfig
from tridifleet.sim.engine import SimulationEngine


def test_digital_twin_runs_without_browser_and_generates_decisions():
    engine = SimulationEngine(SimConfig(n_totems=12, radius_km=1.2, seed=7))
    for _ in range(20):
        engine.step(30.0)

    snap = engine.snapshot()
    assert snap["configured"] is True
    assert len(snap["totems"]) == 12
    assert snap["metrics"]["decisions"] > 0
    assert snap["people_count"] > 0


def test_speed_never_exceeds_nominal():
    engine = SimulationEngine(SimConfig(n_totems=2, radius_km=0.8, seed=1))
    assert engine.set_speed(99.0) == 1.0
    assert engine.set_speed(0.4) == 0.5
    assert engine.set_speed(0.1) == 0.25


def test_new_creative_enters_inventory_immediately():
    engine = SimulationEngine(SimConfig(n_totems=3, radius_km=0.8, seed=5))
    ad = Ad(
        ad_id="SOAP-NEW",
        name="Sabonete Novo",
        category="higiene",
        tags=["hygiene", "supermarket", "family"],
        duration_seconds=10,
        daily_budget=400,
    )
    engine.add_creative(ad)
    ids = {row["ad_id"] for row in engine.creative_stats()}
    assert "SOAP-NEW" in ids
