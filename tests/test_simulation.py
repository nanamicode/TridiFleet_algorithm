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


def test_policy_evaluation_metrics_are_well_ordered():
    engine = SimulationEngine(SimConfig(n_totems=18, radius_km=1.3, seed=12))
    for _ in range(30):
        engine.step(30.0)
    m = engine.metrics()
    assert m["mean_regret"] >= 0.0
    assert m["oracle_ceiling"] + 1e-9 >= m["policy_expected"]
    assert m["oracle_ceiling"] + 1e-9 >= m["random_baseline"]
    assert 0.0 <= m["exploration_rate"] <= 1.0
    assert 0.0 <= m["creative_diversity"] <= 1.0


def test_counterfactual_evaluation_is_delayed_until_slot_completion():
    engine = SimulationEngine(SimConfig(n_totems=4, radius_km=0.8, seed=22))
    engine.step(1.0)
    assert engine.total_decisions > 0
    assert len(engine.recent_policy_expected) == 0

    engine.step(engine.config.decision_interval_sim_seconds + 1.0)
    # A completed slot must either produce an audience-scored evaluation or be
    # explicitly skipped only because no pedestrian crossed any totem.
    assert engine.total_feedback > 0
    assert len(engine.pending_evaluation) == engine.config.n_totems


def test_exhausted_budgets_create_no_fill_without_overspend():
    engine = SimulationEngine(SimConfig(n_totems=2, radius_km=0.7, seed=31))
    for ad in engine.store.ads.values():
        engine.spend_today[ad.ad_id] = ad.daily_budget

    engine.step(1.0)

    assert engine.no_fill_slots == engine.config.n_totems
    assert all(t.current_ad_id is None for t in engine.city.totems)
    assert all(
        engine.spend_today.get(ad.ad_id, 0.0) <= ad.daily_budget + 1e-9
        for ad in engine.store.ads.values()
    )
