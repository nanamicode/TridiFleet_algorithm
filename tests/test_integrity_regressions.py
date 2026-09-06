from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from tridifleet.intelligence import HybridRetentionIntelligence
from tridifleet.models import Ad, ContextEvent, Feedback
from tridifleet.sim.domain import SimConfig
from tridifleet.sim.engine import SimulationEngine
from tridifleet.store import MemoryStore


def learner():
    store = MemoryStore()
    store.put_ad(Ad(ad_id='A', name='Teste', duration_seconds=10))
    store.put_context(ContextEvent(totem_id='T', timestamp=datetime.now(timezone.utc)))
    return HybridRetentionIntelligence(store, seed=4)


def test_retries_are_atomic_and_do_not_retrain():
    model = learner()
    decision = model.choose('T')
    fb = Feedback(decision_id=decision.decision_id, reach=10, impressions=5,
                  avg_view_seconds=3, ad_duration_seconds=10)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(model.apply_feedback, [fb] * 30))
    assert model.total_updates == 1
    assert sum(not r['duplicate'] for r in results) == 1
    assert not model._decision_features
    assert not model.store.decisions
    with pytest.raises(ValueError, match='conflicting'):
        model.apply_feedback(fb.model_copy(update={'impressions': 6}))


def test_empty_slots_release_pending_memory():
    model = learner()
    for _ in range(100):
        d = model.choose('T')
        model.apply_feedback(Feedback(decision_id=d.decision_id, reach=0,
            impressions=0, avg_view_seconds=0, ad_duration_seconds=10))
    assert model.total_updates == 0
    assert not model._decision_features and not model.store.decisions


@pytest.mark.parametrize('change', [dict(reach=0, impressions=1),
    dict(avg_view_seconds=11), dict(avg_view_seconds=float('nan')),
    dict(impressions=0, avg_view_seconds=1)])
def test_invalid_telemetry_is_rejected(change):
    values = dict(decision_id='x', reach=10, impressions=2,
                  avg_view_seconds=2, ad_duration_seconds=10)
    values.update(change)
    with pytest.raises(ValidationError):
        Feedback(**values)


def test_observation_and_clock_grouping_do_not_change_world(monkeypatch):
    monkeypatch.setenv('TRIDIFLEET_AUDIT_DB', ':memory:')
    a = SimulationEngine(SimConfig(n_totems=3, radius_km=.7, seed=76))
    b = SimulationEngine(SimConfig(n_totems=3, radius_km=.7, seed=76))
    try:
        a.step(180)
        for _ in range(720):
            b.step(.25)
            b.population.serialized_points(limit=2)
        assert a.population.rng.getstate() == b.population.rng.getstate()
        assert a.population.serialized_points() == b.population.serialized_points()
        ma, mb = a.metrics(), b.metrics()
        ma.pop("audit"); mb.pop("audit")
        assert ma == mb
    finally:
        a.stop(); b.stop()


def test_seed_is_reproducible_across_python_processes():
    code = '''import json
from tridifleet.models import Ad
from tridifleet.sim.ground_truth import GroundTruthModel
from dataclasses import asdict
print(json.dumps(asdict(GroundTruthModel(42).profile(Ad(ad_id='a',name='A',tags=['food','home'],duration_seconds=10))),sort_keys=True))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code],
               env={**os.environ, 'PYTHONHASHSEED': str(seed)}) for seed in (1, 73)]
    assert outputs[0] == outputs[1]


def test_budget_is_reserved_before_other_totems_choose(monkeypatch):
    monkeypatch.setenv('TRIDIFLEET_AUDIT_DB', ':memory:')
    engine = SimulationEngine(SimConfig(n_totems=5, radius_km=.7))
    try:
        engine.store.ads.clear()
        engine.add_creative(Ad(ad_id='only',name='Only',duration_seconds=10,
                              daily_budget=.06,cost_per_play=.06))
        engine.step(1)
        assert engine.total_decisions == 1
        assert engine.no_fill_slots == 4
        assert engine.spend_today['only'] == .06
    finally:
        engine.stop()
