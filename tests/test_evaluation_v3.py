from datetime import datetime,timezone
import numpy as np
from tridifleet.models import Ad
from tridifleet.sim.domain import Person,SimConfig
from tridifleet.sim.ground_truth import GroundTruthModel
from tridifleet.sim.evaluation import outcomes
from tridifleet.sim.engine import SimulationEngine
from tridifleet.sim.checkpoint import save,restore

def person():
    return Person(1,0,0,30,False,4.5,{'food'},.2,[(0,0),(1,0)])

class ConstantWorld(GroundTruthModel):
    def person_expectation(self,*args):return .4,.6,0


def test_null_world_cannot_invent_paired_uplift():
    p=person();ads=[Ad(ad_id=str(i),name=str(i),duration_seconds=10) for i in range(6)]
    b=outcomes(ConstantWorld(1),[(p,0)],ads,datetime.now(timezone.utc),.006,{1:10},'null',10000)
    assert np.ptp(b.expected)==0
    assert np.ptp(b.realized)==0
    assert p.ad_exposures=={}
    assert .28<b.expected[0]<.31  # old plug-in incorrectly returned ~.510


def test_exposure_caps_and_no_audience():
    p=person();ad=Ad(ad_id='a',name='a',duration_seconds=10)
    b=outcomes(ConstantWorld(1),[(p,0)],[ad],datetime.now(timezone.utc),.006,{1:.5},'cap',100)
    assert b.dwell[0]<=.5
    assert b.completions[0]==0
    z=outcomes(ConstantWorld(1),[],[ad],datetime.now(timezone.utc),.006,{},'empty')
    assert z.expected[0]==z.realized[0]==0


def test_checkpoint_restores_pending_learning_and_future_exactly(tmp_path,monkeypatch):
    monkeypatch.setenv('TRIDIFLEET_AUDIT_DB',':memory:')
    a=SimulationEngine(SimConfig(n_totems=3,radius_km=.7,seed=76))
    a.step(125.5)
    path=tmp_path/'state.gz';save(a,path);b=restore(path)
    try:
        a.step(180);b.step(180)
        assert a.population.serialized_points()==b.population.serialized_points()
        ma,mb=a.metrics(),b.metrics();ma.pop('audit');mb.pop('audit')
        assert ma==mb
        assert np.array_equal(a.intelligence.shared.mean,b.intelligence.shared.mean)
        assert np.array_equal(a.intelligence.shared.var,b.intelligence.shared.var)
    finally:a.stop();b.stop()
