"""Versioned JSON checkpoints. No pickle or dynamic class imports."""
import gzip
import json
import os
from pathlib import Path
from dataclasses import fields, is_dataclass
from collections import deque, OrderedDict
from datetime import datetime, date
import numpy as np
from pydantic import BaseModel
from ..models import Ad, ContextEvent, Decision, Feedback
from ..store import Posterior
from .domain import Road, Zone, Person, TotemState, SimConfig, MetricPoint
from .city import CityMap
from .ground_truth import HiddenCreativeProfile

TYPES={c.__name__:c for c in (Ad,ContextEvent,Decision,Feedback,Posterior,Road,Zone,Person,TotemState,SimConfig,MetricPoint,CityMap,HiddenCreativeProfile)}

def pack(v):
    if isinstance(v,np.ndarray):return {'@':'array','v':v.tolist()}
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,datetime):return {'@':'datetime','v':v.isoformat()}
    if isinstance(v,date):return {'@':'date','v':v.isoformat()}
    if isinstance(v,BaseModel):return {'@':type(v).__name__,'v':pack(v.model_dump())}
    if is_dataclass(v):return {'@':type(v).__name__,'v':pack({f.name:getattr(v,f.name) for f in fields(v)})}
    if isinstance(v,dict):return {'@':'ordered' if isinstance(v,OrderedDict) else 'dict','v':[[pack(k),pack(x)] for k,x in v.items()]}
    if isinstance(v,deque):return {'@':'deque','maxlen':v.maxlen,'v':[pack(x) for x in v]}
    if isinstance(v,(tuple,set)):return {'@':'tuple' if isinstance(v,tuple) else 'set','v':[pack(x) for x in v]}
    if isinstance(v,list):return [pack(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)):return v
    raise TypeError(type(v).__name__)

def unpack(v):
    if isinstance(v,list):return [unpack(x) for x in v]
    if not isinstance(v,dict):return v
    kind=v['@'];data=v['v']
    if kind=='array':return np.asarray(data,dtype=float)
    if kind=='datetime':return datetime.fromisoformat(data)
    if kind=='date':return date.fromisoformat(data)
    if kind in ('dict','ordered'):return (OrderedDict if kind=='ordered' else dict)((unpack(k),unpack(x)) for k,x in data)
    if kind=='deque':return deque((unpack(x) for x in data),maxlen=v['maxlen'])
    if kind=='tuple':return tuple(unpack(x) for x in data)
    if kind=='set':return set(unpack(x) for x in data)
    if kind not in TYPES:raise ValueError('Unsupported checkpoint record')
    return TYPES[kind](**unpack(data))

EXCLUDE={'lock','rng','population','truth','sensor','store','intelligence','audit','run_id','_thread','_stop_event','evidence','_last_checkpoint_wall'}

def save(engine,path):
    with engine.lock, engine.intelligence._lock:
        engine.audit.flush()
        m=engine.intelligence
        data=dict(version=1,evaluation='paired_outcomes_v3',parent_run=engine.run_id,parent_tip=engine.audit.last_hash,
            engine={k:v for k,v in vars(engine).items() if k not in EXCLUDE},
            rng=engine.rng.getstate(),population_rng=engine.population.rng.getstate(),
            people=engine.population.people,next_id=engine.population.next_id,
            profiles=engine.truth._profiles,sensor_rng=engine.sensor.rng.getstate(),
            store={k:v for k,v in vars(engine.store).items() if k!='_lock'},
            model={k:v for k,v in vars(m).items() if k not in {'store','encoder','shared','residual','rng','_lock'}},
            shared=vars(m.shared),model_rng=m.rng.bit_generator.state,residual_rng=m.residual.rng.bit_generator.state,
            evidence=vars(engine.evidence))
        payload=json.dumps(pack(data),allow_nan=False,separators=(',',':')).encode()
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+'.tmp')
        with open(tmp,'wb') as f:
            f.write(gzip.compress(payload));f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)

def restore(path):
    from .engine import SimulationEngine
    with gzip.open(path,'rt') as f:data=unpack(json.load(f))
    if data['version']!=1 or data['evaluation']!='paired_outcomes_v3':raise ValueError('Incompatible checkpoint')
    e=SimulationEngine(data['engine']['config'])
    vars(e).update(data['engine']);e.rng.setstate(data['rng'])
    e.population.city=e.city;e.population.people=data['people'];e.population.next_id=data['next_id']
    e.population.rng.setstate(data['population_rng']);e.sensor.rng.setstate(data['sensor_rng'])
    e.truth._profiles=data['profiles'];vars(e.store).update(data['store'])
    vars(e.intelligence).update(data['model']);vars(e.intelligence.shared).update(data['shared'])
    e.intelligence.rng.bit_generator.state=data['model_rng'];e.intelligence.residual.rng.bit_generator.state=data['residual_rng']
    vars(e.evidence).update(data['evidence'])
    e.audit.append('checkpoint_restore',e.sim_time.isoformat(),dict(parent_run=data['parent_run'],parent_tip=data['parent_tip']))
    return e
