"""Read-only, paired potential outcomes. Never exposed to the learner."""
from dataclasses import dataclass
import hashlib
import math
from collections import OrderedDict
import numpy as np
from ..models import Feedback
from ..config import settings

VERSION = 'paired_outcomes_v3'

@dataclass
class OutcomeBatch:
    expected: np.ndarray
    realized: np.ndarray
    impressions: np.ndarray
    dwell: np.ndarray
    completions: np.ndarray

    def feedback(self, index, decision_id, ad, reach):
        return Feedback(decision_id=decision_id, reach=reach,
            impressions=int(self.impressions[index]), avg_view_seconds=float(self.dwell[index]),
            ad_duration_seconds=ad.duration_seconds, completion_rate=float(self.completions[index]))


def outcomes(truth, audience, ads, timestamp, radius, exposure_seconds, key, draws=64):
    """Integrate the actual nonlinear reward, including zero-impression windows.

    All ads share exogenous uniforms/normals. Row 0 supplies actual feedback;
    disjoint rows 1..draws estimate its expectation. No sampling depends on choice.
    """
    n, k = len(audience), len(ads)
    if not n:
        z=np.zeros(k)
        return OutcomeBatch(z.copy(),z.copy(),z.copy(),z.copy(),z.copy())
    seed=int.from_bytes(hashlib.blake2b(f'{truth.seed}|{key}'.encode(),digest_size=8).digest(),'big')
    rng=np.random.default_rng(seed)
    uniforms=rng.random((draws+1,n)); noise=rng.normal(0,.13,(draws+1,n))
    expected=[]; realized=[]; impressions=[]; dwell=[]; completions=[]
    for ad in ads:
        params=[truth.person_expectation(p,d,radius,ad,timestamp,n) for p,d in audience]
        gaze=np.array([v[0] for v in params]); retention=np.array([v[1] for v in params])
        caps=np.array([exposure_seconds.get(p.person_id,0)/ad.duration_seconds for p,d in audience])
        ratios=np.minimum(np.clip(retention+noise,.02,1),caps)
        looked=uniforms<gaze
        counts=looked.sum(axis=1)
        means=np.divide((ratios*looked).sum(axis=1),counts,out=np.zeros(draws+1),where=counts>0)
        rewards=(counts/n)**settings.impression_exponent * means**settings.dwell_exponent
        expected.append(float(rewards[1:].mean()));realized.append(float(rewards[0]))
        impressions.append(int(counts[0]));dwell.append(float(means[0]*ad.duration_seconds))
        completions.append(float(((ratios[0]>=.85)&looked[0]).sum()/counts[0]) if counts[0] else 0)
    return OutcomeBatch(*[np.asarray(x) for x in (expected,realized,impressions,dwell,completions)])


class PairedEvidence:
    """Bounded block summaries; uncertainty is descriptive, not a sequential test."""
    def __init__(self):
        self.blocks=OrderedDict()
        self.total_slots=0
        self.total_people=0
        self.selected_sum=0.
        self.baseline_sum=0.

    def add(self, timestamp, totem_id, chosen, random, reach):
        if not reach:return
        key=f'{timestamp.date()}:{timestamp.hour}:{totem_id}'
        b=self.blocks.setdefault(key,[0,0.,0.])
        b[0]+=1;b[1]+=chosen;b[2]+=random
        if len(self.blocks)>10000:self.blocks.popitem(last=False)
        self.total_slots+=1;self.total_people+=reach
        self.selected_sum+=chosen;self.baseline_sum+=random

    def summary(self):
        means=[(b[1]-b[2])/b[0] for b in self.blocks.values()]
        count=len(means)
        delta=float(np.mean(means)) if count else 0.
        se=float(np.std(means,ddof=1)/math.sqrt(count)) if count>1 else None
        return dict(version=VERSION, slots=self.total_slots, exposed_people=self.total_people,
            observed_policy=self.selected_sum/self.total_slots if self.total_slots else 0,
            paired_random=self.baseline_sum/self.total_slots if self.total_slots else 0,
            block_count=count, block_mean_difference=delta,
            descriptive_interval_95=[delta-2*se,delta+2*se] if count>=20 else None,
            status='descriptive_only' if count>=20 else 'insufficient_blocks',
            limitation='Same-audience one-slot comparison, conditional on policy history. Blocks may remain correlated; not an independent campaign A/B or sequential significance test.')
