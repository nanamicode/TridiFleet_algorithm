"""Multi-seed evidence; no cherry-picking or guaranteed real-world uplift."""
import argparse
import json
import statistics
import time
from pathlib import Path
from benchmark_learning import run

parser = argparse.ArgumentParser()
parser.add_argument('--hours', type=float, default=3)
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 76, 31415, 2026, 90210])
parser.add_argument('--output', default='data/validation.json')
args = parser.parse_args()
results = []
for seed in args.seeds:
    started = time.monotonic()
    m = run(args.hours, seed)
    row = {k: m[k] for k in ('decisions','feedback_events','observed_reward',
        'policy_expected','random_baseline','oracle_ceiling','uplift_vs_random','mean_regret','paired_evidence')}
    row.update(seed=seed, elapsed_seconds=round(time.monotonic()-started,3))
    results.append(row)
    print(json.dumps(row), flush=True)
report = dict(hours_per_seed=args.hours, detection_radius_m=6, results=results,
    mean_uplift=statistics.mean(r['uplift_vs_random'] for r in results),
    min_uplift=min(r['uplift_vs_random'] for r in results),
    interpretation='v3: Monte Carlo of actual nonlinear rewards; paired observed control conditional on same history. Not independent campaign A/B.')
p = Path(args.output)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(report, indent=2)+'\n')
