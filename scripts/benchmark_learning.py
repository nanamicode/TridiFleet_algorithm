"""Deterministic end-to-end learning benchmark for the digital twin.

This is deliberately independent from browser/UI code. It exercises the same
SimulationEngine and HybridRetentionIntelligence used by the local lab.
"""
from __future__ import annotations

import argparse
import os

os.environ["TRIDIFLEET_AUDIT_DB"] = ":memory:"

from tridifleet.sim.domain import SimConfig
from tridifleet.sim.engine import SimulationEngine


def run(hours: float, seed: int = 31415) -> dict:
    engine = SimulationEngine(
        SimConfig(
            n_totems=10,
            radius_km=0.9,
            seed=seed,
            decision_interval_sim_seconds=60.0,
        )
    )
    steps = int(hours * 3600 / 10)
    for _ in range(steps):
        engine.step(10.0)

    metrics = engine.metrics()
    engine.stop()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    hours = 3.0 if args.quick else 12.0
    m = run(hours)

    print("=== TridiFleet learning benchmark ===")
    print(f"simulated_hours={hours:.1f}")
    print(f"decisions={m['decisions']}")
    print(f"feedback_events={m['feedback_events']}")
    print(f"observed_reward={m['observed_reward']:.6f}")
    print(f"policy_expected={m['policy_expected']:.6f}")
    print(f"random_baseline={m['random_baseline']:.6f}")
    print(f"oracle_ceiling={m['oracle_ceiling']:.6f}")
    print(f"uplift_vs_random={m['uplift_vs_random']:.6f}")
    print(f"mean_regret={m['mean_regret']:.6f}")
    print(f"exploration_rate={m['exploration_rate']:.6f}")
    print(f"creative_diversity={m['creative_diversity']:.6f}")
    print(
        "model_uncertainty="
        f"{m['intelligence']['mean_parameter_uncertainty']:.6f}"
    )

    assert m["decisions"] > 500, "benchmark did not generate enough decisions"
    assert m["feedback_events"] > 400, "benchmark did not generate enough feedback"
    assert m["oracle_ceiling"] + 1e-9 >= m["policy_expected"]
    assert m["oracle_ceiling"] + 1e-9 >= m["random_baseline"]
    assert m["mean_regret"] >= 0.0
    # Deterministic guardrail: a material regression in actual learning should
    # break CI instead of being hidden behind API/unit-test correctness.
    assert m["uplift_vs_random"] >= 0.08, (
        "retention policy lost its learning advantage over random rotation: "
        f"{m['uplift_vs_random']:.3%}"
    )


if __name__ == "__main__":
    main()
