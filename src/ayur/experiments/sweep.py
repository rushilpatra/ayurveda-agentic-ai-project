"""Accuracy-vs-question-budget curves - the paper's headline figure.

One pass per (policy, case): the posterior's argmax is recorded after every
question, so a single run yields the whole curve rather than one point per
budget. Run under both the well-specified and the misspecified environment so
the two curves can be shown together.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import UNKNOWN, PatientEnvironment
from ayur.planner.policy import EIGPolicy, FrequencyPolicy, RandomPolicy
from ayur.planner.posterior import Posterior

RESULTS = Path("results")

POLICIES = {
    "random": RandomPolicy,
    "greedy-frequency": FrequencyPolicy,
    "max-eig": EIGPolicy,
}


def curve_for_policy(space, env, policy_factory, n_cases, budget, assumed_noise, start=0):
    """Returns (accuracy_at_q, mean_entropy_at_q, mean_confidence_at_q), each length budget+1."""
    hits = np.zeros(budget + 1)
    entropy = np.zeros(budget + 1)
    confidence = np.zeros(budget + 1)

    for case in env.cases(n_cases, start=start):
        post = Posterior(space.matrix, noise=assumed_noise)
        asked = np.zeros(space.n_features, dtype=bool)
        for k, v in case.revealed.items():
            post.update(k, v)
            asked[k] = True

        rng = np.random.default_rng([env.seed, case.index, 7919])
        policy = policy_factory()

        hits[0] += post.argmax == case.condition
        entropy[0] += post.entropy
        confidence[0] += post.confidence

        for t in range(1, budget + 1):
            k = policy.select(post, asked, rng)
            asked[k] = True
            ans = env.ask(case, k)
            if ans != UNKNOWN:
                post.update(k, ans)
            hits[t] += post.argmax == case.condition
            entropy[t] += post.entropy
            confidence[t] += post.confidence

    return hits / n_cases, entropy / n_cases, confidence / n_cases


def run(n_cases: int, budget: int, settings: dict) -> dict:
    space = build_condition_space()
    out = {
        "n_cases": n_cases,
        "budget": budget,
        "n_conditions": space.n_conditions,
        "n_features": space.n_features,
        "settings": {},
    }

    for setting_name, cfg in settings.items():
        env = PatientEnvironment(
            space, noise=cfg["env_noise"], omission_rate=cfg["omission"], seed=0
        )
        block = {"config": cfg, "policies": {}}
        print(f"\n  [{setting_name}] env_noise={cfg['env_noise']} "
              f"omission={cfg['omission']} agent_assumes={cfg['assumed_noise']}")
        for pname, factory in POLICIES.items():
            t0 = time.perf_counter()
            acc, ent, conf = curve_for_policy(
                space, env, factory, n_cases, budget, cfg["assumed_noise"]
            )
            block["policies"][pname] = {
                "accuracy": [round(float(x), 5) for x in acc],
                "entropy": [round(float(x), 5) for x in ent],
                "confidence": [round(float(x), 5) for x in conf],
            }
            print(f"     {pname:<18} q={budget}: {100*acc[budget]:5.1f}%   "
                  f"({time.perf_counter()-t0:.1f}s)")
        out["settings"][setting_name] = block

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "budget_sweep.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--budget", type=int, default=20)
    args = ap.parse_args()

    settings = {
        "well-specified": {"env_noise": 0.05, "omission": 0.10, "assumed_noise": 0.05},
        "misspecified": {"env_noise": 0.15, "omission": 0.25, "assumed_noise": 0.05},
    }

    print("=" * 70)
    print("BUDGET SWEEP: accuracy vs number of questions")
    print("=" * 70)
    print(f"  cases {args.n}   budget {args.budget}   policies {list(POLICIES)}")
    out = run(args.n, args.budget, settings)

    print()
    print("-" * 70)
    for name, block in out["settings"].items():
        print(f"  {name}")
        header = "    " + "".join(f"q{q:<6}" for q in range(0, args.budget + 1, 4))
        print(header)
        for pname, data in block["policies"].items():
            row = "".join(f"{100*data['accuracy'][q]:<7.1f}"
                          for q in range(0, args.budget + 1, 4))
            print(f"    {row}  {pname}")
    print("=" * 70)
    print(f"written to {RESULTS}/budget_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
