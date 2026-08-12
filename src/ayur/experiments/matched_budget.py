"""Action-count-matched control for the heterogeneous action space.

## The confound this exists to resolve

The headline multi-tool result compares agents under an equal *cost* budget.
The cost sweep exposed what that hides: when tools are priced at 0.02 of a
patient question, a budget of 15.0 buys the tool agent ~565 observations while
the patient-only agent gets 15. It wins, but partly because it simply saw 38x
more evidence.

That makes the equal-cost comparison ambiguous between two very different claims:

  (A) "choosing cheaper sources buys more information per unit cost"   - economic
  (B) "choosing *among* sources is better than always asking the patient
       at equal information" - a claim about selection

Both are defensible, but they are not the same claim and the paper must make the
right one. This experiment isolates (B) by giving every agent **exactly N
observations**, ignoring cost entirely. If tool-EIG still wins, source selection
carries real signal beyond economics. If it does not, the contribution is purely
economic and must be described that way.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import UNKNOWN, PatientEnvironment
from ayur.experiments import metrics as M
from ayur.kg.graph import KnowledgeGraph
from ayur.planner.actions import ActionKind, ActionSpace, action_mutual_information
from ayur.planner.posterior import Posterior

RESULTS = Path("results")


def run_agent(case, env, action_space, space, n_actions: int, noise: float,
              patient_only: bool, rng_seed: int):
    """Take exactly `n_actions` observations, choosing by pure EIG (no cost)."""
    post = Posterior(space.matrix, noise=noise)
    taken = np.zeros(len(action_space), dtype=bool)
    for k, v in case.revealed.items():
        post.update(k, v)
        action_space.block_feature(taken, k)

    if patient_only:
        allowed = action_space.kind == ActionKind.ASK_PATIENT.value
        if not allowed.any():
            raise RuntimeError("patient-only agent has no actions available")
    else:
        allowed = np.ones(len(action_space), dtype=bool)

    rng = np.random.default_rng([rng_seed, case.index])
    kinds = Counter()

    for _ in range(n_actions):
        mi = action_mutual_information(action_space, post.belief)
        mi = np.where(allowed & ~taken, mi, -np.inf)
        if not np.isfinite(mi.max()):
            break
        best = mi.max()
        tied = np.flatnonzero(mi >= best - 1e-12)
        idx = int(rng.choice(tied)) if tied.size > 1 else int(mi.argmax())

        action = action_space.actions[idx]
        taken[idx] = True
        kinds[action.kind.value] += 1

        feature = action_space.feature_of[idx]
        if action.kind is ActionKind.ASK_PATIENT:
            obs = env.ask(case, feature)
            if obs == UNKNOWN:
                continue
        elif action.kind is ActionKind.VERIFY_HERB:
            compat = action_space._herb_compat_cache[case.condition, action.target]
            eps = action_space.noise[ActionKind.VERIFY_HERB]
            r = np.random.default_rng([rng_seed, case.index, 5_000_000 + action.target])
            obs = int(1 - compat) if r.random() < eps else int(compat)
        else:
            truth = int(case.truth[feature])
            eps = action_space.noise[action.kind]
            r = np.random.default_rng([rng_seed, case.index, feature,
                                       hash(action.kind.value) % 10_000])
            obs = int(1 - truth) if r.random() < eps else truth

        post._logp = post._logp + (
            action_space._log_p1[idx] if obs else action_space._log_p0[idx])
        if feature >= 0:
            action_space.block_feature(taken, feature)

    return post.argmax == case.condition, post.confidence, kinds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budgets", type=int, nargs="*", default=[5, 10, 15, 20, 30])
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.10)
    ap.add_argument("--exclusive", action="store_true",
                    help="partition channel access by who can actually answer")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    space = build_condition_space()
    kg = KnowledgeGraph.build()
    action_space = ActionSpace(space, kg=kg, exclusive=args.exclusive)
    env = PatientEnvironment(space, noise=args.noise,
                             omission_rate=args.omission, seed=0)
    tag = args.tag or ("exclusive" if args.exclusive else "redundant")

    print("=" * 92)
    print(f"ACTION-COUNT-MATCHED CONTROL  [{tag} channels]")
    print("=" * 92)
    print(f"  n={args.n} cases, budgets {args.budgets} OBSERVATIONS (cost ignored)")
    print("  isolates source selection from cost economics")
    print(f"  channel scope: {action_space.summary()['by_kind']}")
    if args.exclusive:
        print("  patient cannot report dosha/prakriti; only the KG can supply them")
    print("-" * 92)
    print(f"{'budget':>8}{'all-sources':>14}{'patient-only':>15}{'delta':>9}"
          f"{'95% CI':>20}{'p':>11}{'tool share':>12}")
    print("-" * 92)

    rows = []
    t0 = time.perf_counter()
    for budget in args.budgets:
        multi, single, mix = [], [], Counter()
        for case in env.cases(args.n):
            c1, _, k1 = run_agent(case, env, action_space, space, budget,
                                  args.noise, patient_only=False, rng_seed=11)
            c2, _, _ = run_agent(case, env, action_space, space, budget,
                                 args.noise, patient_only=True, rng_seed=11)
            multi.append(c1)
            single.append(c2)
            mix.update(k1)

        a = np.array(multi, dtype=bool)
        b = np.array(single, dtype=bool)
        delta, lo, hi, _ = M.paired_bootstrap_ci(a, b)
        _, _, p = M.mcnemar(a, b)
        total = max(sum(mix.values()), 1)
        share = 1 - mix.get(ActionKind.ASK_PATIENT.value, 0) / total

        row = {
            "budget_actions": budget,
            "n": args.n,
            "all_sources_accuracy": round(float(a.mean()), 4),
            "patient_only_accuracy": round(float(b.mean()), 4),
            "delta": round(delta, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_mcnemar": p,
            "significant": bool(p < 0.05),
            "non_patient_share": round(share, 3),
            "tool_mix": {k: round(v / total, 3) for k, v in sorted(mix.items())},
        }
        rows.append(row)
        ci = f"[{lo:+.3f}, {hi:+.3f}]"
        pstr = "<1e-16" if p < 1e-16 else f"{p:.1e}"
        print(f"{budget:>8}{100*row['all_sources_accuracy']:>13.1f}%"
              f"{100*row['patient_only_accuracy']:>14.1f}%{delta:>+9.3f}{ci:>20}"
              f"{pstr:>11}{share:>12.2f}", flush=True)

    sig_pos = [r for r in rows if r["significant"] and r["delta"] > 0]
    out = {
        "tag": tag,
        "exclusive_channels": args.exclusive,
        "channel_scope": action_space.summary()["by_kind"],
        "config": vars(args),
        "rows": rows,
        "summary": {
            "n_significant_positive": len(sig_pos),
            "n_budgets": len(rows),
            "mean_delta": round(float(np.mean([r["delta"] for r in rows])), 4),
            "interpretation": (
                "source selection helps at matched observation count"
                if len(sig_pos) >= len(rows) / 2
                else "advantage is primarily economic, not selection"),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"matched_budget_{tag}.json").write_text(json.dumps(out, indent=2))

    print("-" * 92)
    print(f"  significant and positive at "
          f"{out['summary']['n_significant_positive']}/{out['summary']['n_budgets']} budgets")
    print(f"  mean delta {out['summary']['mean_delta']:+.3f}")
    print(f"  -> {out['summary']['interpretation']}")
    print("=" * 92)
    print(f"written to {RESULTS}/matched_budget_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
