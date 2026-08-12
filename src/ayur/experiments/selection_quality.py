"""Does the agent choose the *right source*? A properly posed test.

## Why the earlier control asked the wrong question

`matched_budget.py` compared an all-sources agent against a patient-only agent
at equal observation count and found no benefit. That tests whether *having*
tools helps. It does not test whether the agent *chooses among them well*, which
is the claim we actually want to make.

Here every agent has the identical toolset and identical observation budget.
The only thing that varies is the rule for picking which source answers the next
question:

  eig-source       full EIG over (source, feature) pairs            [proposed]
  random-source    feature chosen by EIG, then a source drawn at random from
                   those able to answer it
  fixed-patient    feature chosen by EIG, answered by the patient when possible,
                   else by whichever source can
  fixed-cheapest   feature chosen by EIG, answered by the cheapest able source
  worst-source     feature chosen by EIG, answered by the *least* informative
                   able source                                      [floor]

Because the feature is chosen identically in every arm except the first, any
difference is attributable to source choice alone. `worst-source` bounds how
much source choice can matter at all: if the gap between best and worst is
small, sources are interchangeable and no selection claim is defensible
regardless of which rule wins.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import UNKNOWN, PatientEnvironment
from ayur.experiments import metrics as M
from ayur.kg.graph import KnowledgeGraph
from ayur.planner.actions import ActionKind, ActionSpace, action_mutual_information
from ayur.planner.posterior import Posterior

RESULTS = Path("results")

STRATEGIES = ("eig-source", "decoupled-eig", "random-source", "fixed-patient",
              "fixed-cheapest", "worst-source")

#: Fidelity at which every feature is scored when ranking *what to learn*.
#: Using one value for all features removes channel noise from the comparison,
#: so features compete on diagnostic merit alone. The value itself is arbitrary
#: (any fixed reference gives the same ranking); 0.05 matches the patient
#: channel so the numbers stay interpretable.
REFERENCE_NOISE = 0.05


def feature_information(space, post, reference_noise: float = REFERENCE_NOISE):
    """I(D; feature) with fidelity held constant across every feature.

    This is the "what to learn" half of the decoupled criterion. Because every
    feature is scored at the same noise level, a feature cannot win merely
    because some channel happens to read it accurately - which is exactly the
    failure mode diagnosed in PREREGISTRATION.md.
    """
    belief = post.belief
    m = space.matrix.astype(np.float64)
    p1 = np.where(m > 0, 1 - reference_noise, reference_noise)   # (D, K)

    marginal = np.clip(belief @ p1, 1e-12, 1 - 1e-12)            # (K,)
    h_marginal = -(marginal * np.log(marginal)
                   + (1 - marginal) * np.log1p(-marginal))
    h_cond = belief @ -(p1 * np.log(p1) + (1 - p1) * np.log1p(-p1))
    return h_marginal - h_cond


def observe(action, case, env, action_space, rng_seed: int) -> int:
    """Return 1/0, or -1 if the source could not answer."""
    feature = action_space.feature_of[action_space.index_of_label[str(action)]]
    if action.kind is ActionKind.ASK_PATIENT:
        obs = env.ask(case, feature)
        return -1 if obs == UNKNOWN else obs
    if action.kind is ActionKind.VERIFY_HERB:
        compat = action_space._herb_compat_cache[case.condition, action.target]
        eps = action_space.noise[ActionKind.VERIFY_HERB]
        r = np.random.default_rng([rng_seed, case.index, 5_000_000 + action.target])
        return int(1 - compat) if r.random() < eps else int(compat)
    truth = int(case.truth[feature])
    eps = action_space.noise[action.kind]
    r = np.random.default_rng([rng_seed, case.index, feature,
                               hash(action.kind.value) % 10_000])
    return int(1 - truth) if r.random() < eps else truth


def run_agent(case, env, action_space, space, budget, noise, strategy, rng_seed,
              by_feature):
    post = Posterior(space.matrix, noise=noise)
    taken = np.zeros(len(action_space), dtype=bool)
    for k, v in case.revealed.items():
        post.update(k, v)
        action_space.block_feature(taken, k)

    rng = np.random.default_rng([rng_seed, case.index])
    kinds = Counter()

    for _ in range(budget):
        mi = action_mutual_information(action_space, post.belief)
        avail = ~taken
        if not avail.any():
            break

        if strategy == "eig-source":
            scores = np.where(avail, mi, -np.inf)
            best = scores.max()
            if not np.isfinite(best):
                break
            tied = np.flatnonzero(scores >= best - 1e-12)
            idx = int(rng.choice(tied)) if tied.size > 1 else int(scores.argmax())

        elif strategy == "decoupled-eig":
            # (1) WHAT to learn: rank features at a constant reference fidelity,
            #     so channel noise cannot distort which attribute looks useful.
            fi = feature_information(space, post)
            reachable = np.full(space.n_features, -np.inf)
            for f, acts in by_feature.items():
                if any(not taken[i] for i in acts):
                    reachable[f] = fi[f]
            # Herb-verification actions observe no feature; score them directly
            # so they remain reachable under this policy.
            featureless = [i for i in range(len(action_space))
                           if action_space.feature_of[i] < 0 and not taken[i]]
            best_feature_score = reachable.max() if np.isfinite(reachable.max()) else -np.inf
            best_featureless = max((mi[i] for i in featureless), default=-np.inf)

            if best_featureless > best_feature_score:
                idx = max(featureless, key=lambda i: mi[i])
            else:
                if not np.isfinite(best_feature_score):
                    break
                tied_f = np.flatnonzero(reachable >= best_feature_score - 1e-12)
                feature = int(rng.choice(tied_f)) if tied_f.size > 1 else int(reachable.argmax())
                # (2) WHO to ask: among sources that can answer it, take the
                #     most informative (equivalently the highest fidelity).
                candidates = [i for i in by_feature[feature] if not taken[i]]
                idx = max(candidates, key=lambda i: mi[i])

        else:
            # Pick the FEATURE by best available EIG, then choose a source among
            # the actions that can answer it. Identical feature choice across
            # all non-eig arms, so only source choice differs.
            scores = np.where(avail, mi, -np.inf)
            if not np.isfinite(scores.max()):
                break
            seed_idx = int(scores.argmax())
            feature = action_space.feature_of[seed_idx]
            if feature < 0:
                candidates = [seed_idx]
            else:
                candidates = [i for i in by_feature.get(feature, [seed_idx])
                              if not taken[i]] or [seed_idx]

            if strategy == "random-source":
                idx = int(rng.choice(candidates))
            elif strategy == "fixed-patient":
                pat = [i for i in candidates
                       if action_space.kind[i] == ActionKind.ASK_PATIENT.value]
                idx = pat[0] if pat else int(candidates[0])
            elif strategy == "fixed-cheapest":
                idx = min(candidates, key=lambda i: action_space.cost[i])
            elif strategy == "worst-source":
                idx = min(candidates, key=lambda i: mi[i])
            else:
                raise ValueError(strategy)

        action = action_space.actions[idx]
        taken[idx] = True
        kinds[action.kind.value] += 1

        obs = observe(action, case, env, action_space, rng_seed)
        if obs < 0:
            continue
        post._logp = post._logp + (
            action_space._log_p1[idx] if obs else action_space._log_p0[idx])
        f = action_space.feature_of[idx]
        if f >= 0:
            action_space.block_feature(taken, f)

    return bool(post.argmax == case.condition), kinds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--budgets", type=int, nargs="*", default=[10, 15, 20])
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.10)
    ap.add_argument("--exclusive", action="store_true")
    args = ap.parse_args()

    space = build_condition_space()
    kg = KnowledgeGraph.build()
    action_space = ActionSpace(space, kg=kg, exclusive=args.exclusive)
    env = PatientEnvironment(space, noise=args.noise,
                             omission_rate=args.omission, seed=0)

    by_feature: dict[int, list[int]] = defaultdict(list)
    for i, f in enumerate(action_space.feature_of):
        if f >= 0:
            by_feature[f].append(i)
    multi = sum(1 for v in by_feature.values() if len(v) > 1)

    print("=" * 96)
    print("SOURCE-SELECTION QUALITY  (identical toolset, identical budget)")
    print("=" * 96)
    print(f"  n={args.n}, budgets {args.budgets}, exclusive={args.exclusive}")
    print(f"  features answerable by >1 source: {multi}/{len(by_feature)}"
          "   <- only these give source choice any room")
    print("-" * 96)

    all_rows, results = [], {}
    t0 = time.perf_counter()
    for budget in args.budgets:
        print(f"\n  budget = {budget} observations")
        print(f"    {'strategy':<18}{'accuracy':>10}{'vs eig':>10}{'95% CI':>20}"
              f"{'p':>11}  tool mix")
        acc = {}
        for strategy in STRATEGIES:
            correct, mix = [], Counter()
            for case in env.cases(args.n):
                c, k = run_agent(case, env, action_space, space, budget,
                                 args.noise, strategy, 11, by_feature)
                correct.append(c)
                mix.update(k)
            acc[strategy] = np.array(correct, dtype=bool)
            total = max(sum(mix.values()), 1)
            results.setdefault(budget, {})[strategy] = {
                "accuracy": round(float(acc[strategy].mean()), 4),
                "tool_mix": {k: round(v / total, 2) for k, v in sorted(mix.items())},
            }

        ref = acc["eig-source"]
        for strategy in STRATEGIES:
            a = acc[strategy]
            info = results[budget][strategy]
            if strategy == "eig-source":
                line = f"{'—':>10}{'':>20}{'':>11}"
            else:
                delta, lo, hi, _ = M.paired_bootstrap_ci(ref, a)
                _, _, p = M.mcnemar(ref, a)
                info.update({"delta_vs_eig": round(delta, 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "p_mcnemar": p, "significant": bool(p < 0.05)})
                pstr = "<1e-16" if p < 1e-16 else f"{p:.1e}"
                line = f"{delta:>+10.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}{pstr:>11}"
            mix = " ".join(f"{k.split('_')[0]}:{v:.2f}"
                           for k, v in info["tool_mix"].items())
            print(f"    {strategy:<18}{100*info['accuracy']:>9.1f}%{line}  {mix}",
                  flush=True)
            all_rows.append({"budget": budget, "strategy": strategy, **info})

    # Headroom: how much can source choice matter at all?
    spreads = []
    for budget, per in results.items():
        best = max(v["accuracy"] for v in per.values())
        worst = min(v["accuracy"] for v in per.values())
        spreads.append(best - worst)

    beats = [r for r in all_rows
             if r["strategy"] != "eig-source" and r.get("significant")
             and r.get("delta_vs_eig", 0) > 0]

    out = {
        "config": vars(args),
        "features_with_multiple_sources": multi,
        "features_total": len(by_feature),
        "rows": all_rows,
        "summary": {
            "mean_best_worst_spread": round(float(np.mean(spreads)), 4),
            "n_strategies_eig_significantly_beats": len(beats),
            "n_comparisons": len([r for r in all_rows
                                  if r["strategy"] != "eig-source"]),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_exclusive" if args.exclusive else ""
    (RESULTS / f"selection_quality{suffix}.json").write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print("\n" + "-" * 96)
    print(f"  best-worst spread (headroom for source choice): "
          f"{100*s['mean_best_worst_spread']:.1f} pts")
    print(f"  EIG-source significantly beats "
          f"{s['n_strategies_eig_significantly_beats']}/{s['n_comparisons']} alternatives")
    print("=" * 96)
    print(f"written to {RESULTS}/selection_quality{suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
