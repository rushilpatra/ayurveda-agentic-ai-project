"""The heterogeneous-action-space experiment.

Measures the claims the single-source planner cannot address:
  * does weighing information against source cost beat asking the patient only?
  * tool-selection accuracy against an oracle that maximises true EIG/cost
  * evidence grounding - every conclusion traceable to the actions that produced it
  * cost efficiency - accuracy per unit of consultation budget
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import PatientEnvironment
from ayur.experiments import metrics as M
from ayur.kg.graph import KnowledgeGraph
from ayur.planner.actions import ActionKind, ActionSpace, action_mutual_information
from ayur.planner.multitool import build_tool_agents
from ayur.planner.posterior import Posterior

RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"


def oracle_tool_selection_rate(trajectories, space, action_space, env, noise) -> float:
    """Fraction of chosen actions that an EIG/cost oracle would also have chosen.

    Recomputed by replaying each trajectory: at every step, compare the agent's
    action against argmax of true information-per-cost given the same belief.
    """
    agree = total = 0
    for tr in trajectories:
        case = env.make_case(tr["case_index"])
        post = Posterior(space.matrix, noise=noise)
        taken = np.zeros(len(action_space), dtype=bool)
        for k, v in case.revealed.items():
            post.update(k, v)
            action_space.block_feature(taken, k)

        for label in tr["actions"]:
            scores = action_mutual_information(action_space, post.belief) / action_space.cost
            scores[taken] = -np.inf
            best = int(scores.argmax())
            chosen = action_space.index_of_label.get(label)
            if chosen is None:
                break
            total += 1
            agree += (action_space.kind[chosen] == action_space.kind[best])
            taken[chosen] = True
            f = action_space.feature_of[chosen]
            obs = int(case.truth[f]) if f >= 0 else 0
            post._logp = post._logp + (
                action_space._log_p1[chosen] if obs else action_space._log_p0[chosen])
            if f >= 0:
                action_space.block_feature(taken, f)
    return agree / total if total else float("nan")


def run(n_cases: int, max_cost: float, env_noise: float, assumed_noise: float,
        omission: float, tag: str) -> dict:
    space = build_condition_space()
    kg = KnowledgeGraph.build()
    action_space = ActionSpace(space, kg=kg)
    env = PatientEnvironment(space, noise=env_noise, omission_rate=omission, seed=0)
    agents = build_tool_agents(max_cost=max_cost, noise=assumed_noise)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINTS / f"tools_{tag}_trajectories.jsonl"
    by_agent: dict[str, list[dict]] = {a.name: [] for a in agents}

    print("=" * 96)
    print(f"HETEROGENEOUS ACTION SPACE: {tag}")
    print("=" * 96)
    print(f"  actions {len(action_space)}  {action_space.summary()['by_kind']}")
    print(f"  cost budget {max_cost}  (1.0 = one patient question)")
    print(f"  env noise {env_noise}  omission {omission}  agent assumes {assumed_noise}")
    print(f"  cases {n_cases}")
    print("-" * 96)

    t0 = time.perf_counter()
    with out_path.open("w") as fh:
        for case in env.cases(n_cases):
            for agent in agents:
                tr = agent.run(case, env, action_space, kg)
                by_agent[agent.name].append(tr.to_json())
                fh.write(json.dumps(tr.to_json()) + "\n")
            if (case.index + 1) % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  case {case.index+1}/{n_cases}  elapsed {el:.0f}s  "
                      f"eta {el/(case.index+1)*(n_cases-case.index-1):.0f}s", flush=True)

    rows = []
    for name, trs in by_agent.items():
        correct = np.array([t["correct"] for t in trs], dtype=bool)
        cost = np.array([t["total_cost"] for t in trs], dtype=float)
        nact = np.array([t["n_actions"] for t in trs], dtype=float)
        conf = np.array([t["confidence"] for t in trs], dtype=float)
        kinds = Counter(k for t in trs for k in t["kinds"])
        total_kinds = max(sum(kinds.values()), 1)
        patient_actions = np.array(
            [sum(1 for k in t["kinds"] if k == ActionKind.ASK_PATIENT.value) for t in trs],
            dtype=float)
        rows.append({
            "agent": name,
            "n": len(trs),
            "accuracy": round(float(correct.mean()), 4),
            "mean_cost": round(float(cost.mean()), 3),
            "mean_actions": round(float(nact.mean()), 2),
            "mean_patient_questions": round(float(patient_actions.mean()), 2),
            "accuracy_per_cost": round(float(correct.mean() / max(cost.mean(), 1e-9)), 4),
            "ece": round(M.expected_calibration_error(conf, correct), 4),
            "aurc": round(M.aurc(conf, correct), 4),
            "tool_mix": {k: round(v / total_kinds, 3) for k, v in kinds.items()},
            "evidence_per_case": round(
                float(np.mean([len(t["evidence"]) for t in trs])), 2),
        })

    ref = "T1-tool-eig-per-cost"
    comparisons = []
    if ref in by_agent:
        a = np.array([t["correct"] for t in by_agent[ref]], dtype=bool)
        raw_p = {}
        for name, trs in by_agent.items():
            if name == ref:
                continue
            b = np.array([t["correct"] for t in trs], dtype=bool)
            delta, lo, hi, _ = M.paired_bootstrap_ci(a, b)
            _, _, p = M.mcnemar(a, b)
            raw_p[name] = p
            comparisons.append({"agent": name, "delta_accuracy": round(delta, 4),
                                "ci95": [round(lo, 4), round(hi, 4)], "p_mcnemar": p})
        adj = M.holm_correction(raw_p)
        for c in comparisons:
            c["p_holm"] = round(adj[c["agent"]], 6)
            c["significant"] = bool(c["p_holm"] < 0.05)

    print("  computing tool-selection agreement vs EIG/cost oracle ...", flush=True)
    oracle = {}
    for name in (ref, "T4-cheapest-first"):
        if name in by_agent:
            oracle[name] = round(oracle_tool_selection_rate(
                by_agent[name][:60], space, action_space, env, assumed_noise), 4)

    out = {
        "tag": tag,
        "config": {"n_cases": n_cases, "max_cost": max_cost, "env_noise": env_noise,
                   "assumed_noise": assumed_noise, "omission": omission},
        "action_space": action_space.summary(),
        "results": rows,
        "comparisons": comparisons,
        "tool_selection_agreement_vs_oracle": oracle,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "trajectories": str(out_path),
    }
    (RESULTS / f"tools_{tag}.json").write_text(json.dumps(out, indent=2))

    print("-" * 96)
    print(f"{'agent':<26}{'n':>5}{'acc':>8}{'cost':>8}{'acts':>7}{'pt-Qs':>7}"
          f"{'acc/cost':>10}{'ECE':>8}  tool mix")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: -x["accuracy"]):
        mix = " ".join(f"{k.split('_')[0]}:{v:.2f}" for k, v in sorted(r["tool_mix"].items()))
        print(f"{r['agent']:<26}{r['n']:>5}{100*r['accuracy']:>7.1f}%{r['mean_cost']:>8.2f}"
              f"{r['mean_actions']:>7.1f}{r['mean_patient_questions']:>7.1f}"
              f"{r['accuracy_per_cost']:>10.4f}{r['ece']:>8.3f}  {mix}")
    print("-" * 96)
    if comparisons:
        print(f"  paired vs {ref}:")
        for c in sorted(comparisons, key=lambda x: -x["delta_accuracy"]):
            print(f"    {c['agent']:<26} delta {c['delta_accuracy']:+.3f}  "
                  f"CI {c['ci95']}  p_holm {c['p_holm']:.2e}  "
                  f"{'sig' if c['significant'] else 'ns'}")
    if oracle:
        print(f"  tool-kind agreement with EIG/cost oracle: {oracle}")
    print("=" * 96)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-cost", type=float, default=15.0)
    ap.add_argument("--env-noise", type=float, default=0.05)
    ap.add_argument("--assumed-noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.10)
    ap.add_argument("--tag", default="wellspec")
    args = ap.parse_args()
    run(args.n, args.max_cost, args.env_noise, args.assumed_noise,
        args.omission, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
