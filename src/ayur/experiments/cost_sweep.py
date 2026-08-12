"""Cost-table sensitivity sweep - does the multi-tool advantage survive reprating?

## Why this experiment exists

The headline result in POSITIONING.md section 12 is that tool-EIG beats a
patient-only agent by +16.0 points at equal cost. But the cost table
(patient 1.0, kg 0.1, retrieval 0.3, herb 0.4) is a *design choice*, not
something measured from clinical practice. The obvious reviewer question is:

    "Is the advantage real, or did you pick costs that produce it?"

This sweep answers it. It varies how cheap the non-patient channels are relative
to a patient question, from 1.0 (tools cost exactly as much as asking - no
economic reason to prefer them) down to 0.02 (nearly free), and reports the
tool-EIG minus patient-only gap at each point.

Two things would falsify the claim:
  * the advantage vanishing except in a narrow band of cost ratios, or
  * the advantage appearing even at ratio 1.0, which would mean it comes from
    something other than cost-awareness and the mechanism story is wrong.

A third setting - ratio > 1.0, tools *more* expensive than asking - is included
as a sanity check: there the planner should stop using them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import PatientEnvironment
from ayur.experiments import metrics as M
from ayur.kg.graph import KnowledgeGraph
from ayur.planner.actions import ActionKind, ActionSpace
from ayur.planner.multitool import (
    MultiToolAgent,
    PatientOnlyPolicy,
    ToolEIGPolicy,
    ToolStoppingRule,
)

RESULTS = Path("results")

#: Multipliers on the default non-patient costs. 1.0 = every channel costs the
#: same as a patient question; 2.0 = tools are twice as expensive as asking.
DEFAULT_RATIOS = (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0)


def costs_for_ratio(ratio: float) -> dict:
    """Patient stays at 1.0; every other channel is priced at `ratio`.

    Deliberately collapses the three tool costs to one value so the sweep has a
    single interpretable axis: "how much cheaper than asking the patient".
    """
    return {
        ActionKind.ASK_PATIENT: 1.0,
        ActionKind.QUERY_KG: ratio,
        ActionKind.RETRIEVE_TEXT: ratio,
        ActionKind.VERIFY_HERB: ratio,
    }


def run_one(space, kg, env, ratio: float, n_cases: int, max_cost: float,
            noise: float) -> dict:
    action_space = ActionSpace(space, kg=kg, costs=costs_for_ratio(ratio))
    stopping = ToolStoppingRule(tau_confidence=1.01, max_cost=max_cost,
                                min_information_gain=-1.0, allow_abstain=False)
    tool_agent = MultiToolAgent("tool-eig", ToolEIGPolicy(), stopping, noise)
    patient_agent = MultiToolAgent("patient-only", PatientOnlyPolicy(), stopping, noise)

    tool_correct, patient_correct = [], []
    tool_patient_qs, tool_mix = [], {}

    for case in env.cases(n_cases):
        a = tool_agent.run(case, env, action_space, kg)
        b = patient_agent.run(case, env, action_space, kg)
        tool_correct.append(a.correct)
        patient_correct.append(b.correct)
        tool_patient_qs.append(sum(1 for k in a.kinds
                                   if k == ActionKind.ASK_PATIENT.value))
        for k in a.kinds:
            tool_mix[k] = tool_mix.get(k, 0) + 1

    tc = np.array(tool_correct, dtype=bool)
    pc = np.array(patient_correct, dtype=bool)
    delta, lo, hi, _ = M.paired_bootstrap_ci(tc, pc)
    _, _, p = M.mcnemar(tc, pc)
    total = max(sum(tool_mix.values()), 1)

    return {
        "cost_ratio": ratio,
        "n": n_cases,
        "tool_eig_accuracy": round(float(tc.mean()), 4),
        "patient_only_accuracy": round(float(pc.mean()), 4),
        "delta": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_mcnemar": p,
        "significant": bool(p < 0.05),
        "tool_mean_patient_questions": round(float(np.mean(tool_patient_qs)), 2),
        "tool_mix": {k: round(v / total, 3) for k, v in sorted(tool_mix.items())},
        "non_patient_action_share": round(
            1 - tool_mix.get(ActionKind.ASK_PATIENT.value, 0) / total, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-cost", type=float, default=15.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.10)
    ap.add_argument("--ratios", type=float, nargs="*", default=None)
    args = ap.parse_args()

    ratios = tuple(args.ratios) if args.ratios else DEFAULT_RATIOS
    space = build_condition_space()
    kg = KnowledgeGraph.build()
    env = PatientEnvironment(space, noise=args.noise,
                             omission_rate=args.omission, seed=0)

    print("=" * 96)
    print("COST-TABLE SENSITIVITY SWEEP")
    print("=" * 96)
    print(f"  n={args.n} cases per point, budget {args.max_cost}, "
          f"ratios {list(ratios)}")
    print("  ratio = cost of every non-patient channel, with a patient question fixed at 1.0")
    print("-" * 96)
    print(f"{'ratio':>7}{'tool-EIG':>11}{'patient':>10}{'delta':>9}"
          f"{'95% CI':>20}{'p':>11}{'ptQs':>7}{'tool share':>12}")
    print("-" * 96)

    rows = []
    t0 = time.perf_counter()
    for ratio in ratios:
        r = run_one(space, kg, env, ratio, args.n, args.max_cost, args.noise)
        rows.append(r)
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        pstr = "<1e-16" if r["p_mcnemar"] < 1e-16 else f"{r['p_mcnemar']:.1e}"
        print(f"{ratio:>7.2f}{100*r['tool_eig_accuracy']:>10.1f}%"
              f"{100*r['patient_only_accuracy']:>9.1f}%{r['delta']:>+9.3f}{ci:>20}"
              f"{pstr:>11}{r['tool_mean_patient_questions']:>7.1f}"
              f"{r['non_patient_action_share']:>12.2f}", flush=True)

    deltas = [r["delta"] for r in rows]
    sig = [r for r in rows if r["significant"] and r["delta"] > 0]
    below_one = [r for r in rows if r["cost_ratio"] < 1.0]
    at_or_above = [r for r in rows if r["cost_ratio"] >= 1.0]

    out = {
        "config": vars(args),
        "ratios": list(ratios),
        "rows": rows,
        "summary": {
            "min_delta": round(min(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "n_points_significant_positive": len(sig),
            "n_points": len(rows),
            "all_cheap_ratios_positive": all(r["delta"] > 0 for r in below_one),
            "delta_when_tools_not_cheaper": [
                {"ratio": r["cost_ratio"], "delta": r["delta"],
                 "significant": r["significant"]} for r in at_or_above],
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cost_sweep.json").write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print("-" * 96)
    print(f"  advantage significant and positive at "
          f"{s['n_points_significant_positive']}/{s['n_points']} cost ratios")
    print(f"  delta range {s['min_delta']:+.3f} .. {s['max_delta']:+.3f}")
    print(f"  positive at every ratio below 1.0: {s['all_cheap_ratios_positive']}")
    print(f"  at ratio >= 1.0 (tools no cheaper): "
          f"{s['delta_when_tools_not_cheaper']}")
    print(f"  elapsed {out['elapsed_seconds']}s")
    print("=" * 96)
    print(f"written to {RESULTS}/cost_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
