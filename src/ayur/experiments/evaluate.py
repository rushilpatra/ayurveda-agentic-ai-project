"""Turn trajectory JSONL into result tables, calibration numbers and statistics.

Reads whatever a run actually produced and always states the exact n. A stage
that stopped early is reported as incomplete rather than silently averaged.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ayur.experiments import metrics as M

RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"

#: The agent every other agent is compared against.
REFERENCE_AGENT = "B7-max-eig"


def load_trajectories(path: Path) -> dict[str, list[dict]]:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_agent[rec["agent"]].append(rec)
    for agent in by_agent:
        by_agent[agent].sort(key=lambda r: r["case_index"])
    return dict(by_agent)


def summarise(agent: str, rows: list[dict], n_conditions: int) -> M.AgentResult:
    truth = np.array([r["true_condition"] for r in rows])
    correct = np.array([r["correct"] for r in rows], dtype=bool)
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    nq = np.array([r["n_questions"] for r in rows], dtype=float)
    secs = np.array([r["seconds"] for r in rows], dtype=float)
    decided = np.array([r["decision"] == "diagnose" for r in rows], dtype=bool)
    top5 = [r["top5"] for r in rows]

    pred = np.array([r["predicted"] if r["predicted"] is not None else -1 for r in rows])

    return M.AgentResult(
        agent=agent,
        n=len(rows),
        n_diagnosed=int(decided.sum()),
        n_abstained=int((~decided).sum()),
        coverage=float(decided.mean()),
        accuracy_overall=M.accuracy(correct),
        # Selective accuracy: of the cases it chose to answer, how many were right.
        accuracy_selective=M.accuracy(correct[decided]) if decided.any() else float("nan"),
        top3=M.top_k_accuracy(top5, truth, 3),
        top5=M.top_k_accuracy(top5, truth, 5),
        macro_f1=M.macro_f1(pred[decided], truth[decided], n_conditions) if decided.any() else float("nan"),
        mean_questions=float(nq.mean()),
        median_questions=float(np.median(nq)),
        ece=M.expected_calibration_error(conf[decided], correct[decided]) if decided.any() else float("nan"),
        brier=M.brier_score(conf[decided], correct[decided]) if decided.any() else float("nan"),
        aurc=M.aurc(conf, correct),
        mean_seconds=float(secs.mean()),
    )


def compare(by_agent: dict[str, list[dict]], reference: str = REFERENCE_AGENT) -> list[dict]:
    """Paired comparisons against the reference agent, Holm-corrected."""
    if reference not in by_agent:
        return []
    ref_rows = {r["case_index"]: r for r in by_agent[reference]}

    raw_p: dict[str, float] = {}
    partial: list[dict] = []
    for agent, rows in by_agent.items():
        if agent == reference:
            continue
        shared = [r for r in rows if r["case_index"] in ref_rows]
        if not shared:
            continue
        a = np.array([ref_rows[r["case_index"]]["correct"] for r in shared], dtype=bool)
        b = np.array([r["correct"] for r in shared], dtype=bool)
        delta, lo, hi, p_boot = M.paired_bootstrap_ci(a, b)
        b01, b10, p_mc = M.mcnemar(a, b)
        raw_p[agent] = p_mc
        partial.append(
            {
                "agent": agent,
                "vs": reference,
                "n_paired": len(shared),
                "delta_accuracy": round(delta, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "p_bootstrap": round(p_boot, 6),
                "mcnemar_b01": b01,
                "mcnemar_b10": b10,
                "p_mcnemar": round(p_mc, 6),
            }
        )

    adjusted = M.holm_correction(raw_p)
    for row in partial:
        row["p_mcnemar_holm"] = round(adjusted[row["agent"]], 6)
        row["significant_005"] = bool(row["p_mcnemar_holm"] < 0.05)
    return partial


def evaluate_stage(stage: str) -> dict:
    path = CHECKPOINTS / f"{stage}_trajectories.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make {stage}` first")

    manifest_path = RESULTS / f"manifest_{stage}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    n_conditions = manifest.get("n_conditions", 446)

    by_agent = load_trajectories(path)
    results = [summarise(a, rows, n_conditions) for a, rows in sorted(by_agent.items())]
    comparisons = compare(by_agent)

    counts = {a: len(rows) for a, rows in by_agent.items()}
    ragged = len(set(counts.values())) > 1

    out = {
        "stage": stage,
        "n_conditions": n_conditions,
        "n_features": manifest.get("n_features"),
        "cases_per_agent": counts,
        "incomplete": bool(manifest.get("stopped_early", False)) or ragged,
        "reason": manifest.get("reason"),
        "results": [r.row() for r in results],
        "comparisons": comparisons,
    }
    (RESULTS / f"evaluation_{stage}.json").write_text(json.dumps(out, indent=2))

    # --- printed table --------------------------------------------------------
    print("=" * 118)
    print(f"EVALUATION: {stage}")
    if out["incomplete"]:
        print(f"  ** INCOMPLETE RUN ({out['reason']}) - numbers below are over the cases "
              "that finished **")
    print("=" * 118)
    hdr = (f"{'agent':<26}{'n':>6}{'cov':>7}{'acc':>8}{'sel-acc':>9}{'top3':>7}"
           f"{'top5':>7}{'macroF1':>9}{'q':>6}{'ECE':>8}{'Brier':>8}{'AURC':>8}{'ms':>8}")
    print(hdr)
    print("-" * 118)
    for r in sorted(results, key=lambda x: -(x.accuracy_overall if np.isfinite(x.accuracy_overall) else -1)):
        print(
            f"{r.agent:<26}{r.n:>6}{100*r.coverage:>6.1f}%{100*r.accuracy_overall:>7.1f}%"
            f"{100*r.accuracy_selective:>8.1f}%{100*r.top3:>6.1f}%{100*r.top5:>6.1f}%"
            f"{r.macro_f1:>9.3f}{r.mean_questions:>6.1f}{r.ece:>8.3f}{r.brier:>8.3f}"
            f"{r.aurc:>8.3f}{1000*r.mean_seconds:>8.1f}"
        )
    print("-" * 118)
    print("  cov = coverage (fraction answered rather than abstained);  "
          "sel-acc = accuracy on answered cases only")

    if comparisons:
        print()
        print(f"PAIRED COMPARISONS vs {REFERENCE_AGENT}  (positive delta = reference is better)")
        print("-" * 118)
        print(f"{'agent':<26}{'n':>6}{'delta':>9}{'95% CI':>20}{'p (McNemar)':>14}"
              f"{'p Holm':>10}{'sig':>6}")
        print("-" * 118)
        for c in sorted(comparisons, key=lambda x: -x["delta_accuracy"]):
            ci = f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]"
            print(f"{c['agent']:<26}{c['n_paired']:>6}{c['delta_accuracy']:>+9.3f}{ci:>20}"
                  f"{c['p_mcnemar']:>14.2e}{c['p_mcnemar_holm']:>10.2e}"
                  f"{'yes' if c['significant_005'] else 'no':>6}")
    print("=" * 118)
    print(f"written to {RESULTS}/evaluation_{stage}.json")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    args = ap.parse_args()
    evaluate_stage(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
