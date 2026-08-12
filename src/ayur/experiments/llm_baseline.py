"""B6 - LLM-as-planner baseline. The MAI-DxO analogue, scaled to local hardware.

## Design: isolate planning, not inference

MAI-DxO's contribution is that an LLM, prompted to deliberate, chooses good next
actions. Comparing it end-to-end against this system would confound two things:
the quality of the *planner* and the quality of the *belief update*.

So B6 shares this project's posterior and differs only in how the next question
is chosen:

    B7  next question = argmax closed-form expected information gain
    B6  next question = whichever the LLM picks from the same shortlist

Both then update the identical posterior with the identical answer and diagnose
from it. Any accuracy difference is attributable to action selection alone.

The shortlist is built **without** using EIG - it is the K most frequent unasked
attributes - so the LLM is not handed the answer, and neither planner gets a
shortlist the other could not have produced.

Cost: one LLM call per question per case. This is exactly the expense the
algorithmic planner avoids, and the measured seconds-per-case ratio is itself a
reported result.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import UNKNOWN, PatientEnvironment
from ayur.env.templates import build as build_templates
from ayur.experiments import metrics as M
from ayur.planner.policy import EIGPolicy
from ayur.planner.posterior import Posterior

RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"

PROMPT = """You are a diagnostician taking a patient history. Choose the single most \
informative question to ask next.

What you already know about this patient:
{history}

Candidate questions:
{options}

Which question is most useful for narrowing down the diagnosis? \
Reply with only its number."""


def parse_index(text: str, n: int) -> int | None:
    if not text:
        return None
    m = re.search(r"\b(\d+)\b", text)
    if not m:
        return None
    i = int(m.group(1)) - 1
    return i if 0 <= i < n else None


def shortlist(space, asked: np.ndarray, k: int) -> list[int]:
    """K most frequent unasked attributes - constructed without reference to EIG."""
    counts = space.matrix.sum(0).astype(float)
    counts[asked] = -np.inf
    return [int(i) for i in np.argsort(-counts)[:k]]


def history_text(space, bank, observed: dict[int, int], limit: int = 12) -> str:
    if not observed:
        return "  (nothing yet)"
    lines = []
    for feat, val in list(observed.items())[-limit:]:
        column, value = space.features[feat].split("::", 1)
        lines.append(f"  - {column}: {value} = {'yes' if val else 'no'}")
    return "\n".join(lines)


def run(n_cases: int, budget: int, k_options: int, noise: float,
        omission: float, backend_name: str) -> dict:
    from ayur.llm.backend import get_backend

    space = build_condition_space()
    bank = build_templates(space.features)
    env = PatientEnvironment(space, noise=noise, omission_rate=omission, seed=0)
    backend = get_backend(backend_name)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINTS / "llm_planner_trajectories.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["case_index"])
                except Exception:
                    pass

    print("=" * 78)
    print("B6 - LLM AS PLANNER  (shared posterior; only action selection differs)")
    print("=" * 78)
    print(f"  cases {n_cases}   budget {budget}   shortlist {k_options}")
    print(f"  backend {backend_name}   already done {len(done)}")
    print("-" * 78)

    llm_rows, eig_rows = [], []
    t0 = time.perf_counter()
    n_unparsed = n_calls = 0

    with out_path.open("a") as fh:
        for case in env.cases(n_cases):
            if case.index in done:
                continue

            # --- B6: LLM chooses ------------------------------------------------
            post = Posterior(space.matrix, noise=noise)
            asked = np.zeros(space.n_features, dtype=bool)
            observed: dict[int, int] = {}
            for k, v in case.revealed.items():
                post.update(k, v)
                asked[k] = True
                observed[k] = v

            t_case = time.perf_counter()
            for _ in range(budget):
                options = shortlist(space, asked, k_options)
                listing = "\n".join(
                    f"  {i+1}. {bank.ask(f, 'en')}" for i, f in enumerate(options))
                gen = backend.generate(
                    PROMPT.format(history=history_text(space, bank, observed),
                                  options=listing),
                    max_tokens=12, temperature=0.0)
                n_calls += 1
                idx = parse_index(gen.text, len(options))
                if idx is None:
                    n_unparsed += 1
                    idx = 0            # deterministic fallback: first option
                feat = options[idx]
                asked[feat] = True
                ans = env.ask(case, feat)
                if ans != UNKNOWN:
                    post.update(feat, ans)
                    observed[feat] = ans
            llm_seconds = time.perf_counter() - t_case
            llm_correct = bool(post.argmax == case.condition)
            llm_conf = post.confidence

            # --- B7: EIG chooses, identical everything else ---------------------
            post2 = Posterior(space.matrix, noise=noise)
            asked2 = np.zeros(space.n_features, dtype=bool)
            for k, v in case.revealed.items():
                post2.update(k, v)
                asked2[k] = True
            rng = np.random.default_rng([env.seed, case.index, 7919])
            policy = EIGPolicy()
            t_case = time.perf_counter()
            for _ in range(budget):
                f = policy.select(post2, asked2, rng)
                asked2[f] = True
                ans = env.ask(case, f)
                if ans != UNKNOWN:
                    post2.update(f, ans)
            eig_seconds = time.perf_counter() - t_case

            rec = {
                "case_index": case.index,
                "true_condition": int(case.condition),
                "llm_correct": llm_correct,
                "llm_confidence": float(llm_conf),
                "llm_seconds": round(llm_seconds, 3),
                "eig_correct": bool(post2.argmax == case.condition),
                "eig_confidence": float(post2.confidence),
                "eig_seconds": round(eig_seconds, 4),
            }
            fh.write(json.dumps(rec) + "\n")
            llm_rows.append(rec)
            eig_rows.append(rec)

            if (case.index + 1) % 10 == 0:
                el = time.perf_counter() - t0
                fh.flush()
                print(f"  case {case.index+1}/{n_cases}  elapsed {el:.0f}s  "
                      f"eta {el/(case.index+1)*(n_cases-case.index-1)/60:.1f} min",
                      flush=True)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    llm_c = np.array([r["llm_correct"] for r in rows], dtype=bool)
    eig_c = np.array([r["eig_correct"] for r in rows], dtype=bool)
    llm_s = np.array([r["llm_seconds"] for r in rows], dtype=float)
    eig_s = np.array([r["eig_seconds"] for r in rows], dtype=float)

    delta, lo, hi, _ = M.paired_bootstrap_ci(eig_c, llm_c)
    b01, b10, p = M.mcnemar(eig_c, llm_c)

    out = {
        "n": len(rows),
        "budget": budget,
        "shortlist_size": k_options,
        "llm_accuracy": round(float(llm_c.mean()), 4),
        "eig_accuracy": round(float(eig_c.mean()), 4),
        "delta_eig_minus_llm": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_mcnemar": p,
        "llm_seconds_per_case": round(float(llm_s.mean()), 3),
        "eig_seconds_per_case": round(float(eig_s.mean()), 5),
        "speedup": round(float(llm_s.mean() / max(eig_s.mean(), 1e-9)), 1),
        "llm_calls": n_calls,
        "unparsed_responses": n_unparsed,
        "unparsed_rate": round(n_unparsed / max(n_calls, 1), 4),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "llm_planner.json").write_text(json.dumps(out, indent=2))

    print("-" * 78)
    print(f"  n = {out['n']} cases, identical posterior, identical answers")
    print(f"  B6 LLM planner      {100*out['llm_accuracy']:5.1f}%   "
          f"{out['llm_seconds_per_case']:.2f} s/case")
    print(f"  B7 EIG planner      {100*out['eig_accuracy']:5.1f}%   "
          f"{out['eig_seconds_per_case']:.4f} s/case")
    print(f"  delta (EIG - LLM)   {out['delta_eig_minus_llm']:+.3f}  "
          f"CI {out['ci95']}  p={out['p_mcnemar']:.2e}")
    print(f"  speed               EIG is {out['speedup']}x faster")
    print(f"  unparsed LLM replies {100*out['unparsed_rate']:.1f}% "
          f"of {out['llm_calls']} calls")
    print("=" * 78)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--k-options", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.10)
    ap.add_argument("--backend", default="mlx")
    args = ap.parse_args()
    run(args.n, args.budget, args.k_options, args.noise, args.omission, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
