"""Fit calibration on a held-out split, then measure whether it repairs
abstention under misspecification.

This is the experiment that answers the failure found in the pilot: abstention
hurt because the posterior was overconfident. Split is strict - parameters are
fitted on calibration cases and evaluated on disjoint test cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import UNKNOWN, PatientEnvironment
from ayur.experiments import metrics as M
from ayur.planner.calibration import (
    apply_temperature,
    fit_effective_noise,
    fit_tau_for_risk,
    fit_temperature,
)
from ayur.planner.policy import EIGPolicy
from ayur.planner.posterior import Posterior

RESULTS = Path("results")


def collect(space, env, n_cases, start, budget, assumed_noise):
    """Run the EIG policy for a fixed budget; return beliefs, truths, observations."""
    beliefs, truths, observations = [], [], []
    for case in env.cases(n_cases, start=start):
        post = Posterior(space.matrix, noise=assumed_noise)
        asked = np.zeros(space.n_features, dtype=bool)
        obs: dict[int, int] = {}
        for k, v in case.revealed.items():
            post.update(k, v)
            asked[k] = True
            obs[k] = v
        rng = np.random.default_rng([env.seed, case.index, 7919])
        policy = EIGPolicy()
        for _ in range(budget):
            k = policy.select(post, asked, rng)
            asked[k] = True
            ans = env.ask(case, k)
            if ans != UNKNOWN:
                post.update(k, ans)
                obs[int(k)] = int(ans)
        beliefs.append(post.belief)
        truths.append(case.condition)
        observations.append(obs)
    return np.array(beliefs), np.array(truths), observations


def evaluate_at(beliefs, truths, temperature, tau):
    conf = np.empty(len(beliefs))
    correct = np.empty(len(beliefs), dtype=bool)
    for i, b in enumerate(beliefs):
        s = apply_temperature(b, temperature)
        conf[i] = s.max()
        correct[i] = s.argmax() == truths[i]
    answered = conf >= tau
    return {
        "temperature": round(float(temperature), 4),
        "tau": round(float(tau), 4),
        "coverage": round(float(answered.mean()), 4),
        "accuracy_overall": round(float(np.mean(correct & answered)), 4),
        "accuracy_selective": round(float(correct[answered].mean()), 4) if answered.any() else None,
        "selective_risk": round(float(1 - correct[answered].mean()), 4) if answered.any() else None,
        "ece": round(M.expected_calibration_error(conf, correct), 4),
        "brier": round(M.brier_score(conf, correct), 4),
        "aurc": round(M.aurc(conf, correct), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--env-noise", type=float, default=0.15)
    ap.add_argument("--assumed-noise", type=float, default=0.05)
    ap.add_argument("--omission", type=float, default=0.25)
    ap.add_argument("--target-risk", type=float, default=0.2)
    args = ap.parse_args()

    space = build_condition_space()
    env = PatientEnvironment(
        space, noise=args.env_noise, omission_rate=args.omission, seed=0
    )

    print("=" * 78)
    print("CALIBRATION UNDER MISSPECIFICATION")
    print("=" * 78)
    print(f"  environment noise   {args.env_noise}   omission {args.omission}")
    print(f"  agent assumes       {args.assumed_noise}   (misspecified)")
    print(f"  calibration cases   {args.n_calib}   test cases {args.n_test} (disjoint)")
    print(f"  question budget     {args.budget}")
    print("-" * 78)

    print("  collecting calibration split ...", flush=True)
    cb, ct, cobs = collect(space, env, args.n_calib, 0, args.budget, args.assumed_noise)
    print("  collecting test split ...", flush=True)
    tb, tt, _ = collect(space, env, args.n_test, 10_000, args.budget, args.assumed_noise)

    baseline = evaluate_at(tb, tt, 1.0, 0.5)

    print("  fitting temperature ...", flush=True)
    temperature, calib_ece = fit_temperature(cb, ct)

    conf_c = np.array([apply_temperature(b, temperature).max() for b in cb])
    corr_c = np.array([apply_temperature(b, temperature).argmax() == t for b, t in zip(cb, ct)])
    tau, achieved_risk, achieved_cov = fit_tau_for_risk(conf_c, corr_c, args.target_risk)
    met_target = achieved_risk <= args.target_risk

    calibrated = evaluate_at(tb, tt, temperature, tau)

    print("  fitting effective noise ...", flush=True)
    eff_noise, nll = fit_effective_noise(space.matrix, cobs[:120], ct[:120])

    out = {
        "config": vars(args),
        "fitted": {
            "temperature": round(temperature, 4),
            "calibration_ece_at_temperature": round(calib_ece, 4),
            "tau": round(tau, 4),
            "target_selective_risk": args.target_risk,
            "achieved_risk_on_calibration": round(achieved_risk, 4),
            "target_met": bool(met_target),
            "effective_noise_mle": round(eff_noise, 4),
            "true_env_noise": args.env_noise,
            "assumed_noise": args.assumed_noise,
        },
        "test_uncalibrated": baseline,
        "test_calibrated": calibrated,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2))

    print("-" * 78)
    print(f"  fitted temperature      {temperature:.3f}   (calibration ECE {calib_ece:.4f})")
    print(f"  fitted tau              {tau:.4f}   target risk {args.target_risk}"
          f"   achieved {achieved_risk:.4f}"
          f"   {'OK' if met_target else '** TARGET NOT MET **'}")
    print(f"  effective noise (MLE)   {eff_noise:.4f}"
          f"   vs true {args.env_noise}, assumed {args.assumed_noise}")
    print("-" * 78)
    print(f"  {'metric':<22}{'uncalibrated':>16}{'calibrated':>16}")
    for key in ("coverage", "accuracy_overall", "accuracy_selective",
                "selective_risk", "ece", "brier", "aurc"):
        a, b = baseline[key], calibrated[key]
        astr = f"{a:.4f}" if isinstance(a, float) else str(a)
        bstr = f"{b:.4f}" if isinstance(b, float) else str(b)
        print(f"  {key:<22}{astr:>16}{bstr:>16}")
    print("=" * 78)
    print(f"written to {RESULTS}/calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
