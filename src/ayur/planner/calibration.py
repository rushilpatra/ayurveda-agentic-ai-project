"""Posterior calibration and threshold fitting.

Motivated by a measured failure. Under a misspecified likelihood (environment
noise 0.15, agent assumes 0.05) the pilot showed:

    B8 forced, no abstention   43.0% accuracy   ECE 0.154
    B7 with abstention         28.5% accuracy   ECE 0.396

Abstention made things *worse*, because the confidence it thresholds on was
wrong. An abstention mechanism is only as good as the calibration underneath it.
This module fixes the calibration rather than tuning the threshold around it.

Two knobs, fitted on a held-out calibration split and never on test:

  * temperature   - flattens an over-sharp posterior:  b ∝ exp(log b / T)
  * effective noise - the likelihood parameter the agent should have assumed

Both are fitted with plain grid search. The objective is small and the search is
one-dimensional; anything more elaborate would obscure a simple result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ayur.experiments.metrics import expected_calibration_error


def apply_temperature(belief: np.ndarray, temperature: float) -> np.ndarray:
    """Flatten (T > 1) or sharpen (T < 1) a probability vector."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return belief
    logb = np.log(np.clip(belief, 1e-300, None)) / temperature
    logb -= logb.max()
    b = np.exp(logb)
    return b / b.sum()


@dataclass
class Calibrator:
    """Fitted calibration parameters."""

    temperature: float = 1.0
    tau: float = 0.5
    target_selective_risk: float | None = None
    fitted_on: int = 0

    def confidence(self, belief: np.ndarray) -> float:
        return float(apply_temperature(belief, self.temperature).max())

    def should_answer(self, belief: np.ndarray) -> bool:
        return self.confidence(belief) >= self.tau


def fit_temperature(
    beliefs: np.ndarray,
    truth: np.ndarray,
    grid: np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Grid-search the temperature that minimises ECE on a calibration split.

    beliefs : (n, D) posterior per case
    truth   : (n,) true condition index

    Returns (temperature, ece_at_that_temperature).
    """
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.0, 11)[:-1], np.linspace(1.0, 8.0, 71)])

    best_t, best_ece = 1.0, np.inf
    for t in grid:
        conf = np.empty(len(beliefs))
        correct = np.empty(len(beliefs), dtype=bool)
        for i, b in enumerate(beliefs):
            scaled = apply_temperature(b, float(t))
            conf[i] = scaled.max()
            correct[i] = scaled.argmax() == truth[i]
        ece = expected_calibration_error(conf, correct, n_bins=n_bins)
        if ece < best_ece:
            best_t, best_ece = float(t), float(ece)
    return best_t, best_ece


def fit_tau_for_risk(
    confidence: np.ndarray,
    correct: np.ndarray,
    target_risk: float = 0.2,
    min_coverage: float = 0.05,
) -> tuple[float, float, float]:
    """Smallest threshold whose selective risk on the calibration split is
    within target, so that coverage is as high as the risk budget allows.

    Returns (tau, achieved_risk, achieved_coverage). If no threshold reaches the
    target, returns the most conservative one tried and its actual risk - the
    caller must not silently present that as meeting the target.
    """
    candidates = np.unique(np.round(confidence, 4))
    best = (1.0, 1.0, 0.0)
    for tau in np.sort(candidates):
        mask = confidence >= tau
        coverage = float(mask.mean())
        if coverage < min_coverage:
            continue
        risk = float(1.0 - correct[mask].mean())
        if risk <= target_risk:
            return float(tau), risk, coverage
        best = (float(tau), risk, coverage)
    return best


def fit_effective_noise(
    matrix: np.ndarray,
    observations: list[dict[int, int]],
    truth: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose the likelihood noise that best explains calibration data.

    The agent does not know the environment's true noise. Rather than assume a
    value, estimate the one under which its own observations are most likely.
    Returns (noise, mean_negative_log_likelihood).
    """
    from ayur.planner.posterior import Posterior

    if grid is None:
        grid = np.linspace(0.02, 0.45, 44)

    best_noise, best_nll = 0.05, np.inf
    for noise in grid:
        post_template = Posterior(matrix, noise=float(noise))
        total = 0.0
        for obs, t in zip(observations, truth):
            p = post_template.copy()
            p.reset()
            for k, v in obs.items():
                p.update(k, v)
            total += -np.log(max(p.belief[t], 1e-300))
        nll = total / max(len(observations), 1)
        if nll < best_nll:
            best_noise, best_nll = float(noise), float(nll)
    return best_noise, best_nll
