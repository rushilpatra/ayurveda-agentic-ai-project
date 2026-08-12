"""Metrics, organised by the npj review's two evaluation tiers.

Basic indicators        - objective correctness, task completion
Developmental indicators - efficiency, uncertainty/calibration

The calibration and selective-prediction metrics are the ones neither MAI-DxO
nor AgentClinic report; they are the paper's clearest open lane, so they are
implemented carefully rather than approximated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- basic: correctness -------------------------------------------------------


def accuracy(correct: np.ndarray) -> float:
    return float(np.mean(correct)) if correct.size else float("nan")


def top_k_accuracy(top5: list[list], truth: np.ndarray, k: int = 3) -> float:
    """top5 is a list of [[condition, prob], ...] per case, already sorted."""
    hits = [int(t) in [int(c) for c, _ in row[:k]] for row, t in zip(top5, truth)]
    return float(np.mean(hits)) if hits else float("nan")


def macro_f1(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over conditions.

    Mandatory here, not optional: the condition distribution is long-tailed, so
    micro-accuracy is dominated by whichever conditions happen to be sampled.
    Classes absent from both prediction and truth are skipped rather than
    counted as F1 = 0, which would otherwise make the score a function of how
    many classes went unsampled.
    """
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (truth == c)))
        fp = int(np.sum((pred == c) & (truth != c)))
        fn = int(np.sum((pred != c) & (truth == c)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


# --- developmental: uncertainty ----------------------------------------------


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE with equal-width bins on [0, 1]."""
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Brier on the top-1 confidence treated as P(correct)."""
    if confidence.size == 0:
        return float("nan")
    return float(np.mean((confidence - correct.astype(float)) ** 2))


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return out


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk as a function of coverage, ordering by confidence.

    Returns (coverage, risk). Answering only the most confident cases should
    lower risk; if it does not, the confidence signal is worthless.
    """
    order = np.argsort(-confidence)
    c = correct[order].astype(float)
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(1.0 - c) / np.arange(1, n + 1)
    return coverage, risk


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    if confidence.size == 0:
        return float("nan")
    coverage, risk = risk_coverage(confidence, correct)
    return float(np.trapz(risk, coverage))


# --- statistics ---------------------------------------------------------------


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """CI on mean(a) - mean(b) over paired cases, plus a two-sided p-value.

    Paired because every agent sees exactly the same cases - which is only true
    because the environment is deterministic.
    """
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    observed = float(d.mean())
    # p-value by inverting the bootstrap around zero
    centred = boots - observed
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    return observed, float(lo), float(hi), p


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar for paired binary outcomes. Returns (b01, b10, p)."""
    from math import comb

    b01 = int(np.sum(~a_correct.astype(bool) & b_correct.astype(bool)))
    b10 = int(np.sum(a_correct.astype(bool) & ~b_correct.astype(bool)))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return b01, b10, float(min(1.0, 2 * tail))


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni across a family of comparisons."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)   # enforce monotonicity
        adjusted[key] = running
    return adjusted


# --- aggregation --------------------------------------------------------------


@dataclass
class AgentResult:
    agent: str
    n: int
    n_diagnosed: int
    n_abstained: int
    coverage: float
    accuracy_overall: float
    accuracy_selective: float
    top3: float
    top5: float
    macro_f1: float
    mean_questions: float
    median_questions: float
    ece: float
    brier: float
    aurc: float
    mean_seconds: float

    def row(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
