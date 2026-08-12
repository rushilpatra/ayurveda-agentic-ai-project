"""Explicit belief state over conditions.

This is the component that distinguishes the approach from LLM-internal
deliberation (MAI-DxO): the distribution actually exists, so it can be
inspected, audited and calibrated. Updates are pure numpy - no LLM calls.
"""
from __future__ import annotations

import numpy as np

from ayur.data.schema import DEFAULT_NOISE


class Posterior:
    """Naive-Bayes belief over D conditions given binary feature observations.

    The conditional-independence assumption is a stated modelling choice, not an
    oversight; its calibration cost is measured and reported rather than hidden.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        prior: np.ndarray | None = None,
        noise: float = DEFAULT_NOISE,
    ):
        if not 0.0 < noise < 0.5:
            raise ValueError(f"noise must be in (0, 0.5), got {noise}")
        self.matrix = matrix.astype(np.float64)
        self.n_conditions, self.n_features = self.matrix.shape
        self.noise = noise

        # P(feature = 1 | condition)
        self.p1 = np.where(self.matrix > 0, 1.0 - noise, noise)
        self._log_p1 = np.log(self.p1)
        self._log_p0 = np.log1p(-self.p1)

        if prior is None:
            prior = np.full(self.n_conditions, 1.0 / self.n_conditions)
        prior = np.asarray(prior, dtype=np.float64)
        if prior.shape != (self.n_conditions,):
            raise ValueError(f"prior must have shape ({self.n_conditions},)")
        if not np.isclose(prior.sum(), 1.0):
            prior = prior / prior.sum()
        self.log_prior = np.log(np.clip(prior, 1e-300, None))

        self.reset()

    # --- state ---------------------------------------------------------------

    def reset(self) -> None:
        self._logp = self.log_prior.copy()
        self.observed: dict[int, int] = {}

    def copy(self) -> "Posterior":
        other = object.__new__(Posterior)
        other.__dict__.update(self.__dict__)
        other._logp = self._logp.copy()
        other.observed = dict(self.observed)
        return other

    @property
    def belief(self) -> np.ndarray:
        """Normalised posterior over conditions."""
        z = self._logp - self._logp.max()
        b = np.exp(z)
        return b / b.sum()

    def update(self, feature: int, value: int) -> None:
        """Condition on observing `feature` = value (1 present, 0 absent)."""
        if feature in self.observed:
            return  # idempotent: never double-count evidence
        self._logp = self._logp + (
            self._log_p1[:, feature] if value else self._log_p0[:, feature]
        )
        self.observed[feature] = int(value)

    # --- summaries -----------------------------------------------------------

    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief, in nats."""
        b = self.belief
        nz = b[b > 0]
        return float(-(nz * np.log(nz)).sum())

    @property
    def argmax(self) -> int:
        return int(self.belief.argmax())

    @property
    def confidence(self) -> float:
        return float(self.belief.max())

    def top_k(self, k: int = 5) -> list[tuple[int, float]]:
        b = self.belief
        idx = np.argpartition(-b, min(k, len(b) - 1))[:k]
        idx = idx[np.argsort(-b[idx])]
        return [(int(i), float(b[i])) for i in idx]

    def margin(self) -> float:
        """Gap between the top two hypotheses - a sharper stopping signal than
        max probability alone when the belief is bimodal."""
        b = np.sort(self.belief)[::-1]
        return float(b[0] - b[1]) if len(b) > 1 else 1.0
