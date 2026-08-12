"""Deterministic interactive patient environment.

Contrast with prior work: AgentClinic simulates the patient with an LLM, and
reports that swapping the patient model moves measured doctor accuracy from 52%
to 46%; SDBench's Gatekeeper synthesises findings, so the same case differs
between runs. In both, the measuring instrument is itself stochastic and part of
every score belongs to the environment rather than the agent under test.

Here a case is a pure function of (dataset, seed, case index). Two runs on two
machines produce identical trajectories, so a score change means the agent
changed. Realism lost to determinism is recovered by sweeping an explicit noise
model rather than by inheriting an LLM's unspecified one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ayur.data.prep import ConditionSpace
from ayur.data.schema import DEFAULT_NOISE

#: Column whose features can serve as a presenting complaint.
CHIEF_COMPLAINT_COLUMN = "Symptoms"

#: Patient response codes.
YES, NO, UNKNOWN = 1, 0, -1


@dataclass
class Case:
    """One interactive case. Fully determined by (space, seed, index)."""

    index: int
    condition: int                 # ground-truth condition index
    truth: np.ndarray              # (K,) sampled binary attribute values
    revealed: dict[int, int]       # features visible before any question
    seed: int

    @property
    def chief_complaint(self) -> int | None:
        for k, v in self.revealed.items():
            if v == YES:
                return k
        return None


@dataclass
class PatientEnvironment:
    """Generates cases and answers attribute queries.

    Parameters
    ----------
    noise
        P(feature = 1 | condition) is ``1 - noise`` when listed, ``noise`` when
        not. Governs how faithfully a patient presents the textbook picture.
    omission_rate
        Probability the patient cannot or will not answer a given question,
        returning UNKNOWN. Models real consultation friction; swept as an
        experimental variable rather than fixed.
    n_revealed
        Number of attributes visible up front (the presenting complaint).
    """

    space: ConditionSpace
    noise: float = DEFAULT_NOISE
    omission_rate: float = 0.0
    n_revealed: int = 1
    seed: int = 0
    _p1: np.ndarray = field(init=False, repr=False)
    _complaint_features: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        m = self.space.matrix.astype(np.float64)
        self._p1 = np.where(m > 0, 1.0 - self.noise, self.noise)
        self._complaint_features = np.array(
            [
                k
                for k, f in enumerate(self.space.features)
                if f.split("::", 1)[0] == CHIEF_COMPLAINT_COLUMN
            ],
            dtype=int,
        )
        if self._complaint_features.size == 0:
            raise ValueError(
                f"no features from column {CHIEF_COMPLAINT_COLUMN!r}; "
                "cannot construct a presenting complaint"
            )

    # --- case construction ----------------------------------------------------

    def _rng(self, index: int) -> np.random.Generator:
        """Per-case generator: case i is identical regardless of what ran before."""
        return np.random.default_rng([self.seed, index])

    def make_case(self, index: int, condition: int | None = None) -> Case:
        rng = self._rng(index)
        if condition is None:
            condition = int(rng.integers(self.space.n_conditions))

        truth = (rng.random(self.space.n_features) < self._p1[condition]).astype(np.int8)

        # Presenting complaint: a symptom the patient actually has. Falls back to
        # any present feature, then to a declared-absent symptom, so that every
        # case starts with something rather than silently starting blank.
        present = self._complaint_features[truth[self._complaint_features] == 1]
        if present.size == 0:
            present = np.flatnonzero(truth == 1)
        if present.size == 0:
            present = self._complaint_features[:1]
            revealed = {int(present[0]): NO}
        else:
            chosen = rng.choice(present, size=min(self.n_revealed, present.size), replace=False)
            revealed = {int(k): int(truth[k]) for k in np.atleast_1d(chosen)}

        return Case(index=index, condition=condition, truth=truth, revealed=revealed, seed=self.seed)

    def cases(self, n: int, start: int = 0):
        for i in range(start, start + n):
            yield self.make_case(i)

    # --- interaction ----------------------------------------------------------

    def ask(self, case: Case, feature: int) -> int:
        """Answer a question about one attribute. Deterministic given the case.

        Returns YES, NO, or UNKNOWN. Omission is derived from a per-(case,
        feature) generator so that asking the same question twice - or asking
        questions in a different order - always yields the same answer.
        """
        if feature in case.revealed:
            return case.revealed[feature]
        if self.omission_rate > 0.0:
            r = np.random.default_rng([self.seed, case.index, int(feature)])
            if r.random() < self.omission_rate:
                return UNKNOWN
        return int(case.truth[feature])
