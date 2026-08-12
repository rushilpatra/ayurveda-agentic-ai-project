"""Action-selection policies, including the cost-aware EIG planner.

All policies are pure numpy. Evaluating thousands of interactive cases requires
no LLM call at any step - which is what makes the full evaluation feasible on a
16 GB laptop, and what makes it exactly reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .posterior import Posterior


def mutual_information(post: Posterior, belief: np.ndarray | None = None) -> np.ndarray:
    """I(condition ; feature_k) under the current belief, for every feature.

        I = H(o_k) - E_d[ H(o_k | d) ]

    Equivalently the expected reduction in posterior entropy from observing
    feature k. Computed in closed form for all K features at once.
    """
    b = post.belief if belief is None else belief

    p1 = np.clip(b @ post.p1, 1e-12, 1.0 - 1e-12)      # P(o_k = 1)
    h_marginal = -(p1 * np.log(p1) + (1 - p1) * np.log1p(-p1))

    # H(o_k | d) is constant per feature-condition pair; weight by belief.
    p = post.p1
    h_cond_per_condition = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    h_conditional = b @ h_cond_per_condition

    return h_marginal - h_conditional


@dataclass
class Policy:
    name: str

    def select(self, post: Posterior, asked: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError


@dataclass
class RandomPolicy(Policy):
    """B3 - floor. Picks uniformly among unasked features."""

    name: str = "random"

    def select(self, post, asked, rng):
        candidates = np.flatnonzero(~asked)
        if candidates.size == 0:
            raise RuntimeError("no features left to ask about")
        return int(rng.choice(candidates))


@dataclass
class FrequencyPolicy(Policy):
    """B5 - greedy most-frequent unasked feature.

    Belief-independent: it asks popular questions, but never adapts to what it
    has already learned. Isolates how much of the planner's benefit comes from
    information gain rather than from merely asking things.
    """

    name: str = "greedy-frequency"
    _counts: np.ndarray | None = None

    def select(self, post, asked, rng):
        if self._counts is None:
            self._counts = post.matrix.sum(0)
        scores = self._counts.astype(np.float64).copy()
        scores[asked] = -np.inf
        return int(scores.argmax())


@dataclass
class EIGPolicy(Policy):
    """B7 - maximise expected information gain, net of action cost.

        a* = argmax_k [ I(D ; o_k) - lambda * cost(k) ]

    With uniform costs and lambda = 0 this is pure max-EIG.
    """

    name: str = "max-eig"
    cost: np.ndarray | None = None
    lambda_cost: float = 0.0

    def select(self, post, asked, rng):
        mi = mutual_information(post)
        if self.lambda_cost and self.cost is not None:
            mi = mi - self.lambda_cost * self.cost
        mi = mi.copy()
        mi[asked] = -np.inf
        best = mi.max()
        # Break ties randomly rather than by index, so feature ordering in the
        # matrix cannot silently advantage the policy.
        tied = np.flatnonzero(mi >= best - 1e-12)
        return int(rng.choice(tied)) if tied.size > 1 else int(mi.argmax())


@dataclass
class StoppingRule:
    """When to commit, when to keep asking, and when to abstain.

    Neither MAI-DxO nor AgentClinic has an abstention mechanism; both force a
    commitment. `tau_confidence` should be fitted on a calibration split to hit
    a target selective risk, not hand-tuned.
    """

    tau_confidence: float = 0.5
    min_information_gain: float = 1e-3
    max_questions: int = 20
    #: When False the agent must always commit, exactly like MAI-DxO and
    #: AgentClinic. Kept separate from `tau_confidence`: a high tau means
    #: "keep asking", it must not silently become "refuse to answer".
    allow_abstain: bool = True

    def _exhausted(self, post: Posterior) -> str:
        return "abstain" if self.allow_abstain else "diagnose"

    def decide(self, post: Posterior, asked: np.ndarray, n_asked: int) -> str:
        """Returns one of: 'diagnose', 'abstain', 'ask'."""
        if post.confidence >= self.tau_confidence:
            return "diagnose"
        if n_asked >= self.max_questions or asked.all():
            return self._exhausted(post)
        best_gain = float(np.max(np.where(asked, -np.inf, mutual_information(post))))
        if best_gain < self.min_information_gain:
            # Nothing left worth asking, and we are still not confident.
            return self._exhausted(post)
        return "ask"


POLICIES = {
    "random": RandomPolicy,
    "greedy-frequency": FrequencyPolicy,
    "max-eig": EIGPolicy,
}
