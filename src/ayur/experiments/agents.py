"""Diagnostic agents = (policy, stopping rule) pairs, plus the non-interactive
baselines. These are B1-B8 from POSITIONING.md section 7.

Every agent here is pure numpy. The LLM-planner baseline (B6) lives separately
because it is the only one that needs a model loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ayur.env.patient import UNKNOWN, Case, PatientEnvironment
from ayur.planner.policy import (
    EIGPolicy,
    FrequencyPolicy,
    Policy,
    RandomPolicy,
    StoppingRule,
)
from ayur.planner.posterior import Posterior


@dataclass
class Trajectory:
    """One case, start to finish. This is the unit that gets checkpointed."""

    case_index: int
    agent: str
    true_condition: int
    predicted: int | None
    decision: str                      # diagnose | abstain
    confidence: float
    entropy: float
    n_questions: int
    asked: list[int]
    answers: list[int]
    top5: list[tuple[int, float]]
    correct: bool
    seconds: float

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["top5"] = [[int(i), round(float(p), 6)] for i, p in self.top5]
        d["confidence"] = round(self.confidence, 6)
        d["entropy"] = round(self.entropy, 6)
        d["seconds"] = round(self.seconds, 5)
        return d


@dataclass
class InteractiveAgent:
    """B3 / B5 / B7 / B8 - sequential agents differing only in policy and stopping."""

    name: str
    policy_factory: type[Policy] | None
    stopping: StoppingRule
    prior: np.ndarray | None = None
    noise: float = 0.05

    def run(self, case: Case, env: PatientEnvironment, matrix: np.ndarray) -> Trajectory:
        import time

        t0 = time.perf_counter()
        post = Posterior(matrix, prior=self.prior, noise=self.noise)
        asked_mask = np.zeros(matrix.shape[1], dtype=bool)
        asked: list[int] = []
        answers: list[int] = []

        # Presenting complaint is free - it is not a question the agent asked.
        for k, v in case.revealed.items():
            post.update(k, v)
            asked_mask[k] = True

        rng = np.random.default_rng([env.seed, case.index, 7919])
        policy = self.policy_factory() if self.policy_factory else None

        decision = "diagnose"
        while policy is not None:
            decision = self.stopping.decide(post, asked_mask, len(asked))
            if decision != "ask":
                break
            k = policy.select(post, asked_mask, rng)
            asked_mask[k] = True
            ans = env.ask(case, k)
            asked.append(int(k))
            answers.append(int(ans))
            if ans != UNKNOWN:
                post.update(k, ans)

        predicted = post.argmax if decision == "diagnose" else None
        return Trajectory(
            case_index=case.index,
            agent=self.name,
            true_condition=int(case.condition),
            predicted=None if predicted is None else int(predicted),
            decision=decision,
            confidence=post.confidence,
            entropy=post.entropy,
            n_questions=len(asked),
            asked=asked,
            answers=answers,
            top5=post.top_k(5),
            correct=bool(predicted is not None and predicted == case.condition),
            seconds=time.perf_counter() - t0,
        )


@dataclass
class PriorOnlyAgent:
    """B4 - answers from the prior, asks nothing.

    The most dangerous baseline: if it is competitive, interaction adds nothing
    and the paper has no contribution. Measured at 0.30% vs 0.22% chance.
    """

    name: str = "B4-prior-only"
    prior: np.ndarray | None = None
    noise: float = 0.05

    def run(self, case: Case, env: PatientEnvironment, matrix: np.ndarray) -> Trajectory:
        import time

        t0 = time.perf_counter()
        post = Posterior(matrix, prior=self.prior, noise=self.noise)
        for k, v in case.revealed.items():
            post.update(k, v)
        return Trajectory(
            case_index=case.index,
            agent=self.name,
            true_condition=int(case.condition),
            predicted=int(post.argmax),
            decision="diagnose",
            confidence=post.confidence,
            entropy=post.entropy,
            n_questions=0,
            asked=[],
            answers=[],
            top5=post.top_k(5),
            correct=bool(post.argmax == case.condition),
            seconds=time.perf_counter() - t0,
        )


@dataclass
class FullInformationAgent:
    """B1 - ceiling. Sees every attribute; no interaction needed.

    Bounds what any amount of questioning could achieve under this likelihood.
    """

    name: str = "B1-full-information"
    prior: np.ndarray | None = None
    noise: float = 0.05

    def run(self, case: Case, env: PatientEnvironment, matrix: np.ndarray) -> Trajectory:
        import time

        t0 = time.perf_counter()
        post = Posterior(matrix, prior=self.prior, noise=self.noise)
        for k in range(matrix.shape[1]):
            post.update(k, int(case.truth[k]))
        return Trajectory(
            case_index=case.index,
            agent=self.name,
            true_condition=int(case.condition),
            predicted=int(post.argmax),
            decision="diagnose",
            confidence=post.confidence,
            entropy=post.entropy,
            n_questions=matrix.shape[1],
            asked=[],
            answers=[],
            top5=post.top_k(5),
            correct=bool(post.argmax == case.condition),
            seconds=time.perf_counter() - t0,
        )


def build_agents(
    max_questions: int = 15,
    tau: float = 0.5,
    prior: np.ndarray | None = None,
    noise: float = 0.05,
) -> list:
    """The full baseline ladder. Order matters for report tables."""
    # Forced agents spend the whole budget then commit - the protocol MAI-DxO
    # and AgentClinic use. tau is unreachable so they never stop early, and
    # allow_abstain=False makes them commit rather than decline at the budget.
    forced = StoppingRule(
        tau_confidence=1.01,
        max_questions=max_questions,
        min_information_gain=-1.0,
        allow_abstain=False,
    )
    adaptive = StoppingRule(
        tau_confidence=tau, max_questions=max_questions, allow_abstain=True
    )

    return [
        FullInformationAgent(prior=prior, noise=noise),
        PriorOnlyAgent(prior=prior, noise=noise),
        InteractiveAgent("B3-random", RandomPolicy, forced, prior, noise),
        InteractiveAgent("B5-greedy-frequency", FrequencyPolicy, forced, prior, noise),
        InteractiveAgent("B7-max-eig", EIGPolicy, adaptive, prior, noise),
        InteractiveAgent("B8-max-eig-no-abstain", EIGPolicy, forced, prior, noise),
    ]
