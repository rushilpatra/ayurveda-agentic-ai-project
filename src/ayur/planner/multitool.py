"""Cost-aware planner over the heterogeneous action space, and the agent loop.

This implements the paper's central claim: choosing *which knowledge source to
consult* is a planning problem distinct from choosing which question to ask.

Policies compared:
  ToolEIGPolicy       maximise information per unit cost   (proposed)
  ToolEIGLagrangian   maximise information minus lambda*cost
  PatientOnlyPolicy   the single-source ablation - what every prior system does
  CheapestFirstPolicy always take the cheapest informative action (cost-blind to value)
  RandomToolPolicy    floor
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ayur.planner.actions import (
    Action,
    ActionKind,
    ActionSpace,
    action_mutual_information,
)
from ayur.planner.posterior import Posterior


@dataclass
class ToolPolicy:
    name: str

    def score(self, space: ActionSpace, belief: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def select(self, space: ActionSpace, post: Posterior,
               taken: np.ndarray, rng: np.random.Generator) -> int:
        scores = self.score(space, post.belief).copy()
        scores[taken] = -np.inf
        best = scores.max()
        if not np.isfinite(best):
            raise RuntimeError("no actions remain")
        tied = np.flatnonzero(scores >= best - 1e-12)
        return int(rng.choice(tied)) if tied.size > 1 else int(scores.argmax())


@dataclass
class ToolEIGPolicy(ToolPolicy):
    """Proposed: expected information gain per unit cost."""

    name: str = "tool-eig-per-cost"

    def score(self, space, belief):
        return action_mutual_information(space, belief) / space.cost


@dataclass
class ToolEIGLagrangian(ToolPolicy):
    """Information minus a price on cost. Equivalent family, different knob."""

    name: str = "tool-eig-lagrangian"
    lambda_cost: float = 0.05

    def score(self, space, belief):
        return action_mutual_information(space, belief) - self.lambda_cost * space.cost


@dataclass
class PatientOnlyPolicy(ToolPolicy):
    """Ablation: the single-source agent. What MAI-DxO / AgentClinic do."""

    name: str = "patient-only"

    def score(self, space, belief):
        mi = action_mutual_information(space, belief)
        mi = np.where(space.kind == ActionKind.ASK_PATIENT.value, mi, -np.inf)
        return mi


@dataclass
class CheapestFirstPolicy(ToolPolicy):
    """Always take the cheapest action with any information at all.

    Isolates whether the benefit comes from *weighing* information against cost,
    or merely from having cheap actions available.
    """

    name: str = "cheapest-first"

    def score(self, space, belief):
        mi = action_mutual_information(space, belief)
        informative = mi > 1e-6
        return np.where(informative, -space.cost, -np.inf)


@dataclass
class RandomToolPolicy(ToolPolicy):
    name: str = "random-tool"

    def score(self, space, belief):
        return np.zeros(len(space))  # all tied -> uniform random via tie-break


@dataclass
class ToolStoppingRule:
    tau_confidence: float = 0.5
    min_information_gain: float = 1e-3
    max_cost: float = 15.0
    allow_abstain: bool = True

    def decide(self, space, post, taken, spent) -> str:
        if post.confidence >= self.tau_confidence:
            return "diagnose"
        if spent >= self.max_cost or taken.all():
            return "abstain" if self.allow_abstain else "diagnose"
        mi = action_mutual_information(space, post.belief)
        mi[taken] = -np.inf
        # Budget-aware: an action we cannot afford is not available.
        affordable = space.cost <= (self.max_cost - spent)
        mi = np.where(affordable, mi, -np.inf)
        if not np.isfinite(mi.max()) or mi.max() < self.min_information_gain:
            return "abstain" if self.allow_abstain else "diagnose"
        return "act"


@dataclass
class ToolTrajectory:
    case_index: int
    agent: str
    true_condition: int
    predicted: int | None
    decision: str
    confidence: float
    n_actions: int
    total_cost: float
    actions: list[str]
    kinds: list[str]
    observations: list[int]
    evidence: list[str]
    correct: bool
    seconds: float

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["confidence"] = round(self.confidence, 6)
        d["total_cost"] = round(self.total_cost, 4)
        d["seconds"] = round(self.seconds, 5)
        return d


@dataclass
class MultiToolAgent:
    name: str
    policy: ToolPolicy
    stopping: ToolStoppingRule
    noise: float = 0.05

    def run(self, case, env, space: ActionSpace, kg=None) -> ToolTrajectory:
        import time

        t0 = time.perf_counter()
        post = Posterior(space.space.matrix, noise=self.noise)
        taken = np.zeros(len(space), dtype=bool)
        # Two actions can observe the same underlying attribute through
        # different channels; observing it twice must not double-count.
        observed_features: set[int] = set()

        for k, v in case.revealed.items():
            post.update(k, v)
            observed_features.add(k)
            space.block_feature(taken, k)

        rng = np.random.default_rng([env.seed, case.index, 104729])
        labels, kinds, observations, evidence = [], [], [], []
        spent = 0.0

        while True:
            decision = self.stopping.decide(space, post, taken, spent)
            if decision != "act":
                break
            # Mask actions we cannot afford before selecting. The stopping rule
            # only guarantees that *some* affordable action exists; without this
            # the policy would pick the highest-scoring one and overspend.
            unaffordable = space.cost > (self.stopping.max_cost - spent) + 1e-9
            idx = self.policy.select(space, post, taken | unaffordable, rng)
            action: Action = space.actions[idx]
            taken[idx] = True
            spent += action.cost

            obs = self._observe(action, case, env, space, kg)
            labels.append(str(action))
            kinds.append(action.kind.value)
            observations.append(int(obs))

            if obs >= 0:
                # Update through this channel's likelihood, not the patient's.
                post._logp = post._logp + (
                    space._log_p1[idx] if obs else space._log_p0[idx]
                )
                feature = space.feature_of[idx]
                if feature >= 0:
                    observed_features.add(feature)
                    # Block the same attribute via other channels.
                    space.block_feature(taken, feature)
                evidence.append(self._evidence(action, obs, kg))

        predicted = post.argmax if decision == "diagnose" else None
        return ToolTrajectory(
            case_index=case.index,
            agent=self.name,
            true_condition=int(case.condition),
            predicted=None if predicted is None else int(predicted),
            decision=decision,
            confidence=post.confidence,
            n_actions=len(labels),
            total_cost=spent,
            actions=labels,
            kinds=kinds,
            observations=observations,
            evidence=evidence,
            correct=bool(predicted is not None and predicted == case.condition),
            seconds=time.perf_counter() - t0,
        )

    def _observe(self, action, case, env, space, kg) -> int:
        """Return 1 / 0, or -1 for an unusable answer."""
        from ayur.env.patient import UNKNOWN

        if action.kind is ActionKind.ASK_PATIENT:
            ans = env.ask(case, action.target)
            return -1 if ans == UNKNOWN else ans

        if action.kind in (ActionKind.QUERY_KG, ActionKind.RETRIEVE_TEXT):
            # These consult a source about the patient's true attribute, but
            # through a lossier channel: the source may disagree with reality.
            truth = int(case.truth[action.target])
            eps = space.noise[action.kind]
            r = np.random.default_rng([env.seed, case.index, action.target,
                                       hash(action.kind.value) % 10_000])
            return int(1 - truth) if r.random() < eps else truth

        if action.kind is ActionKind.VERIFY_HERB:
            # Confirmatory: is this herb compatible with the patient's true
            # condition? Answered from the symbolic pharmacology, with noise.
            compat = space._herb_compat_cache[case.condition, action.target]
            eps = space.noise[ActionKind.VERIFY_HERB]
            r = np.random.default_rng([env.seed, case.index, 5_000_000 + action.target])
            return int(1 - compat) if r.random() < eps else int(compat)

        return -1

    def _evidence(self, action, obs, kg) -> str:
        if action.kind is ActionKind.VERIFY_HERB:
            return f"[herb-verify] {action.label} compatible={bool(obs)} (amidha pharmacology)"
        if action.kind is ActionKind.QUERY_KG:
            return f"[kg] {action.label} = {obs}"
        if action.kind is ActionKind.RETRIEVE_TEXT:
            return f"[retrieval] {action.label} = {obs}"
        return f"[patient] {action.label} = {obs}"


def build_tool_agents(max_cost: float = 15.0, tau: float = 0.5,
                      noise: float = 0.05) -> list[MultiToolAgent]:
    forced = ToolStoppingRule(tau_confidence=1.01, max_cost=max_cost,
                              min_information_gain=-1.0, allow_abstain=False)
    return [
        MultiToolAgent("T1-tool-eig-per-cost", ToolEIGPolicy(), forced, noise),
        MultiToolAgent("T2-tool-eig-lagrangian", ToolEIGLagrangian(), forced, noise),
        MultiToolAgent("T3-patient-only", PatientOnlyPolicy(), forced, noise),
        MultiToolAgent("T4-cheapest-first", CheapestFirstPolicy(), forced, noise),
        MultiToolAgent("T5-random-tool", RandomToolPolicy(), forced, noise),
    ]
