"""Tests for the heterogeneous action space and the cost-aware planner."""
from __future__ import annotations

import numpy as np
import pytest

from ayur.data.prep import build_condition_space
from ayur.env.patient import PatientEnvironment
from ayur.kg.graph import KnowledgeGraph
from ayur.planner.actions import (
    DEFAULT_COSTS,
    ActionKind,
    ActionSpace,
    action_mutual_information,
)
from ayur.planner.multitool import (
    CheapestFirstPolicy,
    MultiToolAgent,
    PatientOnlyPolicy,
    ToolEIGPolicy,
    ToolStoppingRule,
    build_tool_agents,
)
from ayur.planner.posterior import Posterior


@pytest.fixture(scope="module")
def space():
    return build_condition_space()


@pytest.fixture(scope="module")
def kg():
    return KnowledgeGraph.build()


@pytest.fixture(scope="module")
def action_space(space, kg):
    return ActionSpace(space, kg=kg)


# --- action space -------------------------------------------------------------


def test_all_four_channels_present(action_space):
    kinds = set(action_space.summary()["by_kind"])
    assert kinds == {"ask_patient", "query_kg", "retrieve_text", "verify_herb"}


def test_patient_channel_covers_every_feature(space, action_space):
    n = action_space.summary()["by_kind"]["ask_patient"]
    assert n == space.n_features


def test_kg_channel_restricted_to_dosha_attributes(space, action_space):
    for a in action_space.actions:
        if a.kind is ActionKind.QUERY_KG:
            column = a.label.split("::", 1)[0]
            assert column in ("Doshas", "Constitution/Prakriti")


def test_costs_are_ordered_as_designed(action_space):
    c = action_space.costs
    assert c[ActionKind.QUERY_KG] < c[ActionKind.RETRIEVE_TEXT] \
        < c[ActionKind.VERIFY_HERB] < c[ActionKind.ASK_PATIENT]


def test_likelihoods_are_valid_probabilities(action_space):
    assert (action_space.p1 > 0).all() and (action_space.p1 < 1).all()


def test_noisier_channel_carries_less_information(space, action_space):
    """retrieve_text observes the same attribute as ask_patient but noisier,
    so it must never be more informative for the same feature."""
    post = Posterior(space.matrix)
    mi = action_mutual_information(action_space, post.belief)
    by_feature = {}
    for i, a in enumerate(action_space.actions):
        if a.kind in (ActionKind.ASK_PATIENT, ActionKind.RETRIEVE_TEXT):
            by_feature.setdefault(a.target, {})[a.kind] = mi[i]
    checked = 0
    for feat, d in by_feature.items():
        if len(d) == 2:
            assert d[ActionKind.RETRIEVE_TEXT] <= d[ActionKind.ASK_PATIENT] + 1e-12
            checked += 1
    assert checked > 100


def test_uninformative_herbs_are_excluded(action_space):
    """A herb compatible with every condition tells you nothing."""
    post = Posterior(action_space.space.matrix)
    mi = action_mutual_information(action_space, post.belief)
    for i, a in enumerate(action_space.actions):
        if a.kind is ActionKind.VERIFY_HERB:
            assert mi[i] > 0


def test_reverse_index_matches_linear_scan(action_space):
    """The precomputed index must agree with the scan it replaced."""
    for feature in (0, 5, 100, 500):
        expected = {i for i, f in enumerate(action_space.feature_of) if f == feature}
        got = set(action_space.actions_for_feature.get(feature, np.array([], int)).tolist())
        assert got == expected


def test_label_index_is_consistent(action_space):
    for i in (0, 10, 1000, len(action_space) - 1):
        assert action_space.index_of_label[str(action_space.actions[i])] == i


# --- policies -----------------------------------------------------------------


def test_patient_only_never_uses_a_tool(space, action_space):
    post = Posterior(space.matrix)
    taken = np.zeros(len(action_space), dtype=bool)
    rng = np.random.default_rng(0)
    policy = PatientOnlyPolicy()
    for _ in range(20):
        idx = policy.select(action_space, post, taken, rng)
        assert action_space.actions[idx].kind is ActionKind.ASK_PATIENT
        taken[idx] = True


def test_eig_per_cost_prefers_cheap_informative_actions(space, action_space):
    """The KG channel is near-free and informative, so it should go first."""
    post = Posterior(space.matrix)
    taken = np.zeros(len(action_space), dtype=bool)
    rng = np.random.default_rng(0)
    idx = ToolEIGPolicy().select(action_space, post, taken, rng)
    assert action_space.actions[idx].kind is ActionKind.QUERY_KG


def test_cheapest_first_ignores_information(space, action_space):
    post = Posterior(space.matrix)
    taken = np.zeros(len(action_space), dtype=bool)
    rng = np.random.default_rng(0)
    idx = CheapestFirstPolicy().select(action_space, post, taken, rng)
    assert action_space.actions[idx].cost == min(
        a.cost for a in action_space.actions)


def test_policies_never_repeat_an_action(space, action_space):
    rng = np.random.default_rng(0)
    for policy in (ToolEIGPolicy(), CheapestFirstPolicy(), PatientOnlyPolicy()):
        post = Posterior(space.matrix)
        taken = np.zeros(len(action_space), dtype=bool)
        for _ in range(25):
            idx = policy.select(action_space, post, taken, rng)
            assert not taken[idx]
            taken[idx] = True


# --- agent loop ---------------------------------------------------------------


def test_agent_respects_cost_budget(space, action_space, kg):
    env = PatientEnvironment(space, seed=0)
    agent = MultiToolAgent("t", ToolEIGPolicy(),
                           ToolStoppingRule(tau_confidence=1.01, max_cost=5.0,
                                            allow_abstain=False))
    for case in env.cases(5):
        tr = agent.run(case, env, action_space, kg)
        assert tr.total_cost <= 5.0 + 1e-9, tr.total_cost


def test_agent_does_not_observe_same_attribute_twice(space, action_space, kg):
    """Blocking must span channels: KG and patient observe the same feature."""
    env = PatientEnvironment(space, seed=1)
    agent = MultiToolAgent("t", ToolEIGPolicy(),
                           ToolStoppingRule(tau_confidence=1.01, max_cost=12.0,
                                            allow_abstain=False))
    for case in env.cases(5):
        tr = agent.run(case, env, action_space, kg)
        features = [action_space.feature_of[action_space.index_of_label[a]]
                    for a in tr.actions]
        observed = [f for f in features if f >= 0]
        assert len(observed) == len(set(observed)), "an attribute was observed twice"


def test_every_used_observation_produces_evidence(space, action_space, kg):
    env = PatientEnvironment(space, seed=2)
    agent = MultiToolAgent("t", ToolEIGPolicy(),
                           ToolStoppingRule(tau_confidence=1.01, max_cost=10.0,
                                            allow_abstain=False))
    for case in env.cases(5):
        tr = agent.run(case, env, action_space, kg)
        usable = sum(1 for o in tr.observations if o >= 0)
        assert len(tr.evidence) == usable


def test_trajectories_are_deterministic(space, action_space, kg):
    env = PatientEnvironment(space, seed=3)
    agent = MultiToolAgent("t", ToolEIGPolicy(),
                           ToolStoppingRule(tau_confidence=0.5, max_cost=10.0))
    case = env.make_case(4)
    a = agent.run(case, env, action_space, kg)
    b = agent.run(case, env, action_space, kg)
    assert a.actions == b.actions
    assert a.predicted == b.predicted
    assert a.total_cost == b.total_cost


def test_forced_tool_agents_always_commit(space, action_space, kg):
    env = PatientEnvironment(space, seed=4)
    for agent in build_tool_agents(max_cost=8.0):
        for case in env.cases(3):
            tr = agent.run(case, env, action_space, kg)
            assert tr.decision == "diagnose"
            assert tr.predicted is not None
