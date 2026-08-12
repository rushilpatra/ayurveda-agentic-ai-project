"""Tests for the posterior, the EIG planner and the deterministic environment."""
from __future__ import annotations

import numpy as np
import pytest

from ayur.data.prep import build_condition_space
from ayur.env.patient import NO, UNKNOWN, YES, PatientEnvironment
from ayur.planner.policy import (
    EIGPolicy,
    FrequencyPolicy,
    RandomPolicy,
    StoppingRule,
    mutual_information,
)
from ayur.planner.posterior import Posterior


@pytest.fixture(scope="module")
def space():
    return build_condition_space()


@pytest.fixture(scope="module")
def toy_matrix():
    """4 conditions x 3 features, hand-checkable.

    f0 splits {0,1} from {2,3} - perfectly informative under a uniform belief.
    f2 is present in every condition - carries no information at all.
    """
    return np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )


# --- posterior ----------------------------------------------------------------


def test_uniform_prior_is_uniform(toy_matrix):
    p = Posterior(toy_matrix)
    assert np.allclose(p.belief, 0.25)
    assert p.entropy == pytest.approx(np.log(4))


def test_belief_is_normalised_after_updates(toy_matrix):
    p = Posterior(toy_matrix)
    p.update(0, YES)
    p.update(1, NO)
    assert p.belief.sum() == pytest.approx(1.0)
    assert (p.belief >= 0).all()


def test_evidence_moves_belief_the_right_way(toy_matrix):
    p = Posterior(toy_matrix)
    p.update(0, YES)          # only conditions 0 and 1 have f0
    b = p.belief
    assert b[0] + b[1] > b[2] + b[3]
    assert p.argmax in (0, 1)


def test_update_is_idempotent(toy_matrix):
    """Asking the same question twice must not double-count the evidence."""
    a = Posterior(toy_matrix)
    a.update(0, YES)
    b = Posterior(toy_matrix)
    b.update(0, YES)
    b.update(0, YES)
    assert np.allclose(a.belief, b.belief)


def test_update_order_does_not_matter(toy_matrix):
    a = Posterior(toy_matrix)
    a.update(0, YES)
    a.update(1, NO)
    b = Posterior(toy_matrix)
    b.update(1, NO)
    b.update(0, YES)
    assert np.allclose(a.belief, b.belief)


def test_entropy_decreases_with_informative_evidence(toy_matrix):
    p = Posterior(toy_matrix)
    before = p.entropy
    p.update(0, YES)
    assert p.entropy < before


def test_reset_restores_prior(toy_matrix):
    p = Posterior(toy_matrix)
    p.update(0, YES)
    p.reset()
    assert np.allclose(p.belief, 0.25)
    assert p.observed == {}


def test_rejects_bad_noise(toy_matrix):
    with pytest.raises(ValueError):
        Posterior(toy_matrix, noise=0.0)
    with pytest.raises(ValueError):
        Posterior(toy_matrix, noise=0.7)


# --- information gain ---------------------------------------------------------


def test_uninformative_feature_has_near_zero_gain(toy_matrix):
    """f2 is present in every condition, so observing it tells you nothing."""
    p = Posterior(toy_matrix)
    mi = mutual_information(p)
    assert mi[2] == pytest.approx(0.0, abs=1e-9)
    assert mi[0] > mi[2]


def test_splitting_feature_has_highest_gain(toy_matrix):
    p = Posterior(toy_matrix)
    mi = mutual_information(p)
    assert mi.argmax() in (0, 1)


def test_mutual_information_is_non_negative(space):
    p = Posterior(space.matrix)
    assert (mutual_information(p) >= -1e-12).all()


def test_information_gain_matches_actual_entropy_reduction(toy_matrix):
    """Closed-form EIG must equal the belief-weighted entropy drop it predicts."""
    p = Posterior(toy_matrix)
    k = 0
    predicted = mutual_information(p)[k]

    p_yes = float(p.belief @ p.p1[:, k])
    a, b = p.copy(), p.copy()
    a.update(k, YES)
    b.update(k, NO)
    actual = p.entropy - (p_yes * a.entropy + (1 - p_yes) * b.entropy)

    assert predicted == pytest.approx(actual, abs=1e-9)


# --- policies -----------------------------------------------------------------


def test_policies_never_repeat_a_question(space):
    rng = np.random.default_rng(0)
    for policy in (RandomPolicy(), FrequencyPolicy(), EIGPolicy()):
        p = Posterior(space.matrix)
        asked = np.zeros(space.n_features, dtype=bool)
        for _ in range(15):
            k = policy.select(p, asked, rng)
            assert not asked[k], f"{policy.name} repeated feature {k}"
            asked[k] = True
            p.update(k, YES)


def test_eig_beats_frequency_on_average(space):
    """The core empirical claim, as a regression test.

    Small n and a modest margin: this guards against the planner silently
    breaking, it is not the paper's headline experiment.
    """
    def run(policy_factory, n_cases=60, budget=10):
        env = PatientEnvironment(space, seed=3)
        hits = 0
        for case in env.cases(n_cases):
            policy = policy_factory()
            p = Posterior(space.matrix)
            asked = np.zeros(space.n_features, dtype=bool)
            for k, v in case.revealed.items():
                p.update(k, v)
                asked[k] = True
            rng = np.random.default_rng(case.index)
            for _ in range(budget):
                k = policy.select(p, asked, rng)
                asked[k] = True
                ans = env.ask(case, k)
                if ans != UNKNOWN:
                    p.update(k, ans)
            hits += p.argmax == case.condition
        return hits / n_cases

    eig = run(EIGPolicy)
    freq = run(FrequencyPolicy)
    assert eig > freq, f"EIG {eig:.3f} did not beat frequency {freq:.3f}"


# --- stopping rule ------------------------------------------------------------


def test_stopping_rule_diagnoses_when_confident(toy_matrix):
    p = Posterior(toy_matrix)
    p._logp = np.log(np.array([0.97, 0.01, 0.01, 0.01]))
    asked = np.zeros(3, dtype=bool)
    assert StoppingRule(tau_confidence=0.5).decide(p, asked, 1) == "diagnose"


def test_stopping_rule_abstains_when_nothing_left_to_ask(toy_matrix):
    p = Posterior(toy_matrix)
    asked = np.ones(3, dtype=bool)
    assert StoppingRule(tau_confidence=0.99).decide(p, asked, 3) == "abstain"


def test_stopping_rule_asks_when_uncertain_and_gain_available(toy_matrix):
    p = Posterior(toy_matrix)
    asked = np.zeros(3, dtype=bool)
    assert StoppingRule(tau_confidence=0.9).decide(p, asked, 0) == "ask"


# --- environment determinism --------------------------------------------------


def test_cases_are_reproducible(space):
    """The central methodological claim: the environment is not stochastic."""
    a = PatientEnvironment(space, seed=42).make_case(7)
    b = PatientEnvironment(space, seed=42).make_case(7)
    assert a.condition == b.condition
    assert np.array_equal(a.truth, b.truth)
    assert a.revealed == b.revealed


def test_case_independent_of_generation_order(space):
    """Case 5 is the same whether or not cases 0-4 were drawn first."""
    env = PatientEnvironment(space, seed=1)
    direct = env.make_case(5)
    sequential = list(env.cases(6))[5]
    assert direct.condition == sequential.condition
    assert np.array_equal(direct.truth, sequential.truth)


def test_different_seeds_give_different_cases(space):
    a = PatientEnvironment(space, seed=1).make_case(0)
    b = PatientEnvironment(space, seed=2).make_case(0)
    assert not (a.condition == b.condition and np.array_equal(a.truth, b.truth))


def test_answers_are_stable_and_order_independent(space):
    env = PatientEnvironment(space, omission_rate=0.3, seed=9)
    case = env.make_case(2)
    forward = [env.ask(case, k) for k in range(0, 40)]
    backward = [env.ask(case, k) for k in range(39, -1, -1)][::-1]
    assert forward == backward


def test_revealed_features_are_consistent_with_truth(space):
    env = PatientEnvironment(space, seed=5)
    for case in env.cases(20):
        for k, v in case.revealed.items():
            assert env.ask(case, k) == v


def test_omission_produces_unknown(space):
    env = PatientEnvironment(space, omission_rate=1.0, seed=0)
    case = env.make_case(0)
    unasked = [k for k in range(space.n_features) if k not in case.revealed]
    assert all(env.ask(case, k) == UNKNOWN for k in unasked[:20])


def test_forced_agent_commits_at_budget_instead_of_abstaining(toy_matrix):
    """Regression: tau=1.01 must mean 'never stop early', not 'never answer'."""
    p = Posterior(toy_matrix)
    asked = np.zeros(3, dtype=bool)
    forced = StoppingRule(tau_confidence=1.01, max_questions=2, allow_abstain=False)
    assert forced.decide(p, asked, 2) == "diagnose"
    allowed = StoppingRule(tau_confidence=1.01, max_questions=2, allow_abstain=True)
    assert allowed.decide(p, asked, 2) == "abstain"


def test_forced_agent_never_abstains_end_to_end(space):
    from ayur.experiments.agents import build_agents

    env = PatientEnvironment(space, seed=11)
    agents = {a.name: a for a in build_agents(max_questions=5)}
    for name in ("B3-random", "B5-greedy-frequency", "B8-max-eig-no-abstain"):
        for case in env.cases(5):
            traj = agents[name].run(case, env, space.matrix)
            assert traj.decision == "diagnose", f"{name} abstained but must not"
            assert traj.predicted is not None
