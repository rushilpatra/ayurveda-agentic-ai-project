"""Tests that lock in the corrections measured in DATA_AUDIT.md.

If one of these fails, either the raw data changed or a preprocessing bug was
introduced that the audit already ruled out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ayur.data import prep, schema


# --- normalisation ------------------------------------------------------------


def test_none_specific_is_treated_as_null():
    """'none specific' fills 57% of `Ayurvedic Herbs`; it is a placeholder."""
    assert prep.split_cell("none specific") == []
    assert prep.split_cell("Ashwagandha, none specific") == ["ashwagandha"]
    assert prep.split_cell("N/A") == []
    assert prep.split_cell(None) == []
    assert prep.split_cell(float("nan")) == []


def test_split_cell_normalises_and_splits():
    assert prep.split_cell("Sore throat, chest congestion") == [
        "sore throat",
        "chest congestion",
    ]
    assert prep.split_cell("Dry; Cold Air / Dusty Air") == [
        "dry",
        "cold air",
        "dusty air",
    ]


def test_dosha_canonicalisation_is_order_invariant():
    """'Vata, Kapha' and 'Kapha, Vata' are the same dosha set."""
    assert prep.canonical_dosha("Vata, Kapha") == prep.canonical_dosha("Kapha, Vata")
    assert prep.canonical_dosha("Vata, Kapha") == ("kapha", "vata")
    assert prep.canonical_dosha("Pitta") == ("pitta",)


def test_tridosha_expands_to_all_three():
    assert set(prep.canonical_dosha("tridosha")) == set(schema.DOSHAS)


# --- condition space ----------------------------------------------------------


@pytest.fixture(scope="module")
def space():
    return prep.build_condition_space()


def test_row_count_matches_audit(space):
    """446 rows, not the 15,160 claimed in the project brief."""
    assert space.n_conditions == 446


def test_dosha_collapses_to_six_canonical_sets(space):
    assert len(set(space.dosha)) == 6


def test_no_leaky_column_reaches_the_feature_space(space):
    """`Ayurvedic Herbs` etc. would make diagnosis trivial."""
    cols = {f.split("::", 1)[0] for f in space.features}
    for leaky in schema.LEAKY_COLUMNS:
        assert leaky not in cols, f"{leaky} leaked into the observation space"
    assert cols <= set(schema.ASKABLE_COLUMNS)


def test_singleton_features_are_dropped(space):
    """Every retained feature appears in >= keep_min conditions."""
    counts = space.matrix.sum(0)
    assert counts.min() >= space.keep_min


def test_feature_space_is_discriminative_but_not_trivial(space):
    """Guards the regime the viability probe validated."""
    per_condition = space.matrix.sum(1).mean()
    per_feature = space.matrix.sum(0).mean()
    assert 15 < per_condition < 35, per_condition
    assert per_feature > 5, per_feature


def test_condition_ids_are_unique(space):
    """Disease names repeat (Asthma x4); ids must not."""
    assert len(set(space.condition_ids)) == space.n_conditions


def test_matrix_is_binary(space):
    assert set(np.unique(space.matrix)) <= {0, 1}


def test_build_is_deterministic():
    a = prep.build_condition_space()
    b = prep.build_condition_space()
    assert np.array_equal(a.matrix, b.matrix)
    assert a.features == b.features
    assert a.condition_ids == b.condition_ids


def test_keep_min_sweep_changes_feature_count():
    """The threshold is a design decision; the sweep must actually vary."""
    k1 = prep.build_condition_space(keep_min=1).n_features
    k2 = prep.build_condition_space(keep_min=2).n_features
    k3 = prep.build_condition_space(keep_min=3).n_features
    assert k1 > k2 > k3


# --- herbs --------------------------------------------------------------------


def test_herb_lookup_resolves_known_aliases():
    lookup = prep.herb_lookup()
    assert "tulsi" in lookup
    assert "ashwagandha" in lookup
    # Sanskrit synonym resolves to the canonical name
    assert lookup["surasa"] == "Tulsi"


def test_amidha_has_360_herbs():
    assert len(prep.load_herbs()) == 360


def test_herb_pharmacology_vocabularies_are_closed():
    """Rasa/virya/vipaka are small closed sets - this is what makes
    herb verification checkable without an LLM."""
    herbs = prep.load_herbs()
    rasa = {prep.normalise(x) for h in herbs for x in h.get("rasa", [])}
    virya = {prep.normalise(h["virya"]) for h in herbs if h.get("virya")}
    vipaka = {prep.normalise(h["vipaka"]) for h in herbs if h.get("vipaka")}
    assert rasa <= set(schema.RASA), rasa - set(schema.RASA)
    assert virya <= set(schema.VIRYA), virya - set(schema.VIRYA)
    assert vipaka <= set(schema.VIPAKA), vipaka - set(schema.VIPAKA)
