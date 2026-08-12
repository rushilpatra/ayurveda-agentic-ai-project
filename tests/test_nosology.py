"""Tests for the Sanskrit->English nosology mapping and its effect on the KG."""
from __future__ import annotations

import pytest

from ayur.data import prep
from ayur.kg import nosology as N
from ayur.kg.graph import KnowledgeGraph, NosologyProvider, evaluate_two_hop


@pytest.fixture(scope="module")
def known_conditions():
    return {prep.normalise(d) for d in prep.load_ayurgenix()["Disease"]}


# --- table integrity ----------------------------------------------------------


def test_every_entry_has_a_valid_confidence_level():
    for term, (targets, confidence, gloss) in N.MAPPING.items():
        assert confidence in N.CONFIDENCE_LEVELS, f"{term}: {confidence}"
        assert gloss.strip(), f"{term} has no gloss"


def test_keys_are_normalised():
    """Keys are matched against prep.normalise output, so they must match it."""
    for term in N.MAPPING:
        assert term == prep.normalise(term), f"{term!r} will never match"


def test_all_targets_exist_in_ayurgenix(known_conditions):
    """A target that names no real condition silently contributes nothing."""
    missing = []
    for term, (targets, _, _) in N.MAPPING.items():
        for t in targets:
            if t not in known_conditions:
                missing.append(f"{term} -> {t}")
    assert not missing, f"targets not present in AyurGenixAI: {missing}"


def test_unmappable_terms_have_no_targets():
    """Dosha-defined classes have no biomedical counterpart; claiming one
    would be worse than admitting the gap."""
    for term, (targets, confidence, _) in N.MAPPING.items():
        if confidence == "unmappable":
            assert targets == (), f"{term} is unmappable but has targets"
        else:
            assert targets, f"{term} is {confidence} but has no targets"


def test_nothing_is_marked_expert_reviewed():
    """Guards against the flag being flipped without an actual review."""
    for c in N.correspondences():
        assert c.expert_reviewed is False
    assert N.coverage_report()["expert_reviewed"] is False


def test_well_established_correspondences():
    assert "fever" in N.english_for("jwara")
    assert "cough" in N.english_for("kasa")
    assert "anemia" in N.english_for("pandu")
    assert "epilepsy" in N.english_for("apasmara")
    assert "sciatica" in N.english_for("gridhrasi")


def test_contested_mappings_are_not_labelled_exact():
    """Prameha is a disease class, not diabetes; kushta spans ~18 conditions."""
    assert N.confidence_of("prameha") in ("broad", "approximate")
    assert N.confidence_of("kushta") in ("broad", "approximate")
    assert N.confidence_of("vata vyadhi") == "unmappable"


def test_coverage_is_substantial():
    rep = N.coverage_report()
    assert rep["edge_coverage"] > 0.8, rep["edge_coverage"]
    assert rep["targets_not_in_ayurgenix"] == []


def test_only_dosha_classes_remain_unmapped():
    """Everything with >=3 herbs should be mapped unless genuinely unmappable."""
    rep = N.coverage_report()
    for term, _ in rep["unmapped_terms_with_3plus_herbs"]:
        assert N.confidence_of(term) == "unmappable", f"{term} is simply missing"


def test_confidence_filter_is_respected():
    strict = N.coverage_report(min_confidence="close")
    loose = N.coverage_report(min_confidence="approximate")
    assert strict["edges_unlocked"] < loose["edges_unlocked"]


# --- integration with the graph -----------------------------------------------


@pytest.fixture(scope="module")
def kg():
    return KnowledgeGraph.build()


def test_provider_emits_classical_term_edges(kg):
    assert kg.relations.get("has_classical_term", 0) > 50


def test_edges_record_their_confidence_tier(kg):
    sources = {t.source for t in kg.triples if t.relation == "has_classical_term"}
    assert sources
    assert all(s.startswith("nosology:") for s in sources)
    tiers = {s.split(":", 1)[1] for s in sources}
    assert tiers <= set(N.CONFIDENCE_LEVELS)


def test_strict_provider_emits_fewer_edges():
    strict = list(NosologyProvider(min_confidence="exact").triples())
    loose = list(NosologyProvider(min_confidence="approximate").triples())
    assert 0 < len(strict) < len(loose)


def test_indication_path_improves_two_hop_recall(kg):
    """The whole point of the mapping: it must beat the dosha-only baseline.

    15.1% Hit@10 was measured before the mapping existed (DATA_AUDIT.md section 6).
    """
    result = evaluate_two_hop(kg)
    assert result["hit@10"] > 0.151
    assert result["hit@5"] > 0.10


def test_recommendations_cite_the_classical_edge(kg):
    """Provenance must include the mapping edge, not just the dosha edge."""
    conditions = [t.subject for t in kg.triples if t.relation == "has_classical_term"]
    found = False
    for cond in conditions[:20]:
        if not kg.gold_herbs(cond):
            continue
        for _, _, support in kg.recommend_herbs(cond, top_k=5):
            if any(t.relation == "has_classical_term" for t in support):
                found = True
                break
        if found:
            break
    assert found, "no recommendation cited a classical-term edge"


def test_conditions_without_a_classical_term_still_get_recommendations(kg):
    """The dosha path must keep working where no mapping exists."""
    cond = next(c for c in {t.subject for t in kg.triples
                            if t.relation == "involves_dosha"}
                if not kg.objects(c, "has_classical_term"))
    assert kg.recommend_herbs(cond, top_k=5)
