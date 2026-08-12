"""Tests for the KG, herb verification, bilingual templates and calibration."""
from __future__ import annotations

import re

import numpy as np
import pytest

from ayur.data import prep
from ayur.data.schema import DOSHAS, RASA, VIPAKA, VIRYA
from ayur.env import templates, translate
from ayur.kg.graph import (
    AmidhaProvider,
    AyurGenixProvider,
    AyurKOSHProvider,
    KnowledgeGraph,
    Triple,
    evaluate_two_hop,
)
from ayur.planner.calibration import apply_temperature, fit_tau_for_risk
from ayur.tools.herb_verify import HerbVerifier

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# --- knowledge graph ----------------------------------------------------------


@pytest.fixture(scope="module")
def kg():
    return KnowledgeGraph.build()


def test_graph_builds_from_all_available_sources(kg):
    """Amidha, AyurGenixAI and the nosology bridge. AyurKOSH is unobtainable
    (DATA_AUDIT.md section 4) and contributes nothing."""
    families = {s.split(":", 1)[0] for s in kg.sources}
    assert families == {"amidha", "ayurgenix", "nosology"}
    assert "ayurkosh" not in families
    assert len(kg.triples) > 5000


def test_every_triple_carries_provenance(kg):
    """Evidence grounding is only checkable if every edge knows its source."""
    assert all(t.source for t in kg.triples)


def test_expected_relations_present(kg):
    for rel in ("pacifies", "aggravates", "indicated_for", "involves_dosha",
                "has_rasa", "has_virya", "has_vipaka", "treated_by_gold"):
        assert kg.relations.get(rel, 0) > 0, f"missing relation {rel}"


def test_dosha_objects_are_canonical(kg):
    for rel in ("pacifies", "aggravates", "involves_dosha"):
        objects = {t.object for t in kg.triples if t.relation == rel}
        assert objects <= set(DOSHAS), f"{rel} has non-canonical doshas: {objects - set(DOSHAS)}"


def test_tridosha_expands_rather_than_leaking(kg):
    """'tridosha' must become three edges, not survive as a fourth dosha."""
    assert "tridosha" not in {t.object for t in kg.triples if t.relation == "pacifies"}


def test_ayurkosh_provider_is_wired_but_inert():
    """The paywalled source must not break the build while unavailable."""
    p = AyurKOSHProvider()
    assert p.available is False
    assert list(p.triples()) == []


def test_two_hop_beats_random(kg):
    result = evaluate_two_hop(kg)
    assert result["n_conditions_evaluated"] > 100
    assert result["hit@10"] > result["random_hit@10_baseline"] * 2


def test_recommendations_return_supporting_triples(kg):
    conditions = [t.subject for t in kg.triples if t.relation == "treated_by_gold"]
    recs = kg.recommend_herbs(conditions[0], top_k=5)
    assert recs
    for herb, score, support in recs:
        assert support, f"{herb} recommended with no supporting triples"
        assert all(isinstance(t, Triple) for t in support)


def test_gold_edges_are_not_used_for_scoring(kg):
    """Leakage guard: recommendation must not consult treated_by_gold."""
    conditions = sorted({t.subject for t in kg.triples if t.relation == "treated_by_gold"})
    cond = conditions[0]
    gold = kg.gold_herbs(cond)
    ranked = [h for h, _, _ in kg.recommend_herbs(cond, top_k=360)]
    # If gold were used, every gold herb would occupy the top positions.
    assert not set(ranked[: len(gold)]) == gold or len(gold) > 3


# --- herb verification --------------------------------------------------------


@pytest.fixture(scope="module")
def verifier():
    return HerbVerifier()


def test_vocabularies_are_closed(verifier):
    check = verifier.check_vocabularies()
    assert check["rasa_unexpected"] == []
    assert check["guna_unexpected"] == []
    assert check["virya_unexpected"] == []
    assert check["vipaka_unexpected"] == []
    assert check["counts"] == {"rasa": len(RASA), "guna": 6,
                               "virya": len(VIRYA), "vipaka": len(VIPAKA)}


def test_resolves_spelling_variant(verifier):
    assert verifier.resolve("guggul") == "Guggulu"


def test_resolves_sanskrit_synonym(verifier):
    assert verifier.resolve("surasa") == "Tulsi"


def test_rejects_non_ayurvedic_substances(verifier):
    """A wrong herb is worse than an unresolved one."""
    for junk in ("fish oil", "tea tree oil", "lifestyle changes", "none specific"):
        assert verifier.resolve(junk) is None, junk


def test_ashwagandha_pacifies_vata_kapha_aggravates_pitta(verifier):
    """Classical pharmacology: Ashwagandha is ushna, so it aggravates Pitta."""
    good = verifier.verify("Ashwagandha", {"vata", "kapha"})
    assert good.resolved and good.compatible and good.score > 0

    bad = verifier.verify("Ashwagandha", {"pitta"})
    assert bad.resolved and not bad.compatible and bad.score < 0


def test_verdicts_carry_evidence(verifier):
    v = verifier.verify("Tulsi", {"kapha", "vata"}, indication="kasa")
    assert v.evidence
    assert any("pacifies" in e for e in v.evidence)


def test_unresolved_herb_has_no_verdict(verifier):
    v = verifier.verify("fish oil", {"vata"})
    assert v.resolved is False
    assert v.compatible is None


# --- bilingual templates ------------------------------------------------------


@pytest.fixture(scope="module")
def space():
    return prep.build_condition_space()


def test_every_askable_column_has_both_languages():
    from ayur.data.schema import ASKABLE_COLUMNS

    for col in ASKABLE_COLUMNS:
        assert col in templates.TEMPLATES, f"no template for {col}"
        for lang in templates.LANGUAGES:
            assert templates.TEMPLATES[col][lang].strip()


def test_unknown_column_raises_rather_than_fabricating():
    with pytest.raises(KeyError):
        templates.render("Not A Column", "x", "en")


def test_hindi_frames_are_devanagari():
    for col, frame in templates.TEMPLATES.items():
        assert DEVANAGARI.search(frame["hi"]), f"{col} Hindi frame is not Devanagari"


def test_question_bank_covers_all_features(space):
    bank = templates.build(space.features, use_cache=False)
    assert len(bank) == space.n_features
    for lang in templates.LANGUAGES:
        assert bank.ask(0, lang)


def test_rejects_unsupported_language(space):
    bank = templates.build(space.features, use_cache=False)
    with pytest.raises(ValueError):
        bank.ask(0, "fr")


def test_english_and_hindi_are_paired_per_feature(space):
    """Same feature index -> same question in both languages, so runs pair exactly."""
    bank = templates.build(space.features, use_cache=False)
    assert len(bank.questions) == len(space.features)
    assert bank.ask(5, "en") != bank.ask(5, "hi")


# --- Hindi term rendering -----------------------------------------------------


def test_glossary_keys_are_normalised():
    """Values arrive normalised (hyphens become spaces), so keys must match."""
    for key in translate.GLOSSARY:
        assert key == prep.normalise(key), (
            f"glossary key {key!r} will never match; use {prep.normalise(key)!r}"
        )


def test_age_ranges_render_by_rule():
    assert translate.render_term("20 40 years") == ("20 से 40 वर्ष", "rule")
    assert translate.render_term("60 years")[1] == "rule"


def test_doshas_render_in_devanagari():
    for d in DOSHAS:
        hi, source = translate.render_term(d)
        assert source == "glossary"
        assert DEVANAGARI.search(hi)


def test_unknown_terms_code_switch_rather_than_guess():
    """Deliberate: an unverified translation is worse than an honest anglicism."""
    hi, source = translate.render_term("kidney stones")
    assert source == "code-switched"
    assert hi == "kidney stones"


def test_weighted_devanagari_coverage_is_reported_and_reasonable(space):
    w = space.matrix.sum(0).astype(float)
    cov = np.array(
        [bool(DEVANAGARI.search(translate.render_term(f.split("::", 1)[1])[0]))
         for f in space.features],
        dtype=float,
    )
    weighted = float((cov * w).sum() / w.sum())
    assert weighted > 0.5, f"weighted Devanagari coverage fell to {weighted:.3f}"


# --- calibration --------------------------------------------------------------


def test_temperature_one_is_identity():
    b = np.array([0.7, 0.2, 0.1])
    assert np.allclose(apply_temperature(b, 1.0), b)


def test_higher_temperature_flattens():
    b = np.array([0.9, 0.05, 0.05])
    flat = apply_temperature(b, 3.0)
    assert flat.max() < b.max()
    assert flat.sum() == pytest.approx(1.0)


def test_temperature_preserves_argmax():
    """Temperature scaling is monotonic - it cannot change the prediction."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rng.dirichlet(np.ones(10))
        for t in (0.5, 1.5, 4.0):
            assert apply_temperature(b, t).argmax() == b.argmax()


def test_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        apply_temperature(np.array([0.5, 0.5]), 0.0)


def test_tau_fitting_meets_risk_target():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 500)
    correct = rng.uniform(0, 1, 500) < conf     # well-calibrated by construction
    tau, risk, coverage = fit_tau_for_risk(conf, correct, target_risk=0.2)
    assert risk <= 0.2 + 1e-9
    assert 0.0 < coverage <= 1.0
