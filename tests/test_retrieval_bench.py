"""Tests for BM25 retrieval and the BhashaBench harness."""
from __future__ import annotations

import numpy as np
import pytest

from ayur.data import prep
from ayur.experiments import bhashabench as bb
from ayur.tools import retrieval as R


# --- retrieval ----------------------------------------------------------------


@pytest.fixture(scope="module")
def space():
    return prep.build_condition_space()


@pytest.fixture(scope="module")
def retriever(space):
    return R.BM25Retriever(R.load_corpus(space.conditions))


def test_tokenizer_drops_stopwords_and_short_tokens():
    toks = R.tokenize("What is the Ayurvedic approach to cough?")
    assert "the" not in toks and "is" not in toks
    assert "ayurvedic" in toks and "cough" in toks


def test_corpus_combines_both_sources(retriever):
    sources = {p.source for p in retriever.passages}
    assert sources == {"ayurveda-llm", "amidha"}
    assert len(retriever.passages) > 1500


def test_backend_is_real_bm25(retriever):
    assert retriever.backend == "rank_bm25"


def test_search_returns_relevant_passages(retriever):
    hits = retriever.search("fever jwara", top_k=5)
    assert hits
    text = " ".join(retriever.passages[i].text.lower() for i, _ in hits)
    assert "jwara" in text or "fever" in text


def test_search_scores_are_descending(retriever):
    hits = retriever.search("cough kapha", top_k=10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("") == []
    assert retriever.search("the is of") == []


def test_ties_are_not_broken_by_corpus_order(retriever):
    """Guards the artefact documented in DATA_AUDIT.md section 6."""
    hits = retriever.search("ayurveda", top_k=50)
    ids = [i for i, _ in hits]
    # A pure corpus-order ranking would be strictly increasing in doc id.
    assert ids != sorted(ids), "ranking looks like corpus order, not relevance"


def test_condition_tagging_prefers_longer_names(space):
    passages = R.load_corpus(space.conditions)
    tagged = [p for p in passages if p.conditions]
    assert len(tagged) > 300


def test_sparse_metrics_reported_not_raw_agreement(space, retriever):
    """The headline must be lift over base rate, never raw agreement."""
    lik = R.build_text_likelihood(space, retriever, top_k=10, max_features=120)
    s = lik.summary()
    assert "lift_over_base_rate" in s
    assert "precision" in s and "recall" in s
    # The trap must stay visible next to its refutation.
    assert "raw_agreement_UNINFORMATIVE" in s
    assert "all_zeros_would_score" in s
    assert s["all_zeros_would_score"] > 0.9


def test_text_signal_beats_chance(space, retriever):
    lik = R.build_text_likelihood(space, retriever, top_k=25, max_features=200)
    assert lik.lift > 1.0, "corpus carries no signal at all"


# --- BhashaBench --------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("A", "A"),
    ("  b ", "B"),
    ("(C)", "C"),
    ("D. Vagbhat", "D"),
    ("The answer is B", "B"),
    ("उत्तर C है", "C"),
    ("Vata and Pitta", None),
    ("", None),
    (None, None),
])
def test_choice_parsing(text, expected):
    assert bb.parse_choice(text) == expected


def test_prompts_are_in_their_stated_language():
    import re

    devanagari = re.compile(r"[ऀ-ॿ]")
    assert not devanagari.search(bb.PROMPT_EN)
    assert devanagari.search(bb.PROMPT_HI)


def test_both_splits_load_with_expected_sizes():
    en, hi = bb.load_split("en"), bb.load_split("hi")
    assert len(en) == 9348
    assert len(hi) == 5615
    assert list(en.columns) == list(hi.columns)


def test_splits_share_no_item_ids():
    """The constraint that forces domain-stratified comparison."""
    en, hi = bb.load_split("en"), bb.load_split("hi")
    assert len(set(en["id"]) & set(hi["id"])) == 0


def test_stratified_gap_can_reverse_the_naive_one():
    """The exact confound the stratified comparison exists to prevent.

    English is better in *both* domains, but the Hindi split is dominated by the
    easy domain, so the naive difference of means says Hindi is better. This is
    Simpson's paradox, and BhashaBench's disjoint splits make it a live risk:
    per-domain size ratios there range from 0.36 to 1.04.
    """
    # Domain A (hard): en 0.50, hi 0.20  -> English +0.30
    # Domain B (easy): en 0.90, hi 0.85  -> English +0.05
    # but English is mostly sampled from A, Hindi mostly from B.
    en_overall = (100 * 0.50 + 10 * 0.90) / 110      # 0.536
    hi_overall = (10 * 0.20 + 100 * 0.85) / 110      # 0.791

    en = {"accuracy": en_overall, "accuracy_parsed_only": en_overall,
          "chance_corrected_accuracy": 0.35,
          "by_domain": {"A": {"n": 100, "accuracy": 0.50},
                        "B": {"n": 10, "accuracy": 0.90}}}
    hi = {"accuracy": hi_overall, "accuracy_parsed_only": hi_overall,
          "chance_corrected_accuracy": 0.20,
          "by_domain": {"A": {"n": 10, "accuracy": 0.20},
                        "B": {"n": 100, "accuracy": 0.85}}}

    gap = bb.stratified_language_gap(en, hi)
    assert gap["n_shared_domains"] == 2

    # Naive comparison wrongly favours Hindi ...
    assert gap["naive_delta_do_not_report"] < 0
    # ... while the stratified comparison correctly favours English.
    assert gap["weighted_delta_en_minus_hi"] > 0
    assert gap["weighted_delta_en_minus_hi"] == pytest.approx(0.175)

    assert "delta_chance_corrected" in gap
    assert "delta_attributable_to_format" in gap
