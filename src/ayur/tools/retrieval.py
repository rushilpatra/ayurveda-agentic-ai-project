"""Real lexical retrieval over the local Ayurvedic corpus.

## Why this exists

The first version of the `retrieve_text` action was a *simulated* noisy channel:
it observed the same attribute as `ask_patient` with a hand-set noise of 0.25.
That behaves plausibly, but calling it "text retrieval" in a paper would be an
overclaim - no text was ever read.

This module replaces it with a real BM25 index and, more importantly, changes
what the channel *means*:

  * `ask_patient(k)`  observes attribute k **of this patient** - ground truth
  * `retrieve_text(k)` observes what **the corpus says about attribute k** for
    conditions like this one - a statement about the world, not the individual

The second is genuinely weaker evidence, and now its likelihood is **measured
from text statistics** rather than assumed. The effective noise of the channel
is estimated by comparing text-derived attribute-condition associations against
the curated AyurGenixAI matrix, so the number in the results table is empirical.

Corpus: Ayurveda-LLM (1,529 QA passages, largely Sushruta Samhita exposition)
plus the Amidha herb entries (360). Both are local; no network access.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ayur.data import prep

AYURVEDA_LLM = Path("data/raw/ayurveda_llm/AYURVEDIC_DATASETFULL.json")
CACHE = Path("data/processed/retrieval_index.json")

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was", "were",
    "for", "with", "as", "by", "on", "at", "it", "this", "that", "be", "which",
    "from", "has", "have", "not", "can", "may", "also", "its", "these", "such",
    "what", "how", "does", "do", "you", "your", "their", "there", "when",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", str(text).lower())
            if len(t) > 2 and t not in STOPWORDS]


@dataclass
class Passage:
    doc_id: int
    source: str
    text: str
    conditions: list[str] = field(default_factory=list)


def load_corpus(conditions: list[str]) -> list[Passage]:
    """Assemble passages and tag each with the conditions it mentions."""
    passages: list[Passage] = []

    # Longest-first so "Chronic Kidney Disease" wins over "Kidney Disease".
    names = sorted({c for c in conditions}, key=len, reverse=True)
    lowered = [(n, n.lower()) for n in names]

    def mentions(text: str) -> list[str]:
        t = text.lower()
        return [n for n, nl in lowered if len(nl) > 3 and nl in t]

    if AYURVEDA_LLM.exists():
        for i, rec in enumerate(json.load(open(AYURVEDA_LLM))):
            text = " ".join(str(rec.get(k, "")) for k in
                            ("question", "Context_Cot", "response"))
            passages.append(Passage(len(passages), "ayurveda-llm", text, mentions(text)))

    for h in prep.load_herbs():
        parts = [h.get("name", ""), h.get("botanical_name", ""),
                 h.get("english_name", "") or "",
                 " ".join(h.get("sanskrit_synonyms", [])),
                 " ".join(h.get("main_indications", [])),
                 " ".join(h.get("rasa", [])), h.get("virya", "") or "",
                 h.get("preview", "") or ""]
        text = " ".join(p for p in parts if p)
        passages.append(Passage(len(passages), "amidha", text, mentions(text)))

    return passages


class BM25Retriever:
    """Okapi BM25 over the local corpus."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(p.text) for p in passages]

        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.tokenized, k1=k1, b=b)
            self._backend = "rank_bm25"
        except ImportError:  # pragma: no cover - dependency is pinned
            self._bm25 = None
            self._backend = "unavailable"

    @property
    def backend(self) -> str:
        return self._backend

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Ties broken by document id, never by corpus order preference - see the
        # tie-breaking lesson recorded in DATA_AUDIT.md section 6 (J4).
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


@dataclass
class TextLikelihood:
    """Attribute-condition associations estimated from the corpus.

    ## Do not use raw agreement here

    Both the curated and the text-derived matrix are ~97% zeros. Raw cell
    agreement is therefore dominated by shared zeros: an all-zeros matrix scores
    ~97.6% and looks excellent while containing no information at all. The first
    version of this class reported 95.65% agreement, which was *worse than
    predicting nothing* and would have been a badly misleading number in a paper.

    Precision, recall and lift over the base rate are reported instead. Lift is
    the honest headline: how much more often the corpus asserts a true
    attribute-condition pair than chance would.
    """

    matrix: np.ndarray            # (D, K) in [0, 1], text-derived P(feature | condition)
    coverage: np.ndarray          # (K,) fraction of conditions with any text evidence
    precision: float              # of text-asserted pairs, fraction curated as true
    recall: float                 # of curated true pairs, fraction the text finds
    f1: float
    base_rate: float              # density of the curated matrix
    lift: float                   # precision / base_rate
    trivial_agreement: float      # what an all-zeros matrix would score
    raw_agreement: float          # kept only to show it is uninformative
    n_passages: int
    n_features_with_evidence: int

    def summary(self) -> dict:
        return {
            "n_passages": self.n_passages,
            "n_features_with_evidence": self.n_features_with_evidence,
            "mean_feature_coverage": round(float(self.coverage.mean()), 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "curated_base_rate": round(self.base_rate, 4),
            "lift_over_base_rate": round(self.lift, 2),
            "raw_agreement_UNINFORMATIVE": round(self.raw_agreement, 4),
            "all_zeros_would_score": round(self.trivial_agreement, 4),
        }


def build_text_likelihood(space, retriever: BM25Retriever, top_k: int = 25,
                          max_features: int | None = None) -> TextLikelihood:
    """Estimate P(attribute | condition) from retrieved passages.

    For each attribute, retrieve passages matching its surface text and count
    which conditions those passages mention. The resulting column is a noisy,
    text-derived version of the curated column - and the disagreement between
    the two *is* the channel's noise, measured rather than assumed.
    """
    condition_index: dict[str, list[int]] = {}
    for i, name in enumerate(space.conditions):
        condition_index.setdefault(name, []).append(i)

    D, K = space.matrix.shape
    text_matrix = np.zeros((D, K), dtype=np.float64)
    coverage = np.zeros(K)

    features = range(K if max_features is None else min(K, max_features))
    for k in features:
        value = space.features[k].split("::", 1)[1]
        hits = retriever.search(value, top_k=top_k)
        if not hits:
            continue
        weight = np.zeros(D)
        for doc_id, score in hits:
            for cond in retriever.passages[doc_id].conditions:
                for i in condition_index.get(cond, ()):
                    weight[i] += score
        if weight.sum() <= 0:
            continue
        text_matrix[:, k] = weight / weight.max()
        coverage[k] = float((weight > 0).mean())

    evaluated = [k for k in features if coverage[k] > 0]
    if evaluated:
        curated = space.matrix[:, evaluated] > 0
        derived = text_matrix[:, evaluated] > 0.05
        tp = float((curated & derived).sum())
        fp = float((~curated & derived).sum())
        fn = float((curated & ~derived).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        base_rate = float(curated.mean())
        lift = precision / base_rate if base_rate else float("nan")
        raw_agreement = float((curated == derived).mean())
        trivial = float((~curated).mean())   # score of predicting all zeros
    else:
        precision = recall = f1 = base_rate = lift = raw_agreement = trivial = float("nan")

    return TextLikelihood(
        matrix=text_matrix,
        coverage=coverage,
        precision=precision,
        recall=recall,
        f1=f1,
        base_rate=base_rate,
        lift=lift,
        trivial_agreement=trivial,
        raw_agreement=raw_agreement,
        n_passages=len(retriever.passages),
        n_features_with_evidence=len(evaluated),
    )


def build(space=None, top_k: int = 25, max_features: int | None = None):
    if space is None:
        space = prep.build_condition_space()
    passages = load_corpus(space.conditions)
    retriever = BM25Retriever(passages)
    likelihood = build_text_likelihood(space, retriever, top_k=top_k,
                                       max_features=max_features)
    return retriever, likelihood


def main() -> int:
    space = prep.build_condition_space()
    passages = load_corpus(space.conditions)
    retriever = BM25Retriever(passages)

    tagged = sum(1 for p in passages if p.conditions)
    by_source: dict[str, int] = {}
    for p in passages:
        by_source[p.source] = by_source.get(p.source, 0) + 1

    print("=" * 74)
    print("LEXICAL RETRIEVAL (BM25, local corpus)")
    print("=" * 74)
    print(f"  backend                 {retriever.backend}")
    print(f"  passages                {len(passages)}   {by_source}")
    print(f"  passages naming >=1 condition  {tagged} ({100*tagged/len(passages):.1f}%)")
    print(f"  mean tokens/passage     {np.mean([len(t) for t in retriever.tokenized]):.0f}")
    print("-" * 74)

    for q in ("fever", "abdominal pain", "ashwagandha", "kapha imbalance cough"):
        hits = retriever.search(q, top_k=3)
        print(f"  query {q!r}")
        for doc_id, score in hits:
            p = retriever.passages[doc_id]
            print(f"     [{p.source}] score={score:.2f} conds={p.conditions[:2]} "
                  f"| {p.text[:80].strip()}...")
    print("-" * 74)

    print("  estimating text-derived likelihood (this reads the corpus) ...", flush=True)
    likelihood = build_text_likelihood(space, retriever, top_k=25)
    s = likelihood.summary()
    for k, v in s.items():
        print(f"  {k:<32} {v}")
    print("-" * 74)
    print(f"  Corpus assertions are {s['lift_over_base_rate']}x more likely to be")
    print(f"  true than chance (precision {s['precision']} vs base rate "
          f"{s['curated_base_rate']}),")
    print(f"  but recall is only {s['recall']} - the corpus is thin.")
    print(f"  NOTE: raw agreement reads {s['raw_agreement_UNINFORMATIVE']}, yet an")
    print(f"  all-zeros matrix scores {s['all_zeros_would_score']}. Never report it.")
    print("=" * 74)

    out = Path("results/retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "backend": retriever.backend,
        "n_passages": len(passages),
        "by_source": by_source,
        "passages_with_condition_mention": tagged,
        **s,
    }, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
