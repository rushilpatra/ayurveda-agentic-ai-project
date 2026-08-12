"""Symbolic knowledge graph assembled from pluggable edge providers.

AyurKOSH sits behind an IEEE DataPort paywall (see DATA_AUDIT.md section 4), so
the graph is built from the sources we hold. The provider interface exists so
that AyurKOSH becomes one more provider when it arrives - additional edges and
classical-text provenance, not a rewrite.

Every edge carries its source, which is what makes evidence grounding checkable:
a recommendation can be traced to the exact triples that produced it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from ayur.data import prep
from ayur.data.schema import DOSHAS


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    source: str

    def __str__(self) -> str:
        return f"({self.subject} -{self.relation}-> {self.object})  [{self.source}]"


class EdgeProvider(Protocol):
    name: str

    def triples(self) -> Iterable[Triple]: ...


# --- providers ----------------------------------------------------------------


@dataclass
class AmidhaProvider:
    """herb -> dosha, indication, and pharmacological properties. 360 herbs."""

    name: str = "amidha"
    herbs: list[dict] | None = None

    def triples(self):
        herbs = self.herbs if self.herbs is not None else prep.load_herbs()
        for h in herbs:
            herb = h["name"]
            for d in h.get("pacify") or []:
                d = prep.normalise(d)
                if d == "tridosha":
                    for x in DOSHAS:
                        yield Triple(herb, "pacifies", x, self.name)
                elif d in DOSHAS:
                    yield Triple(herb, "pacifies", d, self.name)
            for d in h.get("aggravate") or []:
                d = prep.normalise(d)
                if d in DOSHAS:
                    yield Triple(herb, "aggravates", d, self.name)
            for ind in h.get("main_indications", []):
                yield Triple(herb, "indicated_for", prep.normalise(ind), self.name)
            for r in h.get("rasa", []):
                yield Triple(herb, "has_rasa", prep.normalise(r), self.name)
            for g in h.get("guna", []):
                yield Triple(herb, "has_guna", prep.normalise(g), self.name)
            if h.get("virya"):
                yield Triple(herb, "has_virya", prep.normalise(h["virya"]), self.name)
            if h.get("vipaka"):
                yield Triple(herb, "has_vipaka", prep.normalise(h["vipaka"]), self.name)


@dataclass
class AyurGenixProvider:
    """condition -> dosha, prakriti, and (sparsely) herb. 446 conditions.

    The condition -> herb edges are held out as gold labels for evaluating
    two-hop inference, so they are tagged with a distinct relation and are not
    used when scoring.
    """

    name: str = "ayurgenix"
    df: object = None

    def triples(self):
        import pandas as pd

        df = self.df if self.df is not None else prep.load_ayurgenix()
        lookup = prep.herb_lookup()
        for _, r in df.iterrows():
            cond = str(r["Disease"])
            for d in prep.canonical_dosha(r["Doshas"]):
                yield Triple(cond, "involves_dosha", d, self.name)
            for p in prep.split_cell(r.get("Constitution/Prakriti")):
                yield Triple(cond, "prakriti", p, self.name)
            for tok in prep.split_cell(r.get("Ayurvedic Herbs")):
                canonical = lookup.get(tok)
                if canonical:
                    yield Triple(cond, "treated_by_gold", canonical, self.name)


@dataclass
class NosologyProvider:
    """condition -> classical Sanskrit indication, via the curated mapping.

    This is what makes the 1,738 Amidha `herb -indicated_for-> <sanskrit>` edges
    reachable from the English condition space. Without it the two vocabularies
    share exactly one string and the herb-indication edges are unusable
    (DATA_AUDIT.md section 6, J1).

    Each edge records the confidence tier of the correspondence it came from, so
    a downstream consumer can require `exact`/`close` only.
    """

    name: str = "nosology"
    min_confidence: str = "approximate"
    df: object = None

    def triples(self):
        from ayur.kg.nosology import CONFIDENCE_LEVELS, MAPPING

        allowed = set(
            CONFIDENCE_LEVELS[: CONFIDENCE_LEVELS.index(self.min_confidence) + 1])

        df = self.df if self.df is not None else prep.load_ayurgenix()
        english_to_sanskrit: dict[str, list[tuple[str, str]]] = {}
        for sanskrit, (targets, confidence, _) in MAPPING.items():
            if confidence not in allowed:
                continue
            for t in targets:
                english_to_sanskrit.setdefault(t, []).append((sanskrit, confidence))

        seen = set()
        for name in df["Disease"].astype(str):
            key = prep.normalise(name)
            for sanskrit, confidence in english_to_sanskrit.get(key, ()):
                edge = (name, sanskrit)
                if edge in seen:
                    continue
                seen.add(edge)
                yield Triple(name, "has_classical_term", sanskrit,
                             f"{self.name}:{confidence}")


@dataclass
class AyurKOSHProvider:
    """Vyadhi -> Lakshana and classical-text triples. Not yet available.

    Drops in unchanged once the IEEE DataPort files are obtained: point `path`
    at the .xlsx/.xml and implement the parse. Until then it contributes
    nothing rather than failing, so the pipeline runs without it.
    """

    name: str = "ayurkosh"
    path: object = None

    @property
    def available(self) -> bool:
        from pathlib import Path

        return self.path is not None and Path(self.path).exists()

    def triples(self):
        if not self.available:
            return iter(())
        raise NotImplementedError(
            "AyurKOSH parser not written - implement once the data is in hand"
        )


DEFAULT_PROVIDERS = [AmidhaProvider(), AyurGenixProvider(), NosologyProvider(),
                     AyurKOSHProvider()]


# --- graph --------------------------------------------------------------------


@dataclass
class KnowledgeGraph:
    triples: list[Triple] = field(default_factory=list)
    _spo: dict = field(default_factory=lambda: defaultdict(list), repr=False)
    _ops: dict = field(default_factory=lambda: defaultdict(list), repr=False)

    @classmethod
    def build(cls, providers: list[EdgeProvider] | None = None) -> "KnowledgeGraph":
        providers = providers if providers is not None else DEFAULT_PROVIDERS
        kg = cls()
        for provider in providers:
            for t in provider.triples():
                kg.add(t)
        return kg

    def add(self, t: Triple) -> None:
        self.triples.append(t)
        self._spo[(t.subject, t.relation)].append(t)
        self._ops[(t.object, t.relation)].append(t)

    # --- queries --------------------------------------------------------------

    def objects(self, subject: str, relation: str) -> list[str]:
        return [t.object for t in self._spo.get((subject, relation), [])]

    def subjects(self, obj: str, relation: str) -> list[str]:
        return [t.subject for t in self._ops.get((obj, relation), [])]

    def edges(self, subject: str, relation: str) -> list[Triple]:
        return list(self._spo.get((subject, relation), []))

    @property
    def relations(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for t in self.triples:
            counts[t.relation] += 1
        return dict(counts)

    @property
    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for t in self.triples:
            counts[t.source] += 1
        return dict(counts)

    def herbs(self) -> list[str]:
        return sorted({t.subject for t in self.triples if t.relation == "pacifies"}
                      | {t.subject for t in self.triples if t.relation == "has_virya"})

    # --- two-hop reasoning ----------------------------------------------------

    def _indication_index(self) -> dict[str, list[str]]:
        """sanskrit indication -> herbs, built once."""
        if not hasattr(self, "_ind_cache"):
            index: dict[str, list[str]] = defaultdict(list)
            for t in self.triples:
                if t.relation == "indicated_for":
                    index[t.object].append(t.subject)
            self._ind_cache = dict(index)
        return self._ind_cache

    def recommend_herbs(
        self, condition: str, top_k: int = 10, indication_weight: float = 2.0
    ) -> list[tuple[str, float, list[Triple]]]:
        """Rank herbs for a condition over two symbolic paths.

            condition -> dosha -> herb            (pharmacological compatibility)
            condition -> sanskrit term -> herb    (classical indication)

        The indication path is weighted higher: "this herb is indicated for
        Jwara" is a far more specific claim than "this herb pacifies Pitta",
        which is true of a third of the pharmacopoeia. The provenance list is
        what evidence grounding is scored against.
        """
        doshas = set(self.objects(condition, "involves_dosha"))
        terms = set(self.objects(condition, "has_classical_term"))
        if not doshas and not terms:
            return []

        index = self._indication_index()
        indicated: dict[str, set[str]] = defaultdict(set)
        for term in terms:
            for herb in index.get(term, ()):
                indicated[herb].add(term)

        scored = []
        for herb in self.herbs():
            pac = set(self.objects(herb, "pacifies")) & doshas
            agg = set(self.objects(herb, "aggravates")) & doshas
            hits = indicated.get(herb, set())
            if not pac and not hits:
                continue

            score = 0.0
            support: list[Triple] = []
            if doshas:
                score += len(pac) / len(doshas) - 0.5 * len(agg) / len(doshas)
                support += [t for t in self.edges(condition, "involves_dosha")
                            if t.object in pac]
                support += [t for t in self.edges(herb, "pacifies") if t.object in pac]
            if hits:
                score += indication_weight * len(hits)
                support += [t for t in self.edges(condition, "has_classical_term")
                            if t.object in hits]
                support += [t for t in self.edges(herb, "indicated_for")
                            if t.object in hits]

            scored.append((herb, score, support))

        # Alphabetical tie-break, never source order (DATA_AUDIT.md section 6).
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def gold_herbs(self, condition: str) -> set[str]:
        """Held-out condition -> herb edges, used only for evaluation."""
        return set(self.objects(condition, "treated_by_gold"))


def evaluate_two_hop(kg: KnowledgeGraph, k_values=(1, 5, 10, 20)) -> dict:
    """Link prediction: does condition -> dosha -> herb recover the gold herbs?

    Ground truth we did not construct, so 'KG reasoning quality' has a real
    definition instead of an LLM judge.
    """
    conditions = sorted({t.subject for t in kg.triples if t.relation == "treated_by_gold"})
    n_herbs = len(kg.herbs())
    hits = {k: 0 for k in k_values}
    ranks: list[int] = []
    evaluated = 0

    for cond in conditions:
        gold = kg.gold_herbs(cond)
        if not gold:
            continue
        ranked = [h for h, _, _ in kg.recommend_herbs(cond, top_k=n_herbs)]
        if not ranked:
            continue
        evaluated += 1
        positions = [ranked.index(g) for g in gold if g in ranked]
        if not positions:
            ranks.append(n_herbs)
            continue
        best = min(positions)
        ranks.append(best + 1)
        for k in k_values:
            hits[k] += best < k

    arr = np.array(ranks) if ranks else np.array([0])
    return {
        "n_conditions_evaluated": evaluated,
        "n_candidate_herbs": n_herbs,
        **{f"hit@{k}": round(hits[k] / evaluated, 4) if evaluated else None for k in k_values},
        "median_rank": float(np.median(arr)),
        "mean_reciprocal_rank": round(float(np.mean(1.0 / arr)), 4),
        "random_hit@10_baseline": round(10 / n_herbs, 4) if n_herbs else None,
    }


def main() -> int:
    import json
    from pathlib import Path

    kg = KnowledgeGraph.build()
    stats = {
        "n_triples": len(kg.triples),
        "relations": kg.relations,
        "sources": kg.sources,
        "n_herbs": len(kg.herbs()),
        "two_hop_evaluation": evaluate_two_hop(kg),
        "ayurkosh_available": AyurKOSHProvider().available,
    }
    out = Path("results/kg_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))

    print("=" * 66)
    print("KNOWLEDGE GRAPH")
    print("=" * 66)
    print(f"  triples   {stats['n_triples']}")
    print(f"  herbs     {stats['n_herbs']}")
    print("  relations:")
    for r, c in sorted(kg.relations.items(), key=lambda x: -x[1]):
        print(f"      {r:<20} {c}")
    print("  sources:")
    for s, c in sorted(kg.sources.items(), key=lambda x: -x[1]):
        print(f"      {s:<20} {c}")
    print("-" * 66)
    print("  two-hop condition -> dosha -> herb (gold = direct condition->herb):")
    for k, v in stats["two_hop_evaluation"].items():
        print(f"      {k:<28} {v}")
    print("-" * 66)
    print(f"  AyurKOSH available: {stats['ayurkosh_available']}"
          "   (provider is wired in; supply the file to activate)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
