"""Herb verification against Ayurvedic pharmacology - fully symbolic, no LLM.

The Amidha vocabularies are small and closed (6 rasa, 2 virya, 3 vipaka, 6 guna,
3 dosha; all verified complete in tests). That is what lets a recommendation be
checked programmatically instead of judged by a model - which is the reason this
domain makes the planning contribution measurable.

Every verdict carries the triples that justify it, so evidence-grounding
precision is computable rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ayur.data import prep
from ayur.data.schema import DOSHAS, GUNA, RASA, VIPAKA, VIRYA

#: Classical dosha-property relationships used for compatibility checking.
#: Ushna (hot) virya aggravates Pitta and pacifies Vata/Kapha; Sheeta the reverse.
VIRYA_EFFECT = {
    "ushna": {"aggravates": {"pitta"}, "pacifies": {"vata", "kapha"}},
    "sheeta": {"aggravates": {"vata", "kapha"}, "pacifies": {"pitta"}},
}

#: Rasa (taste) -> doshas classically aggravated / pacified.
RASA_EFFECT = {
    "madhura": {"pacifies": {"vata", "pitta"}, "aggravates": {"kapha"}},
    "amla":    {"pacifies": {"vata"},          "aggravates": {"pitta", "kapha"}},
    "lavana":  {"pacifies": {"vata"},          "aggravates": {"pitta", "kapha"}},
    "katu":    {"pacifies": {"kapha"},         "aggravates": {"vata", "pitta"}},
    "tikta":   {"pacifies": {"pitta", "kapha"}, "aggravates": {"vata"}},
    "kashaya": {"pacifies": {"pitta", "kapha"}, "aggravates": {"vata"}},
}


@dataclass
class Verdict:
    herb: str
    resolved: bool
    compatible: bool | None
    score: float
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["score"] = round(self.score, 4)
        return d


class HerbVerifier:
    """Resolves herb names and checks dosha compatibility from first principles."""

    def __init__(self, herbs: list[dict] | None = None):
        self.herbs = herbs if herbs is not None else prep.load_herbs()
        self.by_name = {h["name"]: h for h in self.herbs}
        self.lookup = prep.herb_lookup(self.herbs)

    # --- resolution -----------------------------------------------------------

    def resolve(self, name: str) -> str | None:
        """Map a free-form herb string to a canonical Amidha entry.

        Exact alias match first, then a conservative prefix match to catch
        spelling variants such as 'guggul' for 'Guggulu'. Deliberately does not
        do fuzzy matching: a wrong herb is worse than an unresolved one.
        """
        n = prep.normalise(name)
        if not n or prep.is_null_token(n):
            return None
        if n in self.lookup:
            return self.lookup[n]
        for alias, canonical in self.lookup.items():
            if len(n) >= 5 and (alias.startswith(n) or n.startswith(alias)) and \
                    abs(len(alias) - len(n)) <= 2:
                return canonical
        return None

    def properties(self, herb: str) -> dict:
        h = self.by_name.get(herb)
        if not h:
            return {}
        return {
            "rasa": [prep.normalise(x) for x in h.get("rasa", [])],
            "guna": [prep.normalise(x) for x in h.get("guna", [])],
            "virya": prep.normalise(h.get("virya", "")),
            "vipaka": prep.normalise(h.get("vipaka", "")),
            "pacifies": [prep.normalise(x) for x in (h.get("pacify") or [])],
            "aggravates": [prep.normalise(x) for x in (h.get("aggravate") or [])],
            "indications": [prep.normalise(x) for x in h.get("main_indications", [])],
            "botanical_name": h.get("botanical_name"),
        }

    # --- verification ---------------------------------------------------------

    def verify(self, name: str, target_doshas: set[str], indication: str | None = None) -> Verdict:
        """Is this herb appropriate for a condition involving `target_doshas`?"""
        canonical = self.resolve(name)
        if canonical is None:
            return Verdict(
                herb=name,
                resolved=False,
                compatible=None,
                score=0.0,
                reasons=[f"'{name}' does not resolve to any of the {len(self.herbs)} "
                         "herbs in the Amidha database"],
            )

        p = self.properties(canonical)
        target = {d for d in target_doshas if d in DOSHAS}
        reasons, evidence = [], []
        score = 0.0

        pacified = target & set(p["pacifies"])
        aggravated = target & set(p["aggravates"])

        if pacified:
            score += len(pacified) / max(len(target), 1)
            reasons.append(f"pacifies {', '.join(sorted(pacified))}")
            evidence += [f"({canonical} -pacifies-> {d}) [amidha]" for d in sorted(pacified)]
        if aggravated:
            score -= 0.5 * len(aggravated) / max(len(target), 1)
            reasons.append(f"AGGRAVATES {', '.join(sorted(aggravated))}")
            evidence += [f"({canonical} -aggravates-> {d}) [amidha]" for d in sorted(aggravated)]

        # Independent cross-check from virya, which should agree with the
        # explicit pacify/aggravate fields. Disagreement is worth surfacing.
        virya = p["virya"]
        if virya in VIRYA_EFFECT:
            implied_agg = VIRYA_EFFECT[virya]["aggravates"] & target
            if implied_agg and not (implied_agg & set(p["aggravates"])):
                reasons.append(
                    f"virya {virya} classically aggravates "
                    f"{', '.join(sorted(implied_agg))}, which the herb's own "
                    "aggravate field does not list"
                )
            evidence.append(f"({canonical} -has_virya-> {virya}) [amidha]")

        if indication:
            ind = prep.normalise(indication)
            if ind in p["indications"]:
                score += 0.5
                reasons.append(f"explicitly indicated for {ind}")
                evidence.append(f"({canonical} -indicated_for-> {ind}) [amidha]")

        return Verdict(
            herb=canonical,
            resolved=True,
            compatible=bool(pacified and not aggravated),
            score=score,
            reasons=reasons or ["no dosha relationship recorded for the target doshas"],
            evidence=evidence,
            properties=p,
        )

    def check_vocabularies(self) -> dict:
        """Confirm the closed vocabularies really are closed. Used by tests."""
        seen = {"rasa": set(), "guna": set(), "virya": set(), "vipaka": set()}
        for h in self.herbs:
            seen["rasa"] |= {prep.normalise(x) for x in h.get("rasa", [])}
            seen["guna"] |= {prep.normalise(x) for x in h.get("guna", [])}
            if h.get("virya"):
                seen["virya"].add(prep.normalise(h["virya"]))
            if h.get("vipaka"):
                seen["vipaka"].add(prep.normalise(h["vipaka"]))
        return {
            "rasa_unexpected": sorted(seen["rasa"] - set(RASA)),
            "guna_unexpected": sorted(seen["guna"] - set(GUNA)),
            "virya_unexpected": sorted(seen["virya"] - set(VIRYA)),
            "vipaka_unexpected": sorted(seen["vipaka"] - set(VIPAKA)),
            "counts": {k: len(v) for k, v in seen.items()},
        }


def main() -> int:
    import json
    from pathlib import Path

    v = HerbVerifier()
    print("=" * 70)
    print("HERB VERIFICATION  (symbolic, zero LLM calls)")
    print("=" * 70)
    print(f"  herbs {len(v.herbs)}   aliases {len(v.lookup)}")
    print(f"  vocabulary check: {json.dumps(v.check_vocabularies()['counts'])}")
    print("-" * 70)

    demos = [
        ("Ashwagandha", {"vata", "kapha"}, None),
        ("Ashwagandha", {"pitta"}, None),
        ("guggul", {"kapha"}, None),          # spelling variant
        ("Tulsi", {"kapha", "vata"}, "kasa"),
        ("fish oil", {"vata"}, None),          # not an Ayurvedic herb
    ]
    for name, doshas, ind in demos:
        r = v.verify(name, doshas, ind)
        status = ("UNRESOLVED" if not r.resolved
                  else "compatible" if r.compatible else "NOT compatible")
        print(f"  {name:<14} vs {str(sorted(doshas)):<20} -> {status:<15} score={r.score:+.2f}")
        for reason in r.reasons:
            print(f"       - {reason}")
        if r.resolved and r.herb != name:
            print(f"       resolved to: {r.herb}")
    print("=" * 70)

    out = Path("results/herb_verification_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [v.verify(n, d, i).to_json() for n, d, i in demos], indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
