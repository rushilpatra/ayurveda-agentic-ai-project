"""Sanskrit (Ayurvedic) → English (biomedical) disease-term mapping.

## Why this file exists

Amidha indexes herbs by classical Sanskrit indication (Jwara, Kasa, Prameha);
AyurGenixAI indexes conditions by English biomedical name (Fever, Cough,
Diabetes). Measured overlap between the two vocabularies: **1 string out of 363**
(`malaria`). That single gap strands **1,738 herb→indication edges**, leaving the
knowledge graph to run on the dosha path alone (see DATA_AUDIT.md §6, J1/J4).

This table bridges them. It covers the 91 indications carrying ≥3 herbs, which
account for 1,598 of the 1,738 edges (92%).

## How to read the confidence levels

Classical Ayurvedic nosology does not partition disease the way biomedical
nosology does, so these are correspondences of varying tightness, not synonyms.
Every entry is labelled:

  `exact`        near one-to-one and uncontroversial (Jwara = fever)
  `close`        standard equivalence in the literature, minor scope difference
                 (Amavata ≈ rheumatoid arthritis)
  `broad`        the Sanskrit term names a *class* wider than any single English
                 disease (Kushta covers most skin disease); mapped to several
  `approximate`  contested or genuinely partial; use with care

**Nothing here is authoritative.** These are drawn from standard correspondences
in Ayurvedic textbooks and secondary literature, compiled without a domain
expert. Entries marked `approximate` in particular need review by an Ayurvedic
practitioner before publication. `expert_reviewed` is False for every entry
until that happens, and the flag is checked by the tests.

## Known scope mismatches worth stating in the paper

  * `prameha` is a class of urinary disorders of which madhumeha (≈ diabetes
    mellitus) is one member. Mapping it to "diabetes" alone is the common
    simplification and it is wrong at the margin.
  * `kushta` covers ~18 classical skin conditions including leprosy; mapping it
    to modern dermatological categories is inherently lossy.
  * `vata vyadhi` is a dosha-defined class spanning neurological and
    musculoskeletal disease with no biomedical counterpart at all.
"""
from __future__ import annotations

from dataclasses import dataclass

#: sanskrit term -> (english targets, confidence, gloss)
#: English targets are matched against normalised AyurGenixAI disease names.
MAPPING: dict[str, tuple[tuple[str, ...], str, str]] = {
    # --- high-frequency, well-established -----------------------------------
    "jwara": (("fever", "dengue fever", "relapsing fever"), "exact", "fever"),
    "kasa": (("cough",), "exact", "cough"),
    "shwasa": (("asthma", "respiratory issues"), "close", "dyspnoea / asthma"),
    "atisara": (("gastroenteritis",), "close", "diarrhoea"),
    "pravahika": (("amebiasis",), "close", "dysentery"),
    "kamala": (("hepatitis", "liver disorders"), "close", "jaundice"),
    "pandu": (("anemia",), "exact", "anaemia / pallor"),
    "arsha": (("hidradenitis suppurativa",), "approximate", "haemorrhoids"),
    "prameha": (("diabetes", "diabetes mellitus"), "broad",
                "urinary disorders; madhumeha ≈ diabetes"),
    "medoroga": (("obesity", "weight gain"), "close", "obesity / lipid disorder"),
    "hridroga": (("heart disease", "coronary artery disease", "angina"),
                 "close", "heart disease"),
    "amavata": (("rheumatoid arthritis",), "close", "rheumatoid arthritis"),
    "vatarakta": (("gout",), "close", "gout"),
    "sandhivata": (("osteoarthritis",), "close", "osteoarthritis"),
    "gridhrasi": (("sciatica",), "exact", "sciatica"),
    "pakshaghata": (("stroke", "bell s palsy"), "close", "hemiplegia / paralysis"),
    "apasmara": (("epilepsy",), "exact", "epilepsy"),
    "unmada": (("bipolar disorder", "mental health disorders"),
               "approximate", "psychosis / insanity"),
    "anidra": (("insomnia",), "exact", "insomnia"),
    "bhrama": (("meniere s disease",), "approximate", "vertigo / dizziness"),
    "shirashoola": (("migraine",), "close", "headache"),
    "ashmari": (("kidney stones", "gallstones"), "close", "calculi"),
    "mutrakrichha": (("urinary tract infection uti", "cystitis"),
                     "close", "dysuria"),
    "mutraghata": (("hydronephrosis",), "approximate", "urinary obstruction"),
    "klaibya": (("erectile dysfunction", "sexual dysfunction"),
                "close", "erectile dysfunction"),
    "pradara": (("menstrual disorders",), "close", "menorrhagia / leucorrhoea"),
    "khalitya": (("hair loss", "alopecia"), "exact", "alopecia"),
    "kandu": (("skin allergy", "atopic dermatitis"), "close", "pruritus"),
    "dadru": (("skin infections",), "close", "ringworm / tinea"),
    "kushta": (("skin diseases", "psoriasis", "leprosy", "eczema atopic dermatitis"),
               "broad", "class of skin diseases incl. leprosy"),
    "visarpa": (("herpes simplex virus hsv",), "approximate", "erysipelas / herpes"),
    "shotha": (("inflammation",), "close", "oedema / swelling"),
    "vrana": (("skin infections", "buruli ulcer"), "close", "wound / ulcer"),
    "bhagna": (("spinal cord injury",), "approximate", "fracture"),
    "kshaya": (("tuberculosis", "weight loss"), "close", "wasting / consumption"),
    "karshya": (("weight loss",), "close", "emaciation"),
    "daurbalya": (("chronic fatigue syndrome",), "close", "debility / weakness"),
    "shrama": (("chronic fatigue syndrome",), "close", "fatigue"),
    "agnimandya": (("indigestion", "digestive disorders"),
                   "close", "low digestive fire"),
    "ajeerna": (("indigestion",), "exact", "indigestion"),
    "aruchi": (("indigestion",), "approximate", "anorexia / distaste for food"),
    "grahani": (("irritable bowel syndrome ibs", "celiac disease"),
                "close", "malabsorption syndrome"),
    "vibandha": (("constipation",), "exact", "constipation"),
    "malabaddhata": (("constipation",), "exact", "constipation"),
    "anaha": (("constipation",), "close", "distension with constipation"),
    "adhmana": (("digestive disorders",), "close", "flatulence / distension"),
    "chhardi": (("gastroenteritis",), "approximate", "vomiting"),
    "chardi": (("gastroenteritis",), "approximate", "vomiting"),
    "hikka": (("gastroesophageal reflux disease gerd",), "approximate", "hiccups"),
    "shoola": (("peptic ulcers",), "approximate", "colic pain"),
    "udara shoola": (("peptic ulcers",), "approximate", "abdominal pain"),
    "udararoga": (("liver cirrhosis", "liver disorders"), "broad", "abdominal diseases"),
    "yakrit roga": (("liver disorders", "hepatitis"), "close", "liver disease"),
    "yakrit vikara": (("liver disorders",), "close", "liver disorder"),
    "pliharoga": (("lymphadenopathy",), "approximate", "splenic disorder"),
    "krimi": (("worm infections", "ascariasis"), "close", "worms / parasites"),
    "krimiroga": (("worm infections",), "close", "parasitic disease"),
    "krimi vyadhi": (("worm infections",), "close", "parasitic disease"),
    "raktapitta": (("hemophilia",), "approximate", "bleeding disorders"),
    "raktadosha": (("skin diseases",), "approximate", "vitiated blood"),
    "trishna": (("diabetes",), "approximate", "excessive thirst"),
    "daha": (("acidity",), "approximate", "burning sensation"),
    "gulma": (("irritable bowel syndrome ibs",), "approximate", "abdominal mass"),
    "granthi": (("brain tumors",), "approximate", "cyst / benign tumour"),
    "vidradhi": (("skin infections",), "approximate", "abscess"),
    "apachi": (("lymphadenopathy",), "close", "cervical lymphadenitis"),
    "gandamala": (("goiter", "lymphadenopathy"), "close", "cervical lymphadenopathy"),
    "netraroga": (("conjunctivitis", "glaucoma"), "broad", "eye diseases"),
    "mukharoga": (("gingivitis",), "broad", "oral diseases"),
    "mukhapaka": (("gingivitis",), "close", "stomatitis / mouth ulcers"),
    "dantaroga": (("gingivitis",), "close", "dental disease"),
    "kantharoga": (("strep throat",), "close", "throat disease"),
    "pratishyaya": (("common cold", "cold"), "exact", "coryza / common cold"),
    "pinasa": (("chronic sinusitis", "sinusitis"), "close", "chronic rhinitis"),
    "shiroroga": (("migraine",), "broad", "diseases of the head"),
    "visha": (("ciguatera poisoning",), "close", "poisoning / toxicity"),
    "mada": (("ciguatera poisoning",), "approximate", "intoxication"),
    "smritidaurbalya": (("dementia", "alzheimer s disease"), "close", "memory loss"),
    "sandhishoola": (("joint pain", "arthritis"), "exact", "joint pain"),
    "urakshata": (("tuberculosis",), "approximate", "chest injury / phthisis"),
    "kshata": (("spinal cord injury",), "approximate", "injury / trauma"),
    "kshata kshaya": (("tuberculosis",), "approximate", "phthisis with wasting"),
    "vranashotha": (("inflammation",), "close", "inflamed wound"),
    "stanya kshaya": (("menopause",), "approximate", "deficient lactation"),
    "yonidosha": (("menstrual disorders",), "broad", "vaginal / gynaecological"),
    "ojo kshaya": (("hiv aids",), "approximate", "loss of ojas / immunodeficiency"),

    # --- dosha-defined classes: no biomedical counterpart -------------------
    "vata vyadhi": ((), "unmappable", "class of vata disorders (neuro/musculoskeletal)"),
    "vatavyadhi": ((), "unmappable", "class of vata disorders"),
    "kapha roga": ((), "unmappable", "class of kapha disorders"),
    "kaphaja roga": ((), "unmappable", "class of kapha disorders"),
    "pitta roga": ((), "unmappable", "class of pitta disorders"),
}

CONFIDENCE_LEVELS = ("exact", "close", "broad", "approximate", "unmappable")


@dataclass(frozen=True)
class Correspondence:
    sanskrit: str
    english: tuple[str, ...]
    confidence: str
    gloss: str
    expert_reviewed: bool = False   # nothing here is reviewed yet


def correspondences() -> list[Correspondence]:
    return [Correspondence(k, v[0], v[1], v[2]) for k, v in sorted(MAPPING.items())]


def english_for(sanskrit: str) -> tuple[str, ...]:
    entry = MAPPING.get(sanskrit)
    return entry[0] if entry else ()


def confidence_of(sanskrit: str) -> str | None:
    entry = MAPPING.get(sanskrit)
    return entry[1] if entry else None


def coverage_report(min_confidence: str = "approximate") -> dict:
    """How many herb->indication edges the table actually unlocks."""
    import collections

    from ayur.data import prep

    allowed = set(CONFIDENCE_LEVELS[: CONFIDENCE_LEVELS.index(min_confidence) + 1])

    herbs = prep.load_herbs()
    counts = collections.Counter()
    for h in herbs:
        for i in h.get("main_indications", []):
            counts[prep.normalise(i)] += 1

    df = prep.load_ayurgenix()
    known = {prep.normalise(d) for d in df.Disease}

    total_edges = sum(counts.values())
    mapped_edges = unmatched_targets = 0
    missing: list[tuple[str, int]] = []
    bad_targets: list[str] = []

    for term, n in counts.items():
        entry = MAPPING.get(term)
        if not entry or entry[1] not in allowed or not entry[0]:
            if n >= 3:
                missing.append((term, n))
            continue
        hits = [t for t in entry[0] if t in known]
        for t in entry[0]:
            if t not in known:
                bad_targets.append(f"{term} -> {t}")
                unmatched_targets += 1
        if hits:
            mapped_edges += n

    by_conf = collections.Counter(v[1] for v in MAPPING.values())
    return {
        "n_mapped_terms": len(MAPPING),
        "by_confidence": dict(by_conf),
        "total_herb_indication_edges": total_edges,
        "edges_unlocked": mapped_edges,
        "edge_coverage": round(mapped_edges / total_edges, 4) if total_edges else 0.0,
        "unmapped_terms_with_3plus_herbs": sorted(missing, key=lambda x: -x[1]),
        "targets_not_in_ayurgenix": sorted(set(bad_targets)),
        "expert_reviewed": False,
    }


def main() -> int:
    import json
    from pathlib import Path

    rep = coverage_report()
    out = Path("results/nosology_mapping.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {**rep, "mapping": {k: {"english": list(v[0]), "confidence": v[1],
                                "gloss": v[2]} for k, v in sorted(MAPPING.items())}},
        indent=2, ensure_ascii=False))

    print("=" * 74)
    print("SANSKRIT -> ENGLISH NOSOLOGY MAPPING")
    print("=" * 74)
    print(f"  terms mapped              {rep['n_mapped_terms']}")
    print(f"  by confidence             {rep['by_confidence']}")
    print(f"  herb->indication edges    {rep['total_herb_indication_edges']}")
    print(f"  edges unlocked            {rep['edges_unlocked']}  "
          f"({100*rep['edge_coverage']:.1f}%)")
    print(f"  targets not found in AyurGenixAI: {len(rep['targets_not_in_ayurgenix'])}")
    for t in rep["targets_not_in_ayurgenix"][:10]:
        print(f"      ! {t}")
    print(f"  unmapped terms with >=3 herbs: "
          f"{len(rep['unmapped_terms_with_3plus_herbs'])}")
    for term, n in rep["unmapped_terms_with_3plus_herbs"][:10]:
        print(f"      - {term} ({n} herbs)")
    print("-" * 74)
    print("  ** expert_reviewed = False for every entry. An Ayurvedic")
    print("     practitioner must review these before publication, especially")
    print("     the 'approximate' tier. **")
    print("=" * 74)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
