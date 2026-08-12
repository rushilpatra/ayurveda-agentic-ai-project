"""Hindi term rendering: rule-based + curated glossary, deliberately NOT LLM.

## Why the LLM approach was abandoned (measured, 2026-07-27)

The obvious design is to translate each unique attribute value once with the
frozen local model and cache it. It was implemented and tested on Qwen3-4B-4bit.
The output was confidently wrong in ways that would have silently corrupted
every Hindi result:

    "abdominal cramps"  -> अंगुली के दर्द       ("finger pain")
    "50 80 years"       -> पचास आठ सौ वर्ष      ("fifty eight hundred years")
    "20 60 years"       -> 20 छह सौ वर्ष        ("20 six hundred years")
    "40 80 years"       -> 40 अस्तित्व 80 वर्ष  ("40 existence 80 years")

A mistranslated symptom is worse than an untranslated one: it is undetectable in
the results table and it would make the bilingual comparison meaningless. A 4B
model is not a medical translator, and nothing in the pipeline could have caught
this automatically.

## What is done instead

1. **Rule-based** rendering for structured values - age ranges, numbers,
   severity and frequency levels. These are regular, so they are handled exactly
   and need no model.
2. **Curated glossary** for the Ayurvedic vocabulary and high-frequency clinical
   terms, written out rather than generated.
3. **Deliberate code-switching** for the remaining specialist terms, which is
   how Hindi-speaking clinicians actually refer to them. The rate is measured
   and reported rather than hidden.

Any bilingual claim must state the code-switching rate and note that the Hindi
surface forms have not been validated by a native speaker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CACHE = Path("data/processed/hi_terms.json")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# --- rule 1: numbers ----------------------------------------------------------

HI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

AGE_RANGE = re.compile(r"^(\d+)\s+(\d+)\s*years?$")
AGE_SINGLE = re.compile(r"^(\d+)\s*years?$")
AGE_PLUS = re.compile(r"^(\d+)\s*\+\s*years?$")


def rule_based(term: str) -> str | None:
    """Exact renderings for structured values. None if no rule applies."""
    t = term.strip().lower()

    m = AGE_RANGE.match(t)
    if m:
        return f"{m.group(1)} से {m.group(2)} वर्ष"          # "20 to 40 years"
    m = AGE_PLUS.match(t)
    if m:
        return f"{m.group(1)} वर्ष से अधिक"                   # "over 40 years"
    m = AGE_SINGLE.match(t)
    if m:
        return f"{m.group(1)} वर्ष"
    if t in {"all ages", "any age"}:
        return "सभी आयु वर्ग"
    return None


# --- rule 2: curated glossary -------------------------------------------------

#: NOTE ON KEYS: values arrive already normalised by `prep.normalise`, which
#: replaces punctuation with spaces. Keys must therefore be written with spaces,
#: never hyphens - "pitta-prakriti" would never match "pitta prakriti".
GLOSSARY: dict[str, str] = {
    # doshas and constitution
    "vata": "वात", "pitta": "पित्त", "kapha": "कफ", "tridosha": "त्रिदोष",
    "vata prakriti": "वात प्रकृति", "pitta prakriti": "पित्त प्रकृति",
    "kapha prakriti": "कफ प्रकृति",
    "vata kapha": "वात-कफ", "vata pitta": "वात-पित्त", "pitta kapha": "पित्त-कफ",
    "kapha pitta": "कफ-पित्त", "pitta vata": "पित्त-वात", "kapha vata": "कफ-वात",
    # highest-impact terms, ranked by condition support
    "both genders": "दोनों लिंग", "adults": "वयस्क", "adult": "वयस्क",
    "children": "बच्चे", "childhood": "बचपन", "elderly": "वृद्ध",
    "genetic": "आनुवंशिक", "genetics": "आनुवंशिकी",
    "genetic mutation": "आनुवंशिक उत्परिवर्तन",
    "genetic mutations": "आनुवंशिक उत्परिवर्तन",
    "stress": "तनाव", "family history": "पारिवारिक इतिहास",
    "no family history": "कोई पारिवारिक इतिहास नहीं",
    "low fat": "कम वसा", "high protein": "उच्च प्रोटीन",
    "high protein diet": "उच्च प्रोटीन आहार", "low sodium": "कम नमक",
    "balanced": "संतुलित", "obesity": "मोटापा", "age": "आयु",
    "poor sanitation": "खराब स्वच्छता", "poor hygiene": "खराब स्वच्छता",
    "rainy season": "वर्षा ऋतु", "antibiotics": "एंटीबायोटिक्स",
    "infections": "संक्रमण", "infection": "संक्रमण",
    "viral infections": "विषाणु संक्रमण", "surgery": "शल्य चिकित्सा",
    # levels and severity
    "low": "कम", "moderate": "मध्यम", "high": "अधिक", "severe": "गंभीर",
    "mild": "हल्का", "mild to moderate": "हल्का से मध्यम",
    "moderate to severe": "मध्यम से गंभीर", "sedentary": "निष्क्रिय",
    "high stress": "उच्च तनाव", "moderate stress": "मध्यम तनाव",
    "low stress": "निम्न तनाव",
    # sleep
    "irregular sleep": "अनियमित नींद", "disturbed sleep": "बाधित नींद",
    "poor sleep": "खराब नींद", "insomnia": "अनिद्रा",
    "normal sleep": "सामान्य नींद", "regular sleep": "नियमित नींद",
    # demographics
    "all genders": "सभी लिंग", "male": "पुरुष", "female": "महिला",
    "both": "दोनों",
    # seasons
    "winter": "सर्दी", "summer": "गर्मी", "monsoon": "मानसून",
    "rainy": "वर्षा ऋतु", "spring": "वसंत", "autumn": "शरद",
    "all seasons": "सभी ऋतुएँ",
    # common symptoms - written out, not generated
    "fever": "बुखार", "cough": "खाँसी", "fatigue": "थकान",
    "headache": "सिरदर्द", "nausea": "जी मिचलाना", "vomiting": "उल्टी",
    "diarrhea": "दस्त", "constipation": "कब्ज", "abdominal pain": "पेट दर्द",
    "abdominal cramps": "पेट में ऐंठन", "chest pain": "सीने में दर्द",
    "joint pain": "जोड़ों का दर्द", "muscle pain": "मांसपेशियों में दर्द",
    "back pain": "पीठ दर्द", "weight loss": "वजन घटना",
    "weight gain": "वजन बढ़ना", "swelling": "सूजन", "rash": "चकत्ते",
    "itching": "खुजली", "dizziness": "चक्कर आना",
    "shortness of breath": "साँस फूलना", "difficulty breathing": "साँस लेने में कठिनाई",
    "sore throat": "गले में खराश", "runny nose": "नाक बहना",
    "sneezing": "छींक आना", "chills": "ठंड लगना", "bloating": "पेट फूलना",
    "indigestion": "अपच", "acidity": "अम्लता", "hair loss": "बालों का झड़ना",
    "blurred vision": "धुंधली दृष्टि", "muscle weakness": "मांसपेशियों की कमजोरी",
    "confusion": "भ्रम", "anxiety": "चिंता", "depression": "अवसाद",
    "insulin resistance": "इंसुलिन प्रतिरोध",
    "high blood pressure": "उच्च रक्तचाप", "jaundice": "पीलिया",
    "seizures": "दौरे", "frequent urination": "बार-बार पेशाब आना",
    # diet and lifestyle
    "balanced diet": "संतुलित आहार", "spicy foods": "मसालेदार भोजन",
    "oily foods": "तैलीय भोजन", "cold foods": "ठंडा भोजन",
    "processed foods": "प्रसंस्कृत भोजन", "vegetarian": "शाकाहारी",
    "non vegetarian": "मांसाहारी", "smoking": "धूम्रपान",
    "alcohol": "शराब", "desk job": "डेस्क जॉब",
    "sedentary lifestyle": "गतिहीन जीवनशैली",
    "physical labor": "शारीरिक श्रम",
    # environment
    "dust": "धूल", "pollen": "पराग", "cold air": "ठंडी हवा",
    "dry": "शुष्क", "humid": "आर्द्र", "pollution": "प्रदूषण",
    "air pollution": "वायु प्रदूषण", "viral infection": "विषाणु संक्रमण",
    # negation
    "none specific": "कोई विशेष नहीं",
}


def render_term(term: str) -> tuple[str, str]:
    """Return (hindi_or_english, source)."""
    t = term.strip().lower()
    if t in GLOSSARY:
        return GLOSSARY[t], "glossary"
    ruled = rule_based(t)
    if ruled is not None:
        return ruled, "rule"
    return term, "code-switched"


# --- cache --------------------------------------------------------------------


def build_cache(features: list[str] | None = None) -> dict:
    from ayur.data import prep

    if features is None:
        features = prep.build_condition_space().features

    terms = sorted({f.split("::", 1)[1] for f in features})
    existing = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    cache: dict[str, dict] = {}
    for term in terms:
        prior = existing.get(term)
        # A hand-corrected entry is authoritative and is never overwritten.
        if prior and prior.get("source") == "human":
            cache[term] = prior
            continue
        hindi, source = render_term(term)
        cache[term] = {
            "hi": hindi,
            "source": source,
            "needs_review": source == "code-switched",
        }

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True))
    return cache


def load_cache() -> dict:
    if not CACHE.exists():
        return build_cache()
    return json.loads(CACHE.read_text())


def stats(cache: dict | None = None) -> dict:
    cache = cache if cache is not None else load_cache()
    by_source: dict[str, int] = {}
    for v in cache.values():
        by_source[v["source"]] = by_source.get(v["source"], 0) + 1
    n = max(len(cache), 1)
    devanagari = sum(1 for v in cache.values() if DEVANAGARI.search(v["hi"]))
    return {
        "n_terms": len(cache),
        "by_source": by_source,
        "devanagari_coverage": round(devanagari / n, 4),
        "code_switch_rate": round(by_source.get("code-switched", 0) / n, 4),
    }


def main() -> int:
    cache = build_cache()
    s = stats(cache)

    print("=" * 72)
    print("HINDI TERM RENDERING  (rule-based + curated; no LLM)")
    print("=" * 72)
    print(f"  unique terms          {s['n_terms']}")
    for src, count in sorted(s["by_source"].items(), key=lambda x: -x[1]):
        print(f"    {src:<18} {count:>5}  ({100*count/s['n_terms']:.1f}%)")
    print(f"  Devanagari coverage   {100*s['devanagari_coverage']:.1f}%")
    print(f"  code-switch rate      {100*s['code_switch_rate']:.1f}%")
    print("-" * 72)
    for t in ["20 40 years", "abdominal cramps", "fever", "vata",
              "high stress", "kidney stones"]:
        hi, src = render_term(t)
        print(f"  {t:<22} -> {hi:<28} [{src}]")
    print("-" * 72)
    print("  Code-switched terms keep their English form on purpose: Hindi-speaking")
    print("  clinicians use English for specialist terminology, and an unverified")
    print("  machine translation would be worse than an honest anglicism.")
    print("  Surface forms still need native-speaker review before publication.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
