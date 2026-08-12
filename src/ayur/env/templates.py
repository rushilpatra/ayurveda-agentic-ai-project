"""English and Hindi question templates, generated once and cached.

Per the local-execution requirement: the interactive loop must not call an LLM
to verbalise an attribute. Templates are written per column, filled
deterministically, and cached to disk. Rendering a question is a dict lookup.

Because the environment is deterministic and the templates are fixed, an
English run and a Hindi run of the same case index are *exactly paired* - the
same patient, the same hidden attributes, the same question order. That is the
paired bilingual comparison BhashaBench-Ayur cannot support (its two splits
share no items; see DATA_AUDIT.md section 1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ayur.data import prep
from ayur.data.schema import ASKABLE_COLUMNS

CACHE = Path("data/processed/question_templates.json")

#: Per-column question frames. {v} is the attribute value.
TEMPLATES: dict[str, dict[str, str]] = {
    "Symptoms":                 {"en": "Do you have {v}?",
                                 "hi": "क्या आपको {v} की समस्या है?"},
    "Symptom Severity":         {"en": "Would you describe the severity of your symptoms as {v}?",
                                 "hi": "क्या आपके लक्षणों की तीव्रता {v} है?"},
    "Medical History":          {"en": "Have you previously been diagnosed with {v}?",
                                 "hi": "क्या आपको पहले कभी {v} रहा है?"},
    "Current Medications":      {"en": "Are you currently taking {v}?",
                                 "hi": "क्या आप वर्तमान में {v} ले रहे हैं?"},
    "Risk Factors":             {"en": "Have you been exposed to {v}?",
                                 "hi": "क्या आप {v} के संपर्क में रहे हैं?"},
    "Environmental Factors":    {"en": "Are you regularly in an environment with {v}?",
                                 "hi": "क्या आप नियमित रूप से {v} वाले वातावरण में रहते हैं?"},
    "Sleep Patterns":           {"en": "Would you describe your sleep as {v}?",
                                 "hi": "क्या आपकी नींद {v} है?"},
    "Stress Levels":            {"en": "Would you describe your stress level as {v}?",
                                 "hi": "क्या आपका तनाव स्तर {v} है?"},
    "Physical Activity Levels": {"en": "Is your level of physical activity {v}?",
                                 "hi": "क्या आपकी शारीरिक गतिविधि {v} है?"},
    "Family History":           {"en": "Does anyone in your family have {v}?",
                                 "hi": "क्या आपके परिवार में किसी को {v} है?"},
    "Dietary Habits":           {"en": "Would you describe your diet as {v}?",
                                 "hi": "क्या आपका आहार {v} है?"},
    "Allergies (Food/Env)":     {"en": "Are you allergic to {v}?",
                                 "hi": "क्या आपको {v} से एलर्जी है?"},
    "Seasonal Variation":       {"en": "Do your symptoms worsen during {v}?",
                                 "hi": "क्या आपके लक्षण {v} में बढ़ जाते हैं?"},
    "Age Group":                {"en": "Are you in the {v} age group?",
                                 "hi": "क्या आपकी आयु {v} वर्ग में आती है?"},
    "Gender":                   {"en": "Do you identify as {v}?",
                                 "hi": "क्या आप {v} हैं?"},
    "Occupation and Lifestyle": {"en": "Does {v} describe your occupation or lifestyle?",
                                 "hi": "क्या {v} आपके व्यवसाय या जीवनशैली का वर्णन करता है?"},
    "Cultural Preferences":     {"en": "Does {v} apply to your dietary or cultural practice?",
                                 "hi": "क्या {v} आपके आहार या सांस्कृतिक व्यवहार पर लागू होता है?"},
    "Doshas":                   {"en": "Do you show signs of {v} dosha imbalance?",
                                 "hi": "क्या आपमें {v} दोष असंतुलन के लक्षण हैं?"},
    "Constitution/Prakriti":    {"en": "Is your constitution (prakriti) {v}?",
                                 "hi": "क्या आपकी प्रकृति {v} है?"},
}

#: Ayurvedic terms rendered in Devanagari rather than transliterated, so the
#: Hindi questions read as Hindi rather than as romanised Sanskrit.
GLOSSARY_HI: dict[str, str] = {
    "vata": "वात", "pitta": "पित्त", "kapha": "कफ", "tridosha": "त्रिदोष",
    "vata-prakriti": "वात प्रकृति", "pitta-prakriti": "पित्त प्रकृति",
    "kapha-prakriti": "कफ प्रकृति",
    "vata-kapha": "वात-कफ", "vata-pitta": "वात-पित्त", "pitta-kapha": "पित्त-कफ",
    "kapha-pitta": "कफ-पित्त", "pitta-vata": "पित्त-वात", "kapha-vata": "कफ-वात",
    # frequent lifestyle values
    "high stress": "उच्च तनाव", "moderate stress": "मध्यम तनाव",
    "low stress": "निम्न तनाव",
    "irregular sleep": "अनियमित नींद", "disturbed sleep": "बाधित नींद",
    "low": "कम", "moderate": "मध्यम", "high": "अधिक",
    "all ages": "सभी आयु वर्ग", "all genders": "सभी लिंग",
    "male": "पुरुष", "female": "महिला",
    "winter": "सर्दी", "summer": "गर्मी", "monsoon": "मानसून",
    "rainy": "वर्षा ऋतु", "spring": "वसंत", "autumn": "शरद",
}

LANGUAGES = ("en", "hi")


@dataclass
class QuestionBank:
    """feature index -> {'en': str, 'hi': str}."""

    questions: list[dict[str, str]]
    features: list[str]

    def ask(self, feature: int, language: str = "en") -> str:
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
        return self.questions[feature][language]

    def __len__(self) -> int:
        return len(self.questions)


def render(column: str, value: str, language: str) -> str:
    frame = TEMPLATES.get(column)
    if frame is None:
        # Never fabricate a question for a column we have not written a frame
        # for - that would silently produce untranslated or nonsensical text.
        raise KeyError(
            f"no {language} template for column {column!r}; "
            f"add one to TEMPLATES (columns: {sorted(TEMPLATES)})"
        )
    if language == "hi":
        from ayur.env.translate import render_term

        shown = render_term(value)[0]
    else:
        shown = value
    return frame[language].format(v=shown)


def build(features: list[str] | None = None, use_cache: bool = True) -> QuestionBank:
    if features is None:
        features = prep.build_condition_space().features

    if use_cache and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("features") == features:
            return QuestionBank(questions=cached["questions"], features=features)

    questions = []
    for f in features:
        column, value = f.split("::", 1)
        questions.append({lang: render(column, value, lang) for lang in LANGUAGES})

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(
        {"features": features, "questions": questions}, ensure_ascii=False, indent=1))
    return QuestionBank(questions=questions, features=features)


def main() -> int:
    space = prep.build_condition_space()
    bank = build(space.features, use_cache=False)

    missing = [c for c in ASKABLE_COLUMNS if c not in TEMPLATES]
    print("=" * 72)
    print("BILINGUAL QUESTION TEMPLATES")
    print("=" * 72)
    print(f"  features          {len(bank)}")
    print(f"  languages         {', '.join(LANGUAGES)}")
    print(f"  columns covered   {len(TEMPLATES)}/{len(ASKABLE_COLUMNS)}"
          f"{'  MISSING: ' + str(missing) if missing else '  (complete)'}")
    print(f"  glossary terms    {len(GLOSSARY_HI)}")
    print(f"  cache             {CACHE}")
    print("-" * 72)
    for k in (0, 137, 400, 700, 900):
        if k < len(bank):
            print(f"  [{k}] {bank.features[k]}")
            print(f"       en: {bank.ask(k, 'en')}")
            print(f"       hi: {bank.ask(k, 'hi')}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
