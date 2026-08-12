"""Dataset constants, derived from the measured audit in DATA_AUDIT.md.

Every non-obvious value here has an empirical justification recorded in the audit.
Do not change these without re-running experiments/join_check.py.
"""
from __future__ import annotations

# --- AyurGenixAI columns -----------------------------------------------------

#: Patient-side attributes the agent is allowed to ask about (19 columns).
ASKABLE_COLUMNS = [
    "Symptoms",
    "Symptom Severity",
    "Medical History",
    "Current Medications",
    "Risk Factors",
    "Environmental Factors",
    "Sleep Patterns",
    "Stress Levels",
    "Physical Activity Levels",
    "Family History",
    "Dietary Habits",
    "Allergies (Food/Env)",
    "Seasonal Variation",
    "Age Group",
    "Gender",
    "Occupation and Lifestyle",
    "Cultural Preferences",
    "Doshas",
    "Constitution/Prakriti",
]

#: Columns that reveal or strongly imply the diagnosis. Excluded from the
#: observation space; several are usable as *targets* or as KG edges.
LEAKY_COLUMNS = [
    "Diagnosis & Tests",
    "Ayurvedic Herbs",             # used as KG gold labels, never as an observation
    "Formulation",
    "Medical Intervention",
    "Prognosis",
    "Complications",
    "Prevention",
    "Diet and Lifestyle Recommendations",
    "Yoga & Physical Therapy",
    "Herbal/Alternative Remedies",
    "Patient Recommendations",
    "Duration of Treatment",
    "Hindi Name",
    "Marathi Name",
]

#: Placeholder strings that pandas reports as present but which carry no value.
#: Measured: "none specific" fills 256/446 of `Ayurvedic Herbs` and 276/446 of
#: `Herbal/Alternative Remedies`. Treating it as a value makes it the single most
#: frequent "herb" in the corpus.
NULL_PLACEHOLDERS = {
    "none specific",
    "none",
    "n/a",
    "na",
    "nan",
    "not specified",
    "not applicable",
    "varies",
    "-",
    "",
}

#: Separators used inside multi-value cells.
CELL_SPLIT_PATTERN = r"[,;/]"

# --- Ayurvedic controlled vocabularies (complete, measured from Amidha) -------

DOSHAS = ("vata", "pitta", "kapha")
RASA = ("madhura", "amla", "lavana", "katu", "tikta", "kashaya")
VIRYA = ("ushna", "sheeta")
VIPAKA = ("madhura", "amla", "katu")
GUNA = ("laghu", "guru", "ruksha", "snigdha", "tikshna", "sara")

# --- Feature-space construction ----------------------------------------------

#: Minimum number of conditions a feature must appear in to be retained.
#: A feature present in exactly one condition is a unique identifier, not a
#: clinical finding: retaining them makes 61% of the feature space trivially
#: diagnostic. This is a benchmark design decision - report a sweep over
#: {1, 2, 3} in the paper rather than presenting one value as neutral.
DEFAULT_KEEP_MIN = 2

#: Generative model: P(feature = 1 | condition) when the feature is/isn't listed.
DEFAULT_NOISE = 0.05
