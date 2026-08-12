"""Build the canonical condition x feature matrix from the raw datasets.

Applies every correction recorded in DATA_AUDIT.md:
  * "none specific" and friends are treated as null, not as a value
  * dosha strings are canonicalised to sorted sets ("Vata, Kapha" == "Kapha, Vata")
  * leaky columns are excluded from the observation space
  * singleton features are dropped (configurable; this is a design decision)

Outputs are deterministic: same inputs -> byte-identical artefacts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

AYURGENIX_CSV = RAW / "ayurgenix" / "AyurGenixAI_Dataset.csv"
AMIDHA_JSON = RAW / "amidha_herbs" / "herb-database-main" / "herb.json"


# --- normalisation ------------------------------------------------------------


def normalise(text: object) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(text).lower().strip())
    return re.sub(r"\s+", " ", s).strip()


#: Placeholders compared in normalised form. `normalise` strips punctuation, so
#: the literal "n/a" becomes "n a" and would never match the raw placeholder set.
_NULL_NORMALISED = {normalise(p) for p in schema.NULL_PLACEHOLDERS} | {""}


def is_null_token(token: str) -> bool:
    """True if an already-normalised token carries no information."""
    return token in _NULL_NORMALISED


def split_cell(value: object) -> list[str]:
    """Split a multi-value cell, dropping placeholder nulls.

    The whole-cell placeholder check happens *before* splitting: "N/A" would
    otherwise split on "/" into ["n", "a"] and survive as two features.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if is_null_token(normalise(value)):
        return []
    parts = re.split(schema.CELL_SPLIT_PATTERN, str(value))
    out = []
    for p in parts:
        n = normalise(p)
        if not is_null_token(n):
            out.append(n)
    return out


def canonical_dosha(value: object) -> tuple[str, ...]:
    """"Vata, Kapha" and "Kapha, Vata" both -> ("kapha", "vata").

    Collapses the raw 9 dosha strings to 6 canonical sets.
    """
    toks = {t for t in split_cell(value) if t in schema.DOSHAS or t == "tridosha"}
    if "tridosha" in toks:
        return schema.DOSHAS
    return tuple(sorted(toks))


# --- artefacts ----------------------------------------------------------------


@dataclass
class ConditionSpace:
    """The condition x feature matrix and everything needed to interpret it."""

    matrix: np.ndarray            # (D, K) binary
    conditions: list[str]         # length D, display names
    condition_ids: list[str]      # length D, stable unique ids
    features: list[str]           # length K, "Column::token"
    dosha: list[tuple[str, ...]]  # length D, canonical dosha set per condition
    dropped_singletons: int
    keep_min: int

    @property
    def n_conditions(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    def feature_column(self, k: int) -> str:
        return self.features[k].split("::", 1)[0]

    def feature_value(self, k: int) -> str:
        return self.features[k].split("::", 1)[1]

    def summary(self) -> dict:
        return {
            "n_conditions": self.n_conditions,
            "n_features": self.n_features,
            "keep_min": self.keep_min,
            "dropped_singleton_features": self.dropped_singletons,
            "mean_features_per_condition": float(self.matrix.sum(1).mean()),
            "mean_conditions_per_feature": float(self.matrix.sum(0).mean()),
            "n_askable_columns": len(schema.ASKABLE_COLUMNS),
        }


def load_ayurgenix(path: Path = AYURGENIX_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place AyurGenixAI_Dataset.csv there "
            "(see DATA_AUDIT.md section 2)."
        )
    return pd.read_csv(path)


def build_condition_space(
    df: pd.DataFrame | None = None,
    keep_min: int = schema.DEFAULT_KEEP_MIN,
) -> ConditionSpace:
    """Construct the binary condition x feature matrix."""
    if df is None:
        df = load_ayurgenix()

    missing = [c for c in schema.ASKABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"AyurGenixAI is missing expected columns: {missing}")

    rows: list[set[str]] = []
    for _, r in df.iterrows():
        feats: set[str] = set()
        for col in schema.ASKABLE_COLUMNS:
            for tok in split_cell(r[col]):
                feats.add(f"{col}::{tok}")
        rows.append(feats)

    counts: dict[str, int] = {}
    for s in rows:
        for f in s:
            counts[f] = counts.get(f, 0) + 1

    features = sorted(f for f, c in counts.items() if c >= keep_min)
    dropped = len(counts) - len(features)
    index = {f: i for i, f in enumerate(features)}

    matrix = np.zeros((len(rows), len(features)), dtype=np.int8)
    for i, s in enumerate(rows):
        for f in s:
            k = index.get(f)
            if k is not None:
                matrix[i, k] = 1

    # Disease names repeat (Asthma x4); make ids unique and stable.
    # Count on the *normalised* slug, not the raw name: distinct raw names can
    # normalise to the same slug ("Addison's Disease" / "Addisons Disease"),
    # which would otherwise produce colliding ids.
    names = df["Disease"].astype(str).tolist()
    seen: dict[str, int] = {}
    ids = []
    for n in names:
        slug = normalise(n).replace(" ", "_")
        seen[slug] = seen.get(slug, 0) + 1
        ids.append(f"{slug}__{seen[slug]}")

    return ConditionSpace(
        matrix=matrix,
        conditions=names,
        condition_ids=ids,
        features=features,
        dosha=[canonical_dosha(v) for v in df["Doshas"]],
        dropped_singletons=dropped,
        keep_min=keep_min,
    )


def load_herbs(path: Path = AMIDHA_JSON) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (see DATA_AUDIT.md section 3).")
    return json.load(open(path))


def herb_lookup(herbs: list[dict] | None = None) -> dict[str, str]:
    """Alias -> canonical herb name, over name + Sanskrit synonyms + English name."""
    if herbs is None:
        herbs = load_herbs()
    lookup: dict[str, str] = {}
    for h in herbs:
        canonical = h["name"]
        keys = [canonical, *h.get("sanskrit_synonyms", [])]
        if h.get("english_name"):
            keys.append(h["english_name"])
        for k in keys:
            n = normalise(k)
            if n:
                lookup.setdefault(n, canonical)
    return lookup


def save(space: ConditionSpace, out: Path = PROCESSED) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "condition_feature_matrix.npy", space.matrix)
    meta = {
        "conditions": space.conditions,
        "condition_ids": space.condition_ids,
        "features": space.features,
        "dosha": ["|".join(d) for d in space.dosha],
        "summary": space.summary(),
    }
    (out / "condition_space.json").write_text(json.dumps(meta, indent=2))
    return meta["summary"]


def main() -> int:
    df = load_ayurgenix()
    space = build_condition_space(df)
    summary = save(space)

    herbs = load_herbs()
    lookup = herb_lookup(herbs)

    print("=" * 64)
    print("DATA PREP")
    print("=" * 64)
    print(f"  raw rows                    {len(df)}")
    print(f"  conditions (D)              {summary['n_conditions']}")
    print(f"  features (K)                {summary['n_features']}")
    print(f"  dropped singleton features  {summary['dropped_singleton_features']}"
          f"  (keep_min={summary['keep_min']})")
    print(f"  mean features / condition   {summary['mean_features_per_condition']:.2f}")
    print(f"  mean conditions / feature   {summary['mean_conditions_per_feature']:.2f}")
    print(f"  canonical dosha sets        {len(set(space.dosha))}")
    print(f"  herbs                       {len(herbs)}")
    print(f"  herb aliases                {len(lookup)}")
    print("=" * 64)
    print(f"written to {PROCESSED}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
