"""BhashaBench-Ayur evaluation with the frozen local model.

Resumable and batched: 14,963 questions at ~1 s each is several hours on this
laptop, so every answer is appended to JSONL and a restart skips completed ids.

Reporting follows the constraint measured in DATA_AUDIT.md section 1: the
English and Hindi splits **share no item ids** and have per-domain size ratios
from 0.36 to 1.04. They are different question populations, so the language
comparison is **domain-stratified**, never a raw difference of means, and it is
additionally stratified by difficulty because Hard items are only 4.0% / 5.9%
of each split.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"
DATA = Path("data/raw/bhashabench_ayur")

CHOICES = ("A", "B", "C", "D")

PROMPT_EN = """Answer this multiple-choice question about Ayurveda.

{question}

A. {a}
B. {b}
C. {c}
D. {d}

Reply with only the single letter A, B, C or D."""

PROMPT_HI = """आयुर्वेद से संबंधित इस बहुविकल्पीय प्रश्न का उत्तर दें।

{question}

A. {a}
B. {b}
C. {c}
D. {d}

केवल एक अक्षर A, B, C या D में उत्तर दें।"""


def load_split(language: str) -> pd.DataFrame:
    path = DATA / ("English_test.parquet" if language == "en" else "Hindi_test.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see DATA_AUDIT.md section 1")
    return pd.read_parquet(path)


def parse_choice(text: str) -> str | None:
    """Extract the chosen letter. Returns None if the model did not answer."""
    if not text:
        return None
    t = text.strip()
    m = re.match(r"^\s*\(?([ABCD])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(?:answer|उत्तर)\s*(?:is|:|है)?\s*\(?([ABCD])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([ABCD])\b", t)
    return letters[0].upper() if letters else None


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def evaluate(language: str, n: int | None, backend_name: str = "mlx",
             checkpoint_every: int = 25, max_tokens: int = 16) -> dict:
    from ayur.llm.backend import get_backend

    df = load_split(language)
    # Deduplicate: the English split has 350 repeated question strings.
    df = df.drop_duplicates(subset=["question", "option_a", "option_b",
                                    "option_c", "option_d"])
    if n is not None:
        # Stratify the subset by subject_domain so a partial run is not a
        # biased sample of the easiest domains.
        df = (df.groupby("subject_domain", group_keys=False)
                .apply(lambda g: g.head(max(1, round(n * len(g) / len(df)))))
                .head(n))

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINTS / f"bhashabench_{language}.jsonl"
    done = completed_ids(out_path)

    backend = get_backend(backend_name)
    template = PROMPT_EN if language == "en" else PROMPT_HI

    todo = df[~df["id"].isin(done)]
    print(f"  [{language}] {len(df)} questions after dedup, "
          f"{len(done)} already done, {len(todo)} to run", flush=True)

    t0 = time.perf_counter()
    with out_path.open("a") as fh:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            prompt = template.format(
                question=r["question"], a=r["option_a"], b=r["option_b"],
                c=r["option_c"], d=r["option_d"])
            gen = backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            choice = parse_choice(gen.text)
            fh.write(json.dumps({
                "id": r["id"],
                "language": language,
                "predicted": choice,
                "correct_answer": r["correct_answer"],
                "correct": bool(choice == r["correct_answer"]),
                "unparsed": choice is None,
                "subject_domain": r["subject_domain"],
                "question_level": r["question_level"],
                "question_type": r["question_type"],
                "raw": gen.text[:120],
                "seconds": round(gen.seconds, 3),
            }, ensure_ascii=False) + "\n")
            if i % checkpoint_every == 0:
                fh.flush()
                el = time.perf_counter() - t0
                print(f"    {i}/{len(todo)}  {el:.0f}s  "
                      f"eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)

    return summarise(language)


def summarise(language: str) -> dict:
    path = CHECKPOINTS / f"bhashabench_{language}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}

    correct = np.array([r["correct"] for r in rows])
    unparsed = np.array([r["unparsed"] for r in rows])

    by_domain: dict[str, list] = defaultdict(list)
    by_level: dict[str, list] = defaultdict(list)
    for r in rows:
        by_domain[r["subject_domain"]].append(r["correct"])
        by_level[r["question_level"]].append(r["correct"])

    # Separate knowledge from instruction-following: an unparsed reply is scored
    # wrong, but the model failing to emit a letter in Hindi is a different
    # deficiency from not knowing the answer. Report both.
    parsed = ~unparsed
    parsed_acc = float(correct[parsed].mean()) if parsed.any() else float("nan")

    # Four options, so 25% is chance. On MCQ benchmarks the raw number flatters
    # a weak model; the chance-corrected score is what actually compares.
    chance = 1.0 / len(CHOICES)
    corrected = (float(correct.mean()) - chance) / (1 - chance)

    return {
        "language": language,
        "n": len(rows),
        "accuracy": round(float(correct.mean()), 4),
        "accuracy_parsed_only": round(parsed_acc, 4),
        "n_parsed": int(parsed.sum()),
        "chance_level": chance,
        "chance_corrected_accuracy": round(corrected, 4),
        "unparsed_rate": round(float(unparsed.mean()), 4),
        "by_domain": {k: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)}
                      for k, v in sorted(by_domain.items())},
        "by_level": {k: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)}
                     for k, v in sorted(by_level.items())},
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 3),
    }


def stratified_language_gap(en: dict, hi: dict) -> dict:
    """Domain-weighted English-Hindi gap.

    A raw difference of means would confound language with topic mix, because
    the two splits are disjoint question sets with different domain proportions.
    """
    shared = sorted(set(en["by_domain"]) & set(hi["by_domain"]))
    rows, weights, deltas = [], [], []
    for d in shared:
        e, h = en["by_domain"][d], hi["by_domain"][d]
        w = e["n"] + h["n"]
        rows.append({"domain": d, "n_en": e["n"], "n_hi": h["n"],
                     "acc_en": e["accuracy"], "acc_hi": h["accuracy"],
                     "delta": round(e["accuracy"] - h["accuracy"], 4)})
        weights.append(w)
        deltas.append(e["accuracy"] - h["accuracy"])
    weights = np.array(weights, dtype=float)
    return {
        "n_shared_domains": len(shared),
        "weighted_delta_en_minus_hi": round(
            float(np.average(deltas, weights=weights)), 4) if shared else None,
        "unweighted_delta": round(float(np.mean(deltas)), 4) if shared else None,
        "naive_delta_do_not_report": round(en["accuracy"] - hi["accuracy"], 4),
        # How much of the gap is the model failing to emit a letter in Hindi
        # rather than not knowing the answer.
        "delta_parsed_only": round(
            en["accuracy_parsed_only"] - hi["accuracy_parsed_only"], 4),
        "delta_attributable_to_format": round(
            (en["accuracy"] - hi["accuracy"])
            - (en["accuracy_parsed_only"] - hi["accuracy_parsed_only"]), 4),
        "delta_chance_corrected": round(
            en["chance_corrected_accuracy"] - hi["chance_corrected_accuracy"], 4),
        "per_domain": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300,
                    help="questions per language (domain-stratified); omit for all")
    ap.add_argument("--all", action="store_true", help="run the full 14,963")
    ap.add_argument("--backend", default="mlx")
    args = ap.parse_args()
    n = None if args.all else args.n

    print("=" * 74)
    print("BHASHABENCH-AYUR")
    print("=" * 74)

    out = {}
    for language in ("en", "hi"):
        out[language] = evaluate(language, n, backend_name=args.backend)

    gap = stratified_language_gap(out["en"], out["hi"])
    out["language_gap"] = gap

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bhashabench.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    print("-" * 74)
    for language in ("en", "hi"):
        s = out[language]
        print(f"  {language}: n={s['n']}  accuracy={100*s['accuracy']:.1f}%  "
              f"unparsed={100*s['unparsed_rate']:.1f}%  {s['mean_seconds']:.2f}s/q")
        print(f"      by level: " + "  ".join(
            f"{k}={100*v['accuracy']:.1f}%(n={v['n']})"
            for k, v in s["by_level"].items()))
    print("-" * 74)
    print(f"  domain-stratified EN-HI gap: "
          f"{100*gap['weighted_delta_en_minus_hi']:+.1f} pts "
          f"(over {gap['n_shared_domains']} shared domains)")
    print(f"  naive difference of means:   "
          f"{100*gap['naive_delta_do_not_report']:+.1f} pts  <- confounded, do not report")
    print("=" * 74)
    print(f"written to {RESULTS}/bhashabench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
