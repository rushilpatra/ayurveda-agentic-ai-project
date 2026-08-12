# Value-of-Information Planning over Heterogeneous Knowledge Sources

An uncertainty-aware diagnostic agent that decides **which knowledge source to consult
next**, not merely which question to ask — evaluated on a deterministic bilingual Ayurvedic
environment that runs end-to-end on an Apple-silicon laptop.

Everything here runs locally on CPU + Metal. No CUDA, no cloud APIs, no fine-tuning.
Verified on **Apple M4 / 16 GB unified memory / macOS 15.6.1 / Python 3.11.13**.

---

## What this is

Prior interactive diagnostic systems — [MAI-DxO](https://arxiv.org/abs/2506.22405),
[AgentClinic](https://arxiv.org/abs/2405.07960) — plan *inside* an LLM and are scored by an
LLM patient. That makes them expensive, irreproducible (AgentClinic reports that swapping
the patient model moves measured doctor accuracy 52% → 46%), and uncalibrated.

This project takes a different position on three axes:

| | prior work | here |
|---|---|---|
| planning | LLM chain-of-debate | explicit posterior + closed-form expected information gain |
| environment | LLM patient / gatekeeper | deterministic, bit-reproducible |
| action space | ask patient, order test | ask patient · query KG · verify herb · retrieve text, each with a cost |
| uncertainty | not reported | ECE, Brier, risk–coverage, principled abstention |

The planner makes **zero LLM calls**. A 5,000-case evaluation runs in 54 minutes on a
laptop, which is what makes the full experimental grid feasible at all.

## Headline results

All from real runs; every table in `results/RESULTS.md` states its exact n.

**Single-source planner, n = 5,000 cases, 446 conditions, 20-question budget:**

| agent | accuracy | ECE | AURC |
|---|---|---|---|
| max-EIG | **93.7%** | 0.005 | 0.002 |
| greedy-frequency | 62.6% | 0.009 | 0.130 |
| random | 8.8% | — | — |
| prior-only (no questions) | 3.9% | — | — |

**Heterogeneous action space, n = 200, equal cost budget:**

| agent | well-specified | misspecified |
|---|---|---|
| tool-EIG per cost | **95.5%** | **43.5%** |
| patient-only *(what prior work does)* | 79.5% | 28.5% |
| cheapest-first | 24.0% | 6.5% |

**+16 points over the single-source agent at identical cost**, replicated under
misspecification. `cheapest-first` has the same cheap actions available and collapses —
the gain comes from *weighing* information against cost, not from cheapness.

**Source selection matters — but our criterion does not yet exploit it fully.** Holding
toolset, budget and question choice fixed and varying *only* which source answers each
question (n = 150 per cell, cost ignored):

| selection rule | budget 10 | budget 15 | budget 20 |
|---|---|---|---|
| always-patient *(hand-coded heuristic)* | 58.7% | **83.3%** | **94.0%** |
| **EIG over (source, feature)** | **62.7%** | 80.0% | 87.3% |
| random source | 24.7% | 52.7% | 71.3% |
| cheapest source | 20.7% | 26.7% | 38.0% |
| worst source *(floor)* | 12.0% | 16.7% | 22.7% |

**62.9-point average spread between best and worst policy** — source choice is the
highest-leverage decision measured in this project. The EIG criterion captures most of that
automatically, beating random selection by **16.0–38.0 points** (all p < 1e-4) with no
hand-specified source preference.

> **But it loses to the trivial heuristic, and increasingly so:** +4.0 (ns) → −3.3 (ns) →
> **−6.7 (p = 0.006)** as budget grows. Greedy mutual information conflates *channel
> fidelity* with *feature informativeness*, so it over-selects the 15 low-noise KG
> attributes instead of the most diagnostic ones.
>
> We **pre-registered** a fix (decouple what-to-learn from who-to-ask; see
> `PREREGISTRATION.md`) with an explicit success criterion and a one-run commitment. The
> diagnosis was confirmed — KG over-selection fell 43%→17% and the monotone trend reversed,
> both as predicted — but the **primary prediction failed**: the gap narrowed to 3.3 points
> against a committed threshold of 2.0. Reported as unsuccessful; no further variants
> searched. The remaining gap is plausibly the price of *greedy* one-step lookahead, which
> would require non-myopic planning to test.

> **Two further boundaries.** (1) A matched-observation control finds that *adding* sources
> to an already-rich patient channel buys nothing at equal information (0/5 budgets, mean
> −1.6 pts) — heterogeneity is not free value, but *managing* it is a real problem.
> (2) The equal-cost +16-point result above is economic: cheaper consultations at equal
> accuracy, patient burden 15 → 8.6 questions. A sanity check confirms the mechanism —
> price tools above patient questions and the planner abandons them entirely (tool share
> 0.00, delta exactly 0).

## Quick start

```bash
make setup          # creates .venv, installs pinned deps (never touches an existing env)
make detect         # prints detected hardware and the chosen model tier
make test           # 114 tests
make prep           # builds the 446 x 977 condition-feature matrix
make smoke          # 20 cases, all non-LLM components, ~10 s
make all            # every non-LLM experiment, in dependency order
```

Long runs, safe overnight:

```bash
caffeinate -i bash scripts/overnight.sh
```

Every long job checkpoints every 25 cases, resumes from the last completed case, reports
ETA, and stops gracefully before memory pressure becomes a problem.

## Layout

```
src/ayur/
  env_detect.py          hardware detection -> model tier, thread count
  data/{schema,prep}.py  canonical condition x feature matrix
  env/
    patient.py           deterministic interactive patient environment
    templates.py         English/Hindi question templates
    translate.py         rule-based + curated Hindi terms (NOT LLM - see below)
  planner/
    posterior.py         explicit belief over conditions
    policy.py            EIG, baselines, stopping rules
    actions.py           heterogeneous action space (4 observation channels)
    multitool.py         cost-aware planner + agent loop
    calibration.py       temperature / tau / effective-noise fitting
  kg/
    graph.py             symbolic KG, pluggable edge providers
    nosology.py          91 curated Sanskrit<->English disease correspondences
  tools/
    herb_verify.py       symbolic pharmacology checking, zero LLM
    retrieval.py         real BM25 over 1,889 local passages
  llm/backend.py         frozen MLX model behind a swappable interface
  experiments/           runner, metrics, evaluate, sweep, tools, report
```

## Data

See **`DATA_AUDIT.md`** for the full measured audit. **Four of five datasets acquired**; the fifth is confirmed unobtainable and the system is built not to need it.

| dataset | status |
|---|---|
| AyurGenixAI | ✅ 446 conditions (**not** the 15,160 "records" claimed — that is a cell count) |
| Amidha Herb DB | ✅ 360 herbs (**not** 700 — v2.0 consolidated duplicates) |
| BhashaBench-Ayur | ✅ 14,963 questions (9,348 en + 5,615 hi; splits are **not** parallel) |
| Ayurveda-LLM | ✅ 1,529 passages (⚠️ no license declared) |
| AyurKOSH | ❌ **not obtained** — IEEE DataPort paywall, confirmed unobtainable 2026-08-02 |

## Things that are deliberately *not* claimed

Recorded here because each was measured and each would have been easy to overstate.

1. **Text retrieval is a negative result.** The BM25 index is real, but the corpus is thin:
   precision 0.073 at a 0.025 base rate (2.95× lift) with **6% recall**. It is not a useful
   information channel, and the paper says so.
2. **Raw agreement on sparse matrices is meaningless.** The text-derived matrix scores
   95.65% agreement while an all-zeros matrix scores 97.51%. Reported as precision/recall/lift
   with the majority-class baseline alongside.
3. **Never break ranking ties by source order.** An early KG evaluation reported Hit@10 = 36.1%;
   a stable sort was leaving tied herbs in file order, and gold herbs cluster at median
   position 79/360 vs 180 by chance. The correct figure is **15.1%**.
4. **LLM translation was tried and rejected.** Qwen3-4B produced *"abdominal cramps"* →
   "finger pain" and *"50 80 years"* → "fifty eight hundred years". Hindi terms are now
   rule-based plus a curated glossary; remaining terms code-switch deliberately, and the rate
   (weighted Devanagari coverage 66.8%) is reported rather than hidden.
5. **Absolute accuracies are optimistic.** Cases are generated from a disease reference
   table, not real patients. The misspecified setting is the honest headline; the
   well-specified numbers are an upper bound.
6. **Hindi surface forms have not been reviewed by a native speaker.**

## Status

**Complete and validated** — 114 tests passing, 20 result files, 6 figures, all generated
from real runs on this laptop:

uncertainty planner · heterogeneous action space · calibration and abstention · symbolic KG ·
herb verification · BM25 retrieval · bilingual templates · MLX backend · LLM planner baseline ·
BhashaBench evaluation.

**Remaining work**, in order of value:

1. **Full BhashaBench sweep** — 400 of 14,963 questions run. The harness is resumable and
   domain-stratified; the rest is ~4 hours of local inference (`--all`).
2. **Cost-table sensitivity sweep.** The action costs are a design choice; the paper needs a
   sweep, not a single operating point.
3. **Expert review.** No Ayurvedic practitioner has reviewed the case constructions, and no
   native Hindi speaker has reviewed the question templates. ~50 sampled cases would be the
   highest-value non-compute investment available.
4. ~~AyurKOSH~~ — **confirmed unobtainable.** The two-hop dosha join replaces it; the
   provider stays wired in and inert in case it is ever released openly.
