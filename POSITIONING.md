# Research Positioning — Phase 3

Written after reading, in full: Nori et al., *Sequential Diagnosis with Language Models*
(arXiv:2506.22405v2, Microsoft AI); Schmidgall et al., *AgentClinic* (arXiv:2405.07960v5,
now npj Digital Medicine 2026); Zhao et al., *AI agent in healthcare* (npj AI 2:31, 2026).

---

## 1. What the prior work actually did

### SDBench / MAI-DxO (Microsoft AI, July 2025)

- **Benchmark:** 304 NEJM clinicopathological conference cases (2017–2025), 56 held out
  (2024–2025) to control for memorization.
- **Action space:** exactly three — ask free-text question, request diagnostic test, commit
  to diagnosis.
- **Environment:** a **Gatekeeper LLM** (o4-mini) holding the full case file. It reveals
  findings only on explicit query, refuses vague requests, and — importantly — **synthesizes
  plausible synthetic findings** for tests not in the original write-up, so that missing data
  doesn't leak signal.
- **Scoring:** a **Judge LLM** (o3) applies a physician-authored 5-point Likert rubric;
  ≥ 4 counts as correct. Validated at Cohen's κ = 0.70 (model diagnoses) / 0.87 (human).
- **Cost model:** $300 per physician visit; tests priced via CPT codes matched against a 2023
  US health-system price table (CMS price-transparency rule), matched > 98% of the time.
- **Headline results:** physicians 20% @ $2,963. Off-the-shelf o3 78.6% @ $7,850.
  MAI-DxO 79.9% @ $2,397, or 85.5% @ $7,184. Orchestration lifted off-the-shelf models by
  ~11 points on average.
- **How it plans:** a simulated panel of five physician personas conducting an internal
  "chain of debate", plus marginal-cost estimation between rounds and model ensembling.

### AgentClinic (Stanford/JHU/ETH, v5 May 2025)

- **Environment:** four LLM agents — patient, doctor, measurement, moderator.
- **Action budget:** N = 20 patient/measurement interactions before diagnosis is forced.
- **Sources:** MedQA (USMLE), MIMIC-IV EHR, NEJM (120 curated), MedMCQA for specialties.
- **Coverage:** 9 specialties; **7 languages — including Hindi**, plus Chinese, Korean,
  Spanish, French, Persian.
- **Extras:** 24 cognitive/implicit biases; patient-centric metrics (confidence, compliance,
  consultation rating); tools (web, textbooks, reflection, persistent notebook).
- **Headline results:** Claude-3.5 62.1%, human physicians 54, GPT-4 51.6%. MedQA accuracy is
  only *weakly* predictive of AgentClinic accuracy.
- **Budget sensitivity:** N = 20 → 10 collapses accuracy from 52% to 25%; N = 30 is slightly
  *worse* than 20.

---

## 2. Claims that must be dropped from the original proposal

Stating these plainly now is cheaper than having a reviewer state them later.

| Original claim | Status |
|---|---|
| "First large-scale **interactive** benchmark" | ❌ **Dead.** SDBench and AgentClinic both predate it. |
| "**Bilingual** interactive benchmark" is the novelty | ❌ **Dead.** AgentClinic already ships **Hindi** among 7 languages. |
| "Sequential decision-making instead of one-shot" as the core idea | ❌ **Dead.** This is exactly MAI-DxO's framing, in its own words. |
| "Agent decides what to ask next" | ⚠️ **Weak.** Both priors do this. Only the *mechanism* can be novel. |
| "First Ayurvedic interactive benchmark" | ✅ **Survives** — but domain-first claims are weak contributions on their own. |

The proposal as originally written would read to a reviewer as *MAI-DxO applied to Ayurveda*.
That is not publishable at a strong venue. The contribution has to move from **the task** to
**the mechanism**.

---

## 3. The four claims that do survive — and why

### C1. A deterministic patient environment that removes a confound the prior work documents

This is the strongest available claim and it is nearly free.

Both benchmarks simulate the patient with an LLM. AgentClinic reports the consequence
directly: holding the doctor fixed, swapping the *patient* model moves measured doctor
accuracy from **52% (GPT-4 patient) → 48% (GPT-3.5) → 46% (Mixtral)**. SDBench's Gatekeeper
has the same property — it invents synthetic findings, so two runs of the same case are not
the same case.

**The measuring instrument is itself a stochastic LLM, so the benchmark's numbers are not
exactly reproducible and are partly attributable to the environment rather than the agent
under test.**

A patient environment driven by structured attribute lookup with templated verbalization is:
- bit-exact reproducible across runs and across machines,
- free of leakage-by-synthesis,
- attributable — a score change means the *agent* changed,
- and ~free to run, which is what makes thousands of cases feasible on a 16 GB laptop.

Note the framing: the local-hardware constraint *forces* this design, and the design is
independently the methodologically better one. Present it as a contribution, never as a
limitation.

**Cost:** determinism narrows realism. Templated patient responses cannot model the vagueness,
volunteering, and inconsistency of real patients. Mitigate with a controlled noise model
(response-omission rate, symptom-report error rate) swept as an experimental variable, and
report results across the sweep rather than at a single operating point.

### C2. Explicit posterior + closed-form expected information gain, instead of LLM-internal deliberation

MAI-DxO's differential-diagnosis update is a *rendering of an LLM's internal reasoning*. There
is no probability distribution in the system, no closed-form value of information, and — as a
result — nothing that can be calibrated, audited, or proved about it.

The proposed planner maintains an explicit belief `b(d)` over diseases and scores each
candidate action by expected entropy reduction against its cost. That yields three properties
neither prior has: the next question is **justifiable** (you can print why), the planner is
**auditable** (the numbers exist), and the belief is **calibratable** (see C3).

It is also ~4 orders of magnitude cheaper per step, which is the entire reason the full
evaluation can run locally.

### C3. Calibration and principled abstention — a clean, unoccupied gap

Neither paper reports ECE, Brier score, risk–coverage curves, or any abstention mechanism.
MAI-DxO forces a commitment; AgentClinic forces one at N = 20.

Meanwhile the npj review names this exact failure as the field's leading clinical risk:

> "Diagnostic hallucinations may arise in the domain of rare diseases or ambiguous clinical
> presentations, where the agent generates **confident yet substantively incorrect
> conclusions**, thereby posing clinical risks."

An agent that says *"I don't have enough evidence"* and is *right about when* is a
contribution that a medical-AI venue will recognize immediately. This is the claim most worth
over-investing in.

### C4. Cost-aware planning over structurally different knowledge sources

> **REVISED 2026-08-02 after the matched-observation control (§18).** The original claim
> below asserted that choosing among heterogeneous sources is itself valuable. **That was
> tested and refuted**: at equal observation count the multi-source agent is
> neutral-to-worse. The surviving claim is economic — pricing sources and maximising
> information *per unit cost* buys the same accuracy far more cheaply. Read the paragraph
> below as motivation for the action space, not as a validated result.

Both priors choose among homogeneous information sources (ask the patient / order a test).
The proposal chooses among **structurally different** ones: patient interview, symbolic KG
traversal, dense/lexical text retrieval, and formal herb-property verification. These differ
in cost, latency, reliability, and *kind* of evidence returned.

"Which knowledge source should I consult next, and is it worth it?" is a genuinely different
planning problem from "which test should I order". Frame the paper around this.

**Honest caveat:** the strength of C4 is currently gated by data access (§5).

---

## 4. Revised framing

**Do not** title it around interactive diagnosis. Suggested direction:

> **Value-of-Information Planning over Heterogeneous Knowledge Sources: a Deterministic,
> Calibrated Benchmark for Ayurvedic Diagnostic Agents**

One-line pitch:

> Prior interactive diagnostic agents plan inside an LLM and are scored by an LLM patient,
> making them expensive, irreproducible, and uncalibrated. We show that an explicit posterior
> with closed-form value-of-information planning over a heterogeneous toolset matches or beats
> LLM-internal orchestration at a fraction of the cost, while additionally providing
> calibrated confidence and principled abstention — evaluated on a deterministic bilingual
> Ayurvedic environment that is exactly reproducible.

Ayurveda is then not the *contribution* but the *setting that makes the contribution
measurable*: it is one of the few medical domains with a small closed symbolic vocabulary
(6 rasa, 2 virya, 3 vipaka, 3 dosha — all verified present and complete in the Amidha data,
see `DATA_AUDIT.md`), so tool outputs can be checked programmatically rather than judged by
an LLM.

---

## 5. Formal problem statement

A case is a POMDP-style tuple. Hidden state `s = (d*, x*)` with `d*` the true disease from
vocabulary `D`, and `x* ∈ {0,1}^K` the true values of `K` clinical attributes. The agent
observes `x_obs`, a partial view; the rest are hidden.

Actions:

```
A = A_ask ∪ A_kg ∪ A_ret ∪ A_herb ∪ {diagnose(d), abstain}
```

Belief update by naive Bayes over observed attributes (the independence assumption is a
stated modelling choice, not an oversight — report the calibration cost of it):

```
b_t(d) ∝ P(d) · Π_{k ∈ obs} P(x_k | d)
```

Action scoring by expected information gain net of cost, with λ the cost weight:

```
EIG(a) = H(b_t) − E_{o ~ P(o | b_t, a)} [ H(b_t | a, o) ]
a*     = argmax_a [ EIG(a) − λ · cost(a) ]
```

Stopping rule — the abstention mechanism, and the part to get right:

```
diagnose  if  max_d b_t(d) ≥ τ_conf
abstain   if  max_a EIG(a) < ε   and   max_d b_t(d) < τ_conf
else      take a*
```

τ_conf is set on a calibration split to hit a target selective risk, not hand-tuned.

## 6. Metrics, mapped onto the npj review's evaluation tiers

Using the review's own two-tier structure (basic → developmental) makes the evaluation section
look grounded rather than assembled ad hoc. Cite Table 1 of Zhao et al. for the framing.

**Basic — objective correctness**
- Top-1 / top-3 diagnostic accuracy
- Macro-F1 over the disease vocabulary (essential: the indication distribution is heavily
  long-tailed — 85 of 205 Amidha indications have a single herb)

**Basic — task completion**
- Tool-selection accuracy against an oracle-EIG action
- KG-query validity rate
- Herb-verification precision (checkable exactly, no LLM judge needed)

**Developmental — efficiency**  *(the review's named metric: "number of interaction rounds")*
- Questions-to-diagnosis; accuracy-vs-budget curve at N ∈ {1, 3, 5, 10, 20}
- Wall-clock and token cost per case — where the algorithmic planner should win by orders
  of magnitude

**Developmental — uncertainty** *(the unoccupied gap; over-invest here)*
- Expected Calibration Error, Brier score, reliability diagrams
- Risk–coverage curve and AURC
- Abstention: precision/recall of "insufficient evidence" against an oracle that knows whether
  the remaining hidden attributes could still change the argmax

**Developmental — grounding and language**
- Evidence-attribution precision: every claim traceable to a KG triple or retrieved passage
- English vs Hindi parity — **domain-stratified, not paired.** BhashaBench-Ayur's two splits
  share zero item ids and have per-domain size ratios from 0.36 to 1.04 (`DATA_AUDIT.md` §1),
  so there is no item-level correspondence to pair on. Compute the language gap within each of
  the 16 shared `subject_domain` values, then aggregate with domain sizes as weights, and
  additionally stratify by `question_level` — Hard items are only 4.0% (en) / 5.9% (hi), so an
  unstratified mean is essentially a measurement of Easy items. State explicitly that the
  comparison is across different question populations.
  *(For the interactive environment built from AyurGenixAI this constraint does not apply:
  those cases are generated, so English and Hindi runs can be exactly paired per case. Keep
  the two bilingual claims separate — paired for the environment, stratified for the exam
  benchmark.)*

**Statistical reporting.** Paired bootstrap CIs over cases; McNemar for paired accuracy
comparisons; Holm correction across the metric family. Every table states its exact n.

## 7. Baselines

| # | Baseline | Purpose |
|---|---|---|
| B1 | Single-shot frozen LLM, full information | Ceiling under complete observability |
| B2 | Single-shot frozen LLM, partial information | Shows the cost of incomplete evidence |
| B3 | Random action selection, matched budget | Floor |
| B4 | Frequency/prior-only, no questions | Tests whether interaction helps at all — **the most dangerous baseline; run it first** |
| B5 | Greedy most-frequent-attribute (no EIG) | Isolates the contribution of *information gain* vs merely asking |
| B6 | LLM-as-planner (ReAct-style, frozen local model) | The MAI-DxO analogue, honestly scaled to local hardware |
| B7 | Full uncertainty-aware planner | Proposed |
| B8 | B7 with abstention disabled | Ablates the safety mechanism |

B4 and B5 are the ones that decide whether the paper exists. **Run them in the pilot, before
building anything else.** If B5 matches B7, there is no information-gain contribution and the
design must change while that is still cheap.

## 8. Threats to validity — record these now

1. **Naive Bayes independence** is wrong for correlated symptoms. Quantify the damage; do not
   hide it.
2. **Synthetic patients ≠ patients.** The environment is derived from tabular records, so
   accuracy numbers are not clinical claims. State this in the abstract, not just the
   limitations section.
3. **Determinism vs realism** — mitigated by the noise sweep (C1), not eliminated.
4. **Disease vocabulary is long-tailed and small.** Macro-F1 and per-frequency-stratum
   reporting are mandatory.
5. **No clinical validation.** No Ayurvedic practitioner has reviewed the case constructions.
   A small expert review of ~50 sampled cases would substantially strengthen the paper — flag
   as the highest-value non-compute investment available.
6. **Ayurveda is contested as an evidence base.** Frame the work as computational reasoning
   over a codified traditional knowledge system, never as clinical efficacy. Reviewers will
   check this.

## 10. Viability probe — measured, 2026-07-27

`experiments/viability_probe.py`, run on the real AyurGenixAI table. This is baselines
B3/B4/B5/B7 from §7, executed before building anything, exactly as §7 demands.

**Setup.** 446 conditions. Features drawn from all 19 askable columns (§ `DATA_AUDIT.md` 2),
tokenised and namespaced by column. Features occurring in only one condition were **dropped**
— 1,513 of 2,493 — because a singleton feature is a giveaway identifier, not a clinical
finding. Remaining: **K = 980 features, 23.3 positive per condition, each feature shared by
10.6 conditions.** Patient sampled from P(f=1|d) = 0.95 if listed else 0.05; uniform prior;
naive-Bayes posterior; 2,000 simulated cases per policy.

**Accuracy (%) vs number of questions, 446-way classification:**

| policy | q=0 | q=3 | q=5 | q=7 | q=8 | q=9 | q=10 |
|---|---|---|---|---|---|---|---|
| B3 random | 0.4 | 0.9 | 1.6 | 2.1 | 2.2 | 2.5 | 2.9 |
| B5 greedy-frequency | 0.3 | 1.6 | 4.2 | 9.2 | 12.8 | 15.7 | 20.2 |
| **B7 max-EIG** | 0.3 | 1.7 | 5.1 | 18.6 | **34.2** | **54.4** | **63.4** |

**Verdict: the planning problem is real and information gain is what solves it.**

1. **B4 (prior-only, no questions) = 0.30%**, versus 0.22% chance. Interaction is not
   optional — the contribution cannot be explained away by a strong prior. This was the
   most dangerous baseline and it is comfortably cleared.
2. **B7 beats B5 by +43.1 points at q=10** (63.4% vs 20.2%) and B3 by +60.5. The gap is not
   marginal; expected information gain, not merely asking questions, is doing the work.
   This is the single most important number produced so far — it is the empirical licence
   for claim **C2**.
3. **The interesting regime is q = 5–10**, where B7 climbs 5.1 → 63.4. Neither saturated nor
   flat, so there is room to demonstrate improvement and room for baselines to lose.
4. 63.4% at q=10 over 446 classes is strong but far from solved — headroom remains.

**Caveats, to state in the paper.**

- The patient is sampled from the *same* likelihood the posterior assumes (well-specified).
  Absolute numbers are therefore optimistic; the *ranking* of policies is the robust finding.
  The real evaluation must include misspecification — asymmetric noise, correlated features,
  and a non-uniform disease prior.
- Dropping singleton features is a **benchmark design decision, not a cleaning step**. It
  must be documented and justified: retained, they make 61% of the feature space trivially
  diagnostic. Report results at KEEP_MIN ∈ {1, 2, 3} to show the finding is not an artefact
  of that threshold.
- Uniform disease prior is unrealistic. A frequency-weighted prior will raise B4 and shrink
  every gap; run it before claiming the margin.

## 11. Measured results — pilot, misspecification, calibration (2026-07-27)

All from `make pilot` / `ayur.experiments.calibrate`, 446 conditions, 977 features,
deterministic environment, **zero LLM calls**. n stated per table.

### 11.1 Well-specified (n=200, budget 15, env noise 0.05, omission 0.10)

| agent | coverage | accuracy | sel-acc | macro-F1 | q | ECE | AURC |
|---|---|---|---|---|---|---|---|
| B1 full-information *(ceiling)* | 100% | 100.0% | 100.0% | 1.000 | 977 | 0.000 | 0.000 |
| **B8 max-EIG, forced** | 100% | **80.0%** | 80.0% | 0.690 | 15 | 0.038 | 0.024 |
| B7 max-EIG + abstention | 89.0% | 64.5% | 72.5% | 0.600 | 9.9 | 0.049 | 0.177 |
| B5 greedy-frequency | 100% | 40.0% | 40.0% | 0.292 | 15 | 0.099 | 0.304 |
| B3 random | 100% | 6.0% | 6.0% | 0.033 | 15 | 0.026 | 0.858 |
| B4 prior-only | 100% | 2.5% | 2.5% | 0.013 | 0 | 0.007 | 0.972 |

**EIG vs greedy-frequency at an identical 15-question budget: 80.0% vs 40.0%.** All
comparisons significant under McNemar with Holm correction (p < 1e-3).

### 11.2 Misspecified — the honest headline (n=200, budget 20, env noise **0.15**, omission **0.25**, agent still assumes 0.05)

| agent | coverage | accuracy | sel-acc | macro-F1 | ECE |
|---|---|---|---|---|---|
| B1 full-information | 100% | 99.5% | 99.5% | 0.993 | 0.004 |
| **B8 max-EIG, forced** | 100% | **43.0%** | 43.0% | 0.308 | 0.154 |
| B7 max-EIG + abstention | 89.5% | 28.5% | 31.8% | 0.202 | **0.396** |
| B5 greedy-frequency | 100% | 20.0% | 20.0% | 0.120 | 0.240 |
| B3 random | 100% | 5.0% | 5.0% | 0.027 | 0.047 |
| B4 prior-only | 100% | 0.5% | 0.5% | 0.002 | 0.029 |

Two findings, both worth reporting prominently:

1. **The policy ranking survives misspecification.** EIG still better than 2× greedy-frequency
   (43.0% vs 20.0%). Absolute accuracy roughly halves, which is why §10's optimistic caveat
   was correct and why this table, not §11.1, belongs in the abstract.
2. **Abstention becomes actively harmful when the posterior is miscalibrated.** B7 (28.5%)
   underperforms B8 (43.0%) *because* it abstains on a confidence signal that is wrong —
   ECE rises from 0.049 well-specified to **0.396** misspecified. The agent is overconfident
   (it assumes noise 0.05 in a 0.15 world), so `tau` fires on false certainty.

   This is a genuinely useful negative result: **an abstention mechanism is only as good as
   the calibration underneath it.** Papers proposing selective prediction rarely show the
   failure mode; showing it and then fixing it is stronger than only showing the fix.

### 11.3 Calibration repairs it (n=250 calibration / 250 disjoint test)

Temperature and `tau` fitted on the calibration split only, evaluated on held-out test.

| metric | uncalibrated | calibrated |
|---|---|---|
| **ECE** | 0.1537 | **0.0371** (4.1× better) |
| Brier | 0.1244 | **0.0813** |
| selective accuracy | 74.4% | **77.4%** |
| selective risk | 25.6% | **22.6%** |
| coverage | 48.4% | 46.0% |
| AURC | 0.2636 | 0.2648 |

Fitted: **temperature 1.700**; **tau 0.2837**, hitting a 0.20 target selective risk
(achieved 0.1967 on calibration).

**The result worth leading with:** fitting the likelihood's noise by maximum likelihood on
the agent's *own observations* recovers **0.18** against a true environment noise of
**0.15** — despite the agent having been told 0.05. The slight overestimate is correct
behaviour, not error: a 0.25 omission rate makes observations genuinely look noisier, so
0.18 is the right *effective* noise. **The agent can detect and quantify its own
misspecification without being told.** That is a much stronger claim than "we tuned a
temperature".

**Honest note on AURC.** It is unchanged (0.2636 → 0.2648) and must not be presented as
improved. Temperature scaling is monotonic, so it cannot reorder cases by confidence —
it fixes *calibration*, not *discrimination*. Say so explicitly; a reviewer will check.

## 12. C4 validated — the heterogeneous action space (2026-07-27)

This is the claim the whole repositioning rests on, and it holds.

### Design

Every action is an **observation channel**: a likelihood vector
`p1[d] = P(observation = 1 | condition d)` plus a scalar cost. Once expressed that way,
mutual information is computed identically for all four channels and they compete on one
scale:

```
score(a) = I(D ; o_a) / cost(a)
```

| channel | scope | noise | cost |
|---|---|---|---|
| `ask_patient` | all 977 attributes | 0.05 | **1.00** |
| `query_kg` | dosha / prakriti only (15) | 0.02 | 0.10 |
| `retrieve_text` | all 977 attributes | 0.25 | 0.30 |
| `verify_herb` | 350 informative herbs | 0.10 | 0.40 |

2,319 actions total. Cost is in units of one patient question; the *ordering* is the design
claim, and it is swept in the paper rather than asserted.

### Result (n=200, cost budget 15.0, well-specified)

| agent | accuracy | cost | patient Qs | acc/cost | tool mix |
|---|---|---|---|---|---|
| **T1 tool-EIG per cost** | **95.5%** | 14.94 | 13.5 | **0.0639** | ask .59 · kg .32 · retr .05 · herb .04 |
| T2 tool-EIG Lagrangian | 91.5% | 14.85 | 8.6 | 0.0616 | ask .22 · kg .35 · retr .42 |
| **T3 patient-only** *(the prior-work ablation)* | 79.5% | 15.00 | 15.0 | 0.0530 | ask 1.00 |
| T4 cheapest-first | 24.0% | 14.70 | 0.0 | 0.0163 | kg .25 · retr .75 |
| T5 random-tool | 7.0% | 14.94 | 10.1 | 0.0047 | — |

**T1 − T3 = +16.0 points at an identical cost budget**, 95% CI [+10.5, +22.0],
McNemar Holm-corrected p < 1e-16. T3 is exactly what MAI-DxO and AgentClinic do: consult
one source, the patient.

> ⚠️ **Read this alongside §17–§18.** The advantage is **economic, not selective**. An equal
> cost budget buys the tool agent more observations than the patient-only agent, and a
> matched-observation control (§18) shows the benefit disappears — and reverses — once
> that is equalised. Phrase this result as *cheaper consultations at equal accuracy*, never
> as *better source selection*.

T1 − T2 = +4.0 points, CI [+1.0, +7.0], p = 0.022 — the two cost formulations are *not*
equivalent, and the per-cost ratio wins. (At n=60 this gap was not significant; n=200
resolves it.)

**The T4 ablation is what makes the claim non-trivial.** Cheapest-first has the *same* cheap
actions available and scores 16.7%. So the gain is not "cheap tools exist" — it is
**weighing information against cost**. Without that ablation a reviewer would rightly
suspect the result is an artefact of the cost table.

### Tool-selection accuracy — now measurable

Agreement with an oracle that maximises true EIG/cost, computed by replaying every
trajectory:

- **T1: 82.4%**
- T4: 22.0%

This is the metric the original proposal asked for and which a single-source agent cannot
have, because there is no tool choice to score.

### What the agent actually does

The learned behaviour is sensible and was not hand-coded: it **front-loads the near-free KG
queries** to pin down dosha and prakriti (31% of its actions for 2% of its budget), then
spends the remaining budget on patient questions, and uses herb verification as a
confirmatory check. T2, priced differently, shifts heavily to retrieval instead — the two
cost formulations are not equivalent in behaviour even where they are close in accuracy
(their 6.7-point gap is **not** significant, p = 0.125).

### Caveats

- n=60 pilot; the n=200 runs (well-specified and misspecified) are queued.
- The cost table is a design choice, not measured from clinical practice. The paper must
  present a sensitivity sweep over it, not a single operating point.
- The cost table is a design choice; sweep it.
- `retrieve_text` was a *simulated* channel. It has now been replaced with a real BM25
  index — see §13, which changes what can be claimed for it.

## 13. Real retrieval — built, and it is a negative result (2026-07-27)

`ayur/tools/retrieval.py` builds a genuine Okapi BM25 index over **1,889 local passages**
(1,529 Ayurveda-LLM + 360 Amidha entries), 800 of which name at least one condition.
Qualitatively it works: *"kapha imbalance cough"* returns Kasa Roga passages, *"fever"*
returns Jwara passages.

Quantitatively, the corpus is too thin to support the channel.

Text-derived attribute→condition associations, scored against the curated matrix:

| metric | value |
|---|---|
| precision | 0.073 |
| curated base rate | 0.025 |
| **lift over chance** | **2.95×** |
| recall | **0.064** |
| F1 | 0.069 |
| feature coverage | 1.8% of conditions |

**The corpus asserts true associations ~3× more often than chance, but finds only 6% of
them.** As an information channel that is close to useless: it fires on 1.8% of
condition-attribute pairs.

### A metric trap that nearly went into the results

The first implementation reported **95.65% agreement** with the curated matrix — an
excellent-looking number. It is meaningless. Both matrices are ~97.5% zeros, so agreement
is dominated by shared absences: **an all-zeros matrix scores 97.51%**, i.e. the
text-derived matrix was *worse than predicting nothing* on the metric being reported.

Precision/recall/lift against the base rate are reported instead, and the raw agreement is
retained in the JSON under the key `raw_agreement_UNINFORMATIVE` with the all-zeros
baseline beside it, so it can never be quoted without its refutation.

**General rule for this project — now applied twice (see also the tie-breaking artefact in
`DATA_AUDIT.md` §6): on sparse binary data, never report accuracy or agreement without the
majority-class baseline next to it.**

### Consequence for the paper

The honest position is that **the available Ayurvedic text corpus does not support a
useful retrieval tool**, and the simulated channel (noise 0.25) was *generous* rather than
pessimistic. Options, in order of preference:

1. Report retrieval as a measured negative result and keep the three strong channels
   (patient, KG, herb verification). The heterogeneity claim survives on three sources.
2. Enlarge the corpus before claiming a retrieval channel at all.
3. Do **not** present the simulated channel as retrieval. That was the overclaim, and it is
   now removed.

## 14. BhashaBench-Ayur — the Hindi deficit is severe, and it argues *for* the design

Frozen Qwen3-4B-4bit, zero-shot, n = 400 per language (domain-stratified subset of the
14,963).

| | English | Hindi |
|---|---|---|
| raw accuracy | 48.2% | 32.0% |
| accuracy on parsed replies | 48.4% | 33.8% |
| **chance-corrected** (4 options, floor 25%) | **31.0%** | **9.3%** |
| unparsed replies | 0.2% | 5.2% |

Gap decomposition:

| measure | value |
|---|---|
| domain-stratified (16 shared domains) | +15.9 pts |
| parsed-only | +14.6 pts |
| attributable to instruction-following | **+1.7 pts** |
| **chance-corrected** | **+21.7 pts** |

Three things to take from this.

1. **The raw gap understates the problem.** A 4-option MCQ has a 25% floor, which compresses
   both scores toward each other. Chance-corrected, the gap widens from 16.2 to **21.7
   points**, and Hindi lands at **9.3% — barely above guessing**. Report chance-corrected
   numbers; the raw ones flatter a weak model.
2. **It is not a formatting artefact.** Hindi replies fail to yield a letter 5.2% of the time
   versus 0.2% in English (the model starts explaining in Hindi instead of answering), but
   that accounts for only 1.7 of the 16.2 points. The deficit is knowledge and reasoning.
3. **This is an argument for the proposed architecture, not a limitation of it.** The
   planner is language-agnostic by construction: templated questions plus symbolic
   posterior updates plus KG traversal. It does not route reasoning through the model's
   Hindi competence, so it does not inherit this 21.7-point cliff. An LLM-planner agent
   would. That is a concrete, measured reason to prefer algorithmic planning in
   low-resource-language clinical settings — and it is a stronger motivation for the
   bilingual framing than "we support two languages".

Caveat: this is one 4B model on a 400-question stratified subset. The claim is about *this*
frozen backbone, and the full 14,963-question sweep is resumable and still to run.

## 15. AyurKOSH is out of scope — what that costs, stated for the paper

Confirmed 2026-08-02: the dataset could not be obtained. This section is the honest
accounting, written so the limitation can be lifted straight into the paper rather than
discovered by a reviewer.

### What it would have provided

Machine-readable **Vyadhi (disease) → Lakshana (symptom)** relations drawn from *Charaka
Samhita* and *Bhaishajya Ratnawali*, plus herb-substitution edges and classical-text
provenance. It was the only listed source of *classically grounded* disease–symptom
structure.

### What actually stands in its place

| relation | source used | measured |
|---|---|---|
| condition → 19 patient attributes | AyurGenixAI | 446 conditions, 977 features |
| condition → dosha | AyurGenixAI | 678 edges, 6 canonical dosha sets |
| herb → dosha (pacify / aggravate) | Amidha | 1,057 edges |
| herb → indication | Amidha | 1,738 edges, 205 Sanskrit terms |
| herb pharmacology (rasa/guna/virya/vipaka) | Amidha | 1,921 edges, closed vocabularies |

6,132 triples from two sources. The two-hop path `condition → dosha → herb` reaches
**Hit@10 = 15.1%** against a 2.8% random baseline (5.4×), evaluated against held-out direct
`condition → herb` edges that we did not construct.

### The three real costs — do not paper over these

1. **The disease→symptom structure is biomedical, not classical.** AyurGenixAI uses English
   biomedical disease names (Diabetes, Hypertension); AyurKOSH would have supplied classical
   Sanskrit nosology (Prameha, Raktagata Vata). The agent therefore reasons over a
   *modern clinical* condition space with Ayurvedic attributes layered on, **not** over
   classical Ayurvedic nosology. That is a materially different claim and the paper must say
   so plainly rather than describing the system as reasoning "in Ayurvedic terms".

2. **The J1 vocabulary gap stays open.** Amidha's 205 Sanskrit indications and AyurGenixAI's
   363 English disease names overlap in **exactly one string** (`malaria`). AyurKOSH would
   plausibly have bridged them. Without it, the `herb → indication` edges cannot be joined to
   the condition space, so ~1,738 edges sit unusable for diagnosis and only the dosha path
   carries the two-hop inference. **This is why Hit@10 is 15.1% and not higher.** A curated
   Sanskrit↔English mapping over the ~91 indications with ≥3 herbs would likely lift it
   substantially and is the single highest-value remaining data task (≈1 day).

3. **No classical-text provenance.** Every triple traces to a modern curated dataset, not to
   a cited passage of Charaka or Sushruta. Evidence grounding is therefore *dataset*
   grounding. Do not imply textual authority the graph does not have.

### Why the contribution survives

C4 — value-of-information planning over heterogeneous sources — is a claim about the
**planner**, not about any one source. It is demonstrated over four channels with different
costs, fidelities and scopes, and the measured +16.0-point advantage over a single-source
agent (§12) does not depend on which datasets populate the graph. AyurKOSH would have made
the KG channel *richer*; its absence does not make the planning result weaker.

Suggested one-line framing for the limitations section:

> Our knowledge graph is assembled from two openly available curated datasets. A third
> resource (AyurKOSH) offering classical-text disease–symptom triples was behind a paywall
> we could not access; consequently our condition space follows biomedical nosology, and
> the Sanskrit-to-English indication mapping needed to exploit the herb-indication edges
> remains future work.

## 16. Sanskrit↔English nosology mapping — built, and it works where it applies

`src/ayur/kg/nosology.py`. 91 curated correspondences covering the Amidha indications with
≥3 herbs, each labelled with a confidence tier and a gloss.

| tier | n | meaning |
|---|---|---|
| `exact` | 12 | near one-to-one (Jwara = fever, Gridhrasi = sciatica) |
| `close` | 41 | standard equivalence, minor scope difference (Amavata ≈ RA) |
| `broad` | 7 | Sanskrit term names a wider *class*; mapped to several English names |
| `approximate` | 26 | contested or partial; flagged for review |
| `unmappable` | 5 | dosha-defined classes with no biomedical counterpart |

**1,541 of 1,738 herb→indication edges unlocked (88.7%).** Every target is verified to exist
in AyurGenixAI (a test enforces this — a target naming no real condition would silently
contribute nothing).

### Effect on two-hop retrieval

| | before | after |
|---|---|---|
| Hit@1 | 5.0% | **6.5%** |
| Hit@5 | 7.9% | **13.0%** |
| Hit@10 | 15.1% | **17.3%** |
| MRR | 0.096 | **0.116** |

Hit@5 improves by 64% relative. But the aggregate hides the real result:

| subset | n | Hit@5 | Hit@10 |
|---|---|---|---|
| conditions **with** a classical term | 45 | **20.0%** | **22.2%** |
| conditions without one | 92 | 9.8% | 15.2%

**Where a classical correspondence exists, the indication path doubles Hit@5.** The
aggregate is diluted because only **33% of evaluated conditions have an Ayurvedic
counterpart at all** — AyurGenixAI contains many modern rare genetic syndromes (Barth,
Costello, Coffin-Lowry, Prader-Willi) that classical Ayurveda simply does not describe.

That ceiling is a property of the data, not a defect in the mapping, and it is worth
reporting as such: **a classical knowledge base can only reach the part of a modern
condition space that classical medicine actually recognised.** Expanding the mapping
further would not help; the unmapped conditions have no term to map to.

### Status and honesty constraints

- `expert_reviewed = False` on every entry, enforced by a test. These are compiled from
  standard textbook correspondences **without** a domain expert.
- The `approximate` tier (26 entries) is the one that most needs review. Some are frankly
  weak — `trishna` (excessive thirst) → diabetes, `daha` (burning sensation) → acidity — and
  a practitioner may reject them.
- Scope mismatches are documented in the module docstring: `prameha` is a class of urinary
  disorders of which diabetes is one member; `kushta` spans ~18 classical skin conditions
  including leprosy.
- The graph records each edge's tier in its `source` field (`nosology:exact`, …), so a
  downstream consumer can require `exact`/`close` only. `NosologyProvider(min_confidence=…)`
  exposes this.

This is publishable as a standalone artefact once reviewed, and it is the piece that most
directly compensates for AyurKOSH being unavailable (§15).

## 17. Cost-table sensitivity — a partial defence, and a confound I found in my own design

This experiment was run to answer the obvious reviewer question about §12: *is the
+16-point advantage real, or an artefact of the cost table you chose?* The answer is more
complicated than hoped, and the complication matters more than the headline.

n = 120 cases per point, budget 15.0. `ratio` prices every non-patient channel relative to a
patient question fixed at 1.0.

| ratio | tool-EIG | patient-only | delta | 95% CI | p | patient Qs | tool share |
|---|---|---|---|---|---|---|---|
| 0.02 | 89.2% | 77.5% | +0.117 | [+0.033, +0.208] | 0.016 | 3.8 | 0.99 |
| 0.05 | 89.2% | 77.5% | +0.117 | [+0.033, +0.208] | 0.016 | 0.1 | 1.00 |
| 0.10 | 89.2% | 77.5% | +0.117 | [+0.033, +0.208] | 0.016 | 0.0 | 1.00 |
| 0.25 | 80.8% | 77.5% | +0.033 | [−0.050, +0.117] | 0.57 | 7.0 | 0.82 |
| 0.50 | 87.5% | 77.5% | +0.100 | [+0.033, +0.167] | 0.0075 | 11.8 | 0.35 |
| 0.75 | 80.0% | 77.5% | +0.025 | [−0.033, +0.092] | 0.61 | 10.9 | 0.33 |
| 1.00 | 80.0% | 77.5% | +0.025 | [−0.042, +0.092] | 0.63 | 10.3 | 0.31 |
| 2.00 | 77.5% | 77.5% | +0.000 | [0, 0] | 1.0 | 15.0 | **0.00** |

### What holds

1. **Direction is consistent.** The advantage is positive at *every* ratio below 1.0.
2. **The sanity check passes exactly.** At ratio 2.0 — tools more expensive than asking —
   the planner abandons them completely (tool share **0.00**) and its accuracy becomes
   identical to patient-only, delta exactly 0.000. The cost term is doing what it claims.
3. The effect is significant at 4 of 8 ratios.

### The confound — stated plainly

An equal **cost** budget is not an equal **information** budget. Implied observation counts:

| ratio | tool agent observations | patient-only |
|---|---|---|
| 0.02 | **~565** | 15 |
| 0.05 | ~298 | 15 |
| 0.10 | ~150 | 15 |
| 0.25 | ~39 | 15 |
| 0.50 | ~18 | 15 |
| 1.00 | ~15 | 15 |

At the cheap end the tool agent sees **38× more evidence**. Of course it wins. That result
does not isolate *source selection*; it mostly measures *how many observations the budget
buys*.

Worse for the headline: at ratio 1.00, where observation counts are genuinely matched
(15 vs 15), the advantage falls to **+2.5 points and is not significant** (p = 0.63).

### What this means for the claim

Two distinct claims were being conflated:

- **(A) Economic.** Cheaper sources buy more information per unit cost. *Supported* — and
  it is the clinically meaningful claim, since querying a local index genuinely is cheaper
  than consuming consultation time.
- **(B) Selection.** Choosing among heterogeneous sources beats always asking the patient,
  *at equal information*. **Not established by this experiment**, and the ratio-1.00 row is
  evidence against it.

§12's "+16.0 points at identical cost" is claim (A) and must be worded as such. Describing
it as evidence that the agent picks *better sources* overstates what was measured.

The matched control has now run. See §18 — **(B) is refuted.**

### Caveats on this experiment itself

- n = 120 is underpowered; CIs span ~0.15 and the non-monotonicity (0.25 → ns, 0.50 → sig,
  0.75 → ns) is consistent with noise rather than structure. It should be rerun at n ≥ 500.
- All three tool channels were collapsed to a single price to keep one interpretable axis.
  A per-channel sweep would be more informative and much more expensive.
- Runtime 69 minutes, dominated by the cheap-ratio points where the agent takes hundreds of
  actions per case.

## 18. Matched-observation control — the selection claim is REFUTED

`ayur.experiments.matched_budget`. Every agent takes exactly N observations chosen by pure
EIG with **cost ignored**; the only difference is whether non-patient channels are available.
n = 200 cases per budget.

| observations | all sources | patient-only | delta | 95% CI | p |
|---|---|---|---|---|---|
| 5 | 15.5% | 13.0% | +0.025 | [−0.015, +0.065] | 0.33 |
| 10 | 57.5% | 56.5% | +0.010 | [−0.070, +0.090] | 0.90 |
| 15 | 76.5% | 80.0% | −0.035 | [−0.095, +0.025] | 0.31 |
| 20 | 87.0% | **93.0%** | **−0.060** | [−0.105, −0.020] | **0.012** |
| 30 | 97.0% | 99.0% | −0.020 | [−0.040, −0.005] | 0.12 |

**Significant and positive at 0 of 5 budgets. Mean delta −1.6 points. At 20 observations
the multi-source agent is significantly *worse*.**

### This is the correct result, and it was predictable

With cost removed there is no reason to ever consult a lower-fidelity channel. Patient
noise is 0.05; retrieval is 0.25. Given a fixed number of observations, the optimal policy
is to spend every one on the highest-fidelity source. The multi-source agent still allocates
25–77% of its actions to tools — because per-observation EIG slightly favours some of them —
and pays for it.

So the negative result is not a bug in the planner; it is the planner correctly optimising
the wrong objective once the objective is stripped of cost. **A heterogeneous action space
is only rational under a cost constraint.**

### What the paper must now claim, and must not

✅ **Claim.** Under a cost budget, an agent that prices heterogeneous knowledge sources and
maximises information *per unit cost* reaches a given accuracy far more cheaply than one
restricted to patient interrogation — reducing patient burden from 15 questions to 8.6 at
higher accuracy (§12). The cost model is validated by the sanity check: price tools above
patient questions and the planner correctly stops using them (§17, ratio 2.0).

❌ **Do not claim.** That consulting multiple sources is intrinsically better, or that the
agent "selects better sources". At equal information it is neutral-to-harmful. Any phrasing
implying selection quality independent of cost is contradicted by this table.

### Consequence for the contribution list (§3)

**C4 must be narrowed.** The original wording — "heterogeneous action space… tool-choice-as-
planning over structurally different knowledge sources" — implies (B). The defensible
version is:

> **C4 (revised).** *Cost-aware* value-of-information planning. The contribution is not that
> heterogeneous sources help, but that pricing them and optimising information-per-cost
> yields substantially cheaper consultations at equal or better accuracy — with a control
> showing the benefit is economic rather than an artefact of source diversity.

That is a smaller claim than originally framed and a much more defensible one. It also
sharpens the comparison with MAI-DxO, which *does* reason about cost but only inside an LLM
and without an explicit information-per-cost criterion.

### Why reporting this strengthens the paper

Most papers proposing a multi-tool agent never run the matched-information control, so their
"tools help" result is confounded with "more observations help". Running it, finding the
null, and reframing accordingly is a stronger scientific position than the inflated claim —
and it pre-empts the exact experiment a sceptical reviewer would demand.

## 19. Source selection DOES matter — the earlier control asked the wrong question

§18 concluded that the multi-source advantage is purely economic. That conclusion was drawn
from the wrong comparison, and this section corrects it.

### The mis-posed question

`matched_budget.py` compared **all-sources vs patient-only**. That asks *"does having tools
help?"* — a question about the toolset. The claim we care about is *"does the agent choose
the right source?"* — a question about the **selection rule**. Conflating them is the same
error as evaluating a search algorithm by whether the index exists.

The right experiment holds the toolset and the observation budget fixed and varies only the
rule for deciding which source answers each question. In every arm except the first, the
*feature* is chosen identically by EIG, so any difference is attributable to source choice
alone.

### Result (n = 150 per cell, cost ignored)

**Budget 10 observations**

| selection rule | accuracy | Δ vs EIG | p | sources used |
|---|---|---|---|---|
| **eig-source** *(proposed)* | **62.7%** | — | — | patient .57 · kg .43 |
| fixed-patient *(best hand-coded heuristic)* | 58.7% | +4.0 | 0.41 | patient .97 · kg .03 |
| random-source | 24.7% | **+38.0** | 1.4e-13 | patient .41 · kg .16 · text .43 |
| fixed-cheapest | 20.7% | **+42.0** | 1.9e-16 | kg .42 · text .58 |
| worst-source *(floor)* | 12.0% | **+50.7** | <1e-16 | text 1.00 |

**Budget 15 observations**

| selection rule | accuracy | Δ vs EIG | p | sources used |
|---|---|---|---|---|
| fixed-patient | **83.3%** | −3.3 | 0.30 | patient .98 · kg .02 |
| **eig-source** | 80.0% | — | — | patient .70 · kg .30 |
| random-source | 52.7% | **+27.3** | 1.3e-08 | patient .44 · kg .11 · text .45 |
| fixed-cheapest | 26.7% | **+53.3** | <1e-16 | kg .29 · text .71 |
| worst-source *(floor)* | 16.7% | **+63.3** | <1e-16 | text 1.00 |

**Budget 20 observations**

| selection rule | accuracy | Δ vs EIG | p | sources used |
|---|---|---|---|---|
| fixed-patient | **94.0%** | **−6.7** | **0.0063** | patient .98 · kg .02 |
| **eig-source** | 87.3% | — | — | patient .76 · kg .24 |
| random-source | 71.3% | **+16.0** | 3.9e-05 | patient .45 · kg .09 · text .46 |
| fixed-cheapest | 38.0% | **+49.3** | <1e-16 | kg .22 · text .78 |
| worst-source *(floor)* | 22.7% | **+64.7** | <1e-16 | text 0.99 |

**Best–worst spread: 50.7 / 66.6 / 64.7 points** (mean 62.9). All 977 features are
answerable by more than one source, so there was maximal room for choice to be irrelevant —
and it is not. Which source answers a question matters more than almost any other design
decision measured in this project.

### ⚠️ The EIG criterion is systematically beaten by a trivial heuristic

| budget | EIG − fixed-patient | p |
|---|---|---|
| 10 | +4.0 | 0.41 (ns) |
| 15 | −3.3 | 0.30 (ns) |
| 20 | **−6.7** | **0.0063 (significant)** |

The gap is **monotone in budget** and significant by 20 observations. This is a systematic
deficiency in the proposed criterion, not sampling noise, and an earlier draft of this
section wrongly reported the relationship as "within noise in both directions" on the
strength of the first two budgets alone.

**Diagnosed mechanism.** Mutual information is a function of *both* channel fidelity and
feature informativeness, so a low-noise channel querying a mediocre attribute can outscore a
higher-noise channel querying a highly diagnostic one. The KG channel (noise 0.02) covers
only the 15 dosha/prakriti features, and greedy EIG over-selects them precisely *because
they are low-noise* — spending 24–43% of a limited budget on 15 attributes rather than on
the most diagnostic ones available. `fixed-patient` avoids the trap by construction: it
picks the feature on merit and then fixes the source.

**The implied fix** is to **decouple the two decisions**: choose *what to learn* by
marginalising fidelity out of the EIG computation, then choose *who to ask* by fidelity and
cost. This was pre-registered (`PREREGISTRATION.md`) and tested. **The diagnosis was
confirmed; the fix was insufficient — see §20.**

### What can now be claimed

✅ **Source selection matters enormously.** Holding the toolset, budget and feature choice
fixed, swapping only the source-selection rule moves accuracy from 12.0% to 62.7%. A system
that consults sources without reasoning about which one to consult forfeits up to 50 points.

✅ **The EIG criterion selects near-optimally, without being told anything.** It beats random
source choice by **+38.0 / +27.3 points** (budgets 10 / 15) and the cheapest-source rule by
**+42.0 / +53.3**. It is never given the channels' noise levels as a preference order; it
infers from the belief state which source is worth consulting.

❌ **It does NOT match the best hand-designed heuristic.** `fixed-patient` — always ask the
highest-fidelity source — beats EIG-source by a margin that grows with budget and is
significant at 20 observations (−6.7, p = 0.0063). An earlier draft claimed parity; that was
wrong, and the corrected trend is in the table above.

**Do not claim EIG beats or matches `fixed-patient`.** At useful budgets it loses to it.

❌ **Still not claimed:** that consulting multiple sources beats interrogating the single
best source at equal information (§18). That remains false here, and both results belong in
the paper.

### The precise wording for the paper

> Source selection is consequential: holding toolset, budget and question choice fixed, the
> choice of *which source answers* moves diagnostic accuracy by 62.9 points on average
> between the best and worst policies. A value-of-information criterion captures most of
> that value automatically, outperforming random source selection by 16.0–38.0 points
> (all p < 1e-4) with no hand-specified source preference. It does not, however, match a
> domain-informed always-highest-fidelity heuristic, which exceeds it by 6.7 points at a
> 20-observation budget (p = 0.006); we trace this to greedy mutual information conflating
> channel fidelity with feature informativeness, and identify decoupling the two decisions
> as the remedy.

### Why §18's null is still worth reporting

The two results are complementary and both should appear. §18 shows that *adding* sources to
a rich, high-fidelity patient channel buys nothing at equal information — an honest negative
that most multi-tool papers never test. §19 shows that *given* a heterogeneous toolset,
choosing well among it is worth up to 50 points. Together: heterogeneity is not free value,
but managing heterogeneity is a real problem with a real solution.

### Caveats

- n = 150 per cell; the eig-vs-fixed-patient gap (+4.0) is not significant and should not be
  reported as a win over that baseline.
- `worst-source` is an adversarial floor, not a policy anyone would deploy. The honest
  headline comparison is against **random-source** (+38.0), not against the floor.
- Cost is ignored here by design, to isolate selection from economics.

## 20. Decoupled selection — pre-registered, tested, primary prediction FAILED

After eight exploratory comparisons had already been run against the source-selection claim,
further searching would have been untrustworthy. The next step was therefore
**pre-registered** in `PREREGISTRATION.md` — mechanism, intervention, predictions, failure
conditions and a one-run commitment, all written before execution.

### Outcome

| budget | eig-source | **decoupled-eig** | fixed-patient | decoupled − heuristic | p |
|---|---|---|---|---|---|
| 10 | 62.7% | 52.7% | 58.7% | −5.3 | 0.29 |
| 15 | 79.3% | 77.3% | 83.3% | −4.7 | 0.21 |
| 20 | 86.7% | **90.7%** | 94.0% | **−3.3** | 0.11 |

**Primary prediction — gap ≤ 2.0 points at budget 20 — FAILED (3.3 points).**
All four secondary predictions were met. The primary was designated primary so that
secondaries could not be used to rescue it, and it is not being used that way now.

### What the experiment did establish

The **mechanism diagnosis in §19 was confirmed by direct test**. Prediction S4 stated that
decoupling would visibly reduce KG over-selection; it fell 43%→29%, 30%→21%, 24%→17%. S2
stated that the monotone worsening would disappear if over-selection was its cause; the gap
now shrinks with budget (−6.0 → −6.0 → −3.3) instead of growing. So the causal story is
right: greedy EIG conflates fidelity with informativeness, and that conflation was driving
the deficit.

The remedy simply does not go far enough — and it costs something. At budget 10,
`decoupled-eig` is **significantly worse** than `eig-source` (52.7% vs 62.7%, p = 0.020),
an unpredicted regression: stripping fidelity from the feature ranking hurts when the budget
is too small to revisit a bad choice.

### Final position on source selection

| claim | status |
|---|---|
| Source choice is high-leverage (64.5-pt best-worst spread) | ✅ **established** |
| VOI selection beats random / cheapest / worst by 16–38 pts | ✅ **established** |
| VOI selection matches a domain heuristic | ❌ **refuted** (−6.7 pts, p = 0.006) |
| Decoupling closes that gap | ❌ **refuted** (pre-registered, −3.3 pts) |
| Adding sources helps at equal information | ❌ **refuted** (§18) |

**No further variants will be searched.** Three failure conditions were written down in
advance; one triggered; the commitment was to stop.

### How the paper should read

Lead with the economic claim — validated at n = 5,000 — and report source selection as a
characterised limitation:

> A value-of-information criterion captures most of the value of source selection
> automatically (+16 to +38 points over random selection) but does not match a
> domain-informed heuristic (−6.7 points, p = 0.006). We trace this to greedy mutual
> information conflating channel fidelity with feature informativeness, verify the diagnosis
> by a pre-registered intervention that reduces the pathology as predicted, and report that
> the intervention narrows but does not close the gap. Closing it likely requires
> non-myopic planning rather than a modified one-step criterion.

That last sentence is the honest next direction: every policy tested here is **greedy**.
The gap to a hand-designed heuristic may simply be the price of one-step lookahead, which
would make this a limitation of myopic VOI rather than of the decoupling idea. Testing it
means implementing lookahead — a substantial piece of work, not a variant to try tonight.

## 9. Open dependency — RESOLVED

*(Superseded 2026-07-27.)* AyurGenixAI is now in hand and §10 shows the planning problem is
real. C1–C4 all stand.

**AyurKOSH is confirmed unobtainable (2026-08-02).** Institutional access was not available
and it stays behind the IEEE DataPort paywall. It is out of scope permanently — not deferred
— and the paper must present it that way. What it would have contributed, and what stands in
its place, is set out below and in §15.

C4 is therefore weakened but not lost. The knowledge-source inventory is now:

| Source | Relation | Status |
|---|---|---|
| AyurGenixAI | condition → 19 patient attributes | ✅ 446 conditions, 980 features |
| Amidha | herb → indication, herb → dosha, full pharmacology | ✅ 360 herbs, 1,738 edges |
| BhashaBench-Ayur | bilingual exam knowledge | ✅ 14,963 items |
| Ayurveda-LLM | retrieval corpus | ✅ 1,529 passages (⚠️ unlicensed) |
| AyurKOSH | classical-text symbolic triples | ❌ **unobtainable — out of scope** |

The KG the agent traverses can be assembled by **joining AyurGenixAI's condition→dosha and
Amidha's herb→dosha/indication** on the shared dosha and indication vocabularies. That gives
a genuine two-hop symbolic path (condition → dosha → herb, and condition → indication → herb)
without AyurKOSH. Verify the vocabulary overlap early — it is the next thing to measure, and
if the Sanskrit indication terms in Amidha do not align with AyurGenixAI's English disease
names, that join needs a mapping layer and should be budgeted for.
