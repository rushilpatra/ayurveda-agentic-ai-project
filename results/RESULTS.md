# Results
Generated from real runs on Apple M4 / 16.0 GB unified memory, macOS 15.6.1, Python 3.11.13.
Every table states its exact sample size. Experiments that have not been run are listed as missing rather than omitted.

## Single-source planner - structured

n = 5000 cases, 446 conditions, 977 features.

| agent | n | coverage | accuracy | sel. acc | macro-F1 | questions | ECE | AURC |
|---|---|---|---|---|---|---|---|---|
| B1-full-information | 5000 | 100.0% | 100.0% | 100.0% | 1.000 | 977.0 | 0.000 | 0.000 |
| B8-max-eig-no-abstain | 5000 | 100.0% | 93.7% | 93.7% | 0.935 | 20.0 | 0.005 | 0.002 |
| B7-max-eig | 5000 | 98.1% | 72.9% | 74.3% | 0.733 | 10.0 | 0.016 | 0.160 |
| B5-greedy-frequency | 5000 | 100.0% | 62.6% | 62.6% | 0.603 | 20.0 | 0.009 | 0.130 |
| B3-random | 5000 | 100.0% | 8.8% | 8.8% | 0.078 | 20.0 | 0.010 | 0.808 |
| B4-prior-only | 5000 | 100.0% | 3.9% | 3.9% | 0.017 | 0.0 | 0.006 | 0.956 |

Paired comparisons vs `B7-max-eig` (McNemar, Holm-corrected):

| agent | n | delta | p (Holm) | sig. |
|---|---|---|---|---|
| B4-prior-only | 5000 | 69.0% | <1e-16 | yes |
| B3-random | 5000 | 64.1% | <1e-16 | yes |
| B5-greedy-frequency | 5000 | 10.3% | <1e-16 | yes |
| B8-max-eig-no-abstain | 5000 | -20.8% | <1e-16 | yes |
| B1-full-information | 5000 | -27.1% | <1e-16 | yes |

## Single-source planner - structured-misspecified

n = 5000 cases, 446 conditions, 977 features.

| agent | n | coverage | accuracy | sel. acc | macro-F1 | questions | ECE | AURC |
|---|---|---|---|---|---|---|---|---|
| B1-full-information | 5000 | 100.0% | 99.4% | 99.4% | 0.994 | 977.0 | 0.003 | 0.000 |
| B8-max-eig-no-abstain | 5000 | 100.0% | 42.2% | 42.2% | 0.416 | 20.0 | 0.136 | 0.254 |
| B7-max-eig | 5000 | 86.8% | 26.7% | 30.7% | 0.295 | 13.3 | 0.404 | 0.589 |
| B5-greedy-frequency | 5000 | 100.0% | 22.5% | 22.5% | 0.196 | 20.0 | 0.184 | 0.605 |
| B3-random | 5000 | 100.0% | 2.7% | 2.7% | 0.022 | 20.0 | 0.059 | 0.955 |
| B4-prior-only | 5000 | 100.0% | 1.5% | 1.5% | 0.006 | 0.0 | 0.019 | 0.982 |

Paired comparisons vs `B7-max-eig` (McNemar, Holm-corrected):

| agent | n | delta | p (Holm) | sig. |
|---|---|---|---|---|
| B4-prior-only | 5000 | 25.1% | <1e-16 | yes |
| B3-random | 5000 | 24.0% | <1e-16 | yes |
| B5-greedy-frequency | 5000 | 4.1% | <1e-16 | yes |
| B8-max-eig-no-abstain | 5000 | -15.5% | <1e-16 | yes |
| B1-full-information | 5000 | -72.7% | <1e-16 | yes |

## Single-source planner - pilot

n = 200 cases, 446 conditions, 977 features.

| agent | n | coverage | accuracy | sel. acc | macro-F1 | questions | ECE | AURC |
|---|---|---|---|---|---|---|---|---|
| B1-full-information | 200 | 100.0% | 100.0% | 100.0% | 1.000 | 977.0 | 0.000 | 0.000 |
| B8-max-eig-no-abstain | 200 | 100.0% | 80.0% | 80.0% | 0.690 | 15.0 | 0.038 | 0.024 |
| B7-max-eig | 200 | 89.0% | 64.5% | 72.5% | 0.600 | 9.9 | 0.049 | 0.177 |
| B5-greedy-frequency | 200 | 100.0% | 40.0% | 40.0% | 0.292 | 15.0 | 0.099 | 0.304 |
| B3-random | 200 | 100.0% | 6.0% | 6.0% | 0.033 | 15.0 | 0.026 | 0.858 |
| B4-prior-only | 200 | 100.0% | 2.5% | 2.5% | 0.013 | 0.0 | 0.007 | 0.973 |

Paired comparisons vs `B7-max-eig` (McNemar, Holm-corrected):

| agent | n | delta | p (Holm) | sig. |
|---|---|---|---|---|
| B4-prior-only | 200 | 62.0% | <1e-16 | yes |
| B3-random | 200 | 58.5% | <1e-16 | yes |
| B5-greedy-frequency | 200 | 24.5% | <1e-16 | yes |
| B8-max-eig-no-abstain | 200 | -15.5% | <1e-16 | yes |
| B1-full-information | 200 | -35.5% | <1e-16 | yes |

## Single-source planner - smoke

n = 20 cases, 446 conditions, 977 features.

| agent | n | coverage | accuracy | sel. acc | macro-F1 | questions | ECE | AURC |
|---|---|---|---|---|---|---|---|---|
| B1-full-information | 20 | 100.0% | 100.0% | 100.0% | 1.000 | 977.0 | 0.000 | 0.000 |
| B7-max-eig | 20 | 85.0% | 50.0% | 58.8% | 0.435 | 10.2 | 0.127 | 0.355 |
| B3-random | 20 | 0.0% | 0.0% | nan% | nan | 15.0 | nan | 0.950 |
| B4-prior-only | 20 | 100.0% | 0.0% | 0.0% | 0.000 | 0.0 | 0.032 | 0.950 |
| B5-greedy-frequency | 20 | 0.0% | 0.0% | nan% | nan | 15.0 | nan | 0.950 |
| B8-max-eig-no-abstain | 20 | 0.0% | 0.0% | nan% | nan | 15.0 | nan | 0.950 |

Paired comparisons vs `B7-max-eig` (McNemar, Holm-corrected):

| agent | n | delta | p (Holm) | sig. |
|---|---|---|---|---|
| B4-prior-only | 20 | 50.0% | 9.77e-03 | yes |
| B3-random | 20 | 50.0% | 9.77e-03 | yes |
| B5-greedy-frequency | 20 | 50.0% | 9.77e-03 | yes |
| B8-max-eig-no-abstain | 20 | 50.0% | 9.77e-03 | yes |
| B1-full-information | 20 | -50.0% | 9.77e-03 | yes |

## Heterogeneous action space

n = 200 cases. 2319 actions: {'ask_patient': 977, 'query_kg': 15, 'retrieve_text': 977, 'verify_herb': 350}. Costs: {'ask_patient': 1.0, 'query_kg': 0.1, 'retrieve_text': 0.3, 'verify_herb': 0.4}.

| agent | n | accuracy | cost | patient Qs | acc/cost | ECE |
|---|---|---|---|---|---|---|
| T1-tool-eig-per-cost | 200 | 95.5% | 14.944 | 13.5 | 0.0639 | 0.022 |
| T2-tool-eig-lagrangian | 200 | 91.5% | 14.847 | 8.6 | 0.0616 | 0.024 |
| T3-patient-only | 200 | 79.5% | 15.000 | 15.0 | 0.0530 | 0.032 |
| T4-cheapest-first | 200 | 24.0% | 14.701 | 0.0 | 0.0163 | 0.051 |
| T5-random-tool | 200 | 7.0% | 14.939 | 10.1 | 0.0047 | 0.041 |

Paired vs `T1-tool-eig-per-cost` (McNemar, Holm-corrected):

| agent | delta | p (Holm) | sig. |
|---|---|---|---|
| T5-random-tool | 88.5% | <1e-16 | yes |
| T4-cheapest-first | 71.5% | <1e-16 | yes |
| T3-patient-only | 16.0% | <1e-16 | yes |
| T2-tool-eig-lagrangian | 4.0% | 2.15e-02 | yes |

Tool-kind agreement with an EIG/cost oracle: `{'T1-tool-eig-per-cost': 0.8243, 'T4-cheapest-first': 0.2198}`

### Heterogeneous action space - misspecified

n = 200 cases, env noise 0.15, omission 0.25, agent assumes 0.05.

| agent | n | accuracy | cost | patient Qs | acc/cost | ECE |
|---|---|---|---|---|---|---|
| T1-tool-eig-per-cost | 200 | 43.5% | 14.941 | 13.2 | 0.0291 | 0.241 |
| T2-tool-eig-lagrangian | 200 | 39.5% | 14.915 | 12.3 | 0.0265 | 0.192 |
| T3-patient-only | 200 | 28.5% | 15.000 | 15.0 | 0.0190 | 0.173 |
| T4-cheapest-first | 200 | 6.5% | 14.700 | 0.0 | 0.0044 | 0.197 |
| T5-random-tool | 200 | 3.5% | 14.940 | 10.1 | 0.0023 | 0.081 |

### Cost-table sensitivity

The cost table is a design choice, so the advantage is re-measured across cost ratios. `ratio` prices every non-patient channel relative to a patient question fixed at 1.0. n = 120 cases per point.

| ratio | tool-EIG | patient-only | delta | p | tool share |
|---|---|---|---|---|---|
| 0.020 | 89.2% | 77.5% | +0.117 | 1.61e-02 | 0.993 |
| 0.050 | 89.2% | 77.5% | +0.117 | 1.61e-02 | 1.000 |
| 0.100 | 89.2% | 77.5% | +0.117 | 1.61e-02 | 1.000 |
| 0.250 | 80.8% | 77.5% | +0.033 | 5.72e-01 | 0.820 |
| 0.500 | 87.5% | 77.5% | +0.100 | 7.54e-03 | 0.347 |
| 0.750 | 80.0% | 77.5% | +0.025 | 6.07e-01 | 0.330 |
| 1.000 | 80.0% | 77.5% | +0.025 | 6.29e-01 | 0.310 |
| 2.000 | 77.5% | 77.5% | +0.000 | 1.00e+00 | 0.001 |

Advantage significant and positive at **4/8** cost ratios; delta range +0.000 to +0.117. Positive at every ratio below 1.0: **True**.

### Matched-observation control — is the advantage selective or economic?

Every agent takes exactly N observations chosen by pure EIG with **cost ignored**; the only difference is whether non-patient channels are available. n = 200 cases per budget.

| observations | all sources | patient-only | delta | p | tool share |
|---|---|---|---|---|---|
| 5 | 15.5% | 13.0% | +0.025 | 3.32e-01 | 0.772 |
| 10 | 57.5% | 56.5% | +0.010 | 9.04e-01 | 0.448 |
| 15 | 76.5% | 80.0% | -0.035 | 3.11e-01 | 0.311 |
| 20 | 87.0% | 93.0% | -0.060 | 1.18e-02 | 0.254 |
| 30 | 97.0% | 99.0% | -0.020 | 1.25e-01 | 0.249 |

**Significant and positive at 0 of 5 budgets; mean delta -0.016.**

> **The multi-source advantage is economic, not selective.** With cost removed there is no reason to consult a lower-fidelity channel (patient noise 0.05 vs retrieval 0.25), so spending every observation on the patient is optimal — and the multi-source agent, still allocating 25–77% of actions to tools, is neutral-to-worse. A heterogeneous action space is only rational under a cost constraint. Report the equal-cost result as *cheaper consultations at equal accuracy*, never as *better source selection*.

## Sanskrit→English nosology mapping

91 curated correspondences ({'exact': 12, 'close': 41, 'approximate': 26, 'broad': 7, 'unmappable': 5}), unlocking **1541 of 1738** herb→indication edges (88.7%).

> `expert_reviewed = False`. Compiled from standard textbook correspondences **without** a domain expert; the `approximate` tier especially needs practitioner review before publication.

## Text retrieval (BM25)

1889 local passages ({'ayurveda-llm': 1529, 'amidha': 360}), 800 naming a condition. Backend `rank_bm25`.

- precision **0.0734** vs curated base rate 0.0249 -> **2.95x lift**
- recall 0.0643, F1 0.0685, feature coverage 1.8%

> **Do not quote raw agreement.** It reads 0.9565, but both matrices are ~97.5% zeros and an all-zeros matrix scores 0.9751 - i.e. the text-derived matrix is *worse than predicting nothing* on that metric. The corpus carries real but thin signal: ~3x chance precision at 6% recall.

## Calibration under misspecification

Fitted on 400 calibration cases, evaluated on 400 disjoint test cases. Temperature 1.7, tau 0.3216. Effective-noise MLE **0.18** against a true environment noise of **0.15**, with the agent told 0.05.

| metric | uncalibrated | calibrated |
|---|---|---|
| coverage | 0.485 | 0.432 |
| accuracy_selective | 0.753 | 0.815 |
| selective_risk | 0.247 | 0.185 |
| ece | 0.153 | 0.029 |
| brier | 0.125 | 0.083 |
| aurc | 0.262 | 0.262 |

> AURC is unchanged by design: temperature scaling is monotonic, so it repairs calibration but cannot improve discrimination.

## B6 - LLM as planner

n = 60 cases, budget 10, shortlist 12. **Identical posterior and identical patient answers** - only action selection differs, so the gap is attributable to planning alone.

| planner | accuracy | s/case |
|---|---|---|
| B6 LLM (Qwen3-4B-4bit) | 21.7% | 11.013 |
| B7 closed-form EIG | 51.7% | 0.061 |

- delta (EIG − LLM) **+0.300**, 95% CI [0.1667, 0.4333], McNemar p = 2.77e-04
- EIG is **180.3x faster** per case
- 0.0% of 600 LLM replies could not be parsed to a choice (fell back to the first option)

## BhashaBench-Ayur

Frozen Qwen3-4B-4bit, zero-shot, domain-stratified subset. Chance level is 25% (four options).

| language | n | raw acc | parsed-only | chance-corrected | unparsed | s/q |
|---|---|---|---|---|---|---|
| en | 400 | 48.2% | 48.4% | 31.0% | 0.2% | 0.5 |
| hi | 400 | 32.0% | 33.8% | 9.3% | 5.2% | 0.8 |

| gap measure | value |
|---|---|
| domain-stratified (16 shared domains) | +15.9 pts |
| parsed-only | +14.6 pts |
| attributable to instruction-following | +1.7 pts |
| **chance-corrected** | **+21.7 pts** |

> **Report the chance-corrected gap.** The 25% MCQ floor compresses both raw scores toward each other; corrected for it the gap widens to +21.7 points and Hindi sits at 9.3% — barely above guessing. Only 1.7 points come from the model failing to emit a letter in Hindi, so the deficit is knowledge, not formatting.
>
> The naive difference of means (+16.2 pts) **must not be reported**: the two splits share zero item ids and have per-domain size ratios from 0.36 to 1.04, so it confounds language with topic mix (`DATA_AUDIT.md` §1).

## Knowledge graph

6249 triples from ['amidha', 'ayurgenix', 'nosology:exact', 'nosology:broad', 'nosology:approximate', 'nosology:close']. AyurKOSH available: False.

Two-hop `condition -> dosha -> herb`, evaluated against held-out direct `condition -> herb` edges over 139 conditions and 360 candidate herbs:

- Hit@1 6.5%, Hit@5 13.0%, Hit@10 **17.3%**, Hit@20 33.1%
- random Hit@10 baseline 2.8%, MRR 0.1156, median rank 32.0

## Figures

- **budget sweep**: `results/figures/budget_sweep.png`
- **calibration**: `results/figures/calibration.png`
- **tool selection**: `results/figures/tool_selection.png`
- **agent comparison**: `results/figures/agents_structured.png`
