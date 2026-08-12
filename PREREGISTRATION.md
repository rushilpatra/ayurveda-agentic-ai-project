# Pre-registration: decoupled source selection

**Written 2026-08-02, before running the experiment.** Committed in advance so the
success criterion cannot be adjusted after seeing results. Timestamped by git history
and by the results files it precedes.

## Context

`results/selection_quality.json` established:

| budget | eig-source | fixed-patient | Δ (EIG − heuristic) | p |
|---|---|---|---|---|
| 10 | 62.7% | 58.7% | +4.0 | 0.41 |
| 15 | 80.0% | 83.3% | −3.3 | 0.30 |
| 20 | 87.3% | 94.0% | **−6.7** | **0.0063** |

The proposed value-of-information criterion is beaten by a one-line domain heuristic
("always query the highest-fidelity source"), and the gap grows monotonically with budget.

Prior to this, eight comparisons had already been run against the general claim that the
agent selects sources well. That is enough multiplicity that any further exploratory
search would be untrustworthy. This document exists to convert the next step from a
search into a test.

## Diagnosed mechanism

Mutual information `I(D; o_a)` is a joint function of **channel fidelity** and **feature
informativeness**. A low-noise channel reading a mediocre attribute can outscore a
high-noise channel reading a highly diagnostic one.

The knowledge-graph channel has noise 0.02 (versus 0.05 for the patient) but covers only
the 15 `Doshas` / `Constitution/Prakriti` features. Greedy EIG therefore over-selects those
15 attributes *because they are cheap to read accurately*, not because they discriminate
well — spending 24–43% of a scarce budget on them. `fixed-patient` avoids this by
construction: it ranks features on merit and then fixes the source.

## The intervention

Decouple the two decisions that greedy EIG conflates.

1. **What to learn.** Rank *features* by expected information gain computed at a single
   fixed reference fidelity, identical for every feature. This removes channel noise from
   the comparison, so features are ranked on diagnostic merit alone.
2. **Who to ask.** Among the sources able to answer the chosen feature, select by actual
   fidelity (and, in the cost-aware variant, by information per unit cost).

Implemented as `decoupled-eig` in `ayur/experiments/selection_quality.py`.

## Predictions (committed in advance)

**Primary — the test this stands or falls on.**
> At budget 20, `decoupled-eig` closes the gap to `fixed-patient` to within 2 percentage
> points, and the difference is not statistically significant (p > 0.05, McNemar).

**Secondary.**
> S1. `decoupled-eig` beats `eig-source` at budget 20 by ≥ 3 points.
> S2. The monotone worsening across budgets 10 → 15 → 20 seen for `eig-source`
>     disappears: the `decoupled-eig` − `fixed-patient` gap does not grow with budget.
> S3. `decoupled-eig` continues to beat `random-source` by ≥ 15 points at all budgets.
> S4. `decoupled-eig` routes a smaller share of queries to the KG channel than
>     `eig-source` does (i.e. the diagnosed over-selection is visibly reduced).

## What counts as failure

Any of the following, and the intervention is reported as unsuccessful:

- the primary prediction is not met at budget 20;
- `decoupled-eig` is significantly worse than `fixed-patient` at any tested budget;
- the gap to `fixed-patient` still grows monotonically with budget.

## Commitments

- **Parameters are fixed before running:** n = 150 per cell, budgets {10, 15, 20},
  noise 0.05, omission 0.10, seed 11 — identical to the run being compared against.
- **One run.** No re-running at other n, other budgets, or other seeds in search of a
  favourable result. If the effect is genuine at these settings it can be confirmed later
  at higher n; if it is not, that is the answer.
- **All arms reported**, including `eig-source`, whether or not the intervention succeeds.
- **On failure**, the paper leads with the economic claim (validated at n = 5,000) and
  reports source selection as a diagnosed limitation with this fix attempted and
  unsuccessful.

## Result — PRIMARY PREDICTION FAILED

Single committed run executed 2026-08-02, n = 150, budgets {10, 15, 20}, seed 11.
Full output: `results/selection_quality.json`.

| budget | eig-source | **decoupled-eig** | fixed-patient | decoupled − fixed-patient | p |
|---|---|---|---|---|---|
| 10 | 62.7% | 52.7% | 58.7% | −5.3 | 0.29 (ns) |
| 15 | 79.3% | 77.3% | 83.3% | −4.7 | 0.21 (ns) |
| 20 | 86.7% | **90.7%** | 94.0% | **−3.3** | 0.11 (ns) |

### Scoring against the committed criteria

| prediction | criterion | outcome | verdict |
|---|---|---|---|
| **PRIMARY** | gap to fixed-patient ≤ 2.0 pts at budget 20 | **3.3 pts** | ❌ **FAILED** |
| S1 | beats eig-source at budget 20 by ≥ 3 pts | +4.0 | ✅ met |
| S2 | gap does not grow with budget | −6.0 → −6.0 → −3.3 (shrinks) | ✅ met |
| S3 | beats random-source by ≥ 15 pts at all budgets | +24.0 / +24.7 / +20.7 | ✅ met |
| S4 | routes less to KG than eig-source | .29/.21/.17 vs .43/.30/.24 | ✅ met |

**The primary prediction failed. Per the failure criteria above, the intervention is
reported as unsuccessful.** Four of four secondary predictions were met, but the primary
was designated primary precisely so that a majority of secondaries could not be used to
rescue it after the fact.

Neither of the other two failure conditions triggered: decoupled-eig is not significantly
worse than fixed-patient at any budget (p = 0.29 / 0.21 / 0.11), and the gap no longer
grows with budget.

### An unpredicted regression

At budget 10, `decoupled-eig` (52.7%) is **significantly worse** than `eig-source` (62.7%,
p = 0.020). Removing fidelity from the feature ranking costs accuracy when the budget is
too small to revisit the decision. This was not anticipated and counts against the
intervention.

### What was nonetheless learned

The diagnosed mechanism was **confirmed**. S4 was a direct test of it: decoupling reduced
KG over-selection from 43%→29%, 30%→21%, 24%→17% exactly as predicted, and S2 confirmed
the monotone worsening was caused by that over-selection, since removing it reverses the
trend. The diagnosis in `POSITIONING.md` §19 stands; the remedy is directionally right but
insufficient.

### Honest note on reproducibility

An ad-hoc re-computation of the budget-20 decoupled cell gave 90.0% against the committed
run's 90.7% — one case in 150. A within-process determinism check passes (identical result
hashes across repeated runs), so the discrepancy is not stochasticity in the policy, but it
is **unexplained**. It does not affect any verdict above (both values fail the ≤2.0 pt
criterion), but it should be traced before publication.

### Consequence

Per the commitment above: the paper leads with the economic claim, validated at n = 5,000,
and reports source selection as a diagnosed limitation with this fix attempted and
unsuccessful. No further variants will be searched.
