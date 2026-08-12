"""Viability probe: is there a real sequential-planning problem in AyurGenixAI?

Builds a binary disease x attribute matrix from ALL usable columns (not just Symptoms),
then compares question-selection policies on accuracy-vs-#questions.

If max-EIG does not beat random/greedy, there is no information-gain contribution.
"""
import re, collections
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
df = pd.read_csv("archive (2)/AyurGenixAI_Dataset.csv")

# Columns that describe the PATIENT (askable), not the treatment/outcome (leaky).
ASKABLE = [
    "Symptoms", "Symptom Severity", "Medical History", "Current Medications",
    "Risk Factors", "Environmental Factors", "Sleep Patterns", "Stress Levels",
    "Physical Activity Levels", "Family History", "Dietary Habits",
    "Allergies (Food/Env)", "Seasonal Variation", "Age Group", "Gender",
    "Occupation and Lifestyle", "Cultural Preferences", "Doshas",
    "Constitution/Prakriti",
]
# Excluded as label leakage: Diagnosis & Tests, Ayurvedic Herbs, Formulation,
# Medical Intervention, Prognosis, Complications, Prevention, Diet/Lifestyle Recs,
# Yoga, Herbal Remedies, Patient Recommendations, Duration of Treatment, names.

def toks(s):
    if pd.isna(s):
        return []
    return [t.strip().lower() for t in re.split(r"[,;/]| and ", str(s)) if t.strip()]

# Build feature vocabulary namespaced by column so "low" in Activity != "low" elsewhere.
rows = []
for _, r in df.iterrows():
    feats = set()
    for c in ASKABLE:
        for t in toks(r[c]):
            feats.add(f"{c}::{t}")
    rows.append(feats)

vocab = collections.Counter(f for s in rows for f in s)
# Drop singleton features: a feature seen in exactly one row is a giveaway ID, not a symptom.
KEEP_MIN = 2
feats = sorted([f for f, c in vocab.items() if c >= KEEP_MIN])
fidx = {f: i for i, f in enumerate(feats)}
D, K = len(rows), len(feats)
M = np.zeros((D, K), dtype=np.float64)
for i, s in enumerate(rows):
    for f in s:
        if f in fidx:
            M[i, fidx[f]] = 1.0

print(f"conditions D = {D}")
print(f"features   K = {K}   (kept features present in >= {KEEP_MIN} conditions)")
print(f"dropped singleton features: {sum(1 for c in vocab.values() if c < KEEP_MIN)} of {len(vocab)}")
print(f"mean positive features per condition: {M.sum(1).mean():.2f}")
print(f"mean conditions per feature:          {M.sum(0).mean():.2f}")

# Generative model: P(f=1 | d) = 1-eps if listed else eps
EPS = 0.05
P1 = np.where(M > 0, 1 - EPS, EPS)          # D x K
LOG1, LOG0 = np.log(P1), np.log(1 - P1)

def simulate(policy, n_cases=2000, budget=10, seed=0):
    rng = np.random.default_rng(seed)
    acc_at = np.zeros(budget + 1)
    for _ in range(n_cases):
        d_true = rng.integers(D)
        truth = (rng.random(K) < P1[d_true]).astype(int)   # sampled patient
        logb = np.zeros(D)                                  # uniform prior
        asked = np.zeros(K, dtype=bool)
        b = np.exp(logb - logb.max()); b /= b.sum()
        acc_at[0] += (b.argmax() == d_true)
        for t in range(budget):
            k = policy(b, asked, rng)
            asked[k] = True
            o = truth[k]
            logb = logb + (LOG1[:, k] if o else LOG0[:, k])
            b = np.exp(logb - logb.max()); b /= b.sum()
            acc_at[t + 1] += (b.argmax() == d_true)
    return acc_at / n_cases

def pol_random(b, asked, rng):
    cand = np.flatnonzero(~asked)
    return rng.choice(cand)

def pol_freq(b, asked, rng):
    """B5: greedy most-frequent unasked feature (no belief, no EIG)."""
    s = M.sum(0).copy(); s[asked] = -1
    return int(s.argmax())

def pol_eig(b, asked, rng):
    """B7 core: maximise expected entropy reduction under current belief."""
    p1 = b @ P1                       # P(o=1) per feature
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    # Expected posterior entropy == H(b) - I; maximise mutual information I(D; o_k)
    # I = H(o_k) - E_d[H(o_k|d)]
    Ho = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    Hcond = b @ (-(P1 * np.log(P1) + (1 - P1) * np.log(1 - P1)))
    mi = Ho - Hcond
    mi[asked] = -np.inf
    return int(mi.argmax())

BUDGET = 10
print(f"\n{'':22s}" + "".join(f"q={i:<5d}" for i in range(BUDGET + 1)))
res = {}
for name, pol in [("B3 random", pol_random), ("B5 greedy-frequency", pol_freq), ("B7 max-EIG", pol_eig)]:
    a = simulate(pol, budget=BUDGET)
    res[name] = a
    print(f"{name:22s}" + "".join(f"{100*x:<6.1f}" for x in a))

print(f"\nB4 prior-only (0 questions) accuracy = {100*res['B7 max-EIG'][0]:.2f}%  (chance = {100/D:.2f}%)")
print(f"\nEIG vs greedy-frequency at q=5 : {100*(res['B7 max-EIG'][5]-res['B5 greedy-frequency'][5]):+.1f} pts")
print(f"EIG vs random           at q=5 : {100*(res['B7 max-EIG'][5]-res['B3 random'][5]):+.1f} pts")
print(f"EIG vs greedy-frequency at q=10: {100*(res['B7 max-EIG'][10]-res['B5 greedy-frequency'][10]):+.1f} pts")
