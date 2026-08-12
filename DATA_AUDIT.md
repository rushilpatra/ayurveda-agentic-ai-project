# Dataset Audit — Phase 2

Audit date: 2026-07-27. Machine: Apple M4, 16 GB unified memory, macOS 15.6.1.
All figures below are measured from actual downloads, not from dataset descriptions.

## Summary

| # | Dataset | Access | Status | Measured size |
|---|---------|--------|--------|---------------|
| 1 | BhashaBench-Ayur | HF, `gated: auto` | ✅ **downloaded** (gate accepted 2026-07-27) | 14,963 questions (9,348 en + 5,615 hi) |
| 2 | AyurGenixAI | Kaggle API | ❌ **no credentials on this machine** | unknown |
| 3 | Amidha Herb DB v2.0 | figshare, open | ✅ **downloaded** | 360 herbs, 18 fields |
| 4 | AyurKOSH | IEEE DataPort | ❌ **paid subscription required** | ~813 KB xlsx + XML (per listing) |
| 5 | Ayurveda-LLM-dataset | HF, public | ✅ **downloaded** | 1,529 QA records |

**3 of 5 acquired. The remaining blocker on the critical path is AyurGenixAI.**

---

## 1. BhashaBench-Ayur — ACQUIRED ✅

- Repo: `bharatgenai/BhashaBench-Ayur`, license **CC-BY-4.0**, last modified 2025-10-30
- Paper: arXiv:2510.25409 (BhashaBench V1)
- Local: `data/raw/bhashabench_ayur/{English,Hindi}_test.parquet`
- Gate was accepted 2026-07-27; `pd.read_parquet("hf://datasets/...")` now resolves via the
  stored HF token (user `rushilpatra`).

**Measured: 9,348 English + 5,615 Hindi = 14,963.** Exactly matches the published claim.

Schema — 12 columns, identical across both splits, **zero nulls and zero empty strings**:
`id`, `question`, `option_a`…`option_d`, `correct_answer`, `question_type`,
`question_level`, `topic`, `subject_domain`, `language`.

| | English | Hindi |
|---|---|---|
| MCQ | 9,212 | 5,505 |
| Fill in the blanks | 98 | 80 |
| Match the column | 26 | 15 |
| Assertion or Reasoning | 12 | 15 |
| Easy / Medium / Hard | 5,083 / 3,893 / 372 | 2,861 / 2,421 / 333 |
| distinct topics | 201 | 171 |
| subject domains | 16 | 16 |

Answer keys are near-uniform (EN: A 2,696 / B 2,324 / C 2,173 / D 2,155), so there is no
majority-class shortcut. **Hard questions are only 4.0% (EN) and 5.9% (HI)** — report
stratified by `question_level`, since an aggregate number is dominated by Easy items.

16 subject domains, identical set in both languages. Largest: Kayachikitsa (1,934 en / 1,200
hi), Dravyaguna & Bhaishajya (1,882 / 1,090), Samhita & Siddhanta (880 / 661).

### ⚠️ The two splits are NOT parallel

Measured: **0 shared `id`s**, different sizes, and per-domain ratios ranging from 0.36
(Roga Vigyana) to 1.04 (Ayurvedic Literature & History). Only 135 of 201 topics are shared.

**The Hindi set is a disjoint set of questions, not a translation of the English set.**

Consequence: **per-item paired English↔Hindi comparison is impossible.** Any bilingual claim
must be domain-stratified (compare within each of the 16 shared `subject_domain` values and
aggregate), and must state that the comparison is between different question populations.
Reporting two raw means side by side would confound language with question difficulty and
topic mix. See the corrected metric definition in `POSITIONING.md` §6.

Script check: 93.8% of Hindi questions contain Devanagari (the remaining 6.2% are
transliterated or use Latin-script technical terms); 0.1% of English questions do.

**Other notes:**
- Only a `test` split exists — no train/dev. The dataset is **evaluation-only**; any tuning
  must carve a declared held-out slice.
- **350 duplicate question strings in the English split** (ids are unique, so these are
  genuine near-duplicates). Deduplicate before reporting, and state the count.

## 2. AyurGenixAI — ACQUIRED ✅ (2026-07-27)

- Source: kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset, dated 2025-04-15
- Local: `data/raw/ayurgenix/AyurGenixAI_Dataset.{csv,xlsx}`
- **No Kaggle credentials are required.** `kagglehub.dataset_download(...)` fetches this
  public dataset anonymously:

  ```python
  import kagglehub
  path = kagglehub.dataset_download("kagglekirti123/ayurgenixai-ayurvedic-dataset")
  ```

  Verified 2026-07-27: the download is **byte-identical** to the copy in this repo
  (`md5 993e0e5ac59e3b71c2b3f65f59b30b60`), and the API reports **`versions/1`** — the only
  version that exists. Combined with the `.csv`/`.xlsx` cell-by-cell comparison below, the
  446-row table is confirmed as the complete and current dataset.

### ⚠️ Two major corrections to the project brief

**1. It is 446 rows × 34 columns, not "15,160 records × 35 parameters".**
The brief overstates the row count by ~34×. Measured: 446 rows, 367 unique disease names
(some diseases recur 2–4× as variants — Asthma ×4, Hypertension ×3, Tuberculosis ×3).

*Where the wrong number came from — resolved.* **446 × 34 = 15,164.** The "15,160 records"
figure is the **cell count**, not a row count; "447 diseases" ≈ 446 rows and "35 parameters"
≈ 34 columns. The `.xlsx` and `.csv` in the Kaggle archive were compared cell by cell:
identical shape, identical columns, **0 differences across all 15,164 cells** (14,764
non-null). There is no larger version of this dataset to find — 446 rows is all of it.
Do not repeat the 15,160 figure in the paper.

**2. It is a disease reference table, not patient records.**
Each row is a canonical textbook description of one disease, not an observation of one
patient. Row 0 is "Cough → Sore throat, chest congestion → Vata, Kapha → …". There are no
individuals, no per-patient variation, no base rates, no comorbidity.

Consequence for the design: the brief's plan to "convert existing static datasets into an
interactive patient environment" by hiding attributes of real cases **cannot be done as
written — there are no cases to hide.** What this table supports instead is a *generative
model*: sample a condition, then sample attribute observations conditioned on it. See
`POSITIONING.md` §10.

### Columns

Complete at 100% except: `Allergies (Food/Env)` 21.5%, `Herbal/Alternative Remedies` 96.6%,
`Ayurvedic Herbs` 97.8%, `Yoga & Physical Therapy` 96.9%, `Prevention` 98.0%,
`Medical History` / `Family History` 99.8%.

**Askable (patient-side) — 19 columns.** Symptoms, Symptom Severity, Medical History,
Current Medications, Risk Factors, Environmental Factors, Sleep Patterns, Stress Levels,
Physical Activity Levels, Family History, Dietary Habits, Allergies, Seasonal Variation,
Age Group, Gender, Occupation and Lifestyle, Cultural Preferences, Doshas,
Constitution/Prakriti.

**Label-leaking — must be excluded from the observation space.** Diagnosis & Tests,
Ayurvedic Herbs, Formulation, Medical Intervention, Prognosis, Complications, Prevention,
Diet and Lifestyle Recommendations, Yoga & Physical Therapy, Herbal/Alternative Remedies,
Patient Recommendations, Duration of Treatment, Hindi Name, Marathi Name.

Clean categoricals: `Doshas` 9 values (Pitta 132, Vata+Pitta 117, Vata 62, …),
`Constitution/Prakriti` 12 values, `Symptom Severity` 7, `Sleep Patterns` 8,
`Stress Levels` 7, `Physical Activity Levels` 9.

⚠️ Dosha labels are **not normalised**: `"Vata, Kapha"` and `"Kapha, Vata"` are separate
strings, as are `"Pitta-Kapha"` / `"Kapha-Pitta"` in Prakriti. Canonicalise to sorted sets
before use, or the 9 dosha values collapse to 6 and the 12 prakriti values to ~7.

### The `Symptoms` column alone is too thin to support the benchmark

- **3.24 symptoms per row** (min 1, max 5; 238 rows have exactly 3)
- 465 unique symptom tokens, 1,443 mentions
- **65.6% of symptom tokens appear in exactly one row** — they are near-unique disease IDs
- median number of rows sharing a given symptom = **1**
- 414 of 446 symptom sets are unique

Used alone, this makes diagnosis degenerate: two-thirds of symptoms identify the disease
outright, and a disease only *has* ~3 symptoms, so the information budget is exhausted in
three questions. There would be no sequential planning problem.

**The fix — and it works — is to use all 19 askable columns**, not just `Symptoms`. That
yields 23.3 features per condition over a 980-feature space, with each feature shared by
10.6 conditions on average. See the measured viability result in `POSITIONING.md` §10.
This is also the more faithful design: Ayurvedic consultation genuinely does weigh season,
diet, sleep, occupation and prakriti alongside presenting symptoms.

⚠️ **No license is stated on the Kaggle page metadata we could reach.** Confirm terms before
redistributing any derived benchmark.

### ⚠️ Hidden nulls — `"none specific"` is a placeholder, not a value

Pandas reports `Ayurvedic Herbs` at 97.8% non-null. That is misleading: **256 of 446 rows
(57%) contain the literal string `"none specific"`.**

| Column | NaN | `"none specific"` | **Effective coverage** |
|---|---|---|---|
| Ayurvedic Herbs | 10 | **256** | **40.4%** (not 97.8%) |
| Herbal/Alternative Remedies | 15 | **276** | **34.8%** (not 96.6%) |
| Allergies (Food/Env) | 350 | 0 | 21.5% |
| Medical History / Family History | 1 | 0 | 99.8% |

Any preprocessing must treat `"none specific"` as null. Left as a string it becomes the single
most frequent "herb" in the corpus and will dominate any feature or embedding built from that
column.

---

## 6. Cross-dataset joins — measured

AyurKOSH is unavailable, so the symbolic KG must be assembled by joining the datasets we hold.
Four candidate bridges were tested (`experiments/` scripts).

### J1 · disease name ↔ Amidha indication — ❌ **dead**

363 AyurGenixAI disease names vs 202 Amidha indications: **exactly 1 exact match** (`malaria`),
0.3% coverage. The vocabularies are disjoint by construction — Amidha uses classical Sanskrit
nosology (Jwara, Kasa, Prameha, Amavata), AyurGenixAI uses English biomedical names (Fever,
Cough, Diabetes, Rheumatoid Arthritis).

A mapping layer is required. Many correspondences are well established in the literature
(Jwara→Fever, Kasa→Cough, Prameha→Diabetes, Kamala→Jaundice, Atisara→Diarrhoea,
Arsha→Haemorrhoids, Amavata→Rheumatoid arthritis, Medoroga→Obesity, Hridroga→Heart disease),
so a curated table covering the ~91 indications with ≥3 herbs is tractable — roughly a day of
careful work, and it should be published as an artefact in its own right. **Budget for it; do
not assume string matching will do it.**

### J2 · dosha vocabulary — ✅ **clean, exact**

`vata` / `pitta` / `kapha` appear in both, exactly. AyurGenixAI: vata 252, pitta 307,
kapha 119. Amidha pacify/aggravate: vata 342, pitta 347, kapha 341 (+9 `tridosha`).
Canonicalising to sorted sets collapses **9 raw strings → 6 dosha sets**
(Vata+Pitta 133, Pitta 132, Vata 62, Kapha+Vata 57, Kapha+Pitta 42, Kapha 20).

This is the reliable bridge between the two datasets.

### J3 · herb names — ⚠️ **partial**

AyurGenixAI `Ayurvedic Herbs` has only **43 distinct strings**. Matching against an Amidha
lookup of 1,844 keys (name + Sanskrit synonyms + English name): **21 distinct matched (48.8%),
292 of 628 mentions (46.5%)**. Excluding the `"none specific"` placeholder, real coverage is
substantially better than that figure suggests.

Top matches: ashwagandha (100), brahmi (34), turmeric (34), amla (20), guduchi (19),
guggulu (14), tulsi (13), arjuna (12), shatavari (11).
Real misses worth normalising: **neem (20), triphala (14), ginger (10), guggul (9)** —
note `guggul` vs Amidha's `guggulu` is a pure spelling variant, so a fuzzy/alias pass will
recover several of these cheaply.

`Herbal/Alternative Remedies` is much weaker (10% distinct, 12.8% of mentions) and contains
substantial non-Ayurvedic content (fish oil, tea tree oil, "lifestyle changes", "diet
changes"). **Use `Ayurvedic Herbs`, not this column.**

### J4 · two-hop `condition → dosha → herb` — ✅ **works, and is evaluable**

This is the finding that replaces AyurKOSH. Scoring all 360 Amidha herbs by dosha-pacification
overlap with the condition's dosha profile, evaluated against the herbs AyurGenixAI *directly*
lists for that condition:

| | value |
|---|---|
| evaluable conditions | 139 |
| candidate pool | 360 herbs |
| **Hit@1** | **5.0%** |
| **Hit@5** | **7.9%** |
| **Hit@10** | **15.1%** |
| Hit@20 | 33.8% |
| random Hit@10 | 2.8% |
| median rank of best gold herb | 32 |
| MRR | 0.096 |

**~5.4× better than random at Hit@10.** The dosha path carries real signal, but a
weak one.

> ⚠️ **Correction — an earlier draft of this file reported Hit@10 = 36.1%. That number was
> wrong, and the reason is worth recording.** The first implementation ranked herbs with
> `np.argsort(-scores, kind="stable")`. Dosha-overlap scoring produces only a handful of
> distinct score levels, so **ties dominate**, and a stable sort leaves tied herbs in
> *Amidha file order*. The gold herbs are not uniformly distributed in that file — their
> median position is **79 of 360**, against 180 expected by chance (Tulsi 0, Amla 1,
> Ashwagandha 2, Giloy 3, Shatavari 4). The file is ordered roughly by prominence, and the
> gold labels are the prominent herbs, so ties were being resolved in favour of the correct
> answer. The measured effect was to inflate Hit@10 by ~2.4×.
>
> **Rule for this project: never break ranking ties by source order.** Break them
> alphabetically (as `kg/graph.py` now does) or at random with a fixed seed, and report
> which. Any retrieval or KG metric computed with a stable sort over a prominence-ordered
> corpus is suspect.

Two things make this valuable rather than merely adequate:

1. **It is a genuine link-prediction task with ground truth we did not construct.** The direct
   `condition → herb` edges act as gold labels for the two-hop symbolic inference. That gives
   the "KG reasoning quality" metric a real, defensible definition instead of an LLM judge.
2. **There is large headroom, and the failure mode is diagnostic.** Hit@5 == Hit@10 because
   dosha-overlap scoring is coarse — it produces only a handful of distinct score levels, so
   ties dominate. Refining with rasa/guna/virya/vipaka compatibility, indication matching, or
   learned edge weights is exactly the kind of improvement a paper should demonstrate.

**Conclusion: the KG module does not depend on AyurKOSH.** Build the graph with a pluggable
source interface so AyurKOSH can be added later as an additional edge provider; when it
arrives it should improve J1 and add classical-text provenance, not require a rewrite.

## 3. Amidha Ayurveda Herb Database v2.0 — ACQUIRED ✅

- DOI 10.6084/m9.figshare.30491042.v3, **CC BY 4.0**, author Sparsh Varshney, published 2026-06-07
- Also mirrored: github.com/sciencewithsaucee-sudo/herb-database, datahub.io
- Local: `data/raw/amidha_herbs/herb-database-main/herb.json`

**Correction to the project brief:** this is **360 records, not "360–700+"**. v2.0
deliberately consolidated duplicates and removed formulation entries to produce canonical
plant records. Cite 360.

Schema (18 fields, coverage measured over all 360 records):

| Field | Type | Coverage |
|---|---|---|
| `name`, `botanical_name`, `family`, `english_name` | str | 100% |
| `sanskrit_synonyms`, `part_used`, `main_indications` | list | 100% |
| `rasa`, `guna`, `prabhav` | list | 100% |
| `virya`, `vipaka` | str | 100% |
| `tridosha` | bool | 100% |
| `pacify` | list | 99.7% |
| `aggravate` | list | 97.5% |
| `preview`, `link` | str | 100% |
| `image` | str | mostly empty |

Controlled vocabularies (complete, measured):

- **Rasa** — Tikta 211, Kashaya 114, Katu 111, Madhura 108, Amla 20, Lavana 1
- **Virya** — Ushna 192, Sheeta 168
- **Vipaka** — Katu 236, Madhura 114, Amla 10
- **Guna** — Laghu 257, Ruksha 133, Snigdha 99, Guru 86, Tikshna 59, Sara 2
- **Dosha** — pacify: Kapha 235, Vata 201, Pitta 192, tridosha 9; aggravate: Pitta 155, Vata 141, Kapha 106

These vocabularies are small and closed. **The herb-verification module can be built
entirely from this file, today, with no LLM.**

**Unexpected bonus — a usable partial graph.** `main_indications` yields:

- **1,738 herb → indication edges**
- **205 unique indication terms** (Sanskrit: Raktapitta 107, Kushta 88, Kasa 62, Jwara 58,
  Vrana 58, Krimi 55, Shwasa 54, Mutrakrichha 53, …)
- 4.8 indications per herb on average
- long tail: 91 indications have ≥3 herbs; **85 have exactly 1 herb**

So a `herb —treats→ disease` and `herb —pacifies/aggravates→ dosha` graph exists already.
What is missing is the other half — see the critical gap below.

## 4. AyurKOSH — ❌ NOT OBTAINED (confirmed 2026-08-02)

**Final status: unobtainable.** Institutional IEEE access was not available. This dataset
is excluded from the work and must be described in the paper as *unavailable*, never as
future work that was skipped. What it would have supplied — classical-text
Vyadhi→Lakshana triples — is partially covered by the two-hop dosha join (§6, J4), and the
resulting limitation is stated in `POSITIONING.md`.


- DOI 10.21227/58ej-wz87, created 2026-01-03
- Authors: Sharayu Mirasdar, Mangesh Bedekar, Harish Patankar, Yash Gujar
- Format: XML + Excel (.xlsx, 812.69 KB)
- Built from *Charaka Samhita* and *Bhaishajya Ratnawali*
- Page states verbatim: **"This dataset requires an IEEE DataPort Subscription to access."**

The abstract says it "can be used for academic and non-commercial research", but the
subscription paywall is the operative barrier. This is the only listed source of
**Vyadhi (disease) → Lakshana (symptom)** relations.

## 5. Ayurveda-LLM-dataset — ACQUIRED ✅

- `jaychedaa/Ayurveda-LLM-dataset`, public, ungated
- Local: `data/raw/ayurveda_llm/AYURVEDIC_DATASETFULL.json` (595 KB)
- **1,529 records**, perfectly uniform schema — every record has exactly:
  `question`, `Context_Cot`, `response`
- Content is textbook exposition sourced largely from *Sushruta Samhita*

⚠️ **No license is declared on the HF repo** (no `license:` tag, no LICENSE file). For a
Nature-family or ACL-family submission this is a real problem — provenance and terms are
unstated. Either contact the uploader, or use it only as an unshipped auxiliary and do not
redistribute.

Also note: at 1,529 records this is far too small for meaningful instruction tuning. Its
realistic role is as a small retrieval corpus, not a training set.

---

## The critical gap

The interactive-diagnosis design requires a **disease → symptom** relation. It is what
makes the posterior update meaningful and what makes a follow-up question discriminative:
you ask about a symptom precisely because diseases disagree about it.

Of the five datasets:

- Amidha gives **herb → disease** and **herb → dosha**. Not disease → symptom.
- BhashaBench-Ayur is exam MCQs. Not a relational resource.
- Ayurveda-LLM is free-text QA. Not a relational resource.
- **AyurGenixAI** was to supply per-case symptom sets → **blocked (credentials)**
- **AyurKOSH** was to supply Vyadhi → Lakshana → **blocked (paywall)**

**Both sources of the one relation the project depends on are currently inaccessible.**
Everything successfully downloaded is peripheral: herb pharmacology and text.

This does not sink the project — AyurGenixAI is a routine credential fix and is the more
important of the two — but no amount of code written now can substitute for it.

## Unblock actions (in priority order)

1. **Kaggle** — kaggle.com/settings → API → "Create New Token" → save `kaggle.json` to
   `~/.kaggle/kaggle.json`, then `chmod 600`. Unblocks the patient cases.
   **This is now the only blocker on the critical path.**
2. ~~**HuggingFace** — accept the BhashaBench-Ayur gate.~~ ✅ **Done 2026-07-27.**
3. ~~**AyurKOSH**~~ — ❌ **CLOSED 2026-08-02: confirmed unobtainable.** Institutional access
   was not available and the dataset stays behind the IEEE DataPort paywall. It is treated
   as permanently absent, not pending. The two-hop dosha join (J4 below) replaces it; the
   provider stays wired in and inert should it ever become available. *Original options,
   retained for the record:*
   a. check institutional IEEE access through the university library proxy;
   b. email the corresponding author (Mirasdar / Bedekar, MIT-WPU) — authors of recently
      published DataPort sets commonly share for academic use on request;
   c. proceed without it and rebuild the disease→symptom layer from AyurGenixAI alone,
      documenting AyurKOSH as unavailable. This weakens but does not remove the
      "symbolic KG reasoning" contribution, since Amidha still supports the herb and
      dosha subgraphs.
