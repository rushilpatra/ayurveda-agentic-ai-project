"""Measure every candidate join between AyurGenixAI and Amidha.

Three candidate bridges:
  J1  AyurGenixAI.Disease            <-> Amidha.main_indications   (English <-> Sanskrit)
  J2  AyurGenixAI.Doshas             <-> Amidha.pacify/aggravate   (shared dosha vocab)
  J3  AyurGenixAI."Ayurvedic Herbs"  <-> Amidha.name/synonyms      (herb names)
"""
import json, re, collections
import pandas as pd

df = pd.read_csv("data/raw/ayurgenix/AyurGenixAI_Dataset.csv")
herbs = json.load(open("data/raw/amidha_herbs/herb-database-main/herb.json"))

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def toks(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in re.split(r"[,;/]", str(s)) if t.strip()]

print("=" * 72)
print("J1  Disease names  <->  Amidha indications")
print("=" * 72)
ag_dis = {norm(d) for d in df.Disease.dropna().unique()}
am_ind = {norm(i) for h in herbs for i in h.get("main_indications", [])}
print(f"AyurGenixAI unique diseases : {len(ag_dis)}")
print(f"Amidha unique indications   : {len(am_ind)}")
inter = ag_dis & am_ind
print(f"EXACT string overlap        : {len(inter)}  -> {sorted(inter) if inter else '(none)'}")
print(f"coverage of AyurGenixAI     : {100*len(inter)/len(ag_dis):.1f}%")

print()
print("=" * 72)
print("J2  Dosha vocabulary")
print("=" * 72)
ag_dosha = collections.Counter()
for v in df.Doshas.dropna():
    for t in toks(v):
        ag_dosha[norm(t)] += 1
am_dosha = collections.Counter()
for h in herbs:
    for t in (h.get("pacify") or []) + (h.get("aggravate") or []):
        am_dosha[norm(t)] += 1
print("AyurGenixAI Doshas tokens:", dict(ag_dosha))
print("Amidha pacify/aggravate  :", dict(am_dosha))
print("shared tokens            :", sorted(set(ag_dosha) & set(am_dosha)))
# canonicalised dosha SETS
canon = df.Doshas.dropna().apply(lambda v: tuple(sorted({norm(t) for t in toks(v)})))
print(f"raw Doshas strings={df.Doshas.nunique()}  ->  canonicalised sets={canon.nunique()}")
print("canonical sets:", collections.Counter(canon).most_common())

print()
print("=" * 72)
print("J3  Herb names   AyurGenixAI 'Ayurvedic Herbs'  <->  Amidha")
print("=" * 72)
am_names = {}
for h in herbs:
    am_names[norm(h["name"])] = h["name"]
    for syn in h.get("sanskrit_synonyms", []):
        am_names.setdefault(norm(syn), h["name"])
    if h.get("english_name"):
        am_names.setdefault(norm(h["english_name"]), h["name"])
print(f"Amidha lookup keys (name + synonyms + english): {len(am_names)}")

for col in ["Ayurvedic Herbs", "Herbal/Alternative Remedies"]:
    mentions = collections.Counter()
    for v in df[col].dropna():
        for t in toks(v):
            mentions[norm(t)] += 1
    hit = {k: c for k, c in mentions.items() if k in am_names}
    miss = {k: c for k, c in mentions.items() if k not in am_names}
    tot_m = sum(mentions.values())
    tot_h = sum(hit.values())
    print(f"\n--- {col} ---")
    print(f"  distinct herb strings : {len(mentions)}   total mentions: {tot_m}")
    print(f"  MATCHED to Amidha     : {len(hit)} distinct ({100*len(hit)/len(mentions):.1f}%)"
          f" | {tot_h} mentions ({100*tot_h/tot_m:.1f}%)")
    print(f"  top matched  : {sorted(hit.items(), key=lambda x:-x[1])[:12]}")
    print(f"  top UNmatched: {sorted(miss.items(), key=lambda x:-x[1])[:15]}")

# rows fully covered?
def row_cov(v):
    ts = [norm(t) for t in toks(v)]
    if not ts:
        return None
    return sum(t in am_names for t in ts) / len(ts)
cov = df["Ayurvedic Herbs"].apply(row_cov).dropna()
print(f"\nrows with >=1 herb resolvable to Amidha: {(cov>0).sum()}/{len(cov)} ({100*(cov>0).mean():.1f}%)")
print(f"rows with ALL herbs resolvable         : {(cov==1).sum()}/{len(cov)} ({100*(cov==1).mean():.1f}%)")
