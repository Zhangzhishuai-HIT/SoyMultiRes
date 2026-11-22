"""Inventory of the 33 GRIN stress descriptors against SoySNP50K (read-only; no modelling).

usage: python -I inventory_grin_stress.py <grin_dir> <soysnp50k_samples.tsv> <out_dir>

Accessions are matched to SoySNP50K by the raw VCF sample id (column vcf_id), which keeps GRIN suffixes such
as "PI407999-1" (HapGAM's canonical id drops the hyphen and loses ~300 such accessions).
Maturity group comes from the GRIN MATGROUP descriptor, region from the GRIN origin field.

Writes to <out_dir>:
  trait_inventory.tsv      one row per trait: records, accessions, genotyped (max/soja), methods, value levels
  trait_methods.tsv        one row per trait x evaluation method: records, levels, original_value range, scale note
  trait_overlap.tsv        genotyped accessions measured for both traits (33 x 33)
  trait_mg.tsv             maturity-group distribution of genotyped accessions per trait
  trait_origin.tsv         region-of-origin distribution per trait
  replicate_consistency.tsv accessions with >1 record per trait and how often the codes disagree
  long_records.tsv.gz      all genotyped records (accession, trait, method, value, original_value, taxon)
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

GRIN, SAMPLES, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
STRESS_IDS = [51023, 51025, 51033, 51034, 51040, 51054, 51074, 51085, 51091, 51094, 51097, 51099, 51103, 51107,
              51108, 51110, 51111, 51128, 51129, 51131, 51132, 51133, 51136, 51137, 51138, 51139, 51140, 51142,
              51143, 51144, 51150, 51151, 51153]  # the stress descriptors with >= 500 genotyped accessions

geno = set(pd.read_csv(SAMPLES, sep="\t", dtype=str).vcf_id)
MG_NUM = {"000": -3, "00": -2, "0": -1, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
          "IX": 9, "X": 10}


def mg_group(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "MG_NA"
    return "MG000-0" if v <= -1 else "MGI-II" if v <= 2 else "MGIII-IV" if v <= 4 else "MGV-VI" if v <= 6 else "MGVII-X"


def region(o):
    o = str(o)
    if o.startswith("China"):
        return "China"
    if o.startswith("Korea"):
        return "Korea"
    if o.startswith("Japan"):
        return "Japan"
    if o.startswith(("United States", "Canada", "Brazil", "Argentina", "Mexico", "Paraguay")):
        return "Americas"
    if o.startswith(("Russia", "Kazakhstan", "Uzbekistan", "Georgia", "Ukraine", "Moldova")):
        return "Russia/C.Asia"
    if o.startswith(("India", "Nepal", "Indonesia", "Vietnam", "Thailand", "Taiwan", "Philippines", "Pakistan",
                     "Myanmar", "Malaysia", "Sri Lanka", "Bhutan", "Cambodia", "Laos")):
        return "S/SE Asia"
    if o in ("nan", "", "Unknown"):
        return "Unknown"
    return "Other"




def acc_id(df):
    return (df.accession_prefix.fillna("") + df.accession_number.fillna("") + df.accession_suffix.fillna("")).str.replace(" ", "")


mgdf = pd.read_csv(os.path.join(GRIN, "GRIN_crop51_51055_MATGROUP.csv"), skiprows=2, dtype=str)
mgdf["acc"] = (mgdf.accession_prefix.fillna("") + mgdf.accession_number.fillna("") + mgdf.accession_suffix.fillna("")).str.replace(" ", "")
mgdf["mgv"] = mgdf.observation_value.map(MG_NUM)
mg_acc = mgdf.groupby("acc").mgv.median()

frames, rows, mrows = [], [], []
for f in sorted(os.listdir(GRIN)):
    m = re.match(r"GRIN_crop51_(\d+)_(.+)\.csv$", f)
    if not m or int(m.group(1)) not in STRESS_IDS:
        continue
    code = m.group(2)
    df = pd.read_csv(os.path.join(GRIN, f), skiprows=2, dtype=str)
    df["acc"] = acc_id(df)
    df["trait"] = code
    df["geno"] = df.acc.isin(geno)
    g = df[df.geno]
    tax = Counter(g.drop_duplicates("acc").taxon)
    rows.append(dict(trait=code, records=len(df), accessions=df.acc.nunique(), genotyped=g.acc.nunique(),
                     g_max=sum(v for k, v in tax.items() if str(k).startswith("Glycine max")),
                     g_soja=sum(v for k, v in tax.items() if str(k).startswith("Glycine soja")),
                     g_other=sum(v for k, v in tax.items() if not str(k).startswith(("Glycine max", "Glycine soja"))),
                     methods=g.method_name.nunique(), levels=g.observation_value.nunique(),
                     values=",".join(f"{k}:{n}" for k, n in Counter(g.observation_value).most_common(8)),
                     orig_numeric=pd.to_numeric(g.original_value, errors="coerce").notna().mean().round(3),
                     rec_per_acc_max=g.groupby("acc").size().max()))
    for meth, d in g.groupby("method_name"):
        ov = pd.to_numeric(d.original_value, errors="coerce")
        notes = d.note.dropna().unique()
        mrows.append(dict(trait=code, method=meth, records=len(d), accessions=d.acc.nunique(),
                          values=",".join(f"{k}:{n}" for k, n in Counter(d.observation_value).most_common(8)),
                          orig_frac_numeric=round(ov.notna().mean(), 3),
                          orig_min=ov.min(), orig_max=ov.max(), n_notes=len(notes),
                          note=(notes[0][:160] if len(notes) else "")))
    frames.append(g[["acc", "trait", "method_name", "observation_value", "original_value", "taxon", "origin"]])

inv = pd.DataFrame(rows).sort_values("genotyped", ascending=False)
inv.to_csv(f"{OUT}/trait_inventory.tsv", sep="\t", index=False)
pd.DataFrame(mrows).to_csv(f"{OUT}/trait_methods.tsv", sep="\t", index=False)
long = pd.concat(frames, ignore_index=True)
long.to_csv(f"{OUT}/long_records.tsv.gz", sep="\t", index=False)

# replicate consistency (same accession, same trait, >1 record)
rc = []
for t, d in long.groupby("trait"):
    multi = d.groupby("acc").observation_value.agg(lambda s: s.nunique())
    n_multi = (d.groupby("acc").size() > 1).sum()
    rc.append(dict(trait=t, acc_multi_records=int(n_multi), acc_conflicting_codes=int((multi > 1).sum())))
pd.DataFrame(rc).to_csv(f"{OUT}/replicate_consistency.tsv", sep="\t", index=False)

# overlap matrix
traits = list(inv.trait)
sets = {t: set(long.acc[long.trait == t]) for t in traits}
ov = pd.DataFrame([[len(sets[a] & sets[b]) for b in traits] for a in traits], index=traits, columns=traits)
ov.to_csv(f"{OUT}/trait_overlap.tsv", sep="\t")

# coverage per accession
per_acc = long.drop_duplicates(["acc", "trait"]).groupby("acc").size()
cov = {k: int((per_acc >= k).sum()) for k in (1, 2, 3, 5, 10, 15, 20)}
print("accessions with >=k stress traits:", cov)
print("cells observed:", int(per_acc.sum()), "of", len(per_acc) * len(traits),
      f"({per_acc.sum() / (len(per_acc) * len(traits)):.3%} of accessions-with->=1 x {len(traits)})")

# MG / origin per trait (HapGAM accession table covers G. max analysis set only)
lk = long.drop_duplicates(["acc", "trait"]).copy()
lk["mg_group"] = lk.acc.map(mg_acc).map(mg_group)
lk["region"] = lk.origin.map(region)
mg = lk.pivot_table(index="trait", columns="mg_group", values="acc", aggfunc="count", fill_value=0)
mg = mg[[c for c in ["MG000-0", "MGI-II", "MGIII-IV", "MGV-VI", "MGVII-X", "MG_NA"] if c in mg.columns]]
mg.loc[traits].to_csv(f"{OUT}/trait_mg.tsv", sep="\t")
acc_tab = long.drop_duplicates("acc")[["acc", "taxon", "origin"]].copy()
acc_tab["mg"] = acc_tab.acc.map(mg_acc)
acc_tab["mg_group"] = acc_tab.mg.map(mg_group)
acc_tab["region"] = acc_tab.origin.map(region)
acc_tab["n_traits"] = acc_tab.acc.map(per_acc)
acc_tab.to_csv(f"{OUT}/accessions.tsv", sep="\t", index=False)
print("regions of accessions with >=1 trait:", acc_tab.region.value_counts().to_dict())
print("MG groups:", acc_tab.mg_group.value_counts().to_dict())
print("taxon:", acc_tab.taxon.value_counts().to_dict())
orig = lk.pivot_table(index="trait", columns="region", values="acc", aggfunc="count", fill_value=0)
orig.loc[[t for t in traits if t in orig.index]].to_csv(f"{OUT}/trait_origin.tsv", sep="\t")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(inv.to_string(index=False))
print(pd.DataFrame(rc).to_string(index=False))
print(mg.loc[traits].to_string())
