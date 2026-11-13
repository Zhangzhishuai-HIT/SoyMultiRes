"""Step 1 of the plan: harmonised phenotype tables for the 32 GRIN stress traits and the 5 Mendelian calibration traits.

usage: python -I build_pheno.py            (paths are fixed relative to the project root)

Writes data/processed/:
  pheno_long.tsv.gz       one row per accession x trait x evaluation batch: score (higher = more resistant), raw code(s),
                          resistant (0/1), n_records, original_value mean, year
  pheno_matrix.tsv.gz     accession x 32 traits: mean score over batches (empty = never tested)
  resistant_matrix.tsv.gz accession x 32 traits: 1 if resistant in any batch, 0 if tested and never resistant
  hightemp_quant.tsv      the five quantitative values stored in the HIGHTEMP notes (Smith et al. 2008)
  mendel_matrix.tsv.gz    accession x 5 Mendelian calibration traits (HapGAM's binary codings; maturity group numeric)
  accessions.tsv          all 20,087 genotyped accessions: vcf_id, taxon, origin, region, MG, n_stress_traits,
                          near-identical group (phenotyped accessions)
and docs/DATA_CARD.md. Exits non-zero if a per-trait count differs from the 2026-11-11 inventory or a known
resistance source does not score at the resistant end.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import traits as T  # noqa: E402

GRIN = ROOT / "data/raw/grin_descriptors"
SAMPLES = ROOT.parent / "HapGAM/data/interim/soysnp50k/samples.tsv"   # vcf_id list of the SoySNP50K VCF
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
REGION_MAP = {"China": "China", "Taiwan": "China", "Hong Kong": "China", "Korea, South": "Korea",
              "Korea, North": "Korea", "Korea": "Korea", "Japan": "Japan"}


def region(origin: str) -> str:
    o = str(origin)
    if o in REGION_MAP:
        return REGION_MAP[o]
    if o.startswith(("United States", "Canada", "Brazil", "Argentina", "Mexico", "Paraguay", "Colombia", "Peru")):
        return "Americas"
    if o.startswith(("Russia", "Soviet", "Kazakhstan", "Uzbekistan", "Georgia")):
        return "Russia/C.Asia"
    if o.startswith(("India", "Nepal", "Indonesia", "Vietnam", "Thailand", "Philippines", "Pakistan", "Myanmar",
                     "Malaysia", "Sri Lanka", "Bhutan", "Cambodia", "Laos", "Bangladesh")):
        return "S/SE Asia"
    if o in ("nan", "", "Unknown"):
        return "Unknown"
    return "Other"


def read(code: str) -> pd.DataFrame:
    f = next(GRIN.glob(f"GRIN_crop51_*_{code}.csv"))
    d = pd.read_csv(f, skiprows=2, dtype=str)
    d["acc"] = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
    return d


geno = pd.read_csv(SAMPLES, sep="\t", dtype=str).vcf_id
gset = set(geno)

# ------------------------------------------------------------------ 32 stress traits
rows = []
ht_quant = []
for t in T.TRAITS:
    d = read(t)
    d = d[d.acc.isin(gset)].copy()
    d["batch"] = d.method_name.str.strip()
    raw = d.observation_value.str.strip()
    if t in T.NUMERIC:
        val = pd.to_numeric(raw, errors="coerce")
        d["score"] = -val
    else:
        lv = T.levels(t)
        d["score"] = raw.map(lv).astype(float)
        d["resistant"] = raw.isin(T.resistant_codes(t)).astype(int)
    if d.score.isna().any():
        bad = Counter(raw[d.score.isna()])
        sys.exit(f"{t}: unmapped codes {dict(bad)}")
    d["code"] = raw
    d["orig"] = pd.to_numeric(d.original_value, errors="coerce")
    if t == "HIGHTEMP":
        for a, note in zip(d.acc, d.note.fillna("")):
            v = dict(re.findall(r"(Accelerated aging germination|Standard germination|Hard seed|Phomopsis|Wrinkled): (\d+)%", note))
            if v:
                ht_quant.append({"acc": a, **{k: int(x) for k, x in v.items()}})
    agg = d.groupby(["acc", "batch"]).agg(score=("score", "mean"), n_records=("score", "size"),
                                          code=("code", lambda s: ",".join(sorted(set(s)))),
                                          orig_mean=("orig", "mean")).reset_index()
    if t in T.NUMERIC:
        # least-damaged 10% of the batch (ties included) = resistant
        thr = agg.groupby("batch").score.transform(lambda s: s.quantile(1 - T.NUMERIC_TOP_FRACTION))
        agg["resistant"] = (agg.score >= thr).astype(int)
    else:
        agg["resistant"] = d.groupby(["acc", "batch"]).resistant.max().reindex(
            pd.MultiIndex.from_frame(agg[["acc", "batch"]])).to_numpy()
    agg["trait"] = t
    agg["year"] = agg.batch.map(T.batch_year)
    agg["taxon"] = agg.acc.map(d.drop_duplicates("acc").set_index("acc").taxon)
    rows.append(agg)
long = pd.concat(rows, ignore_index=True)[["acc", "trait", "batch", "year", "score", "resistant", "n_records", "code",
                                            "orig_mean", "taxon"]]
long.to_csv(OUT / "pheno_long.tsv.gz", sep="\t", index=False)
mat = long.groupby(["acc", "trait"]).score.mean().unstack("trait").reindex(columns=T.TRAITS)
res = long.groupby(["acc", "trait"]).resistant.max().unstack("trait").reindex(columns=T.TRAITS)
mat.to_csv(OUT / "pheno_matrix.tsv.gz", sep="\t")
res.to_csv(OUT / "resistant_matrix.tsv.gz", sep="\t")
pd.DataFrame(ht_quant).drop_duplicates("acc").to_csv(OUT / "hightemp_quant.tsv", sep="\t", index=False)

# ------------------------------------------------------------------ Mendelian calibration traits
men = {}
for code, (name, pos, neg) in T.MENDEL.items():
    d = read(code)
    d = d[d.acc.isin(gset)]
    v = d.observation_value.str.strip()
    lab = pd.Series(np.where(v.isin(pos), 1.0, np.where(v.isin(neg), 0.0, np.nan)), index=d.index)
    men[name] = lab.groupby(d.acc).mean().round()      # majority over records (ties -> 0.5 rounds to 0 or 1; rare)
d = read("MATGROUP")
d = d[d.acc.isin(gset)]
men["maturity_group"] = d.observation_value.str.strip().str.upper().map(T.MG_VALUE).groupby(d.acc).median()
mendel = pd.DataFrame(men)
mendel.to_csv(OUT / "mendel_matrix.tsv.gz", sep="\t")

# ------------------------------------------------------------------ accession table (all genotyped accessions)
meta = []
for f in sorted(GRIN.glob("GRIN_crop51_*.csv")):
    d = pd.read_csv(f, skiprows=2, dtype=str, usecols=["accession_prefix", "accession_number", "accession_suffix", "taxon", "origin"])
    d["acc"] = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
    meta.append(d[d.acc.isin(gset)][["acc", "taxon", "origin"]])
meta = pd.concat(meta).dropna(subset=["taxon"]).drop_duplicates("acc").set_index("acc")
acc = pd.DataFrame({"vcf_id": geno}).set_index("vcf_id", drop=False)
acc["taxon"] = meta.taxon.reindex(acc.index)
acc["origin"] = meta.origin.reindex(acc.index)
acc["region"] = acc.origin.map(region)
acc["mg"] = mendel.maturity_group.reindex(acc.index)
acc["mg_group"] = acc.mg.map(T.mg_group)
acc["n_stress_traits"] = mat.notna().sum(1).reindex(acc.index).fillna(0).astype(int)
ni = pd.read_csv(ROOT / "results/near_identical/near_identical_groups.tsv", sep="\t").set_index("acc")
acc["dup_group"] = ni.dup_group.reindex(acc.index)
acc["dup_group_size"] = ni.group_size.reindex(acc.index)
acc.to_csv(OUT / "accessions.tsv", sep="\t", index=False)

# ------------------------------------------------------------------ checks
fail = []
inv = pd.read_csv(ROOT / "results/inventory/trait_inventory.tsv", sep="\t").set_index("trait")
for t in T.TRAITS:
    n = int(mat[t].notna().sum())
    if n != int(inv.loc[t, "genotyped"]):
        fail.append(f"{t}: {n} accessions vs inventory {inv.loc[t, 'genotyped']}")
spot = [("PI88788", "NEMATCYST_3", "max"), ("PI88788", "NEMATCYST_14", "max"), ("PI88788", "NEMATCYST_5", "low"),
        ("PI548402", "NEMATCYST_1", "max"), ("PI548402", "NEMATCYST_5", "max"), ("PI437654", "NEMATCYST_4", "max"),
        ("PI229358", "MEXBEANBTL", "top"), ("PI229358", "LOOPER", "top")]
for a, t, want in spot:
    v = mat.loc[a, t]
    col = mat[t].dropna()
    ok = (v == col.max()) if want == "max" else (v <= col.quantile(0.9)) if want == "low" else (v >= col.quantile(0.9))
    if not ok:
        fail.append(f"direction check failed: {a} {t} = {v}")
print(f"long rows {len(long):,}; accessions with >=1 stress trait {int((acc.n_stress_traits > 0).sum()):,}; "
      f"cells {int(mat.notna().sum().sum()):,}; Mendelian rows {len(mendel):,}")
print("Mendelian counts:", {c: int(mendel[c].notna().sum()) for c in mendel})
print("HIGHTEMP quantitative rows:", len(ht_quant))

# ------------------------------------------------------------------ data card
nb = long.groupby("trait").batch.nunique()
prev = res.mean().round(4)
lines = ["# Data card (SoyMultiRes)", "",
         "Built by `scripts/01_data/build_pheno.py` (step 1 of the plan). Sources: GRIN-Global descriptor downloads in",
         "`down_data/GRIN_soybean_descriptors/` (MANIFEST.tsv; definition pages in `definitions_20261111/`) and the SoySNP50K",
         "VCF sample list (`down_data/LIS_glycine_max_diversity/Wm82.gnm2.div.Song_Hyten_2015`).", "",
         "## Rules", "",
         "- Accessions are matched by the raw VCF sample id (GRIN prefix + number + suffix without spaces, e.g. PI407999-1).",
         "- Every score is oriented so that **higher = more resistant/tolerant**. Categorical codes are ordinal levels "
         "(table below); numeric damage/severity scores are negated.",
         "- BRNSTEMROT U = susceptible in preliminary field tests (coded with S); PMV U = field-resistant, not confirmed "
         "(between S and R); PHYTOROT/PYTHIUMROT H = heterogeneous; NSCANKER/HIGHTEMP I = intermediate; BPMV/SALTREACT T = tolerant.",
         "- `resistant` = GRIN resistant classes for categorical traits; for numeric traits the least-damaged 10% of an "
         "evaluation batch (ties included; e.g. FROGEYE2 score 1 = 'resistant with no lesions' covers 66%).",
         "- Records of the same accession in the same batch are averaged; batches stay separate in `pheno_long.tsv.gz`.",
         "- RUST_TAN: numeric 2-5; direction assumed 'higher = more severe' (to be confirmed from the GRIN method page).",
         "- Mendelian calibration traits use HapGAM's binary codings (flower purple vs white, pubescence tawny vs gray, "
         "indeterminate vs determinate, seed coat pigmented vs yellow) and maturity group 000 = -2 ... X = 10.", "",
         "## 32 stress traits", "",
         "| Trait | Short | Category | Accessions | Batches | Levels (low → high resistance) | Resistant share |",
         "|---|---|---|---:|---:|---|---:|"]
for t in T.TRAITS:
    lv = "−score (numeric)" if t in T.NUMERIC else " < ".join(k for k, _ in sorted(T.levels(t).items(), key=lambda kv: kv[1]))
    lines.append(f"| {t} | {T.SHORT[t]} | {T.CATEGORY[t]} | {int(mat[t].notna().sum()):,} | {int(nb[t])} | {lv} | {prev[t]:.3f} |")
lines += ["", "## Mendelian calibration traits", "", "| Trait | Accessions | Positive share |", "|---|---:|---:|"]
for c in mendel:
    s = mendel[c].dropna()
    lines.append(f"| {c} | {len(s):,} | {'' if c == 'maturity_group' else f'{s.mean():.3f}'} |")
lines += ["", f"Accession table: {len(acc):,} genotyped accessions; {int((acc.n_stress_traits > 0).sum()):,} with ≥1 stress trait; "
          f"taxon {acc.taxon.value_counts().to_dict()}.", ""]
(ROOT / "docs/DATA_CARD.md").write_text("\n".join(lines))
if fail:
    print("CHECKS FAILED:\n  " + "\n  ".join(fail))
    sys.exit(1)
print("all checks passed")
