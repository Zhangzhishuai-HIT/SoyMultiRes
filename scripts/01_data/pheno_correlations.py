"""Harmonised resistance scores, between-trait phenotypic correlations and a check for selective testing.

usage: python -I pheno_correlations.py <inventory_dir> <out_dir>

Resistance score: higher = more resistant/tolerant for every trait (GRIN code tables, 2026-11-11):
  SCN races        S 0, MS 1, MR 2, R 3          PHYTOROT, PYTHIUM   S 0, H 1, R 2
  SDS              S 0, MS 1, MR 2               APHID               S 0, MR 1, R 2
  BRNSTEMROT       U 0 (susceptible in preliminary field test), S 0, R 2, MR 3 ("most resistant")
  NSCANKER         S 0, MS 1, I 2, MR 3, R 4     PMV                 S 0, U 1 (field-resistant, unconfirmed), R 2
  BPMV, SALTREACT  S 0, T 1                      HIGHTEMP            S 0, I 1, T 2
  numeric damage/severity scores (BP, FROGEYE2/3, MEXBEANBTL, LOOPER, VBC, POTATOLHOP, RUST_TAN, CHLOROSIS): -score
Records of the same accession are averaged within an evaluation batch; each trait is then centred within batch
(batch = GRIN method_name) before correlating, so batch-mean differences do not create correlations.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INV, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
CODES = {
    "SCN": {"S": 0, "MS": 1, "MR": 2, "R": 3},
    "PHYTOROT": {"S": 0, "H": 1, "R": 2},
    "PYTHIUMROT": {"S": 0, "H": 1, "R": 2},
    "SDS": {"S": 0, "MS": 1, "MR": 2},
    "APHID": {"S": 0, "MR": 1, "R": 2},
    "BRNSTEMROT": {"U": 0, "S": 0, "R": 2, "MR": 3},
    "NSCANKER": {"S": 0, "MS": 1, "I": 2, "MR": 3, "R": 4},
    "PMV": {"S": 0, "U": 1, "R": 2},
    "BPMV": {"S": 0, "T": 1},
    "SALTREACT": {"S": 0, "T": 1},
    "HIGHTEMP": {"S": 0, "I": 1, "T": 2},
}
NUMERIC = {"BP", "FROGEYE2", "FROGEYE3", "MEXBEANBTL", "LOOPER", "VBC", "POTATOLHOP", "RUST_TAN", "CHLOROSIS"}
RESISTANT = {  # what counts as "resistant" for the rare-class summaries
    "SCN": {"R", "MR"}, "PHYTOROT": {"R"}, "PYTHIUMROT": {"R"}, "SDS": {"MR"}, "APHID": {"R", "MR"},
    "BRNSTEMROT": {"R", "MR"}, "NSCANKER": {"R", "MR"}, "PMV": {"R"}, "BPMV": {"T"}, "SALTREACT": {"T"},
    "HIGHTEMP": {"T"},
}


def family(t):
    return "SCN" if t.startswith("NEMATCYST") else "PHYTOROT" if t.startswith("PHYTOROT") else t


long = pd.read_csv(f"{INV}/long_records.tsv.gz", sep="\t", dtype=str)
long["fam"] = long.trait.map(family)


def score(r):
    if r.trait in NUMERIC:
        v = pd.to_numeric(r.observation_value, errors="coerce")
        return -v
    return CODES[r.fam].get(r.observation_value, np.nan)


long["score"] = [score(r) for r in long.itertuples()]
long["resistant"] = [(np.nan if r.trait in NUMERIC else float(r.observation_value in RESISTANT[r.fam]))
                     for r in long.itertuples()]
bad = long.score.isna().sum()
print("records without a score:", bad, long[long.score.isna()].observation_value.value_counts().head().to_dict())
long = long.dropna(subset=["score"])
# accession x batch mean, then centre within batch
ab = long.groupby(["trait", "method_name", "acc"], as_index=False).agg(score=("score", "mean"),
                                                                        resistant=("resistant", "max"))
ab["n_batch"] = ab.groupby(["trait", "method_name"]).acc.transform("size")
ab["sd_batch"] = ab.groupby(["trait", "method_name"]).score.transform("std")
ab["centred"] = ab.score - ab.groupby(["trait", "method_name"]).score.transform("mean")
ab.to_csv(f"{OUT}/scores_by_batch.tsv.gz", sep="\t", index=False)
informative = ab[(ab.n_batch >= 20) & (ab.sd_batch > 0)]
dropped = ab[~ab.index.isin(informative.index)].groupby(["trait", "method_name"]).size()
print("batches dropped (<20 records or a single class):", len(dropped))
print(dropped.to_string())
acc = informative.groupby(["trait", "acc"]).centred.mean().unstack("trait")
raw = ab.groupby(["trait", "acc"]).score.mean().unstack("trait")
res = ab.groupby(["trait", "acc"]).resistant.max().unstack("trait")
acc.to_csv(f"{OUT}/acc_by_trait_centred.tsv.gz", sep="\t")

# resistant fraction per trait (categorical traits)
rf = pd.DataFrame({"n": res.notna().sum(), "resistant": (res == 1).sum()})
rf["frac"] = (rf.resistant / rf.n).round(4)
rf = rf[rf.n > 0].sort_values("frac")
rf.to_csv(f"{OUT}/resistant_fraction.tsv", sep="\t")
print(rf.to_string())

# pairwise Spearman on batch-centred scores
traits = list(acc.columns)
rows = []
for i, a in enumerate(traits):
    for b in traits[i + 1:]:
        m = acc[[a, b]].dropna()
        if len(m) >= 100 and m[a].nunique() > 1 and m[b].nunique() > 1:
            r, p = spearmanr(m[a], m[b])
            rows.append(dict(a=a, b=b, n=len(m), rho=round(r, 3), p=p, fam_a=family(a), fam_b=family(b)))
cor = pd.DataFrame(rows)
cor["same_family"] = cor.fam_a == cor.fam_b
cor.to_csv(f"{OUT}/pheno_corr_pairs.tsv", sep="\t", index=False)
print("pairs with n>=100:", len(cor), " same pathogen family:", int(cor.same_family.sum()))
for lab, d in [("same family", cor[cor.same_family]), ("different families", cor[~cor.same_family])]:
    print(f"{lab}: median rho {d.rho.median():.3f}; |rho|>=0.2: {(d.rho.abs() >= 0.2).sum()} of {len(d)}; "
          f"p<0.001 & rho>0: {((d.p < 1e-3) & (d.rho > 0)).sum()}; p<0.001 & rho<0: {((d.p < 1e-3) & (d.rho < 0)).sum()}")
pd.set_option("display.width", 200)
print(cor.sort_values("rho", ascending=False).head(25).to_string(index=False))
print(cor[~cor.same_family].sort_values("rho", ascending=False).head(15).to_string(index=False))
print(cor.sort_values("rho").head(10).to_string(index=False))

# selective testing: resistance rate of a reference race among accessions tested / not tested for another race
sel = []
for ref, others in [("NEMATCYST_3", ["NEMATCYST_1", "NEMATCYST_4", "NEMATCYST_5", "NEMATCYST_14"]),
                    ("PHYTOROT_1", ["PHYTOROT_3", "PHYTOROT_4", "PHYTOROT_5", "PHYTOROT_7", "PHYTOROT_12",
                                    "PHYTOROT_17", "PHYTOROT_20", "PHYTOROT_25"])]:
    for o in others:
        tested = res[o].notna()
        r_in = res.loc[tested & res[ref].notna(), ref]
        r_out = res.loc[~tested & res[ref].notna(), ref]
        sel.append(dict(reference=ref, other=o, n_tested_both=len(r_in), ref_resistant_if_tested=round(r_in.mean(), 4),
                        n_ref_only=len(r_out), ref_resistant_if_not_tested=round(r_out.mean(), 4)))
sel = pd.DataFrame(sel)
sel.to_csv(f"{OUT}/selective_testing.tsv", sep="\t", index=False)
print(sel.to_string(index=False))
