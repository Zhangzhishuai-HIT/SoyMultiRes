"""Resistant-vs-not associations between trait pairs on shared accessions (odds ratios), raw and MG-stratified.

usage: python -I pair_associations.py <inventory_dir> <pheno_dir> <out_dir> [<folds.tsv>]

"Resistant" = the GRIN resistant classes (pheno_correlations.py) for categorical traits; for numeric damage scores,
the least-damaged 10% within an evaluation batch. Mantel-Haenszel odds ratio and test stratified by maturity-group
band x taxon, so that pairs linked only through maturity group or wild/cultivated status are not counted.
With a fourth argument (splits/folds.tsv), every pair is tested on one accession per group of near-identical accessions
(the first by accession identifier among the accessions scored for both traits; accessions without genotypes form
groups of their own), written to pair_associations_one_per_group.tsv and compared with pair_associations.tsv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact

INV, PH, OUT = sys.argv[1:4]
GROUP = pd.read_csv(sys.argv[4], sep="\t", dtype={"acc": str}).set_index("acc").dup_group if len(sys.argv) > 4 else None
ab = pd.read_csv(f"{PH}/scores_by_batch.tsv.gz", sep="\t", dtype={"acc": str})
ab = ab[(ab.n_batch >= 20) & (ab.sd_batch > 0)].copy()
num = ab.resistant.isna()
q90 = ab[num].groupby(["trait", "method_name"]).score.transform(lambda s: s.quantile(0.9))
ab.loc[num, "resistant"] = (ab.loc[num, "score"] >= q90).astype(float)
lab = ab.groupby(["trait", "acc"]).resistant.max().unstack("trait")
acc = pd.read_csv(f"{INV}/accessions.tsv", sep="\t", dtype=str).set_index("acc")
stratum = (acc.mg_group.fillna("MG_NA") + "|" + acc.taxon.fillna("NA")).reindex(lab.index)


def family(t):
    return "SCN" if t.startswith("NEMATCYST") else "PHYTOROT" if t.startswith("PHYTOROT") else t


def mh(a, b, s):
    """Mantel-Haenszel OR and chi-square (1 df, continuity-corrected) over strata."""
    num = den = 0.0
    sa = ea = va = 0.0
    for _, g in pd.DataFrame({"a": a, "b": b, "s": s}).groupby("s"):
        n = len(g)
        if n < 2:
            continue
        n11 = ((g.a == 1) & (g.b == 1)).sum()
        n10 = ((g.a == 1) & (g.b == 0)).sum()
        n01 = ((g.a == 0) & (g.b == 1)).sum()
        n00 = ((g.a == 0) & (g.b == 0)).sum()
        num += n11 * n00 / n
        den += n10 * n01 / n
        r1, c1 = n11 + n10, n11 + n01
        sa += n11
        ea += r1 * c1 / n
        va += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1)) if n > 1 else 0
    or_ = num / den if den > 0 else np.inf
    stat = max(abs(sa - ea) - 0.5, 0.0) ** 2 / va if va > 0 else 0.0     # continuity correction, not below zero
    return or_, chi2.sf(stat, 1)


traits = list(lab.columns)
rows = []
for i, a in enumerate(traits):
    for b in traits[i + 1:]:
        m = lab[[a, b]].dropna()
        if GROUP is not None:                                            # one accession per near-identical group
            m = m.sort_index()
            g_ = GROUP.reindex(m.index)
            g_ = g_.where(g_.notna(), pd.Series([f"own_{x}" for x in m.index], index=m.index))
            m = m[~g_.duplicated().to_numpy()]
        if len(m) < 100 or m[a].sum() < 5 or m[b].sum() < 5:
            continue
        t = pd.crosstab(m[a], m[b]).reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0).to_numpy()
        or_raw, p_raw = fisher_exact(t[::-1, ::-1])
        or_mh, p_mh = mh(m[a], m[b], stratum.loc[m.index].to_numpy())
        rows.append(dict(a=a, b=b, n=len(m), both=int(t[1, 1]), res_a=int(m[a].sum()), res_b=int(m[b].sum()),
                         or_raw=round(or_raw, 2), p_raw=p_raw, or_mh=round(or_mh, 2), p_mh=p_mh,
                         same_family=family(a) == family(b)))
d = pd.DataFrame(rows)
# Benjamini-Hochberg on the stratified test
order = np.argsort(d.p_mh.to_numpy(), kind="mergesort")              # ascending P
adj = d.p_mh.to_numpy()[order] * len(d) / np.arange(1, len(d) + 1)
q = np.empty(len(d))
q[order] = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0)   # monotone in P
d["q_mh"] = q
d.to_csv(f"{OUT}/pair_associations{'_one_per_group' if GROUP is not None else ''}.tsv", sep="\t", index=False)
if GROUP is not None:
    full = pd.read_csv(f"{OUT}/pair_associations.tsv", sep="\t")
    j = full.merge(d, on=["a", "b"], how="left", suffixes=("", "_1g"))
    sig = j[j.q_mh < 0.05]
    kept = sig[(sig.q_mh_1g < 0.05) & ((sig.or_mh_1g > 1) == (sig.or_mh > 1))]
    print(f"one accession per group: {len(d)} pairs tested (all accessions: {len(full)}); of the {len(sig)} pairs with "
          f"q < 0.05 on all accessions, {len(kept)} have q < 0.05 in the same direction "
          f"(same pathogen {int(kept.same_family.sum())} of {int(sig.same_family.sum())}, different stresses "
          f"{int((~kept.same_family).sum())} of {int((~sig.same_family).sum())}); accessions per pair: median "
          f"{int(j.n_1g.median())} vs. {int(j.n.median())}")
for lab_, g in [("same pathogen", d[d.same_family]), ("different stresses", d[~d.same_family])]:
    pos = ((g.q_mh < 0.05) & (g.or_mh > 1)).sum()
    neg = ((g.q_mh < 0.05) & (g.or_mh < 1)).sum()
    print(f"{lab_}: tested {len(g)}, q<0.05 positive {pos}, negative {neg}")
pd.set_option("display.width", 220)
print(d[~d.same_family & (d.q_mh < 0.05)].sort_values("or_mh", ascending=False).to_string(index=False))
