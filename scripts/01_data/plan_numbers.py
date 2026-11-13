"""Collect the headline numbers quoted in the detailed plan into results/plan_numbers.tsv (key, value).

usage: python -I plan_numbers.py <project_dir>
"""
import sys
from pathlib import Path

import pandas as pd

P = Path(sys.argv[1]).resolve()
R = P / "results"
out = {}
sam = pd.read_csv(P.parent / "HapGAM/data/interim/soysnp50k/samples.tsv", sep="\t", dtype=str)
out["genotyped_total"] = len(sam)
acc = pd.read_csv(R / "inventory/accessions.tsv", sep="\t")
for k in (1, 2, 3, 5, 10, 15):
    out[f"acc_ge{k}"] = int((acc.n_traits >= k).sum())
out["traits_per_acc_max"] = int(acc.n_traits.max())
out["traits_per_acc_median"] = float(acc.n_traits.median())
out["acc_gmax"] = int((acc.taxon == "Glycine max").sum())
out["acc_gsoja"] = int((acc.taxon == "Glycine soja").sum())
out["acc_no_trait"] = out["genotyped_total"] - out["acc_ge1"]
inv = pd.read_csv(R / "inventory/trait_inventory.tsv", sep="\t")
cells = int(inv.genotyped.sum())
out["cells"] = cells
out["fill_ge1_pct"] = round(100 * cells / (out["acc_ge1"] * len(inv)), 1)
out["fill_all_pct"] = round(100 * cells / (out["genotyped_total"] * len(inv)), 1)
meth = pd.read_csv(R / "inventory/trait_methods.tsv", sep="\t")
out["batches"] = len(meth)
out["batches_per_trait_max"] = int(meth.groupby("trait").size().max())
out["traits_single_batch"] = int((meth.groupby("trait").size() == 1).sum())
ab = pd.read_csv(R / "pheno/scores_by_batch.tsv.gz", sep="\t")
bad = ab[(ab.n_batch < 20) | ~(ab.sd_batch > 0)]
out["batches_uninformative"] = int(bad.groupby(["trait", "method_name"]).ngroups)
out["batch_rows_uninformative"] = len(bad)
rf = pd.read_csv(R / "pheno/resistant_fraction.tsv", sep="\t")
out["rare_traits_lt5"] = int((rf.frac < 0.05).sum())
out["rare_traits_list"] = ",".join(rf[rf.frac < 0.05].trait)
ov = pd.read_csv(R / "inventory/trait_overlap.tsv", sep="\t", index_col=0)
tr = list(ov.index)
vals = [ov.loc[a, b] for i, a in enumerate(tr) for b in tr[i + 1:]]
out["pairs"] = len(vals)
for k in (1, 200, 1000):
    out[f"pairs_overlap_ge{k}"] = sum(v >= k for v in vals)
out["pairs_overlap_zero"] = sum(v == 0 for v in vals)
pa = pd.read_csv(R / "pheno/pair_associations.tsv", sep="\t")
same, diff = pa[pa.same_family], pa[~pa.same_family]
out["assoc_same_tested"] = len(same)
out["assoc_same_pos"] = int(((same.q_mh < 0.05) & (same.or_mh > 1)).sum())
out["assoc_diff_tested"] = len(diff)
out["assoc_diff_pos"] = int(((diff.q_mh < 0.05) & (diff.or_mh > 1)).sum())
out["assoc_diff_neg"] = int(((diff.q_mh < 0.05) & (diff.or_mh < 1)).sum())
ni = R / "near_identical/near_identical_groups.tsv"
if ni.exists():
    g = pd.read_csv(ni, sep="\t")
    out["dup_groups"] = int(g.dup_group.nunique())
    out["dup_acc_in_groups"] = int((g.group_size >= 2).sum())
    out["dup_largest"] = int(g.group_size.max())
    out["dup_n_groups_ge2"] = int(g[g.group_size >= 2].dup_group.nunique())
pd.Series(out).to_csv(R / "plan_numbers.tsv", sep="\t", header=["value"], index_label="key")
for k, v in out.items():
    print(f"{k}\t{v}")
