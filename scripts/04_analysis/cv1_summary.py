"""First scenario (new accessions): accuracy of every model per trait.

usage: python -I cv1_summary.py
For each trait, the test records of the five folds are pooled after standardizing each model's predictions within fold
(evaluate.within_fold_z). Writes results/compare/cv1_point_estimates.tsv: one row per trait with the accuracy of each
model (<model>) and the accuracy without the within-fold standardization (<model>_raw); prim marks the sparse traits
(data.sparse_traits()), and cv1_point_estimates_gmax.tsv with the same accuracy among the Glycine max test accessions
only. Prints the means over the sparse and the other traits.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

MODELS = ["gblup", "bayesb", "lightgbm", "lightgbm_avg4", "bglr_mt", "best_single_avg4", "mtboost_v2",
          "mtboost_v2_noshare4", "mtboost_v2_share4", "mtboost_v2_final", "ens_gblup_lightgbm", "ens_lightgbm_noshare"]


def main():
    rare = set(data.rare_traits())
    sparse = set(data.sparse_traits())
    rows = []
    for t in T.TRAITS:
        row = {"trait": t, "short": T.SHORT[t], "prim": t in sparse}
        for m in MODELS:
            r = pd.read_csv(data.RUNS / "cv1" / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
            row[m] = evaluate.metrics(evaluate.within_fold_z(r), t, t in rare)["primary"]
            row[f"{m}_raw"] = evaluate.metrics(r, t, t in rare)["primary"]
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/compare/cv1_point_estimates.tsv", sep="\t", index=False, float_format="%.10g")
    for sel, lab in ((True, "sparse"), (False, "other")):
        s = d[d.prim == sel]
        print(lab, len(s), "traits:", {m: round(float(s[m].mean()), 4) for m in MODELS})
    # the same accuracy among the Glycine max test accessions only (predictions standardised within fold among them)
    taxon = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id").taxon
    rows = []
    for t in T.TRAITS:
        row = {"trait": t, "short": T.SHORT[t], "prim": t in sparse}
        for m in MODELS:
            r = pd.read_csv(data.RUNS / "cv1" / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
            r = r[r.acc.map(taxon) == "Glycine max"]
            g = r.groupby("acc").resistant.max()
            row["accessions"], row["resistant"] = len(g), int((g == 1).sum())
            row[m] = evaluate.metrics(evaluate.within_fold_z(r), t, t in rare)["primary"] if len(g) >= 50 else float("nan")
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results/compare/cv1_point_estimates_gmax.tsv", sep="\t", index=False, float_format="%.10g")
    ok = g[MODELS].notna().all(axis=1)
    for sel, lab in ((True, "sparse"), (False, "other")):
        s = g[(g.prim == sel) & ok]
        print("G. max only,", lab, len(s), "traits:", {m: round(float(s[m].mean()), 4) for m in MODELS})


if __name__ == "__main__":
    main()
