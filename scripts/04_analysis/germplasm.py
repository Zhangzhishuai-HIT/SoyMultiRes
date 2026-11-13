"""Step 10: multi-resistant germplasm. Out-of-fold validation of resistance predictions and the ranking of all
genotyped accessions by the final all-data MT-Boost.

usage: python -I germplasm.py [--top 0.10]
1. Validation with CV1 out-of-fold predictions (runs/cv1/<model>/<trait>.tsv.gz; every accession predicted by the
   fold model that never saw it), for MT-Boost (mtboost_v2_final) and the best single-trait model (best_single_avg4):
   per trait, predictions standardised within fold, accessions ranked; precision = share of observed resistant
   accessions among the top 5% / 10% (categorical traits: the GRIN resistant class; numeric traits: the top 10% of the
   batch-centred score), base rate and enrichment.
2. Multi-resistance consistency: accessions with out-of-fold predictions in the top 10% for >= k traits of
   different stress categories (among the traits they were scored for) -> share of those records observed resistant.
3. Ranking: final predictions (runs/all/mtboost_final/predictions.npz, "final") standardised per trait over all
   20,087 accessions; category score = mean z of the category's traits; multi-resistance index = number of the 8
   categories with a category score in the top 10%; accessions table with taxon, origin, maturity group.
Writes results/germplasm/{precision.tsv, consistency.tsv, ranking.tsv, top_accessions.tsv}.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

CAT = {row[0]: row[4] for row in T.STRESS}


def per_acc(rec, trait):
    r = evaluate.within_fold_z(rec)
    if trait in T.NUMERIC:
        bm = r.groupby("batch").score.transform("mean")
        r = r.assign(score=r.score - bm)
        g = r.groupby("acc").agg(pred=("pred", "mean"), score=("score", "mean"))
        g["resistant"] = g.score >= g.score.quantile(0.9)
    else:
        g = r.groupby("acc").agg(pred=("pred", "mean"), score=("score", "mean"), resistant=("resistant", "mean"))
        g["resistant"] = g.resistant > 0.5
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=float, default=0.10)
    a = ap.parse_args()
    out = ROOT / "results/germplasm"
    out.mkdir(parents=True, exist_ok=True)
    prec, oof = [], {}
    for model in ("mtboost_v2_final", "best_single_avg4"):
        for t in T.TRAITS:
            g = per_acc(pd.read_csv(data.RUNS / "cv1" / model / f"{t}.tsv.gz", sep="\t", dtype={"acc": str}), t)
            if model == "mtboost_v2_final":
                oof[t] = g
            base = g.resistant.mean()
            row = {"model": model, "trait": t, "short": T.SHORT[t], "category": CAT[t], "n": len(g), "base_rate": base}
            for q in (0.05, 0.10):
                k = max(1, int(round(q * len(g))))
                p = g.sort_values("pred", ascending=False).head(k).resistant.mean()
                row[f"precision_top{int(q * 100)}"] = p
                row[f"enrichment_top{int(q * 100)}"] = p / base if base > 0 else np.nan
            prec.append(row)
    P = pd.DataFrame(prec)
    P.to_csv(out / "precision.tsv", sep="\t", index=False, float_format="%.4f")
    w = P.pivot_table(index="short", columns="model", values="enrichment_top10")
    print("enrichment of observed resistance among the top 10% predicted (out of fold), median over traits:",
          w.median().round(2).to_dict())
    print("MT-Boost enrichment higher than the best single-trait model for", int((w.mtboost_v2_final > w.best_single_avg4).sum()),
          "of", len(w), "traits")
    # 2. consistency across stress categories (out of fold)
    flags = []
    for t, g in oof.items():
        thr = g.pred.quantile(1 - a.top)
        flags.append(pd.DataFrame({"acc": g.index, "trait": t, "category": CAT[t], "top": g.pred >= thr,
                                   "resistant": g.resistant}))
    F = pd.concat(flags, ignore_index=True)
    ncat = F[F.top].groupby("acc").category.nunique()
    rows = []
    for k in (1, 2, 3, 4):
        accs = set(ncat[ncat >= k].index)
        sub = F[F.acc.isin(accs) & F.top]
        rows.append({"min_categories_top": k, "accessions": len(accs), "records_predicted_top": len(sub),
                     "observed_resistant_share": sub.resistant.mean() if len(sub) else np.nan,
                     "base_rate_all_records": F.resistant.mean()})
    C = pd.DataFrame(rows)
    C.to_csv(out / "consistency.tsv", sep="\t", index=False, float_format="%.4f")
    print(C.to_string(index=False))
    # 3. ranking of all genotyped accessions by the final all-data model
    if not (data.RUNS / "all/mtboost_final/predictions.npz").exists():
        print("final all-data predictions not there yet: ranking skipped")
        return
    z = np.load(data.RUNS / "all/mtboost_final/predictions.npz", allow_pickle=True)
    pred = pd.DataFrame(z["final"], index=z["acc"], columns=[str(t) for t in z["traits"]])
    Z = (pred - pred.mean()) / pred.std()
    cats = sorted(set(CAT.values()))
    CS = pd.DataFrame({c: Z[[t for t in Z.columns if CAT[t] == c]].mean(1) for c in cats})
    top_cat = (CS >= CS.quantile(1 - a.top)).sum(1)
    meta = pd.read_csv(data.P / "accessions.tsv", sep="\t").set_index("vcf_id")
    R = pd.DataFrame({"multi_resistance_index": top_cat, "mean_category_z": CS.mean(1)}).join(CS.add_prefix("z_"))
    R = R.join(meta[["taxon", "origin", "region", "mg", "n_stress_traits"]], how="left")
    # wild and cultivated soybean ranked separately (plan: G. soja in, reported separately): z and index within taxon
    tax = R.taxon.fillna("unknown")
    within = pd.Series(0, index=R.index)
    for t_, idx in tax.groupby(tax).groups.items():
        Zt = (pred.loc[idx] - pred.loc[idx].mean()) / pred.loc[idx].std()
        CSt = pd.DataFrame({c: Zt[[t for t in Zt.columns if CAT[t] == c]].mean(1) for c in cats})
        within.loc[idx] = (CSt >= CSt.quantile(1 - a.top)).sum(1)
    R["multi_resistance_index_within_taxon"] = within
    R = R.sort_values(["multi_resistance_index_within_taxon", "mean_category_z"], ascending=False)
    R.to_csv(out / "ranking.tsv", sep="\t", float_format="%.3f")
    R.head(100).to_csv(out / "top_accessions.tsv", sep="\t", float_format="%.3f")
    print("accessions by number of stress categories in the top 10%:", top_cat.value_counts().sort_index().to_dict())
    for t_ in ("Glycine max", "Glycine soja"):
        v = R[R.taxon == t_].multi_resistance_index_within_taxon
        print(f"{t_}: within-taxon index distribution {v.value_counts().sort_index().to_dict()}; "
              f">= 4: {int((v >= 4).sum())} (never screened {int(((v >= 4) & (R[R.taxon == t_].n_stress_traits.fillna(0) == 0)).sum())})")
    print(R.head(15)[["multi_resistance_index", "mean_category_z", "taxon", "origin", "mg", "n_stress_traits"]].to_string())


if __name__ == "__main__":
    main()
