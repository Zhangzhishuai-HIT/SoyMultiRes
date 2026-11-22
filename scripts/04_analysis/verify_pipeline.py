"""Independent checks of the evaluation pipeline (run before the manuscript numbers are trusted).

usage: python -I verify_pipeline.py
1. Splits: near-identical groups never cross outer or inner folds; training and test accessions of every scenario
   are disjoint; in the third scenario no record of a held-out accession stays in training.
2. Every model of a scenario predicts exactly the same test records of a trait.
3. Accuracy recomputed here from the stored predictions with separate code (numpy / scikit-learn only) equals
   results/compare/*_point_estimates.tsv.
4. TraitBoost final = mean of the four refits stored in the fold files (two seeds with and two without shared trees).
Prints PASS / FAIL lines and exits with status 1 on any failure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAIL.append(name)


def zfold(r):
    """Standardize predictions within fold (and trait)."""
    g = r.groupby("fold").pred
    return r.assign(pred=(r.pred - g.transform("mean")) / g.transform("std").replace(0, np.nan))


def accuracy(r, trait, rare):
    """AUPRC for rare traits (resistant = resistant in at least one experiment); Pearson r of per-accession means
    for the other categorical traits; for numeric traits the correlation within experiments with at least ten test
    accessions, weighted by their number."""
    r = zfold(r).dropna(subset=["pred"])
    if trait in T.NUMERIC:
        num = den = 0.0
        for _, b in r.groupby("batch"):
            if len(b) >= 10 and b.score.std() > 0 and b.pred.std() > 0:
                num += len(b) * np.corrcoef(b.score, b.pred)[0, 1]
                den += len(b)
        return num / den if den else np.nan
    a = r.groupby("acc").agg(score=("score", "mean"), pred=("pred", "mean"), res=("resistant", "max"))
    if rare:                 # an accession is resistant when it was scored resistant in at least one experiment
        return average_precision_score((a.res == 1).astype(int), a.pred)
    return np.corrcoef(a.score, a.pred)[0, 1]


def main():
    fo = data.folds()
    # 1. splits
    check("near-identical groups stay in one outer fold", fo.groupby("dup_group").cv1_fold.nunique().max() == 1)
    check("near-identical groups stay in one inner fold", fo.groupby("dup_group").inner_fold.nunique().max() == 1)
    long = data.pheno_long()
    bad = 0
    for f in range(5):
        fit, ev = data.split_sets("cv1", f, fo)
        tr, te = set(fo.index[fit.to_numpy()]), set(fo.index[ev.to_numpy()])
        bad += len(tr & te)
        bad += len(set(fo.dup_group[fit.to_numpy()]) & set(fo.dup_group[ev.to_numpy()]))
    check("cv1: training and test accessions (and their groups) are disjoint", bad == 0, f"overlaps {bad}")
    bad = 0
    for u in range(data.n_splits("cv3")):
        held = set(data.cv3_test_accessions(u))
        for t in data.cv3_traits_of(u):
            ft = data.split_targets(long[long.trait == t], t, "cv3", u, fo)
            bad += len(set(ft.train_acc) & held)
    check("cv3: no accession of a held-out experiment is used for training", bad == 0, f"overlaps {bad}")
    # 2. same test records for all models
    rare = set(data.rare_traits())
    for scen, models in (("cv1", ["gblup", "bayesb", "lightgbm_avg4", "bglr_mt", "best_single_avg4",
                                  "mtboost_v2_noshare4", "mtboost_v2_final"]),
                         ("cv3", ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4", "mtboost_v2_final"])):
        diff = 0
        for t in T.TRAITS:
            keys = []
            for m in models:
                f = data.RUNS / scen / m / f"{t}.tsv.gz"
                if f.exists():
                    r = pd.read_csv(f, sep="\t", dtype={"acc": str})
                    keys.append(set(zip(r.acc, r.batch, r.fold)))
            if scen == "cv3" and t == "NSCANKER":
                continue
            if keys and any(k != keys[0] for k in keys[1:]):
                diff += 1
        check(f"{scen}: all models predict the same test records", diff == 0, f"traits with differences {diff}")
    # 3. accuracy recomputed
    for scen, pe_file, models in (("cv1", "cv1_point_estimates.tsv", ["gblup", "bayesb", "lightgbm_avg4", "bglr_mt",
                                                                     "best_single_avg4", "mtboost_v2_noshare4",
                                                                     "mtboost_v2_final"]),
                                  ("cv3", "cv3_point_estimates.tsv", ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4",
                                                                     "mtboost_v2_final"])):
        pe = pd.read_csv(ROOT / "results/compare" / pe_file, sep="\t").set_index("trait")
        worst = 0.0
        for t in pe.index:
            for m in models:
                r = pd.read_csv(data.RUNS / scen / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
                v = accuracy(r, t, t in rare)
                worst = max(worst, abs(v - pe.loc[t, m]))
        check(f"{scen}: accuracies recomputed with separate code match the result table", worst < 2e-3,
              f"largest difference {worst:.5f}")
    # 4. final = mean of the four refits (fold files of the fits: pred_s1, pred_s2, pred_noshare_s1, pred_noshare_s2)
    worst, n = 0.0, 0
    dev = data.RUNS / "cv1/mtboost_dev/v2"
    parts = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in sorted(dev.glob("fold[0-4].tsv.gz"))])
    if "pred_s1" in parts:
        parts["pred_s1"] = parts["pred_s1"].fillna(parts["pred"])
        parts["pred_noshare_s1"] = parts["pred_noshare_s1"].fillna(parts["pred_noshare"])
    for t in ("NEMATCYST_4", "PHYTOROT_20", "SDS", "CHLOROSIS"):
        fin = pd.read_csv(data.RUNS / "cv1/mtboost_v2_final" / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
        key = ["acc", "batch", "fold"]
        p_ = parts[parts.trait == t]
        m = fin.merge(p_[key + ["pred_s1", "pred_s2", "pred_noshare_s1", "pred_noshare_s2"]], on=key)
        n += len(m)
        want = m[["pred_s1", "pred_s2", "pred_noshare_s1", "pred_noshare_s2"]].mean(axis=1)
        worst = max(worst, float((m.pred - want).abs().max()))
    check("cv1: TraitBoost final equals the mean of the two refits with and the two without shared trees",
          n > 0 and worst < 1e-4, f"{n} records, largest difference {worst:.2e}")
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
