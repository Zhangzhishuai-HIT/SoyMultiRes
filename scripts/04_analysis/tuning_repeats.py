"""Variation between tuning runs of TraitBoost for new accessions.

usage: python -I tuning_repeats.py [--tags v2 v2r1 v2r2]
The main run (tag v2) and repeated runs with other Optuna and tree seeds (train_mtboost.py fit --tag v2r<k>
--optuna_seed <k> --config '{"seed": <k>}', then collect) are compared on the same folds. For every run: the accuracy
of TraitBoost (mean of two refits of the chosen option and two of the best option without shared trees) and of
TraitBoost without shared trees with two refits (mtboost_<tag>_noshare2; for the main run computed here from the stored
refits s1 and s2), the mean over the sparse and over the other traits, and the number of traits with higher accuracy
than the best single-trait model, LightGBM, the mean of LightGBM and TraitBoost without shared trees
(ens_lightgbm_noshare) and TraitBoost without shared trees of the same run. Intervals of the mean differences come
from bootstrap_mean_diff.py --models ... --out cv1_mean_diff_<tag> (joint resampling of groups of near-identical
test accessions). Writes results/compare/tuning_repeats.tsv and tuning_repeats_by_trait.tsv.
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

REF = ["best_single_avg4", "lightgbm_avg4", "ens_lightgbm_noshare"]


def noshare2_main(tag):
    """mtboost_<tag>_noshare2 from the stored refits s1 and s2 (the main run was collected with four refits)."""
    out = data.RUNS / "cv1" / f"mtboost_{tag}_noshare2"
    if out.exists() and len(list(out.glob("*.tsv.gz"))) == len(T.TRAITS):
        return
    dev = data.RUNS / "cv1" / "mtboost_dev" / tag
    allp = pd.concat([pd.read_csv(dev / f"fold{f}.tsv.gz", sep="\t", dtype={"acc": str}) for f in range(5)],
                     ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    for t, d in allp.groupby("trait"):
        d[["acc", "batch", "score", "resistant", "fold"]].assign(
            pred=d[["pred_noshare_s1", "pred_noshare_s2"]].mean(1)).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["v2", "v2r1", "v2r2"])
    a = ap.parse_args()
    rare, sparse = set(data.rare_traits()), set(data.sparse_traits())
    acc = {}

    def accuracy(model, t):
        d = pd.read_csv(data.RUNS / "cv1" / model / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
        return evaluate.metrics(evaluate.within_fold_z(d), t, t in rare)["primary"]

    for m in REF:
        acc[m] = {t: accuracy(m, t) for t in T.TRAITS}
    for tag in a.tags:
        if not (data.RUNS / "cv1" / f"mtboost_{tag}_final").exists():
            print(f"{tag}: not collected, skipped")
            continue
        if tag == "v2":
            noshare2_main(tag)
        for v in ("final", "noshare2"):
            acc[f"mtboost_{tag}_{v}"] = {t: accuracy(f"mtboost_{tag}_{v}", t) for t in T.TRAITS}
    bt = pd.DataFrame(acc)
    bt.index.name = "trait"
    bt.insert(0, "short", [T.SHORT[t] for t in bt.index])
    bt.insert(1, "is_sparse", bt.index.isin(sparse))
    rows = []
    for tag in a.tags:
        fin = f"mtboost_{tag}_final"
        if fin not in bt:
            continue
        for g, sel in (("sparse", bt.is_sparse), ("other", ~bt.is_sparse), ("all", bt.is_sparse | ~bt.is_sparse)):
            r = {"run": tag, "traits": g, "n_traits": int(sel.sum()), "mean_traitboost": bt.loc[sel, fin].mean(),
                 "mean_noshare2": bt.loc[sel, f"mtboost_{tag}_noshare2"].mean()}
            for m in REF + [f"mtboost_{tag}_noshare2"]:
                key = m.replace(f"mtboost_{tag}_", "")
                r[f"diff_{key}"] = (bt.loc[sel, fin] - bt.loc[sel, m]).mean()
                r[f"higher_{key}"] = int((bt.loc[sel, fin] > bt.loc[sel, m]).sum())
            rows.append(r)
    out = pd.DataFrame(rows)
    fins = [c for c in bt if c.endswith("_final")]
    bt["sd_traitboost_runs"] = bt[fins].std(axis=1, ddof=1) if len(fins) > 1 else np.nan
    out.to_csv(ROOT / "results/compare/tuning_repeats.tsv", sep="\t", index=False, float_format="%.10f")
    bt.to_csv(ROOT / "results/compare/tuning_repeats_by_trait.tsv", sep="\t", float_format="%.10f")   # 10 decimals: tables round once
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    if len(fins) > 1:
        print("median SD of TraitBoost accuracy between runs:", round(float(bt.sd_traitboost_runs.median()), 4))


if __name__ == "__main__":
    main()
