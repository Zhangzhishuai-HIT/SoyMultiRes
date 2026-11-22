"""Variation of TraitBoost accuracy between refits with different random seeds (new accessions).

usage: python -I seed_variation.py
The prediction of TraitBoost is the mean of four refits (the chosen option and the best option without shared trees,
each with two seeds); four seeds of each option were fitted (runs/cv1/mtboost_dev/v2/fold<f>.tsv.gz, columns pred_s1-s4
and pred_noshare_s1-s4; fold 0 stores seed 1 as pred / pred_noshare). For every trait, the accuracy of every single
refit is computed as for the models (evaluate.metrics after within-fold standardization).
Writes results/compare/seed_variation.tsv: per trait the accuracies of the four single refits of each option, their
standard deviation, and the accuracy of TraitBoost, the chosen option alone and TraitBoost without shared trees.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402


def main():
    rare = set(data.rare_traits())
    parts = []
    for f in range(5):
        R = pd.read_csv(data.RUNS / "cv1/mtboost_dev/v2" / f"fold{f}.tsv.gz", sep="\t", dtype={"acc": str})
        if "pred_s1" not in R:
            R["pred_s1"], R["pred_noshare_s1"] = R["pred"], R["pred_noshare"]
        parts.append(R)
    R = pd.concat(parts, ignore_index=True)
    cols = {f"share_s{k}": f"pred_s{k}" for k in range(1, 5)} | {f"noshare_s{k}": f"pred_noshare_s{k}" for k in range(1, 5)}
    rows = []
    for t in T.TRAITS:
        x = R[R.trait == t]
        row = {"trait": t, "short": T.SHORT[t]}
        for name, c in cols.items():
            row[name] = evaluate.metrics(evaluate.within_fold_z(x[["acc", "batch", "score", "resistant", "fold"]]
                                                               .assign(pred=x[c])), t, t in rare)["primary"]
        for name, cs in (("traitboost", ["pred_s1", "pred_s2", "pred_noshare_s1", "pred_noshare_s2"]),
                         ("share4", [f"pred_s{k}" for k in range(1, 5)]),
                         ("noshare4", [f"pred_noshare_s{k}" for k in range(1, 5)])):
            row[name] = evaluate.metrics(evaluate.within_fold_z(x[["acc", "batch", "score", "resistant", "fold"]]
                                                               .assign(pred=x[cs].mean(axis=1))), t, t in rare)["primary"]
        row["sd_share"] = float(np.std([row[f"share_s{k}"] for k in range(1, 5)], ddof=1))
        row["sd_noshare"] = float(np.std([row[f"noshare_s{k}"] for k in range(1, 5)], ddof=1))
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/compare/seed_variation.tsv", sep="\t", index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    print(d[["short", "traitboost", "share4", "noshare4", "sd_share", "sd_noshare"]].round(4).to_string(index=False))
    print(f"median SD between seeds: chosen option {d.sd_share.median():.4f}, without shared trees {d.sd_noshare.median():.4f}; "
          f"max {d.sd_share.max():.4f} / {d.sd_noshare.max():.4f}")


if __name__ == "__main__":
    main()
