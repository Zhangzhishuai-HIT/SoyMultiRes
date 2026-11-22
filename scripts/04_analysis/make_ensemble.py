"""Ensemble of a linear and a tree-based single-trait model: the mean of the GBLUP and the LightGBM predictions.

usage: python -I make_ensemble.py
       python -I make_ensemble.py --pair lightgbm_avg4 mtboost_v2_noshare4 --name ens_lightgbm_noshare --scenarios cv1 cv3
                (another pair of models: the mean of their standardized predictions, written to runs/<scenario>/<name>)
For new accessions (cv1) and held-out experiments (cv3; GBLUP there is runs/cv3/mtboost_v2_gblup), the record-level
test predictions of GBLUP and of LightGBM (mean of four refits) are standardised within fold
(evaluate.within_fold_z), and their mean is the prediction of the ensemble. Writes
runs/<scenario>/ens_gblup_lightgbm/<trait>.tsv.gz with the columns of the other prediction files.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=2, default=None, help="two model folders (default: GBLUP and lightgbm_avg4)")
ap.add_argument("--name", default="ens_gblup_lightgbm")
ap.add_argument("--scenarios", nargs="+", default=["cv1", "cv3"])
a = ap.parse_args()
KEY = ["acc", "batch", "fold"]
GBLUP = {"cv1": "gblup", "cv3": "mtboost_v2_gblup"}
for sc in a.scenarios:
    m1, m2 = a.pair if a.pair else (GBLUP[sc], "lightgbm_avg4")
    out = data.RUNS / sc / a.name
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for t in T.TRAITS:
        fg, fl = data.RUNS / sc / m1 / f"{t}.tsv.gz", data.RUNS / sc / m2 / f"{t}.tsv.gz"
        if not (fg.exists() and fl.exists()):
            continue
        g = evaluate.within_fold_z(pd.read_csv(fg, sep="\t", dtype={"acc": str}))
        lg = evaluate.within_fold_z(pd.read_csv(fl, sep="\t", dtype={"acc": str}))
        e = g.merge(lg[KEY + ["pred"]], on=KEY, suffixes=("_g", "_l"), validate="one_to_one")
        assert len(e) == len(g) == len(lg), (sc, t)
        e["pred"] = (e.pred_g + e.pred_l) / 2
        e[list(g.columns)].to_csv(out / f"{t}.tsv.gz", sep="\t", index=False, float_format="%.8g")
        k += 1
    print(sc, k, "traits written to", out)
