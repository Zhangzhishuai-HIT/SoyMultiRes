"""Composite baseline for the primary comparison (plan 5.1): for every trait the single-trait model (GBLUP, BayesB or
LightGBM) that was strongest on inner cross-validation (results/gate/cv1_inner_best_single.tsv, chosen without the outer
test folds), with its outer-CV1 test predictions. Writes runs/cv1/best_single/<trait>.tsv.gz and
runs/cv1/best_single/CHOICE.tsv.

usage: python -I make_best_single.py [choice_file] [out_name]
       e.g. make_best_single.py results/gate/cv1_inner_best_single_avg4.tsv best_single_avg4
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

choice = sys.argv[1] if len(sys.argv) > 1 else "results/gate/cv1_inner_best_single.tsv"
name = sys.argv[2] if len(sys.argv) > 2 else "best_single"
best = pd.read_csv(ROOT / choice, sep="\t").set_index("trait")
out = data.RUNS / "cv1" / name
out.mkdir(parents=True, exist_ok=True)
for t, r in best.iterrows():
    shutil.copyfile(data.RUNS / "cv1" / r.best_model / f"{t}.tsv.gz", out / f"{t}.tsv.gz")
best[["short", "best_model", "best_value", "primary_trait"]].rename(columns={"best_value": "inner_cv_value"}).to_csv(
    out / "CHOICE.tsv", sep="\t")
print(best.best_model.value_counts().to_dict())
