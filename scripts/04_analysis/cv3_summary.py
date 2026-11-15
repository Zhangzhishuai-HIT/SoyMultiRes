"""Third scenario (held-out evaluation experiments): accuracy of every model per trait.

usage: python -I cv3_summary.py
For each trait, the test records of all its held-out experiments (units) are pooled after standardizing each model's
predictions within unit (evaluate.within_fold_z), as in the first scenario. Traits whose held-out experiments leave
one class in the training set are excluded (NSCANKER). Writes results/compare/cv3_point_estimates.tsv (one row per
trait: units, test accessions, measure, accuracy of each model) and prints the means over primary and other traits.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

MODELS = ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4", "mtboost_v2_noshare2", "mtboost_v2_final"]
EXCLUDE = {"NSCANKER"}


def main():
    rare = set(data.rare_traits())
    prim = set(data.primary_traits())
    rows = []
    for t in T.TRAITS:
        if t in EXCLUDE:
            continue
        fs = {m: data.RUNS / "cv3" / m / f"{t}.tsv.gz" for m in MODELS}
        if not all(f.exists() for f in fs.values()):
            continue
        row = {"trait": t, "short": T.SHORT[t], "prim": t in prim}
        for m, f in fs.items():
            r = pd.read_csv(f, sep="\t", dtype={"acc": str})
            met = evaluate.metrics(evaluate.within_fold_z(r), t, t in rare)
            row[m] = met["primary"]
            row["metric"] = met["primary_metric"]
            row["units"] = r.fold.nunique()
            row["test_accessions"] = r.acc.nunique()
        rows.append(row)
    d = pd.DataFrame(rows)
    out = ROOT / "results/compare/cv3_point_estimates.tsv"
    d.to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for sel, lab in ((True, "primary"), (False, "other")):
        s = d[d.prim == sel]
        print(lab, len(s), "traits:", {m: round(float(s[m].mean()), 4) for m in MODELS})
    print("all", len(d), "traits:", {m: round(float(d[m].mean()), 4) for m in MODELS})
    print("wrote", out)


if __name__ == "__main__":
    main()
