"""Paired bootstrap comparison of two models on the same test accessions (plan section 5.1).

usage: python -I compare_models.py <scenario> <model_a> <model_b> [--set all|primary|other|hard] [--reps 1000]

Evaluation rule of 2026-11-12 (docs/FREEZE_DEVIATIONS.md): each model's predictions are standardised within fold
(evaluate.within_fold_z) before the folds are pooled; the raw pooled values are written alongside (a_raw, b_raw).

For every trait both models have finished: resample test accessions with replacement (records of an accession move
together), recompute each model's primary metric (soymultires.evaluate.metrics) and the difference a - b. Two-sided P =
2 x the smaller tail share of the bootstrap differences around 0; Holm adjustment across the traits of the set.
Writes results/compare/<scenario>_<a>_vs_<b>_<set>.tsv and prints the two counts
("a higher for N traits, significantly for M").
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


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def hard_traits():
    inv = pd.read_csv(ROOT / "results/inventory/trait_inventory.tsv", sep="\t").set_index("trait")
    rare = set(data.rare_traits())
    small = {t for t in T.TRAITS if inv.loc[t, "genotyped"] < 1000}
    return [t for t in T.TRAITS if t in rare | small]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--set", default="all", choices=["all", "primary", "other", "hard"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--exclude", nargs="*", default=[], help="traits left out (e.g. cv3 traits without evaluable units)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    a = ap.parse_args()
    rare = set(data.rare_traits())
    prim = data.primary_traits()
    traits = {"all": T.TRAITS, "hard": hard_traits(), "primary": prim,
              "other": [t for t in T.TRAITS if t not in prim]}[a.set]
    rng = np.random.default_rng(20261111)
    rows = []
    for t in traits:
        if t in a.exclude:
            continue
        fa = data.RUNS / a.scenario / a.a / f"{t}.tsv.gz"
        fb = data.RUNS / a.scenario / a.b / f"{t}.tsv.gz"
        if not (fa.exists() and fb.exists()):
            continue
        ra0 = pd.read_csv(fa, sep="\t", dtype={"acc": str})
        rb0 = pd.read_csv(fb, sep="\t", dtype={"acc": str})
        ra, rb = evaluate.within_fold_z(ra0), evaluate.within_fold_z(rb0)
        raw_a = evaluate.metrics(ra0, t, t in rare)["primary"]
        raw_b = evaluate.metrics(rb0, t, t in rare)["primary"]
        key = ["acc", "batch", "fold"]
        m = ra.merge(rb[key + ["pred"]], on=key, suffixes=("_a", "_b"))
        assert len(m) == len(ra) == len(rb), f"{t}: test records differ between models"
        ma = evaluate.metrics(m.rename(columns={"pred_a": "pred"}), t, t in rare)
        mb = evaluate.metrics(m.rename(columns={"pred_b": "pred"}), t, t in rare)
        accs = m.acc.unique()
        idx = m.groupby("acc").indices
        diffs = []
        for _ in range(a.reps):
            pick = rng.choice(accs, size=len(accs), replace=True)
            rows_ = np.concatenate([idx[x] for x in pick])
            s = m.iloc[rows_].copy()
            s["acc"] = np.repeat(np.arange(len(pick)), [len(idx[x]) for x in pick]).astype(str)   # duplicates stay distinct
            da = evaluate.metrics(s.rename(columns={"pred_a": "pred"}), t, t in rare)["primary"]
            db = evaluate.metrics(s.rename(columns={"pred_b": "pred"}), t, t in rare)["primary"]
            diffs.append(da - db)
        diffs = np.array(diffs)
        diffs = diffs[~np.isnan(diffs)]
        p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())) if len(diffs) else np.nan
        rows.append(dict(trait=t, short=T.SHORT[t], metric=ma["primary_metric"], a=ma["primary"], b=mb["primary"],
                         a_raw=raw_a, b_raw=raw_b,
                         diff=ma["primary"] - mb["primary"], lo=np.quantile(diffs, 0.025), hi=np.quantile(diffs, 0.975),
                         p=p))
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p.to_numpy())
    out = ROOT / "results/compare"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"{a.scenario}_{a.a}_vs_{a.b}_{a.set}{a.tag}.tsv", sep="\t", index=False, float_format="%.4f")
    hi = d["diff"] > 0
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"{a.a} higher than {a.b} for {int(hi.sum())} of {len(d)} traits "
          f"(significantly for {int((hi & (d.p_holm < 0.05)).sum())}); lower and significant for "
          f"{int((~hi & (d.p_holm < 0.05)).sum())}")


if __name__ == "__main__":
    main()
