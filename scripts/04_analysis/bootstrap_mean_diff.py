"""Mean difference in accuracy over traits between TraitBoost and other models, with a 95% interval from a joint
bootstrap over groups of near-identical test accessions.

usage: python -I bootstrap_mean_diff.py <cv1|cv3> [--reps 10000] [--models A B C ...] [--out NAME] [--drop_from_sparse TRAIT]
       (--models: the first model is compared with each of the others; --out: results/compare/<NAME>.tsv instead of
        <scenario>_mean_diff.tsv)
The groups of near-identical accessions (splits/folds.tsv, dup_group) among the test accessions of all traits of the
scenario are resampled with replacement, one resample for all traits together, and the accuracy of every model and
trait is computed with the resampled counts as weights (bootstrap_compare.Trait, the same computation as for the
per-trait comparisons). For TraitBoost against the best single-trait model, LightGBM and the ensemble of GBLUP and
LightGBM: the mean difference in accuracy over the sparse traits, over the other traits and over all traits, and the
2.5% and 97.5% quantiles of its bootstrap values (resamples in which an accuracy is undefined are dropped);
--drop_from_sparse adds the sparse traits without the given trait as a further set (same resamples). Writes
results/compare/<scenario>_mean_diff.tsv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/04_analysis"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
import bootstrap_compare as bc  # noqa: E402

MODELS = ["mtboost_v2_final", "best_single_avg4", "lightgbm_avg4", "ens_gblup_lightgbm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["cv1", "cv3"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop_from_sparse", default="", help="a further set: the sparse traits without this trait")
    a = ap.parse_args()
    MODELS_ = a.models
    rare, sparse = set(data.rare_traits()), set(data.sparse_traits())
    traits = [t for t in T.TRAITS if t not in bc.EXCLUDE[a.scenario]
              and all((data.RUNS / a.scenario / m / f"{t}.tsv.gz").exists() for m in MODELS_)]
    trs = {t: bc.Trait(a.scenario, t, MODELS_, t in rare) for t in traits}
    allg = np.unique(np.concatenate([tr.groups for tr in trs.values()]))
    gpos = {g: i for i, g in enumerate(allg)}
    idx = {t: np.array([gpos[g] for g in tr.groups])[tr.group_of] for t, tr in trs.items()}   # accession -> joint group
    point = np.array([trs[t].acc(np.ones((1, trs[t].n)))[0] for t in traits])                # traits x models
    rng = np.random.default_rng([bc.SEED, 7])
    G = len(allg)
    B = np.empty((a.reps, len(traits), len(MODELS_)), np.float32)
    for c0 in range(0, a.reps, a.chunk):
        c = min(a.chunk, a.reps - c0)
        cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=c)
        for k, t in enumerate(traits):
            B[c0:c0 + c, k] = trs[t].acc(cnt[:, idx[t]])
    groups = {"sparse": [k for k, t in enumerate(traits) if t in sparse],
              "other": [k for k, t in enumerate(traits) if t not in sparse], "all": list(range(len(traits)))}
    if a.drop_from_sparse:
        assert a.drop_from_sparse in sparse and a.drop_from_sparse in traits, a.drop_from_sparse
        groups[f"sparse_without_{T.SHORT[a.drop_from_sparse]}"] = [k for k in groups["sparse"] if traits[k] != a.drop_from_sparse]
    rows = []
    for mb in MODELS_[1:]:
        ia, ib = 0, MODELS_.index(mb)
        for g, ks in groups.items():
            d = (B[:, ks, ia].astype(float) - B[:, ks, ib].astype(float)).mean(1)
            ok = np.isfinite(d)
            rows.append({"a": MODELS_[ia], "b": mb, "traits": g, "n_traits": len(ks),
                         "mean_a": float(point[ks, ia].mean()), "mean_b": float(point[ks, ib].mean()),
                         "mean_difference": float((point[ks, ia] - point[ks, ib]).mean()),
                         "low95": float(np.quantile(d[ok], 0.025)), "high95": float(np.quantile(d[ok], 0.975)),
                         "higher": int((point[ks, ia] > point[ks, ib]).sum()),
                         "resamples": int(ok.sum()), "dropped": int((~ok).sum()), "test_groups": G})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/compare" / f"{a.out or a.scenario + '_mean_diff'}.tsv", sep="\t", index=False, float_format="%.6f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
