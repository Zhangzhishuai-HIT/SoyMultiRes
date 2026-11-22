"""Held-out experiments: accuracy of every model per trait.

usage: python -I cv3_summary.py
For each trait, the test records of its held-out experiment are standardized within the experiment
(evaluate.within_fold_z), as for new accessions. Traits whose held-out experiment leaves one class in the training
set are excluded (NSCANKER). Models: GBLUP (mtboost_v2_gblup), BayesB, LightGBM (four refits), multi-trait GBLUP
(bglr_mt, if fitted), the best single-trait model (best_single_avg4: the model chosen for the trait in the inner
cross-validation without the records of the held-out experiment, make_best_single.py --nested cv3), TraitBoost
without shared trees (mtboost_v2_noshare4), the chosen option alone (mtboost_v2_share4), TraitBoost (mtboost_v2_final),
the mean of GBLUP and LightGBM (ens_gblup_lightgbm) and the tree ensemble (ens_lightgbm_noshare).
Writes results/compare/cv3_point_estimates.tsv, one row per trait: held-out experiment, training accessions (scored
for the trait and not in the held-out experiment), test accessions, resistant test accessions, measure, accuracy of
each model; test accessions with a near-identical accession (same group in splits/folds.tsv) among the training
accessions of the trait, and the accuracy of each model on the test accessions without such an accession
(<model>_unrelated; empty when fewer than 30 such test accessions, a single class or no resistant accession remain).
prim marks the sparse traits. Prints the means over the sparse and the other traits.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

MODELS = ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4", "bglr_mt", "best_single_avg4", "mtboost_v2_noshare4",
          "mtboost_v2_share4", "mtboost_v2_final", "ens_gblup_lightgbm", "ens_lightgbm_noshare"]
EXCLUDE = {"NSCANKER"}


def main():
    rare = set(data.rare_traits())
    prim = set(data.sparse_traits())
    long = data.pheno_long()
    fo = data.folds()
    units = pd.read_csv(ROOT / "splits/cv3_batches.tsv", sep="\t")
    unit_of = {t: int(u.unit) for u in units.itertuples() for t in u.chosen_for.split(",")}
    batch_of = {t: u.batch for u in units.itertuples() for t in u.chosen_for.split(",")}
    models = [m for m in MODELS if any((data.RUNS / "cv3" / m).glob("*.tsv.gz"))]
    rows = []
    for t in T.TRAITS:
        if t in EXCLUDE or t not in unit_of:
            continue
        fs = {m: data.RUNS / "cv3" / m / f"{t}.tsv.gz" for m in models}
        assert all(f.exists() for f in fs.values()), (t, [m for m, f in fs.items() if not f.exists()])
        tg = data.split_targets(long[long.trait == t], t, "cv3", unit_of[t], fo)
        train_groups = set(fo.dup_group.reindex(list(tg.train_acc)))
        row = {"trait": t, "short": T.SHORT[t], "prim": t in prim, "experiment": batch_of[t],
               "training_accessions": len(set(tg.train_acc))}
        for m, f in fs.items():
            r = pd.read_csv(f, sep="\t", dtype={"acc": str})
            assert r.fold.nunique() == 1 and set(r.acc) == set(tg.test_acc)
            met = evaluate.metrics(evaluate.within_fold_z(r), t, t in rare)
            row[m] = met["primary"]
            row["metric"] = met["primary_metric"]
            g = r.groupby("acc").resistant.max()
            row["test_accessions"], row["resistant_test_accessions"] = len(g), int((g == 1).sum())
            related = r.acc.map(fo.dup_group).isin(train_groups)
            row["test_accessions_with_near_identical_training_accession"] = int(r.acc[related].nunique())
            u = r[~related]
            gu = u.groupby("acc").resistant.max()
            ok = len(gu) >= 30 and u.score.nunique() > 1 and (t not in rare or (gu == 1).sum() >= 1)
            row[f"{m}_unrelated"] = evaluate.metrics(evaluate.within_fold_z(u), t, t in rare)["primary"] if ok else float("nan")
        rows.append(row)
    d = pd.DataFrame(rows)
    first = ["trait", "short", "prim", "experiment", "training_accessions", "test_accessions", "resistant_test_accessions",
             "test_accessions_with_near_identical_training_accession", "metric"]
    d = d[first + models + [f"{m}_unrelated" for m in models]]
    out = ROOT / "results/compare/cv3_point_estimates.tsv"
    d.to_csv(out, sep="\t", index=False, float_format="%.10g")
    pd.set_option("display.width", 250)
    print(d[first[1:] + models].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for sel, lab in ((True, "sparse"), (False, "other")):
        s = d[d.prim == sel]
        print(lab, len(s), "traits:", {m: round(float(s[m].mean()), 4) for m in models})
    print("all", len(d), "traits:", {m: round(float(d[m].mean()), 4) for m in models})
    rel = d.test_accessions_with_near_identical_training_accession.sum() / d.test_accessions.sum()
    u = d.dropna(subset=[f"{m}_unrelated" for m in models])
    print(f"test accessions with a near-identical training accession: {rel:.1%} of the trait x accession cases; "
          f"per trait {100 * (d.test_accessions_with_near_identical_training_accession / d.test_accessions).min():.0f}-"
          f"{100 * (d.test_accessions_with_near_identical_training_accession / d.test_accessions).max():.0f}%")
    print(f"without them ({len(u)} traits):", {m: (round(float(u[m].mean()), 4), round(float(u[m + '_unrelated'].mean()), 4)) for m in models})
    print("wrote", out)


if __name__ == "__main__":
    main()
