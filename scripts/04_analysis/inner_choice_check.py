"""The two choices made from the inner cross-validation pooled over the five training sets, made instead within each
training set.

usage: python -I inner_choice_check.py
The inner predictions (runs/cv1_inner/<model>/<trait>.tsv.gz) of the training set of outer fold f are the records of
column fold == f (validation accessions of inner fold f within that training set). Within each training set:
(1) the best single-trait model of every trait (GBLUP, BayesB or LightGBM, highest inner accuracy), whose outer test
    predictions of fold f are then taken from that model (runs/cv1/<model>; make_best_single.py --nested writes them
    as runs/cv1/best_single_avg4, the comparator of the article), and the accuracy of these combined predictions
    compared with that of the best single-trait model chosen once from the pooled inner predictions
    (runs/cv1/best_single_pooled);
(2) the composition of TraitBoost: mean inner accuracy over the 33 traits of the mean of the four refits
    (mtboost_v2_final) and of the chosen option alone (mtboost_v2_share2, the refits of the chosen option), and the
    outer accuracy of TraitBoost when the composition is chosen within each training set (over all 33 traits, or for
    the sparse and the other traits separately; the chosen option alone as mtboost_v2_share4, four refits).
(3) for held-out experiments: both choices made from the pooled inner predictions without the records of the
    accessions of the held-out experiment: the best single-trait model of each held-out trait (the comparator of the
    article, runs/cv3/best_single_avg4 from make_best_single.py --nested) compared with the pooled choice
    (runs/cv3/best_single_pooled), and the composition of TraitBoost judged by the rule of the pooled choice (mean inner
    accuracy of each composition over the sparse and over the other traits, all 33 traits without the accessions of the
    experiment) for every held-out experiment.
Writes results/gate/inner_choice_by_fold.tsv (per trait and training set: inner accuracies and the model chosen),
results/gate/best_single_by_training_set.tsv (per trait: outer accuracy of the two best single-trait models),
results/gate/composition_by_training_set.tsv, results/gate/composition_outer_accuracy.tsv,
results/gate/heldout_choice_without_heldout.tsv and results/gate/heldout_composition_without_heldout.tsv (per held-out
experiment: mean inner accuracy of the three compositions over the sparse and over the other traits without its
accessions, and whether the mean of four refits is best for the groups of its held-out traits).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

SINGLE = ["gblup", "bayesb", "lightgbm_avg4"]
rare = set(data.rare_traits())
rows = []
for m in SINGLE + ["mtboost_v2_final", "mtboost_v2_share2", "mtboost_v2_noshare2"]:
    for t in T.TRAITS:
        rec = pd.read_csv(data.RUNS / "cv1_inner" / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
        for f in range(5):
            mt = evaluate.metrics(evaluate.within_fold_z(rec[rec.fold == f]), t, t in rare)
            rows.append({"model": m, "trait": t, "fold": f, "value": mt.get("primary")})
d = pd.DataFrame(rows)
w = d.pivot_table(index=["trait", "fold"], columns="model", values="value")
w["chosen_single"] = w[SINGLE].idxmax(axis=1)
w.reset_index().to_csv(ROOT / "results/gate/inner_choice_by_fold.tsv", sep="\t", index=False, float_format="%.6f")
pooled = pd.read_csv(ROOT / "results/gate/cv1_inner_best_single_avg4.tsv", sep="\t").set_index("trait").best_model
out = []
for t in T.TRAITS:
    parts = []
    for f in range(5):
        r = pd.read_csv(data.RUNS / "cv1" / w.loc[(t, f), "chosen_single"] / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
        parts.append(r[r.fold == f])
    v = evaluate.metrics(evaluate.within_fold_z(pd.concat(parts)), t, t in rare)["primary"]
    b = pd.read_csv(data.RUNS / "cv1" / "best_single_pooled" / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
    vb = evaluate.metrics(evaluate.within_fold_z(b), t, t in rare)["primary"]
    out.append({"trait": t, "short": T.SHORT[t], "model_pooled_choice": pooled[t],
                "training_sets_with_other_choice": int((w.loc[t, "chosen_single"] != pooled[t]).sum()),
                "accuracy_pooled_choice": vb, "accuracy_choice_within_training_sets": v})
o = pd.DataFrame(out)
o.to_csv(ROOT / "results/gate/best_single_by_training_set.tsv", sep="\t", index=False, float_format="%.6f")
diff = o.accuracy_choice_within_training_sets - o.accuracy_pooled_choice
print(f"best single-trait model chosen within training sets: other model in {int(o.training_sets_with_other_choice.sum())} "
      f"of {5 * len(o)} combinations; accuracy lower for {int((diff < -1e-9).sum())} traits, higher for "
      f"{int((diff > 1e-9).sum())}, equal for {int((diff.abs() <= 1e-9).sum())}; mean {o.accuracy_choice_within_training_sets.mean():.4f} "
      f"vs. {o.accuracy_pooled_choice.mean():.4f}")
# the three compositions compared in the inner cross-validation (Supplementary Note 2): the mean of the two options
# (mtboost_v2_final), the chosen option alone (share2) and the best option without shared trees alone (noshare2)
COMP = ["mtboost_v2_final", "mtboost_v2_share2", "mtboost_v2_noshare2"]
OUTER = {"mtboost_v2_final": "mtboost_v2_final", "mtboost_v2_share2": "mtboost_v2_share4", "mtboost_v2_noshare2": "mtboost_v2_noshare4"}
c = w.groupby(level="fold")[COMP].mean()
c["choice"] = c[COMP].idxmax(axis=1)
c["final_higher"] = c.choice == "mtboost_v2_final"
c.to_csv(ROOT / "results/gate/composition_by_training_set.tsv", sep="\t", float_format="%.6f")
print(f"composition (mean inner accuracy over the {len(T.TRAITS)} traits):")
print(c.round(4).to_string())
# outer accuracy of TraitBoost with the composition chosen within each training set: in each training set, the test
# predictions of the composition with the highest mean inner accuracy (the chosen option alone and the best option
# without shared trees alone with four refits, mtboost_v2_share4 / _noshare4, as for the other comparisons of the article)
sparse = set(data.sparse_traits())
cs = w[COMP].reset_index()                                                       # per trait and training set
cs["is_sparse"] = cs.trait.isin(sparse)
cg = cs.groupby(["fold", "is_sparse"])[COMP].mean()
pick_rule = {"all traits": {f: c.choice[f] for f in range(5)},
             "sparse and other traits separately": {(f, sp): cg.loc[(f, sp)].idxmax() for f in range(5) for sp in (True, False)}}
rows_c = []
for t in T.TRAITS:
    outer = {m: pd.read_csv(data.RUNS / "cv1" / OUTER[m] / f"{t}.tsv.gz", sep="\t", dtype={"acc": str}) for m in COMP}
    fin = outer["mtboost_v2_final"]
    for m in COMP:
        assert (fin.acc.to_numpy() == outer[m].acc.to_numpy()).all() and (fin.fold.to_numpy() == outer[m].fold.to_numpy()).all()
    r_ = {"trait": t, "short": T.SHORT[t], "is_sparse": t in sparse,
          "accuracy_pooled_choice": evaluate.metrics(evaluate.within_fold_z(fin), t, t in rare)["primary"]}
    for rule, sw in pick_rule.items():
        pick = np.array([sw[f] if rule == "all traits" else sw[(f, t in sparse)] for f in fin.fold])
        comb = fin.copy()
        comb["pred"] = fin.pred.to_numpy()
        for m in COMP[1:]:
            comb.loc[pick == m, "pred"] = outer[m].pred.to_numpy()[pick == m]
        r_[f"accuracy_choice_within_training_sets ({rule})"] = evaluate.metrics(evaluate.within_fold_z(comb), t, t in rare)["primary"]
    rows_c.append(r_)
oc = pd.DataFrame(rows_c)
oc.to_csv(ROOT / "results/gate/composition_outer_accuracy.tsv", sep="\t", index=False, float_format="%.6f")
for rule in pick_rule:
    col = f"accuracy_choice_within_training_sets ({rule})"
    print(f"TraitBoost with the composition chosen within each training set ({rule}; other compositions: "
          f"{sorted((str(k), v) for k, v in pick_rule[rule].items() if v != 'mtboost_v2_final')}): mean accuracy {oc[col].mean():.4f} vs. "
          f"{oc.accuracy_pooled_choice.mean():.4f} (sparse {oc[oc.is_sparse][col].mean():.4f} vs. "
          f"{oc[oc.is_sparse].accuracy_pooled_choice.mean():.4f}; other {oc[~oc.is_sparse][col].mean():.4f} vs. "
          f"{oc[~oc.is_sparse].accuracy_pooled_choice.mean():.4f})")

# (3) held-out experiments: choices without the accessions of the held-out experiment
units = data.cv3_units()
ch = {}
for u, r in units.iterrows():
    for t in str(r.chosen_for).split(","):
        if t and t != "nan":
            ch[t] = int(u)
SCV3 = {"gblup": "mtboost_v2_gblup", "bayesb": "bayesb", "lightgbm_avg4": "lightgbm_avg4"}
rows3 = []
for t, u in ch.items():
    if t == "NSCANKER" or not (data.RUNS / "cv3" / "best_single_pooled" / f"{t}.tsv.gz").exists():
        continue
    excl = data.cv3_test_accessions(u)
    acc_in = {}
    for m in SINGLE:
        rec = pd.read_csv(data.RUNS / "cv1_inner" / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
        rec = rec[~rec.acc.isin(excl)]
        acc_in[m] = evaluate.metrics(evaluate.within_fold_z(rec), t, t in rare)["primary"]
    m_new = max(SINGLE, key=lambda m: acc_in[m])
    r3 = pd.read_csv(data.RUNS / "cv3" / SCV3[m_new] / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
    b3 = pd.read_csv(data.RUNS / "cv3" / "best_single_pooled" / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
    rows3.append({"trait": t, "short": T.SHORT[t], "unit": u, "model_pooled_choice": pooled[t], "model_without_heldout": m_new,
                  "accuracy_pooled_choice": evaluate.metrics(evaluate.within_fold_z(b3), t, t in rare)["primary"],
                  "accuracy_choice_without_heldout": evaluate.metrics(evaluate.within_fold_z(r3), t, t in rare)["primary"]})
h3 = pd.DataFrame(rows3)
h3.to_csv(ROOT / "results/gate/heldout_choice_without_heldout.tsv", sep="\t", index=False, float_format="%.6f")
# composition of TraitBoost without the accessions of each held-out experiment, by the rule of the pooled choice: mean
# inner accuracy of each composition over the sparse and over the other traits (all 33 traits)
inner_rec = {(m, t): pd.read_csv(data.RUNS / "cv1_inner" / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
             for m in COMP for t in T.TRAITS}


def _acc_or_nan(r, t):
    """Inner accuracy of trait t on records r; NaN where too few records or classes remain (traits scored only in the
    held-out experiment)."""
    if len(r) < 10:
        return np.nan
    try:
        return evaluate.metrics(evaluate.within_fold_z(r), t, t in rare)["primary"]
    except ValueError:
        return np.nan


rows_u = []
for u in sorted(set(ch.values())):
    excl = data.cv3_test_accessions(u)
    acc_u = {(m, t): _acc_or_nan(r[~r.acc.isin(excl)], t) for (m, t), r in inner_rec.items()}
    held = [t for t, uu in ch.items() if uu == u and (data.RUNS / "cv3" / "best_single_pooled" / f"{t}.tsv.gz").exists()]
    r_ = {"unit": u, "heldout_traits": ",".join(T.SHORT[t] for t in held),
          "heldout_groups": ",".join(sorted({"sparse" if t in sparse else "other" for t in held}))}
    for grp, sel in (("sparse", [t for t in T.TRAITS if t in sparse]), ("other", [t for t in T.TRAITS if t not in sparse])):
        sel = [t for t in sel if all(np.isfinite(acc_u[(m, t)]) for m in COMP)]       # traits with records left
        r_[f"traits_{grp}"] = len(sel)
        means = {m: float(np.mean([acc_u[(m, t)] for t in sel])) for m in COMP}
        for m in COMP:
            r_[f"{m}_{grp}"] = means[m]
        r_[f"winner_{grp}"] = max(means, key=means.get)
    r_["final_best_for_heldout_groups"] = all(r_[f"winner_{g}"] == "mtboost_v2_final" for g in r_["heldout_groups"].split(",") if g)
    if held:
        rows_u.append(r_)
c3 = pd.DataFrame(rows_u)
c3.to_csv(ROOT / "results/gate/heldout_composition_without_heldout.tsv", sep="\t", index=False, float_format="%.6f")
print(f"held-out experiments ({len(h3)} traits): best single-trait model chosen without the held-out accessions differs for "
      f"{int((h3.model_pooled_choice != h3.model_without_heldout).sum())} traits; mean accuracy "
      f"{h3.accuracy_choice_without_heldout.mean():.4f} vs. {h3.accuracy_pooled_choice.mean():.4f}; composition of TraitBoost "
      f"(mean of four refits best for the groups of the held-out traits) unchanged for {int(c3.final_best_for_heldout_groups.sum())} "
      f"of {len(c3)} held-out experiments")
