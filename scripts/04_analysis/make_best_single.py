"""Best single-trait model: for every trait the single-trait model (GBLUP, BayesB or LightGBM) with the highest accuracy
in the inner cross-validation, with its predictions.

usage: python -I make_best_single.py --nested <scenario>                       (the comparator of the article)
       python -I make_best_single.py <choice_file> <out_name> [scenario]       (one choice from the pooled inner predictions)
       e.g. make_best_single.py --nested cv1; make_best_single.py --nested cv1_inner; make_best_single.py --nested cv3
            make_best_single.py results/gate/cv1_inner_best_single_avg4.tsv best_single_pooled cv1

--nested (writes runs/<scenario>/best_single_avg4):
  cv1        the test records of fold f are predicted by the model chosen in training set f, i.e. the model with the
             highest accuracy on the inner fold left out of that training set (results/gate/inner_choice_by_fold.tsv,
             column chosen_single, from inner_choice_check.py);
  cv1_inner  the inner predictions of training set f (records with fold == f) come from the model chosen in training
             set f (input of the selection index for measured traits, cv2_selection_index.py);
  cv3        for every held-out trait, the model chosen on the pooled inner predictions without the records of the
             accessions of its held-out experiment (results/gate/heldout_choice_without_heldout.tsv, column
             model_without_heldout); GBLUP there is runs/cv3/mtboost_v2_gblup.
  CHOICE.tsv gives the model per trait and fold (cv1, cv1_inner) or per trait (cv3).
Positional form: the same model per trait in all folds, from a choice file made on the pooled inner predictions (kept
as best_single_pooled for the comparison in Supplementary Note 1).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

SINGLE = ["gblup", "bayesb", "lightgbm_avg4"]


def nested(scenario):
    out = data.RUNS / scenario / "best_single_avg4"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if scenario in ("cv1", "cv1_inner"):
        ch = pd.read_csv(ROOT / "results/gate/inner_choice_by_fold.tsv", sep="\t").pivot(index="trait", columns="fold", values="chosen_single")
        rows = []
        for t in T.TRAITS:
            src = {m: pd.read_csv(data.RUNS / scenario / m / f"{t}.tsv.gz", sep="\t", dtype={"acc": str}) for m in SINGLE}
            ref = src[SINGLE[0]]
            for m in SINGLE[1:]:                                   # the three models hold the same records in the same order
                assert (src[m][["acc", "batch", "fold"]].to_numpy() == ref[["acc", "batch", "fold"]].to_numpy()).all(), (t, m)
            o = ref.copy()
            for f in range(5):
                sel = (ref.fold == f).to_numpy()
                o.loc[sel, "pred"] = src[ch.loc[t, f]].pred.to_numpy()[sel]
            o.to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
            rows.append({"trait": t, "short": T.SHORT[t], **{f"fold{f}": ch.loc[t, f] for f in range(5)}})
        pd.DataFrame(rows).to_csv(out / "CHOICE.tsv", sep="\t", index=False)
        c = ch.stack().value_counts().to_dict()
    elif scenario == "cv3":
        h = pd.read_csv(ROOT / "results/gate/heldout_choice_without_heldout.tsv", sep="\t").set_index("trait")
        src_name = {"gblup": "mtboost_v2_gblup"}
        for t, r in h.iterrows():
            shutil.copyfile(data.RUNS / "cv3" / src_name.get(r.model_without_heldout, r.model_without_heldout) / f"{t}.tsv.gz",
                            out / f"{t}.tsv.gz")
        h[["short", "model_without_heldout"]].rename(columns={"model_without_heldout": "model"}).to_csv(out / "CHOICE.tsv", sep="\t")
        c = h.model_without_heldout.value_counts().to_dict()
    else:
        raise SystemExit(f"--nested is defined for cv1, cv1_inner and cv3, not {scenario}")
    print(scenario, "nested choice:", c)


def pooled(choice, name, scenario):
    best = pd.read_csv(ROOT / choice, sep="\t").set_index("trait")
    out = data.RUNS / scenario / name
    out.mkdir(parents=True, exist_ok=True)
    src_name = {"gblup": "mtboost_v2_gblup"} if scenario == "cv3" else {}
    for t, r in best.iterrows():
        src = data.RUNS / scenario / src_name.get(r.best_model, r.best_model) / f"{t}.tsv.gz"
        if not src.exists() and scenario != "cv1":       # traits without a held-out experiment
            continue
        shutil.copyfile(src, out / f"{t}.tsv.gz")
    best[["short", "best_model", "best_value", "primary_trait"]].rename(columns={"best_value": "inner_cv_value"}).to_csv(
        out / "CHOICE.tsv", sep="\t")
    print(best.best_model.value_counts().to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "--nested":
        nested(sys.argv[2])
    else:
        pooled(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "best_single", sys.argv[3] if len(sys.argv) > 3 else "cv1")
