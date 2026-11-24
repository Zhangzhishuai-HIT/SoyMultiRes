"""Step 5: inputs and collection for BGLR Multitrait by stress category (MegaLMM is infeasible at this size, plan 5.3
rule 2; FREEZE_DEVIATIONS / PROGRESS 11-11). RKHS on the VanRaden kernel, unstructured genetic covariance, diagonal
residual covariance, explicit priors (bglr_mt_fit.R).

usage: python -I bglr_mt_prepare.py prepare --scenario cv1     -> work/bglr_mt/<scenario>/<split>/<category>/
       python -I bglr_mt_prepare.py lists --scenario cv1       -> jobs/lists/bglrmt_<scenario>_todo.txt (largest first)
       python -I bglr_mt_prepare.py collect --scenario cv1     -> runs/<scenario>/bglr_mt/<trait>.tsv.gz
Each directory: Y.tsv (rows = training accessions with a record of a category trait, then the eval accessions with a
record, all NA; targets exactly as the other baselines, data.split_targets), rows.txt (rows of geno/grm.f32.npy),
meta.json (GRM file, header offset, size), test_records_<trait>.tsv.
Held-out experiments: all accessions of the held-out experiment leave the training set, so a trait of the category
can be left without training records or with a single class; such traits are not part of the model of that split,
and a category is fitted for a split only if one of its remaining traits has test records. --split restricts
prepare to one split.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

CAT = {}
for row in T.STRESS:
    CAT.setdefault(row[4], []).append(row[0])


def npy_offset(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        return f.tell()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "lists", "collect"])
    ap.add_argument("--scenario", default="cv1")
    ap.add_argument("--split", type=int, default=None, help="prepare this split only")
    a = ap.parse_args()
    base = ROOT / "work/bglr_mt" / a.scenario
    if a.action == "prepare":
        long = data.pheno_long()
        fo = data.folds()
        pos = {v: i for i, v in enumerate(data.samples())}
        gfile = data.P / "geno/grm.f32.npy"
        meta = {"grm": str(gfile), "offset": npy_offset(gfile), "n_all": len(pos)}
        for f in range(data.n_splits(a.scenario)):
            if a.split is not None and f != a.split:
                continue
            sp = data.split_name(a.scenario, f)
            for cat, traits in CAT.items():
                tg = {t: data.split_targets(long[long.trait == t], t, a.scenario, f, fo) for t in traits}
                dropped = [t for t in traits if len(np.unique(np.asarray(tg[t].y_train, dtype=float))) < 2]
                traits = [t for t in traits if t not in dropped]         # no training record or a single class
                if not any(len(tg[t].test_records) for t in traits):     # held-out experiments: no test record of this category
                    continue
                if dropped:
                    print(sp, cat, "without", ",".join(dropped), "(no training records or one class)")
                d = base / sp / cat
                d.mkdir(parents=True, exist_ok=True)
                tr = sorted(set().union(*[set(tg[t].train_acc) for t in traits]))
                te = sorted(set().union(*[set(tg[t].test_acc) for t in traits]) - set(tr))
                Y = pd.DataFrame(np.nan, index=tr + te, columns=traits)
                for t in traits:
                    Y.loc[tg[t].train_acc, t] = tg[t].y_train
                    tg[t].test_records.assign(fold=f).to_csv(d / f"test_records_{t}.tsv", sep="\t", index=False)
                Y.to_csv(d / "Y.tsv", sep="\t", na_rep="NA")
                np.savetxt(d / "rows.txt", [pos[x] for x in Y.index], fmt="%d")
                json.dump({**meta, "n": len(Y), "n_train": len(tr), "traits": traits}, open(d / "meta.json", "w"))
                (d / "meta.txt").write_text(f"grm\t{meta['grm']}\noffset\t{meta['offset']}\nn_all\t{meta['n_all']}\n")
            print(sp, "prepared")
    elif a.action == "lists":
        todo = []
        for d in sorted(base.glob("*/*")):
            if (d / "pred.tsv").exists():
                continue
            todo.append((json.load(open(d / "meta.json"))["n"], str(d)))
        todo.sort(reverse=True)
        (ROOT / "jobs/lists").mkdir(parents=True, exist_ok=True)
        (ROOT / "jobs/lists" / f"bglrmt_{a.scenario}_todo.txt").write_text("".join(f"{p}\n" for _, p in todo))
        print(len(todo), "directories to fit; largest n", todo[0][0] if todo else 0)
    else:
        out = data.RUNS / a.scenario / "bglr_mt"
        out.mkdir(parents=True, exist_ok=True)
        for cat, traits in CAT.items():
            for t in traits:
                parts, missing = [], 0
                for f in range(data.n_splits(a.scenario)):
                    d = base / data.split_name(a.scenario, f) / cat
                    if not d.exists():                       # held-out experiments: category without test records
                        continue
                    if not (d / "pred.tsv").exists():
                        missing += 1
                        continue
                    P = pd.read_csv(d / "pred.tsv", sep="\t", index_col=0)
                    if t not in P.columns:                   # trait without training records in this split
                        continue
                    rec = pd.read_csv(d / f"test_records_{t}.tsv", sep="\t", dtype={"acc": str})
                    if len(rec):
                        parts.append(rec.assign(pred=rec.acc.map(P[t])))
                if missing:
                    print(t, missing, "folds missing")
                    continue
                if not parts:
                    continue
                pd.concat(parts, ignore_index=True).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
                print(t, "collected")


if __name__ == "__main__":
    main()
