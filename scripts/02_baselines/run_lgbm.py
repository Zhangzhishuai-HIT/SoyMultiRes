"""Step 4: LightGBM per trait and fold, tuned with Optuna on inner folds (HapGAM's search space, src/soymultires/trees.py).

usage: python run_lgbm.py --scenario cv1|cv1_inner --trait NEMATCYST_3 --fold 0 [--threads 8] [--trials 15] [--timeout 1800]

Scenarios and tuning folds from soymultires.data (split_sets, tuning_inner_folds): cv1 tunes on inner folds 0, 1, 2 of
the training part; cv1_inner (inside the training part of outer fold f, validation = inner fold f) tunes on three of the
other inner folds. Near-identical groups never split. Writes runs/<scenario>/lightgbm/parts/<trait>.<split>.tsv.gz and
.json; `--collect` merges the five parts into runs/<scenario>/lightgbm/<trait>.tsv.gz.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.trees import fit_lightgbm, tune_lightgbm  # noqa: E402

CODE = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/soymultires/trees.py").read_bytes()
                      + (ROOT / "src/soymultires/data.py").read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cv1", choices=list(data.SCENARIOS))
    ap.add_argument("--trait")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--reseed", type=int, default=0, help="refit with the stored tuned settings and this many extra "
                    "seeds (no tuning): parts/<trait>.<split>.seeds.tsv.gz with pred_s2..; --collect then also writes "
                    "runs/<scenario>/lightgbm_avg<k>/ (mean of the k seeds)")
    a = ap.parse_args()
    out = data.RUNS / a.scenario / "lightgbm"
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    if a.collect:
        for t in T.TRAITS:
            units = [f for f in range(data.n_splits(a.scenario))
                     if a.scenario != "cv3" or t in data.cv3_traits_of(f)]
            if not units:
                continue
            # units skipped for constant training targets (cv3) are left out
            units = [f for f in units if not (parts / f"{t}.{data.split_name(a.scenario, f)}.skipped.json").exists()]
            if not units:
                print(t, "all units skipped")
                continue
            sf = [parts / f"{t}.{data.split_name(a.scenario, f)}.seeds.tsv.gz" for f in units]
            fs = [parts / f"{t}.{data.split_name(a.scenario, f)}.tsv.gz" for f in units]
            if all(f.exists() for f in sf + fs):     # mean over the tuned refit and the extra seeds
                key = ["acc", "batch", "fold"]
                m = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}).merge(
                    pd.read_csv(g, sep="\t", dtype={"acc": str}), on=key) for f, g in zip(fs, sf)])
                sc = ["pred"] + [c for c in m.columns if c.startswith("pred_s")]
                o2 = data.RUNS / a.scenario / f"lightgbm_avg{len(sc)}"
                o2.mkdir(parents=True, exist_ok=True)
                m[["acc", "batch", "score", "resistant", "fold"]].assign(pred=m[sc].mean(1)).to_csv(
                    o2 / f"{t}.tsv.gz", sep="\t", index=False)
            if all(f.exists() for f in fs):
                pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in fs]).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
                print(t, "collected")
            else:
                print(t, "missing", sum(not f.exists() for f in fs), "folds")
        return
    t0 = time.time()
    if a.reseed:
        sp = data.split_name(a.scenario, a.fold)
        js = json.load(open(parts / f"{a.trait}.{sp}.json"))
        long = data.pheno_long()
        fo = data.folds()
        ft = data.split_targets(long[long.trait == a.trait], a.trait, a.scenario, a.fold, fo)
        pos = {v: i for i, v in enumerate(data.samples())}
        G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
        tr_rows = np.array([pos[x] for x in ft.train_acc])
        o = np.argsort(tr_rows)
        Xtr = np.asarray(G[tr_rows[o]], dtype=np.float32)
        ytr = ft.y_train[o]
        te_rows = np.array([pos[x] for x in ft.test_acc])
        oe = np.argsort(te_rows)
        Xte = np.asarray(G[te_rows[oe]], dtype=np.float32)
        rec = ft.test_records[["acc", "batch"]].assign(fold=a.fold)
        for k in range(1, a.reseed + 1):
            params = {**js["params"], "seed": int(js["params"].get("seed", 0)) + 1000 * k, "num_threads": a.threads}
            bst = fit_lightgbm(Xtr, ytr, params, int(js["rounds"]))
            p = np.empty(len(te_rows))
            p[oe] = bst.predict(Xte)
            rec[f"pred_s{k + 1}"] = rec.acc.map(dict(zip(ft.test_acc, p)))
        rec.to_csv(parts / f"{a.trait}.{sp}.seeds.tsv.gz", sep="\t", index=False)
        print(f"{a.trait} {sp}: {a.reseed} extra seeds, {time.time() - t0:.0f}s")
        return
    long = data.pheno_long()
    fo = data.folds()
    lt = long[long.trait == a.trait]
    ft = data.split_targets(lt, a.trait, a.scenario, a.fold, fo)
    pos = {v: i for i, v in enumerate(data.samples())}
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    Xtr = np.asarray(G[np.sort([pos[x] for x in ft.train_acc])], dtype=np.float32)
    order_tr = np.argsort([pos[x] for x in ft.train_acc])
    train_acc = ft.train_acc[order_tr]
    ytr = ft.y_train[order_tr]
    inner = fo.inner_fold.reindex(train_acc).to_numpy()
    val_sets = [np.flatnonzero(inner == k) for k in data.tuning_inner_folds(a.scenario, a.fold)]
    sp = data.split_name(a.scenario, a.fold)
    if np.unique(ytr).size < 2:      # e.g. cv3 unit whose remaining training records are all one class: not predictable
        json.dump({"trait": a.trait, "fold": a.fold, "scenario": a.scenario, "model": "lightgbm",
                   "skipped": "constant training targets", "n_train": len(ytr)},
                  open(parts / f"{a.trait}.{sp}.skipped.json", "w"), indent=1)
        print(f"{a.trait} {sp}: constant training targets, skipped")
        return
    # validation parts with fewer than 10 records or a single class give no correlation (cv3 can leave such parts)
    val_sets = [v for v in val_sets if len(v) >= 10 and np.unique(ytr[v]).size > 1]
    params, rounds, study = tune_lightgbm(Xtr, ytr, val_sets, n_trials=a.trials, n_threads=a.threads,
                                          seed=20261111, timeout=a.timeout)
    bst = fit_lightgbm(Xtr, ytr, params, rounds)
    del Xtr
    te_rows = np.array([pos[x] for x in ft.test_acc])
    o = np.argsort(te_rows)
    Xte = np.asarray(G[te_rows[o]], dtype=np.float32)
    p = np.empty(len(te_rows))
    p[o] = bst.predict(Xte)
    pm = dict(zip(ft.test_acc, p))
    rec = ft.test_records.assign(fold=a.fold, pred=ft.test_records.acc.map(pm))
    sp = data.split_name(a.scenario, a.fold)
    rec.to_csv(parts / f"{a.trait}.{sp}.tsv.gz", sep="\t", index=False)
    json.dump({"trait": a.trait, "fold": a.fold, "scenario": a.scenario, "model": "lightgbm", "code": CODE, "params": params, "rounds": rounds,
               "trials_done": len(study.trials), "best_inner_r": study.best_value, "n_train": len(ytr),
               "n_test": len(te_rows), "seconds": round(time.time() - t0, 1)},
              open(parts / f"{a.trait}.{sp}.json", "w"), indent=1, default=float)
    print(f"{a.trait} fold {a.fold}: inner r {study.best_value:.3f}, {len(study.trials)} trials, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
