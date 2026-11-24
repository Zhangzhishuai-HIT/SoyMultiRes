"""Probe: multi-task boosted trees (XGBoost multi-output trees, masked Gaussian loss) vs the same trees one trait at a
time, on one cv1_inner split, for trait groups (SCN races, Phytophthora races).

Each tree split serves all traits of the group (shared loci); every trait has its own leaf values. Targets as GBLUP
(data.split_targets; standardised per trait); traits balanced in the loss (weight 1 / n_t). Stage A: fit set minus inner
fold g, per-trait early stopping on g (held-out r, checked every 25 rounds). Stage B: whole fit set, each trait's best
number of rounds, predictions for the eval set (inner fold f). Writes runs/cv1_inner/mtgbm_probe/<split>.<group>.tsv.gz
(records with pred_mt, pred_st) and prints per-trait primary metrics next to the cv1_inner baselines.

usage (CPU: vector-leaf trees are CPU-only in XGBoost 3.0.5): python mtgbm_probe.py --fold 0 [--groups scn,pr] [--depth 4] [--eta 0.05]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

DEVICE = "cpu"   # XGBoost 3.0.5: vector-leaf (multi-output) trees are CPU-only
GROUPS = {"scn": [t for t in T.TRAITS if t.startswith("NEMATCYST")],
          "pr": [t for t in T.TRAITS if t.startswith("PHYTOROT")]}


def targets(traits, train_accs, eval_accs):
    long = data.pheno_long()
    split = pd.Series(np.r_[np.zeros(len(train_accs), int), np.ones(len(eval_accs), int)],
                      index=np.r_[np.asarray(train_accs), np.asarray(eval_accs)])
    tr_set, ev_set = set(train_accs), set(eval_accs)
    out = {}
    for t in traits:
        lt = long[(long.trait == t) & long.acc.isin(tr_set | ev_set)]
        ft = data.fold_targets(lt, t, split, 1)
        te = lt[lt.acc.isin(ev_set)]
        if t in T.NUMERIC:
            trr = lt[lt.acc.isin(tr_set)]
            adj = te.score - te.batch.map(trr.groupby("batch").score.mean()).fillna(trr.score.mean())
            yv = adj.groupby(te.acc).mean()
        else:
            yv = te.groupby("acc").score.mean()
        out[t] = (ft, yv.reindex(ft.test_acc).to_numpy(float))
    return out


def geno(accs):
    pos = {v: i for i, v in enumerate(data.samples())}
    rows = np.array([pos[a] for a in accs])
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    o = np.argsort(rows)
    X = np.empty((len(rows), G.shape[1]), dtype=np.float32)
    X[o] = G[rows[o]]
    return X


def prescreen(tg, traits, acc, X, k):
    """Top-k SNPs by the best squared marginal correlation over the group's traits (training records of the stage)."""
    row = {a: i for i, a in enumerate(acc)}
    best = np.zeros(X.shape[1])
    for t in traits:
        ft = tg[t][0]
        idx = np.array([row[a] for a in ft.train_acc])
        Xt = X[idx]
        Xt = Xt - Xt.mean(0)
        y = ft.y_train - ft.y_train.mean()
        r = (Xt.T @ y) / (np.sqrt((Xt ** 2).sum(0)) * np.sqrt((y ** 2).sum()) + 1e-12)
        best = np.maximum(best, r ** 2)
    return np.sort(np.argsort(-best)[:k])


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fit_group(traits, tg, X_tr, acc_tr, params, rounds, X_ev=None, acc_ev=None, early=True, want=None):
    """One multi-output booster for `traits` (one trait = single-task). Returns per-trait best rounds (early) or the
    eval predictions at the requested rounds (want)."""
    row = {a: i for i, a in enumerate(acc_tr)}
    n, Tg = len(acc_tr), len(traits)
    Y = np.full((n, Tg), np.nan)
    mu, sd = np.zeros(Tg), np.ones(Tg)
    for j, t in enumerate(traits):
        ft = tg[t][0]
        y = ft.y_train
        mu[j], sd[j] = y.mean(), y.std() + 1e-8
        Y[[row[a] for a in ft.train_acc], j] = (y - mu[j]) / sd[j]
    M = np.isfinite(Y)
    c = (M.sum(0).mean() / np.maximum(M.sum(0), 1))[None, :]      # traits balanced
    Y0 = np.where(M, Y, 0.0)

    def obj(predt, dtrain):
        p = predt.reshape(n, Tg)
        g = np.where(M, (p - Y0) * c, 0.0)
        h = np.where(M, c, 1e-6) * np.ones_like(p)
        return (g.ravel(), h.ravel()) if Tg == 1 else (g, h)

    dtr = xgb.DMatrix(X_tr, label=np.zeros((n, Tg)))
    p = {"tree_method": "hist", "device": DEVICE, "multi_strategy": "multi_output_tree" if Tg > 1 else "one_output_per_tree",
         "base_score": 0.0, "verbosity": 0, **params}
    if early:
        rowv = {a: i for i, a in enumerate(acc_ev)}
        dev_ = xgb.DMatrix(X_ev)
        best_r = {t: -np.inf for t in traits}
        best_k = {t: 0 for t in traits}
        bst = None
        for k in range(25, rounds + 1, 25):
            bst = xgb.train(p, dtr, num_boost_round=25, obj=obj, xgb_model=bst)
            pr = bst.predict(dev_).reshape(len(acc_ev), Tg)
            for j, t in enumerate(traits):
                ft, yv = tg[t]
                idx = [rowv[a] for a in ft.test_acc]
                r = pearson(pr[idx, j], yv)
                if np.isfinite(r) and r > best_r[t] + 1e-4:
                    best_r[t], best_k[t] = r, k
        return best_k, best_r
    rowv = {a: i for i, a in enumerate(acc_ev)}
    bst = xgb.train(p, dtr, num_boost_round=max(want.values()), obj=obj)
    dev_ = xgb.DMatrix(X_ev)
    out = {}
    for j, t in enumerate(traits):
        k = max(int(want[t]), 1)
        pr = bst.predict(dev_, iteration_range=(0, k)).reshape(len(acc_ev), Tg)[:, j]
        ft = tg[t][0]
        out[t] = dict(zip(ft.test_acc, pr[[rowv[a] for a in ft.test_acc]] * sd[j] + mu[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--groups", default="scn,pr")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tag", default="d4")
    ap.add_argument("--snps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    sc = "cv1_inner"
    fo = data.folds()
    fit, ev = data.split_sets(sc, a.fold, fo)
    g = (a.fold + 1) % 5
    es = fit & (fo.inner_fold == g)
    params = {"max_depth": a.depth, "learning_rate": a.eta, "subsample": a.subsample, "colsample_bynode": a.colsample,
              "lambda": a.lam, "min_child_weight": 1.0, "seed": 20261112, "nthread": a.threads}
    rare = set(data.rare_traits())
    out_dir = data.RUNS / sc / "mtgbm_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = data.split_name(sc, a.fold)
    for gname in a.groups.split(","):
        traits = GROUPS[gname]
        # stage A
        trA, evA = fo.index[fit & ~es], fo.index[es]
        tgA = targets(traits, trA, evA)
        accA = sorted(set().union(*[set(tgA[t][0].train_acc) for t in traits]))
        accAe = sorted(set().union(*[set(tgA[t][0].test_acc) for t in traits]))
        XA, XAe = geno(accA), geno(accAe)
        colA = prescreen(tgA, traits, accA, XA, a.snps)
        XA, XAe = XA[:, colA], XAe[:, colA]
        rowA = {x: i for i, x in enumerate(accA)}
        kmt, rmt = fit_group(traits, tgA, XA, accA, params, a.rounds, XAe, accAe, early=True)
        kst, rst = {}, {}
        for t in traits:
            accs_t = list(tgA[t][0].train_acc)
            k1, r1 = fit_group([t], tgA, XA[[rowA[x] for x in accs_t]], accs_t, params, a.rounds, XAe, accAe, early=True)
            kst.update(k1)
            rst.update(r1)
        print(f"[{time.time() - t0:5.0f}s] {gname} stage A rounds MT {kmt} ST {kst}", flush=True)
        # stage B
        trB, evB = fo.index[fit], fo.index[ev]
        tgB = targets(traits, trB, evB)
        accB = sorted(set().union(*[set(tgB[t][0].train_acc) for t in traits]))
        accBe = sorted(set().union(*[set(tgB[t][0].test_acc) for t in traits]))
        XB, XBe = geno(accB), geno(accBe)
        colB = prescreen(tgB, traits, accB, XB, a.snps)
        XB, XBe = XB[:, colB], XBe[:, colB]
        rowB = {x: i for i, x in enumerate(accB)}
        pmt = fit_group(traits, tgB, XB, accB, params, a.rounds, XBe, accBe, early=False, want=kmt)
        pst = {}
        for t in traits:
            accs_t = list(tgB[t][0].train_acc)
            pst.update(fit_group([t], tgB, XB[[rowB[x] for x in accs_t]], accs_t, params, a.rounds, XBe, accBe,
                                 early=False, want={t: kst[t]}))
        recs = []
        for t in traits:
            rec = tgB[t][0].test_records
            recs.append(rec.assign(trait=t, pred_mt=rec.acc.map(pmt[t]).to_numpy(), pred_st=rec.acc.map(pst[t]).to_numpy()))
        R = pd.concat(recs, ignore_index=True)
        R.to_csv(out_dir / f"{sp}.{gname}.{a.tag}.tsv.gz", sep="\t", index=False)
        # compare with the cv1_inner baselines on the same split
        rows = []
        for t in traits:
            d = R[R.trait == t].drop(columns="trait")
            row = {"short": T.SHORT[t], "mtgbm": evaluate.metrics(d.assign(pred=d.pred_mt), t, t in rare)["primary"],
                   "stgbm": evaluate.metrics(d.assign(pred=d.pred_st), t, t in rare)["primary"]}
            for m in ("gblup", "bayesb", "lightgbm", "hgmt_g32"):
                f = data.RUNS / sc / m / f"{t}.tsv.gz"
                if f.exists():
                    b = pd.read_csv(f, sep="\t", dtype={"acc": str})
                    b = b[b.fold == a.fold]
                    row[m] = evaluate.metrics(b, t, t in rare)["primary"]
            rows.append(row)
        tab = pd.DataFrame(rows)
        print(f"[{time.time() - t0:5.0f}s] {gname} split {a.fold} ({a.tag}):\n" + tab.round(3).to_string(index=False), flush=True)
        print(tab.drop(columns="short").mean().round(4).to_dict(), flush=True)
        json.dump({"group": gname, "split": sp, "params": params, "rounds_mt": kmt, "rounds_st": kst,
                   "table": tab.to_dict("records")}, open(out_dir / f"{sp}.{gname}.{a.tag}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
