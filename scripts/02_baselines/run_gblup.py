"""Step 4: single-trait GBLUP (VanRaden kernel of step 2, REML via the copied HapGAM SpectralREML).

usage: python run_gblup.py --scenario cv1|cv3 --traits NEMATCYST_3 [...] [--threads 8]

cv1 (考法一): the five frozen folds of splits/folds.tsv. For a single-trait model 考法二 is identical to cv1.
cv3 (考法三): each hold-out batch of splits/cv3_batches.tsv in which the trait was recorded; test = the trait's records
in that batch; training = the trait's other records, minus every record of the test accessions (column fold = unit).
Writes runs/<scenario>/gblup/<trait>.tsv.gz (record-level test predictions: acc, batch, fold, score, resistant, pred)
and <trait>.json (per-fold n, h2, delta, seconds, code hash).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402

CODE = hashlib.sha256((Path(__file__).read_bytes() + (ROOT / "src/soymultires/data.py").read_bytes()
                       + (ROOT / "src/soymultires/reml.py").read_bytes())).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cv1", choices=["cv1", "cv3"])
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    long = data.pheno_long()
    fold_of = data.folds().cv1_fold
    pos = {v: i for i, v in enumerate(data.samples())}
    Kmm = data.grm(mmap=True)
    out = data.RUNS / a.scenario / "gblup"
    out.mkdir(parents=True, exist_ok=True)
    for trait in a.traits:
        t0 = time.time()
        lt = long[long.trait == trait]
        accs = np.array(sorted(lt.acc.unique()))
        rows = np.array([pos[x] for x in accs])
        with threadpool_limits(a.threads):
            Kt = torch.from_numpy(np.asarray(Kmm[np.ix_(rows, rows)], dtype=np.float64))
        loc = {x: i for i, x in enumerate(accs)}
        preds, fits = [], []
        if a.scenario == "cv1":
            plan = [(f, lt, fold_of) for f in range(5)]
        else:
            units = pd.read_csv(data.SPLITS / "cv3_batches.tsv", sep="\t")
            plan = []
            for u in units.itertuples():
                if trait not in u.traits.split(","):
                    continue
                test_acc = set(lt.acc[lt.batch == u.batch])
                keep = (lt.batch == u.batch) | ~lt.acc.isin(test_acc)
                lu = lt[keep]
                plan.append((int(u.unit), lu, pd.Series(np.where(lu.batch == u.batch, int(u.unit), -1), index=lu.acc).groupby(level=0).max()))
        for f, lt_f, fo in plan:
            t1 = time.time()
            ft = data.fold_targets(lt_f, trait, fo, f)
            itr = torch.tensor([loc[x] for x in ft.train_acc])
            ite = torch.tensor([loc[x] for x in ft.test_acc])
            m = SpectralREML().fit(Kt[itr][:, itr], torch.tensor(ft.y_train, dtype=torch.float64))
            p = m.predict(Kt[ite][:, itr]).numpy()
            pm = dict(zip(ft.test_acc, p))
            rec = ft.test_records.assign(fold=f, pred=ft.test_records.acc.map(pm))
            preds.append(rec)
            fits.append(dict(fold=f, n_train=len(ft.y_train), n_test=len(ft.test_acc), h2=round(m.h2, 4),
                             delta=m.delta, seconds=round(time.time() - t1, 1)))
            del m
        pr = pd.concat(preds, ignore_index=True)
        pr.to_csv(out / f"{trait}.tsv.gz", sep="\t", index=False)
        json.dump({"trait": trait, "model": "gblup", "scenario": a.scenario, "code": CODE, "folds": fits,
                   "seconds": round(time.time() - t0, 1)}, open(out / f"{trait}.json", "w"), indent=1)
        print(f"{trait}: n={len(accs)} h2={np.mean([x['h2'] for x in fits]):.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
