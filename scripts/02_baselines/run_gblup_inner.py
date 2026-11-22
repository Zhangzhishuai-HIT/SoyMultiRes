"""GBLUP on the cv1_inner splits (soymultires.data.split_sets): inside the training part of outer fold f, validation
= inner fold f. Same targets and REML as run_gblup.py. Used for the multi-task model's settings and the step-6 check
(plan 3.6) together with BayesB and LightGBM on the same splits.

usage: python run_gblup_inner.py --fold 0 [--threads 8]   -> runs/cv1_inner/gblup/parts/fold<f>_inner<f>.tsv.gz
       python run_gblup_inner.py --fold 0 --traits PHYTOROT_10   -> parts/fold<f>_inner<f>.PHYTOROT_10.tsv.gz (only these
                                                          traits; used when a trait was added to the data later)
       python run_gblup_inner.py --collect                -> runs/cv1_inner/gblup/<trait>.tsv.gz (trait files of a split
                                                          replace the rows of their traits in the split's main file)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402

SC = "cv1_inner"
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--collect", action="store_true")
ap.add_argument("--traits", nargs="*", default=None, help="only these traits (written to a separate part file)")
a = ap.parse_args()
out = data.RUNS / SC / "gblup"
(out / "parts").mkdir(parents=True, exist_ok=True)
if a.collect:
    fs = [out / "parts" / f"{data.split_name(SC, f)}.tsv.gz" for f in range(5)]
    frames = []
    for f in fs:
        if not f.exists():
            continue
        main_ = pd.read_csv(f, sep="\t", dtype={"acc": str})
        extra = [pd.read_csv(e, sep="\t", dtype={"acc": str}) for e in sorted(f.parent.glob(f.name.replace(".tsv.gz", ".*.tsv.gz")))]
        if extra:
            ex = pd.concat(extra, ignore_index=True)
            main_ = pd.concat([main_[~main_.trait.isin(set(ex.trait))], ex], ignore_index=True)
        frames.append(main_)
    allp = pd.concat(frames, ignore_index=True)
    for t, d in allp.groupby("trait"):
        d.drop(columns="trait").to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
    print(f"collected {allp.trait.nunique()} traits from {sum(f.exists() for f in fs)} of 5 splits")
    sys.exit()
torch.set_num_threads(a.threads)
long = data.pheno_long()
fo = data.folds()
pos = {v: i for i, v in enumerate(data.samples())}
Kmm = data.grm(mmap=True)
parts = []
for t in (a.traits or T.TRAITS):
    ft = data.split_targets(long[long.trait == t], t, SC, a.fold, fo)
    accs = np.concatenate([ft.train_acc, ft.test_acc])
    Kt = torch.from_numpy(np.asarray(Kmm[np.ix_([pos[x] for x in accs], [pos[x] for x in accs])], dtype=np.float64))
    n = len(ft.train_acc)
    m = SpectralREML().fit(Kt[:n, :n], torch.tensor(ft.y_train, dtype=torch.float64))
    pm = dict(zip(ft.test_acc, m.predict(Kt[n:, :n]).numpy()))
    parts.append(ft.test_records.assign(fold=a.fold, trait=t, pred=ft.test_records.acc.map(pm)))
    print(t, n, len(ft.test_acc), flush=True)
suffix = ("." + "_".join(a.traits)) if a.traits else ""
pd.concat(parts, ignore_index=True).to_csv(out / "parts" / f"{data.split_name(SC, a.fold)}{suffix}.tsv.gz", sep="\t", index=False)
