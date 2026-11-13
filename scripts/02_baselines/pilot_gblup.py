"""Pilot (plan stage, 2026-11-11): single-trait GBLUP heritability and 5-fold accuracy for the 32 GRIN stress traits.

Not the formal baseline: one random 5-fold split, batch handled by within-batch centring, genotype QC on all
accessions. Purpose: see which traits carry genomic signal and how long the kernel fits take.

usage: python pilot_gblup.py <dosage.npy> <samples.tsv> <scores_by_batch.tsv.gz> <out_dir> [threads] [cpu|cuda] [TRAIT,TRAIT...]
A trait list runs only those traits and merges them into an existing summary (used 2026-11-11 to run the two largest
SCN traits on CPU after the GPU run hit its 5 GiB cap at n = 11,758).
With cuda (gpu10, GPU 0 only, via jobs/launch_gpu0.sh) the per-trait kernels and eigendecompositions run on the GPU
with a 5 GiB cap; the full kernel stays in host memory.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/home/user/zhangzhishuai/myhermes/2080/HapGAM/src")
from hapgam.reml import SpectralREML  # noqa: E402

DOS, SAM, SCORES, OUT = sys.argv[1:5]
TH = int(sys.argv[5]) if len(sys.argv) > 5 else 8
DEV = torch.device(sys.argv[6] if len(sys.argv) > 6 else "cpu")
ONLY = sys.argv[7].split(",") if len(sys.argv) > 7 else None
torch.set_num_threads(TH)
if DEV.type == "cuda":
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 5 * 2**30 / total), 0)
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

sc = pd.read_csv(SCORES, sep="\t", dtype={"acc": str})
sc = sc[(sc.n_batch >= 20) & (sc.sd_batch > 0)]
y_all = sc.groupby(["trait", "acc"]).agg(y=("centred", "mean"), res=("resistant", "max")).reset_index()
sam = pd.read_csv(SAM, sep="\t", dtype=str)
row_of = {v: i for i, v in enumerate(sam.vcf_id)}
y_all = y_all[y_all.acc.isin(row_of)]
accs = sorted(y_all.acc.unique())
rows = np.array([row_of[a] for a in accs])
print(f"accessions with >=1 informative trait value: {len(accs)}", flush=True)

D = np.load(DOS, mmap_mode="r")
X = np.asarray(D[rows], dtype=np.float32)  # n x m, -1 = missing
miss = X < 0
call = 1 - miss.mean(0)
Xm = np.where(miss, np.nan, X)
p = np.nanmean(Xm, 0) / 2
maf = np.minimum(p, 1 - p)
keep = (call >= 0.90) & (maf >= 0.01)
X = X[:, keep]
p = p[keep]
miss = miss[:, keep]
X[miss] = (2 * p)[np.where(miss)[1]]
X -= (2 * p)[None, :]
del Xm, miss
Z = torch.from_numpy(X)
K = Z @ Z.T
del Z, X
K = K.double() / float((2 * p * (1 - p)).sum())
print(f"SNPs kept {keep.sum()} of {len(keep)}; kernel {K.shape[0]} x {K.shape[0]} in {time.time()-t0:.0f}s", flush=True)
pos = {a: i for i, a in enumerate(accs)}

summ = []
rng = np.random.default_rng(20261111)
for trait, d in sorted(y_all.groupby("trait"), key=lambda kv: len(kv[1])):
    if ONLY and trait not in ONLY:
        continue
    t1 = time.time()
    idx = torch.tensor([pos[a] for a in d.acc])
    y = torch.tensor(d.y.to_numpy(), dtype=torch.float64, device=DEV)
    res = d.res.to_numpy()
    Kt = K[idx][:, idx].to(DEV)
    full = SpectralREML().fit(Kt, y)
    folds = rng.permutation(len(d)) % 5
    pred = np.full(len(d), np.nan)
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        mdl = SpectralREML().fit(Kt[tr][:, tr], y[tr])
        pred[te] = mdl.predict(Kt[te][:, tr]).cpu().numpy()
        del mdl
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    yy = d.y.to_numpy()
    r = float(np.corrcoef(pred, yy)[0, 1])
    rec = dict(trait=trait, n=len(d), h2=round(full.h2, 3), r_cv=round(r, 3), seconds=round(time.time() - t1, 1))
    if not np.all(np.isnan(res)):
        lab = (res == 1)
        rec.update(prevalence=round(lab.mean(), 4), n_resistant=int(lab.sum()))
        if 0 < lab.sum() < len(lab):
            rec.update(auroc=round(roc_auc_score(lab, pred), 3), auprc=round(average_precision_score(lab, pred), 3))
            rec["auprc_over_prev"] = round(rec["auprc"] / rec["prevalence"], 2)
    del Kt, full
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    summ.append(rec)
    print(json.dumps(rec), flush=True)
    pd.DataFrame({"acc": d.acc.to_numpy(), "y": yy, "res": res, "fold": folds, "pred": pred}).to_csv(
        out / f"pred_{trait}.tsv.gz", sep="\t", index=False)

new = pd.DataFrame(summ)
old = out / "pilot_gblup_summary.tsv"
if ONLY and old.exists():
    new = pd.concat([pd.read_csv(old, sep="\t").query("trait not in @ONLY"), new], ignore_index=True)
new.sort_values("n", ascending=False).to_csv(old, sep="\t", index=False)
print(f"done in {time.time()-t0:.0f}s", flush=True)
