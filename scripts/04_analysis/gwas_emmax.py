"""Mixed-model GWAS read-out (EMMAX / P3D) for every trait of a trait set, on all accessions with folds (step 8 read-out
(d); also the direction of every SNP effect for the shared / antagonistic locus map, step 9).

usage: python gwas_emmax.py --traitset stress|mendel [--device cuda]
Per trait: targets as for the other models (data.fold_targets on all accessions: mean score; numeric traits adjusted
by batch means); VanRaden kernel of the trait's accessions K = U diag(s) U'; REML variance ratio delta of the null model
(reml.SpectralREML); then for every SNP the GLS test of y = mu + x b + g + e with delta fixed:
  transform by D^1/2 U' (D = 1 / (s / delta + 1)), project out the intercept, b = x'y / x'x, chi2 = b^2 x'x / sigma2.
Writes results/gwas/emmax_<traitset>.npz: chi2 and beta (traits x SNPs, beta per alternative allele copy on the
standardised trait scale), n per trait, delta per trait.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/03_model"))
from soymultires import data  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402
import train_mtboost as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traitset", default="stress", choices=["stress", "mendel"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--gpu_mem_gb", type=float, default=6.0)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / tot), 0)
    t0 = time.time()
    long = tb.use_traitset(a.traitset)
    traits = list(tb.TRAITS)
    fo = data.folds()
    long = long[long.acc.isin(set(fo.index))]
    every = pd.Series(0, index=fo.index)
    pos = {v: i for i, v in enumerate(data.samples())}
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    Kmm = data.grm(mmap=True)
    p = G.shape[1]
    chi2 = np.zeros((len(traits), p), np.float32)
    beta = np.zeros((len(traits), p), np.float32)
    ns, deltas = [], []
    for j, t in enumerate(traits):
        ft = data.fold_targets(long[long.trait == t], t, every, 1)
        rows = np.array([pos[x] for x in ft.train_acc])
        o = np.argsort(rows)
        rows, y = rows[o], ft.y_train[o]
        y = (y - y.mean()) / (y.std() + 1e-12)
        n = len(y)
        K = torch.from_numpy(np.asarray(Kmm[np.ix_(rows, rows)], dtype=np.float64)).to(dev)
        s, U = torch.linalg.eigh(K)
        del K
        s = s.clamp_min(0)
        yt = torch.tensor(y, dtype=torch.float64, device=dev)
        m = SpectralREML().fit(torch.zeros(1, device=dev), yt, eig=(s, U))
        d = 1.0 / (s / m.delta + 1.0)
        dh = torch.sqrt(d).float()
        Uf = U.float()
        del U
        one = dh * (Uf.T @ torch.ones(n, device=dev))
        yv = dh * (Uf.T @ yt.float())
        yc = yv - one * (one @ yv) / (one @ one)
        yy = float(yc @ yc)
        for c0 in range(0, p, a.chunk):
            X = torch.from_numpy(np.asarray(G[rows, c0:c0 + a.chunk], dtype=np.float32)).to(dev)
            X = X - X.mean(0)
            Xt = dh[:, None] * (Uf.T @ X)
            Xt = Xt - one[:, None] * (one @ Xt)[None, :] / (one @ one)
            xx = (Xt * Xt).sum(0).clamp_min(1e-8)
            xy = Xt.T @ yc
            b = xy / xx
            s2 = (yy - b * b * xx) / (n - 2)
            chi2[j, c0:c0 + a.chunk] = (b * b * xx / s2.clamp_min(1e-12)).cpu().numpy()
            beta[j, c0:c0 + a.chunk] = b.cpu().numpy()
        ns.append(n)
        deltas.append(m.delta)
        del Uf
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{time.time() - t0:6.0f}s] {t}: n {n}, delta {m.delta:.3f}, max chi2 {chi2[j].max():.1f}", flush=True)
    out = ROOT / "results/gwas"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"emmax_{a.traitset}.npz", chi2=chi2, beta=beta, traits=np.array(traits),
                        n=np.array(ns), delta=np.array(deltas))
    print("written", out / f"emmax_{a.traitset}.npz")


if __name__ == "__main__":
    main()
