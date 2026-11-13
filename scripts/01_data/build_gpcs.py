"""Genomic principal components for the multi-task model's background (phenotype-free, all 20,087 accessions).

usage: python build_gpcs.py [q] [threads]
Randomised SVD (Halko et al.; 3 power iterations) of the standardised dosage matrix Z (n x p):
Z ~ U S W'. Writes data/processed/geno/gpcs_<q>.npy = U S / sqrt(p) (float32, n x q), so that Psi Psi' approximates
the VanRaden-type kernel Z Z' / p; plus gpcs_<q>_sv.tsv (singular values, share of the total sum of squares).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
G = ROOT / "data/processed/geno"
t0 = time.time()
X = np.load(G / "dosage.npy", mmap_mode="r")
snps = pd.read_csv(G / "snps.tsv", sep="\t")
p_ = snps.alt_freq.to_numpy()
mu = torch.tensor(2 * p_, dtype=torch.float32)
sd = torch.tensor(np.sqrt(np.maximum(2 * p_ * (1 - p_), 1e-6)), dtype=torch.float32)
n, p = X.shape
Z = torch.empty((n, p), dtype=torch.float32)
for i in range(0, n, 2000):
    Z[i:i + 2000] = (torch.from_numpy(np.asarray(X[i:i + 2000])).float() - mu) / sd
total_ss = float((Z ** 2).sum())
print(f"Z {n} x {p} built, {time.time() - t0:.0f}s", flush=True)
g = torch.Generator().manual_seed(20261111)
k = q + 50
Y = Z @ torch.randn(p, k, generator=g)
for _ in range(3):                               # power iterations with re-orthonormalisation
    Q, _ = torch.linalg.qr(Y)
    Y = Z @ (Z.T @ Q)
Q, _ = torch.linalg.qr(Y)
B = Q.T @ Z                                      # k x p
Ub, S, _ = torch.linalg.svd(B, full_matrices=False)
U = (Q @ Ub)[:, :q]
S = S[:q]
psi = (U * S) / np.sqrt(p)
np.save(G / f"gpcs_{q}.npy", psi.numpy().astype(np.float32))
pd.DataFrame({"pc": np.arange(1, q + 1), "singular_value": S.numpy(),
              "share_ss": (S ** 2).numpy() / total_ss}).to_csv(G / f"gpcs_{q}_sv.tsv", sep="\t", index=False)
print(f"q={q}: top {q} PCs explain {float((S ** 2).sum()) / total_ss:.3f} of the sum of squares; {time.time() - t0:.0f}s")
