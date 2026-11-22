"""Near-identical accessions (genebank duplicates / sub-lines) among genotyped accessions with >=1 stress trait.

usage: python near_identical_groups.py <dosage.npy> <samples.tsv> <accessions.tsv> <out_dir> [threads]
Identity = 1 - sum((g_i - g_j)^2) / (4 m) on 0/1/2 dosages with missing calls (0.03%) set to the allele mean;
for these almost fully homozygous accessions this equals the share of identical genotype calls (IBS).
Accessions linked by identity >= 0.99 are grouped (connected components); the split step keeps each group in
one fold. Memory ~5 GB for 17,150 accessions.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DOS, SAM, ACC, OUT = sys.argv[1:5]
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 8)
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
sam = pd.read_csv(SAM, sep="\t", dtype=str)
acc = pd.read_csv(ACC, sep="\t", dtype=str)
row_of = {v: i for i, v in enumerate(sam.vcf_id)}
acc = acc[acc.acc.isin(row_of)].reset_index(drop=True)
G = np.asarray(np.load(DOS, mmap_mode="r")[np.array([row_of[a] for a in acc.acc])], dtype=np.float32)
miss = G < 0
mu = np.where(miss, 0, G).sum(0) / np.maximum((~miss).sum(0), 1)
G[miss] = mu[np.where(miss)[1]]
del miss
m = G.shape[1]
Gt = torch.from_numpy(G)
sq = (Gt * Gt).sum(1)
gram = Gt @ Gt.T
del Gt, G
ibs = (1 - (sq[:, None] + sq[None, :] - 2 * gram) / (4 * m)).numpy()
del gram
np.fill_diagonal(ibs, 0)
n = len(acc)
thr = [0.995, 0.99, 0.98, 0.95]
summary = {}
for t in thr:
    key = "pairs_" + str(t).replace("0.", "0")  # pairs_0995, pairs_099, ...
    summary[key] = int((ibs >= t).sum() // 2)
    print(f"pairs with IBS >= {t}: {summary[key]}", flush=True)
i, j = np.where(np.triu(ibs >= 0.99, 1))
ncomp, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
acc["dup_group"] = lab
sizes = acc.dup_group.map(acc.dup_group.value_counts())
acc["group_size"] = sizes
print(f"accessions {n}; groups {ncomp}; accessions in groups of size>=2: {(sizes >= 2).sum()}; largest group {sizes.max()}")
nn = ibs.max(1)
acc["max_ibs"] = nn.round(4)
print("max IBS to any other accession: quantiles", np.quantile(nn, [0.1, 0.5, 0.9, 0.99]).round(4))
acc.to_csv(out / "near_identical_groups.tsv", sep="\t", index=False)
summary.update(accessions=n, groups=int(ncomp), acc_in_groups=int((sizes >= 2).sum()),
               groups_ge2=int(acc[acc.group_size >= 2].dup_group.nunique()), largest=int(sizes.max()))
pd.Series(summary).to_csv(out / "near_identical_summary.tsv", sep="\t", header=["value"], index_label="key")
print(f"done in {time.time()-t0:.0f}s")
