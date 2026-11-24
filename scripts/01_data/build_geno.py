"""Step 2 of the plan: phenotype-free genotype preparation for all 20,087 SoySNP50K accessions.

usage: python build_geno.py [threads]      (needs about 7.5 GB of memory)

Writes data/processed/geno/:
  dosage.int8       memmap int8 [n_accessions, n_snps], ALT dosage 0/1/2; the 0.0x% missing calls set to the major
                    homozygote (as in HapGAM); row order = samples.tsv, column order = snps.tsv
  dosage.json       shape and dtype
  samples.tsv       vcf_id in row order
  snps.tsv          chrom, pos, id, ref, alt, alt_freq, maf, call_rate, block8, block16, block32
  grm.f32.npy       VanRaden kernel Z Z' / (2 sum p(1-p)) over all accessions (float32)
  pcs.tsv           top 10 eigenvectors of the kernel (with eigenvalues in pcs_eigenvalues.tsv)
(the table of known genes and loci, data/processed/known_genes.tsv, is written by known_loci.py).
QC (all accessions, no phenotypes): call rate >= 0.95 and MAF >= 0.01. Blocks: HapGAM's fixed-size schemes
(8, 16, 32 consecutive SNPs within a chromosome).
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires.blocks import fixed_blocks  # noqa: E402

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HG = ROOT.parent / "HapGAM/data/interim/soysnp50k"            # parsed VCF (read-only)
GFF = ROOT / "data/raw/wm82_gnm2_annotation/glyma.Wm82.gnm2.ann1.RVB6.gene_models_main.gff3.gz"
OUT = ROOT / "data/processed/geno"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

D = np.load(HG / "dosage.npy")                                 # int8 (0.83 GB in memory), -1 = missing
var = pd.read_csv(HG / "variants.tsv", sep="\t")
sam = pd.read_csv(HG / "samples.tsv", sep="\t", dtype=str)
n, m = D.shape
assert len(sam) == n and len(var) == m

# --- per-SNP call rate and ALT frequency, in column chunks (low memory)
called = np.zeros(m); altsum = np.zeros(m)
for s in range(0, m, 4000):
    blk = np.asarray(D[:, s:s + 4000])
    ok = blk >= 0
    called[s:s + 4000] = ok.sum(0)
    altsum[s:s + 4000] = np.where(ok, blk, 0).sum(0)
call_rate = called / n
p = altsum / np.maximum(2 * called, 1)
maf = np.minimum(p, 1 - p)
keep = (call_rate >= 0.95) & (maf >= 0.01)
idx = np.flatnonzero(keep)
print(f"SNPs: {m} -> {keep.sum()} (call rate >= 0.95: {(call_rate >= 0.95).sum()}, MAF >= 0.01: {(maf >= 0.01).sum()})", flush=True)

# --- QC'd dosage memmap, missing -> major homozygote
mm = np.lib.format.open_memmap(OUT / "dosage.npy", mode="w+", dtype=np.int8, shape=(n, len(idx)))
fill = np.where(p[idx] >= 0.5, 2, 0).astype(np.int8)
miss_total = 0
for s in range(0, len(idx), 4000):
    cols = idx[s:s + 4000]
    blk = np.asarray(D[:, cols])
    miss = blk < 0
    miss_total += int(miss.sum())
    blk = np.where(miss, fill[s:s + 4000][None, :], blk).astype(np.int8)
    mm[:, s:s + len(cols)] = blk
mm.flush()
print(f"missing calls imputed: {miss_total} ({miss_total / (n * len(idx)):.5%})", flush=True)
json.dump({"shape": [n, int(len(idx))], "dtype": "int8", "file": "dosage.npy"}, open(OUT / "dosage.json", "w"))
sam[["vcf_id"]].to_csv(OUT / "samples.tsv", sep="\t", index=False)

snps = var.iloc[idx].reset_index(drop=True).copy()
snps["alt_freq"] = p[idx].round(5)
snps["maf"] = maf[idx].round(5)
snps["call_rate"] = call_rate[idx].round(5)
chrom = snps.chrom.to_numpy()
for k in (8, 16, 32):
    snps[f"block{k}"] = fixed_blocks(chrom, k)
snps.to_csv(OUT / "snps.tsv", sep="\t", index=False)
print("blocks:", {k: int(snps[f'block{k}'].max() + 1) for k in (8, 16, 32)}, flush=True)

# --- VanRaden kernel, accumulated over SNP chunks (float32)
pp = snps.alt_freq.to_numpy()
K = np.zeros((n, n), dtype=np.float32)
with threadpool_limits(TH):
    for s in range(0, len(idx), 3000):
        Z = mm[:, s:s + 3000].astype(np.float32) - (2 * pp[s:s + 3000]).astype(np.float32)[None, :]
        K += Z @ Z.T
    K /= np.float32(2 * np.sum(pp * (1 - pp)))
np.save(OUT / "grm.f32.npy", K)
print(f"kernel done ({n} x {n}), mean diag {float(np.mean(np.diag(K))):.3f}, {time.time() - t0:.0f}s", flush=True)
with threadpool_limits(TH):
    w, V = eigsh(K, k=10, which="LA")
order = np.argsort(w)[::-1]
pcs = pd.DataFrame(V[:, order], columns=[f"PC{i + 1}" for i in range(10)])
pcs.insert(0, "vcf_id", sam.vcf_id)
pcs.to_csv(OUT / "pcs.tsv", sep="\t", index=False, float_format="%.6f")
pd.DataFrame({"pc": [f"PC{i + 1}" for i in range(10)], "eigenvalue": w[order],
              "share_of_trace": w[order] / float(np.trace(K))}).to_csv(OUT / "pcs_eigenvalues.tsv", sep="\t", index=False)
del K

# The table of genes and loci of known position (data/processed/known_genes.tsv) is written by known_loci.py.
