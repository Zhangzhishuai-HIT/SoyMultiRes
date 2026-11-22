"""Association of SCN races 1, 3, 5 and 14 with the chromosome 10 lead SNP (ss715606985) when SNP haplotypes at Rhg1
are fitted as fixed effects (Supplementary Note 4).

usage: python -I chr10_haplotype_conditional.py [--threads 8]
For each race, the test of the main association analysis (gwas_emmax.py: the values used for model fitting of all
scored accessions, standardized; REML estimate of the variance ratio with the genomic relationship matrix; GLS test of
the SNP with the ratio fixed) is computed at the chromosome 10 lead SNP only, (a) with the intercept alone, as in the
main analysis, (b) with the indicator of the Peking SNP haplotype as a further fixed effect and (c) with the indicators
of the Peking and of the PI 88788 SNP haplotypes. SNP haplotypes as in rhg1_haplotype_check.py: an accession has the
Peking (PI 88788) SNP haplotype when its calls at the 17 SNPs of chromosome 18 between 1.56 and 1.72 Mb equal those of
PI 548402 (PI 88788). The variance ratio is re-estimated with the fixed effects of each analysis, as in the
conditional analyses of gwas_emmax.py. Writes results/loci/chr10_haplotype_conditional.tsv (race, analysis, scored
accessions, those with the Peking and with the PI 88788 SNP haplotype, homozygous carriers of the alternative allele,
z at the lead SNP; analysis (a) reproduces the main analysis).
"""
import argparse
import sys
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

SNP = "ss715606985"
RHG1_WIN = ("Gm18", 1_560_000, 1_720_000)


def z_at(y, s, U, X, C):
    """GLS test of the column X (centred) given the fixed effects C (centred columns or None), as in gwas_emmax.py."""
    n = len(y)
    m = SpectralREML().fit(torch.zeros(1, dtype=torch.float64), y, C=C, eig=(s, U))
    d = 1.0 / (s / m.delta + 1.0)
    dh = torch.sqrt(d)
    W = torch.ones(n, 1, dtype=torch.float64) if C is None else torch.cat([torch.ones(n, 1, dtype=torch.float64), C], 1)
    Q = torch.linalg.qr(dh[:, None] * (U.T @ W))[0]
    q = Q.shape[1]
    yv = dh * (U.T @ y)
    yc = yv - Q @ (Q.T @ yv)
    xt = dh * (U.T @ X)
    xt = xt - Q @ (Q.T @ xt)
    xx, xy = float(xt @ xt), float(xt @ yc)
    b = xy / xx
    s2 = (float(yc @ yc) - b * b * xx) / (n - q - 1)
    return float(np.sign(b) * np.sqrt(b * b * xx / s2)), float(m.delta), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    long = tb.use_traitset("stress")
    fo = data.folds()
    long = long[long.acc.isin(set(fo.index))]
    every = pd.Series(0, index=fo.index)
    ids = data.samples()
    pos = {v: i for i, v in enumerate(ids)}
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    Kmm = data.grm(mmap=True)
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    j = int(np.flatnonzero(snps.id.to_numpy() == SNP)[0])
    w1 = np.flatnonzero((snps.chrom == RHG1_WIN[0]).to_numpy() & (snps.pos >= RHG1_WIN[1]).to_numpy()
                        & (snps.pos <= RHG1_WIN[2]).to_numpy())
    X1 = np.asarray(G[:, w1])
    hap_pk = (X1 == X1[pos["PI548402"]]).all(1).astype(float)
    hap_88 = (X1 == X1[pos["PI88788"]]).all(1).astype(float)
    main_ = np.load(ROOT / "results/gwas/emmax_stress.npz", allow_pickle=True)
    tr_main = [str(x) for x in main_["traits"]]
    rows = []
    for race in (1, 3, 5, 14):
        t = f"NEMATCYST_{race}"
        ft = data.fold_targets(long[long.trait == t], t, every, 1)
        r_ = np.array([pos[x] for x in ft.train_acc])
        o = np.argsort(r_)
        r_, y = r_[o], ft.y_train[o]
        y = (y - y.mean()) / (y.std() + 1e-12)
        K = torch.from_numpy(np.asarray(Kmm[np.ix_(r_, r_)], dtype=np.float64))
        s, U = torch.linalg.eigh(K)
        del K
        s = s.clamp_min(0)
        yt = torch.tensor(y, dtype=torch.float64)
        x = torch.tensor(np.asarray(G[r_, j], dtype=np.float64))
        x = x - x.mean()
        cen = lambda v: torch.tensor(v - v.mean(), dtype=torch.float64)[:, None]  # noqa: E731
        hp, h8 = hap_pk[r_], hap_88[r_]
        k_ = tr_main.index(t)
        z_main = float(np.sign(main_["beta"][k_][j]) * np.sqrt(main_["chi2"][k_][j]))
        for name, C in (("intercept only (main analysis)", None), ("Peking SNP haplotype fitted", cen(hp)),
                        ("Peking and PI 88788 SNP haplotypes fitted", torch.cat([cen(hp), cen(h8)], 1))):
            z, delta, q = z_at(yt, s, U, x, C)
            rows.append({"race": race, "analysis": name, "accessions": len(y), "peking_haplotype": int(hp.sum()),
                         "pi88788_haplotype": int(h8.sum()), "carriers": int((np.asarray(G[r_, j]) == 2).sum()),
                         "fixed_effects": q, "delta": delta, "z": z, "z_main_analysis": z_main})
        print(pd.DataFrame(rows[-3:]).round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    a_ = out[out.analysis.str.startswith("intercept")]
    assert (abs(a_.z - a_.z_main_analysis) < 0.02).all(), a_[["race", "z", "z_main_analysis"]]   # reproduces the main analysis
    out.to_csv(ROOT / "results/loci/chr10_haplotype_conditional.tsv", sep="\t", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
