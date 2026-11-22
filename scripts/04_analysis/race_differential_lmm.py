"""Carriers and non-carriers of the alleles of the two race-differential loci among the accessions resistant to SCN
race 3, compared with relatedness accounted for (Supplementary Note 4).

usage: python -I race_differential_lmm.py
For each lead SNP (results/loci/loci_emmax.tsv, class race-differential) and SCN race 1, 5 and 14: the race-3-resistant
accessions scored for the race and homozygous at the lead SNP; resistance to the race (0/1, resistant in at least one
experiment) = mean + b * carrier + g + e, g ~ N(0, s2g K) with the genomic relationship matrix K of these accessions,
e ~ N(0, s2e I), fitted by REML (grid over log(s2e/s2g), step 0.05) with the carrier indicator as fixed effect; b is
tested with t = b / SE(b) on n - 2 degrees of freedom; p_wider_range: the same with the grid from -25 to 10;
p_equal_variances: the same test with s2e = s2g. A linear mixed
model for a 0/1 response, used here only to test the difference between carriers and non-carriers when related
accessions are not independent.
Writes results/loci/race_differential_lmm.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402


def lmm_test(K, y, x, grid=np.arange(-10, 10.0001, 0.05)):
    s, U = np.linalg.eigh(K)
    s = np.clip(s, 0, None)
    X = np.column_stack([np.ones_like(y), x])
    Uy, UX = U.T @ y, U.T @ X
    n, q = X.shape
    best = None
    for ld in grid:
        d = s + np.exp(ld)
        XtDX = UX.T @ (UX / d[:, None])
        b = np.linalg.solve(XtDX, UX.T @ (Uy / d))
        r = Uy - UX @ b
        s2 = (r @ (r / d)) / (n - q)
        ll = -0.5 * ((n - q) * np.log(s2) + np.log(d).sum() + np.linalg.slogdet(XtDX)[1])
        if best is None or ll > best[0]:
            best = (ll, ld, b, s2, XtDX)
    _, ld, b, s2, XtDX = best
    se = np.sqrt(s2 * np.linalg.inv(XtDX)[1, 1])
    t = b[1] / se
    return float(b[1]), float(se), float(t), float(2 * stats.t.sf(abs(t), n - q)), float(ld)


def main():
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    lead = loci[loci["class"] == "race-differential"].sort_values(["chrom", "start"])
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    ids = data.samples()
    pos = {a: i for i, a in enumerate(ids)}
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    G = data.grm(mmap=True)
    long = data.pheno_long()
    r = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.max().unstack()
    r = r[r.index.isin(set(ids))]
    sub3 = r[r.NEMATCYST_3 == 1]
    rows = []
    for _, L in lead.iterrows():
        j = int(np.flatnonzero(snps["id"].to_numpy() == L.lead_snp)[0])
        g = pd.Series(np.asarray(D[:, j]), index=ids)
        for race in (1, 5, 14):
            t = f"NEMATCYST_{race}"
            s_ = sub3[sub3[t].notna()]
            dos = g.reindex(s_.index)
            s_ = s_[dos.isin([0, 2]).to_numpy()]
            x = (g.reindex(s_.index).to_numpy() == 2).astype(float)
            y = s_[t].to_numpy().astype(float)
            ix = np.array([pos[a] for a in s_.index])
            K = np.asarray(G[np.ix_(ix, ix)], dtype=float)
            b, se, tt, p, ld = lmm_test(K, y, x)
            p_wide = lmm_test(K, y, x, grid=np.arange(-25, 10.0001, 0.05))[3]   # wider search range
            p_eq = lmm_test(K, y, x, grid=np.array([0.0]))[3]            # residual and genetic variances equal
            rows.append({"block": int(L.block), "chrom": L.chrom, "lead_snp": L.lead_snp, "race": race, "accessions": len(y),
                         "carriers": int(x.sum()), "carriers_resistant": float(y[x == 1].mean()),
                         "noncarriers_resistant": float(y[x == 0].mean()), "effect": b, "se": se, "t": tt, "p": p,
                         "log_delta": ld, "p_wider_range": p_wide, "p_equal_variances": p_eq})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/loci/race_differential_lmm.tsv", sep="\t", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
