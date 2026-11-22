"""Genome-wide significance thresholds of every stress trait from simulated traits without SNP effects.

usage: python -I gwas_null_threshold.py [--traitset stress|mendel] [--reps 1000] [--min_res 0] [--threads 8]
For every trait of the set (or, with --min_res 100, the 11 stress traits with fewer than 100 resistant accessions;
mendel: the four qualitative traits and maturity group used to choose the score): the accessions and values of the main association analysis (gwas_emmax.py:
data.fold_targets on all accessions, standardised), the relationship matrix K of these accessions = U diag(s) U' and the
REML ratio delta of residual to genetic variance of the observed values. A simulated trait is a liability
l = g + e, g ~ N(0, K), e ~ N(0, delta_L I), a polygenic background without effects of single SNPs, whose values are
replaced by the observed values in the order of l: the accession with the highest liability receives the highest
observed value, so that the distribution of the observed values (skew and ties included) is kept. The ratio delta_L
of the liability is chosen (bisection on log delta_L, 200 simulated traits with common random numbers) so that the
median REML ratio of the simulated traits equals the REML ratio delta of the observed trait (--match 0: delta_L =
delta; with skewed scores the simulated traits are then much less heritable than the observed trait). For the SNPs
given with --watch, the signed statistics z of all simulated traits are kept, and the observed z is compared with
them (empirical two-sided P = (k + 1) / (reps + 1), k = simulated traits with |z| at least the observed |z|;
results/gwas/null_threshold_watch.tsv). With --watch_blocks block:SNP,..., the largest |z| over the tested SNPs of
each block is kept for every simulated trait and compared with the observed |z| at the given SNP (the lead SNP of the
block), a comparison that allows for the choice of the lead SNP among the SNPs of the block
(results/gwas/null_threshold_watch_blocks.tsv, written also with --only). Every simulated trait is analysed as the observed
traits in gwas_emmax.py (REML ratio of the simulated trait, maximised on a grid of log delta with step 0.05; GLS
statistic of every SNP tested for the trait, i.e. with at least 30 accessions homozygous for the minor allele), and its
largest statistic over the tested SNPs is kept. The genome-wide threshold at a family-wise error rate of 5% is the 95th
percentile of these largest statistics, given as a P value of the chi-square distribution with one degree of freedom.
The implementation is checked against the statistics of the main analysis (observed values, delta of the main
analysis) for the first chunk of SNPs of every trait.
A block of 16 SNPs is a locus of a trait at a threshold when one of its tested SNPs reaches it (as in locus_map.py).
Writes results/gwas/null_threshold<_mendel>.tsv (per trait: accessions, resistant accessions, delta, replicates,
threshold P, the share of replicates whose largest statistic exceeds the Bonferroni threshold P = 0.05 / 39,627, i.e.
the family-wise error rate of the Bonferroni threshold, and the blocks that are loci at the Bonferroni threshold and
at the stricter of the two thresholds) and results/gwas/null_threshold_loci<_mendel>.tsv (every block that is a locus
of a trait at the Bonferroni threshold, with the smallest P of the block and whether it reaches the stricter threshold).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2 as chi2d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/03_model"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402
import train_mtboost as tb  # noqa: E402


# trait order of the first analysis (32 traits), which sets the random seeds; P. sojae race 10 was added later
SEED_ORDER = ['NEMATCYST_1', 'NEMATCYST_3', 'NEMATCYST_4', 'NEMATCYST_5', 'NEMATCYST_14', 'PHYTOROT_1', 'PHYTOROT_3', 'PHYTOROT_4', 'PHYTOROT_5', 'PHYTOROT_7', 'PHYTOROT_12', 'PHYTOROT_17', 'PHYTOROT_20', 'PHYTOROT_25', 'SDS', 'BRNSTEMROT', 'NSCANKER', 'FROGEYE2', 'FROGEYE3', 'RUST_TAN', 'PYTHIUMROT', 'BP', 'PMV', 'BPMV', 'MEXBEANBTL', 'LOOPER', 'VBC', 'APHID', 'POTATOLHOP', 'CHLOROSIS', 'SALTREACT', 'HIGHTEMP', 'PHYTOROT_10']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--traitset", default="stress", choices=["stress", "mendel"])
    ap.add_argument("--min_res", type=int, default=0, help="only traits with fewer resistant accessions (0: all)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--only", default="", help="test run: these traits only (comma-separated); no files written")
    ap.add_argument("--add_traits", default="", help="only these traits, merged into the existing output files")
    ap.add_argument("--match", type=int, default=1, help="1: liability ratio matched to the observed REML ratio")
    ap.add_argument("--match_reps", type=int, default=200)
    ap.add_argument("--watch", default="ss715606985,ss715628625",
                    help="SNPs whose statistics in the simulated traits are kept (empirical P of the observed z)")
    ap.add_argument("--watch_blocks", default="",
                    help="block:SNP,... largest |z| of the block in every simulated trait vs the observed |z| at SNP")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
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
    main_ = np.load(ROOT / f"results/gwas/emmax_{a.traitset}.npz", allow_pickle=True)
    mac_ = np.load(ROOT / f"results/gwas/mac_{a.traitset}.npz", allow_pickle=True)
    assert [str(x) for x in main_["traits"]] == traits and [str(x) for x in mac_["traits"]] == traits
    if a.traitset == "stress":
        res = pd.read_csv(data.P / "resistant_matrix.tsv.gz", sep="\t", index_col=0).reindex(fo.index)
        nres = (res == 1).sum()
    else:
        nres = pd.Series(-1, index=traits)
    sel = [t for t in traits if a.min_res == 0 or nres[t] < a.min_res]
    if a.only:
        sel = [t for t in sel if t in a.only.split(",")]
    add = [x for x in a.add_traits.split(",") if x]
    if add:
        sel = [t for t in sel if t in add]
    # random numbers of a trait: seed + its position in the trait list of the first analysis (32 traits); traits added
    # to the data later follow at the end (P. sojae race 10: 32), so that every trait keeps its random numbers
    seed_idx = ({t: i for i, t in enumerate(SEED_ORDER)} if a.traitset == "stress" else {t: i for i, t in enumerate(traits)})
    assert set(traits) <= set(seed_idx), set(traits) - set(seed_idx)
    print(f"{len(sel)} traits: {', '.join(sel)}", flush=True)
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    blk = snps.block16.to_numpy()
    chi_bonf = float(chi2d.isf(0.05 / p, 1))
    lds = np.arange(-12.0, 12.0 + 1e-9, 0.05)
    rows, lrows, wrows, wbrows = [], [], [], []
    for t in sel:
        j = traits.index(t)
        ft = data.fold_targets(long[long.trait == t], t, every, 1)
        r_ = np.array([pos[x] for x in ft.train_acc])
        o = np.argsort(r_)
        r_, y = r_[o], ft.y_train[o]
        y = (y - y.mean()) / (y.std() + 1e-12)
        n = len(y)
        assert n == int(main_["n"][j]), (t, n, main_["n"][j])
        K = torch.from_numpy(np.asarray(Kmm[np.ix_(r_, r_)], dtype=np.float64))
        s, U = torch.linalg.eigh(K)
        del K
        s = s.clamp_min(0)
        delta = SpectralREML().fit(torch.zeros(1), torch.from_numpy(y), eig=(s, U)).delta
        assert np.isclose(delta, float(main_["delta"][j]), rtol=1e-6), (t, delta, main_["delta"][j])
        ys = torch.from_numpy(np.sort(y))
        u1 = U.T @ torch.ones(n, dtype=torch.float64)
        ldg = torch.from_numpy(np.exp(lds))
        ddg = s[:, None] + ldg[None, :]                                    # n x grid
        Wg_, logdet_ = 1.0 / ddg, torch.log(ddg).sum(0)
        A_ = (u1 * u1) @ Wg_

        def reml_grid(UY_):
            """REML ratio of every column of U'y (intercept only), maximised on the grid."""
            B_ = (UY_ * u1[:, None]).T @ Wg_
            C_ = (UY_ * UY_).T @ Wg_
            R_ = (C_ - B_ * B_ / A_[None, :]).clamp_min(1e-300)
            ll_ = -0.5 * ((n - 1) * torch.log(2 * math.pi * R_ / (n - 1)) + logdet_[None, :] + torch.log(A_)[None, :])
            return ldg[ll_.argmax(1)]

        # liability ratio: the median REML ratio of the simulated traits equals the observed one
        delta_L, matched = delta, False
        if a.match:
            gb = torch.Generator().manual_seed(a.seed + 10000 + seed_idx[t])
            G0b = U @ (torch.sqrt(s)[:, None] * torch.randn(n, a.match_reps, generator=gb, dtype=torch.float64))
            E0b = torch.randn(n, a.match_reps, generator=gb, dtype=torch.float64)

            def f_(x):
                Lb = G0b + math.exp(0.5 * x) * E0b
                Yb = ys[torch.argsort(torch.argsort(Lb, 0), 0)]
                return math.log(float(reml_grid(U.T @ Yb).median())) - math.log(delta)

            lo, hi = math.log(1e-3), math.log(1e3)
            if f_(lo) > 0:
                delta_L = 1e-3
            elif f_(hi) < 0:
                delta_L = 1e3
            else:
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    if f_(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                delta_L, matched = math.exp(0.5 * (lo + hi)), True
            del G0b, E0b
        # simulated traits: polygenic liability, observed values assigned in the order of the liability
        gen = torch.Generator().manual_seed(a.seed + seed_idx[t])
        L = U @ (torch.sqrt(s)[:, None] * torch.randn(n, a.reps, generator=gen, dtype=torch.float64)) \
            + math.sqrt(delta_L) * torch.randn(n, a.reps, generator=gen, dtype=torch.float64)
        Y = ys[torch.argsort(torch.argsort(L, 0), 0)]                     # n x reps, the observed values in liability order
        del L
        UY = U.T @ Y
        delta_r = reml_grid(UY)                                             # REML ratio of every simulated trait
        del ddg, Wg_
        # GLS statistics: weights d = 1 / (s / delta + 1) in the rotated space, intercept projected out
        D = (1.0 / (s[:, None] / delta_r[None, :] + 1.0)).float()           # n x reps
        Uf, u1f, UYf = U.float(), u1.float(), UY.float()
        Su = (u1f * u1f) @ D                                                # reps
        Sy = (u1f[:, None] * UYf * D).sum(0)
        yy = (UYf * UYf * D).sum(0) - Sy * Sy / Su
        d0 = (1.0 / (s / delta + 1.0)).float()                              # check against the main analysis
        Su0 = float((u1f * u1f * d0).sum())
        uy0 = Uf.T @ torch.from_numpy(y).float()
        Sy0 = float((u1f * uy0 * d0).sum())
        yy0 = float((uy0 * uy0 * d0).sum()) - Sy0 * Sy0 / Su0
        tested = mac_["mac"][j] >= 30
        mx = torch.zeros(a.reps)
        wid = {sn: int(np.flatnonzero(snps["id"].to_numpy() == sn)[0]) for sn in filter(None, a.watch.split(","))}
        zw = {}
        wb = {int(x.split(":")[0]): x.split(":")[1] for x in filter(None, a.watch_blocks.split(","))}
        bmx = {b_: torch.zeros(a.reps) for b_ in wb}                      # largest |z| of the watched blocks
        bn = {b_: 0 for b_ in wb}
        for c0 in range(0, p, a.chunk):
            X = torch.from_numpy(np.asarray(G[r_, c0:c0 + a.chunk], dtype=np.float32))
            X = X - X.mean(0)
            XU = Uf.T @ X                                                   # n x c
            Xu1 = XU * u1f[:, None]
            Su_x = Xu1.T @ D                                                # c x reps
            xx = (XU * XU).T @ D - Su_x * Su_x / Su[None, :]
            xy = XU.T @ (UYf * D) - Su_x * (Sy / Su)[None, :]
            ok = xx > 1e-6
            b = torch.where(ok, xy / xx.clamp_min(1e-6), torch.zeros_like(xy))
            s2 = ((yy[None, :] - b * b * xx) / (n - 2)).clamp_min(1e-12)
            st = torch.where(ok, b * b * xx / s2, torch.zeros_like(xy))
            st[torch.from_numpy(~tested[c0:c0 + a.chunk])] = 0.0
            mx = torch.maximum(mx, st.max(0).values)
            for b_ in wb:                                                   # largest statistic of the watched blocks
                kk = np.flatnonzero((blk[c0:c0 + a.chunk] == b_) & tested[c0:c0 + a.chunk])
                if len(kk):
                    bmx[b_] = torch.maximum(bmx[b_], st[torch.from_numpy(kk)].max(0).values)
                    bn[b_] += len(kk)
            for sn, k_w in wid.items():                                      # statistics at the watched SNPs
                if c0 <= k_w < c0 + a.chunk and tested[k_w]:
                    zw[sn] = (torch.sign(b[k_w - c0]) * torch.sqrt(st[k_w - c0])).numpy().astype(float)
            if c0 == 0:                                                     # the observed trait, delta of the main analysis
                xx0 = (XU * XU * d0[:, None]).sum(0) - (Xu1 * d0[:, None]).sum(0) ** 2 / Su0
                xy0 = (XU * (uy0 * d0)[:, None]).sum(0) - (Xu1 * d0[:, None]).sum(0) * Sy0 / Su0
                b0 = xy0 / xx0.clamp_min(1e-8)
                st0 = (b0 * b0 * xx0 / ((yy0 - b0 * b0 * xx0) / (n - 2))).numpy()
                ref = main_["chi2"][j, :a.chunk]
                k_ = tested[:a.chunk] & (ref > 1)
                assert np.allclose(st0[k_], ref[k_], rtol=2e-3, atol=1e-3), (t, np.abs(st0[k_] - ref[k_]).max())
        mx = mx.numpy().astype(float)
        thr_chi = float(np.quantile(mx, 0.95))
        thr_p = float(chi2d.sf(thr_chi, 1))
        cj = np.where(tested, np.nan_to_num(main_["chi2"][j]).astype(float), 0.0)
        bmax = pd.Series(cj).groupby(blk).max()
        barg = pd.Series(cj).groupby(blk).idxmax()
        lb = bmax.index[bmax >= chi_bonf]
        keep = bmax[lb] >= max(chi_bonf, thr_chi)
        short = T.SHORT.get(t, t)
        for b_ in lb:
            k_ = int(barg[b_])
            lrows.append({"trait": t, "short": short, "block": int(b_), "chrom": snps.chrom[k_], "top_snp": snps["id"][k_],
                          "top_p": float(chi2d.sf(bmax[b_], 1)), "reaches_stricter_threshold": bool(keep[b_])})
        rows.append({"trait": t, "short": short, "accessions": n,
                     "resistant_accessions": int(nres[t]), "delta": delta, "tested_snps": int(tested.sum()),
                     "replicates": a.reps, "threshold_p": thr_p, "bonferroni_p": 0.05 / p,
                     "fwer_of_bonferroni": float((mx > chi_bonf).mean()),
                     "stricter_p": min(thr_p, 0.05 / p),
                     "loci_bonferroni": len(lb), "loci_stricter_threshold": int(keep.sum()),
                     "median_delta_simulated": float(np.median(delta_r.numpy())), "delta_liability": delta_L,
                     "liability_ratio_matched": matched})
        for sn, zr in zw.items():
            k_w = wid[sn]
            z_obs = float(np.sign(main_["beta"][j, k_w]) * np.sqrt(main_["chi2"][j, k_w]))
            wrows.append({"trait": t, "short": short, "snp": sn, "z_observed": z_obs, "replicates": len(zr),
                          "simulated_abs_z_at_least_observed": int((np.abs(zr) >= abs(z_obs)).sum()),
                          "empirical_p": float(((np.abs(zr) >= abs(z_obs)).sum() + 1) / (len(zr) + 1)),
                          "simulated_abs_z_q99": float(np.quantile(np.abs(zr), 0.99)),
                          "simulated_abs_z_max": float(np.abs(zr).max())})
        for b_, sn in wb.items():
            k_w = int(np.flatnonzero(snps["id"].to_numpy() == sn)[0])
            assert blk[k_w] == b_, (b_, sn)
            z_obs = float(np.sign(main_["beta"][j, k_w]) * np.sqrt(main_["chi2"][j, k_w]))
            zb = np.sqrt(bmx[b_].numpy().astype(float))
            wbrows.append({"trait": t, "short": short, "block": b_, "snp": sn, "z_observed": z_obs, "tested_snps_block": bn[b_],
                           "replicates": len(zb), "simulated_block_max_abs_z_at_least_observed": int((zb >= abs(z_obs)).sum()),
                           "empirical_p": float(((zb >= abs(z_obs)).sum() + 1) / (len(zb) + 1)),
                           "simulated_block_max_abs_z_q99": float(np.quantile(zb, 0.99)),
                           "simulated_block_max_abs_z_max": float(zb.max())})
        print(f"[{time.time() - t0:6.0f}s] {t}: n {n}, resistant {int(nres[t])}, delta {delta:.3f}; threshold P "
              f"{thr_p:.3g} (Bonferroni {0.05 / p:.3g}, family-wise error of Bonferroni {(mx > chi_bonf).mean():.3f}); "
              f"loci {len(lb)} -> {int(keep.sum())}", flush=True)
        del U, Uf, UY, UYf, Y, D
    out = ROOT / "results/gwas"
    sfx = "" if a.traitset == "stress" else "_mendel"
    if wbrows:
        pd.DataFrame(wbrows).to_csv(out / f"null_threshold_watch_blocks{sfx}.tsv", sep="\t", index=False, float_format="%.6g")
        print(pd.DataFrame(wbrows).to_string())
    if a.only:
        print(pd.DataFrame(rows).to_string())
        return
    rows_df, lrows_df = pd.DataFrame(rows), pd.DataFrame(lrows)
    if add:                                              # merge with the stored results of the other traits
        old_r = pd.read_csv(out / f"null_threshold{sfx}.tsv", sep="\t")
        old_l = pd.read_csv(out / f"null_threshold_loci{sfx}.tsv", sep="\t")
        assert not set(old_r.trait) & set(add)
        order = {t: i for i, t in enumerate(traits)}
        rows_df = pd.concat([old_r, rows_df], ignore_index=True).sort_values("trait", key=lambda c: c.map(order), kind="stable")
        lrows_df = pd.concat([old_l, lrows_df], ignore_index=True).sort_values("trait", key=lambda c: c.map(order), kind="stable")
        assert list(rows_df.trait) == traits, "every trait needs a row"
    rows_df.to_csv(out / f"null_threshold{sfx}.tsv", sep="\t", index=False, float_format="%.6g")
    lrows_df.to_csv(out / f"null_threshold_loci{sfx}.tsv", sep="\t", index=False, float_format="%.6g")
    if wrows and not add:
        pd.DataFrame(wrows).to_csv(out / f"null_threshold_watch{sfx}.tsv", sep="\t", index=False, float_format="%.6g")
    print("written", out / f"null_threshold{sfx}.tsv", out / f"null_threshold_loci{sfx}.tsv")


if __name__ == "__main__":
    main()
