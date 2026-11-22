"""Mixed-model GWAS read-out (EMMAX / P3D) for every trait of a trait set, on all accessions with folds (step 8 read-out
(d); also the direction of every SNP effect for the shared / antagonistic locus map, step 9).

usage: python gwas_emmax.py --traitset stress|mendel [--device cuda] [--covar none|taxon_pcs] [--pcs 5]
                            [--taxon all|gmax] [--tag TAG]
Per trait: targets as for the other models (data.fold_targets on all accessions: mean score; numeric traits adjusted
by batch means); VanRaden kernel of the trait's accessions K = U diag(s) U'; REML variance ratio delta of the null model
(reml.SpectralREML); then for every SNP the GLS test of y = W a + x b + g + e with delta fixed:
  transform by D^1/2 U' (D = 1 / (s / delta + 1)), project out the fixed effects W, b = x'y / x'x,
  chi2 = b^2 x'x / sigma2 (squared t statistic with n - q - 1 degrees of freedom, q = columns of W).
W is the intercept (--covar none, the analysis of the article) or the intercept, an indicator of Glycine soja and the
first --pcs principal components of K (--covar taxon_pcs; a constant indicator is dropped). --taxon gmax restricts
the analysis to the Glycine max accessions. The two options give the analyses that check the loci for population
structure; their results are written with --tag.
--condition major_genes: conditional analysis (the statistics of the SNPs fitted as fixed effects are set to 0). For the SCN races, the most significant SNP of the main analysis within
300 kb of Rhg1 and within 300 kb of Rhg4, and for the P. sojae races the most significant SNP within 300 kb of Rps1,
are added as fixed effects when they are significant for the race (P < 0.05 / number of SNPs); the other traits keep
the statistics of the main analysis. Shows which loci of the races remain when these major genes are accounted for.
Writes results/gwas/emmax_<traitset><tag>.npz: chi2 and beta (traits x SNPs, beta per alternative allele copy on the
standardised trait scale), n per trait, delta per trait (residual / genetic variance), q per trait.
--add_traits T1,T2: compute only these traits and merge them into the existing results file (for traits added to the
data later, P. sojae race 10 on 2026-11-17); the statistics of the other traits are copied unchanged.
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
    ap.add_argument("--covar", default="none", choices=["none", "taxon_pcs"])
    ap.add_argument("--pcs", type=int, default=5)
    ap.add_argument("--taxon", default="all", choices=["all", "gmax"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--condition", default="none", choices=["none", "major_genes"])
    ap.add_argument("--add_traits", default="", help="only these traits, merged into the existing results file")
    ap.add_argument("--condition_snps", default="",
                    help="'TRAIT,TRAIT:SNP,SNP': these SNPs as fixed effects for these traits (the other traits keep the "
                         "statistics of the main analysis; with --tag)")
    a = ap.parse_args()
    assert a.tag or (a.covar == "none" and a.taxon == "all" and a.condition == "none" and not a.condition_snps), "give --tag for an analysis other than the main one"
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / tot), 0)
    t0 = time.time()
    long = tb.use_traitset(a.traitset)
    traits = list(tb.TRAITS)
    fo = data.folds()
    long = long[long.acc.isin(set(fo.index))]
    taxon = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id").taxon
    if a.taxon == "gmax":
        long = long[long.acc.map(taxon) == "Glycine max"]
    every = pd.Series(0, index=fo.index)
    pos = {v: i for i, v in enumerate(data.samples())}
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    Kmm = data.grm(mmap=True)
    p = G.shape[1]
    chi2 = np.zeros((len(traits), p), np.float32)
    beta = np.zeros((len(traits), p), np.float32)
    ns, deltas, qs = [], [], []
    add = [x for x in a.add_traits.split(",") if x]
    old = None
    if add:
        old = np.load(ROOT / f"results/gwas/emmax_{a.traitset}{a.tag}.npz", allow_pickle=True)
        old_tr = [str(x) for x in old["traits"]]
        assert set(add) <= set(traits) and not set(add) & set(old_tr) and set(old_tr) | set(add) == set(traits), (add, old_tr)
        oi = {t: i for i, t in enumerate(old_tr)}
    cond = {}
    if a.condition == "major_genes":
        main_ = np.load(ROOT / f"results/gwas/emmax_{a.traitset}.npz", allow_pickle=True)
        assert [str(x) for x in main_["traits"]] == traits, "run the main analysis (with --add_traits) first"
        mac_ = np.load(ROOT / f"results/gwas/mac_{a.traitset}.npz", allow_pickle=True)   # tested SNPs: >= 30 minor homozygotes
        assert [str(x) for x in mac_["traits"]] == traits
        snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
        kg = pd.read_csv(data.P / "known_genes.tsv", sep="\t").set_index("locus")
        from scipy.stats import chi2 as chi2dist
        thr = float(chi2dist.isf(0.05 / p, 1))
        genes = {"NEMATCYST": ["Rhg1", "Rhg4"], "PHYTOROT": ["Rps1"]}
        for j, t in enumerate(traits):
            fam = t.split("_")[0]
            for gname in genes.get(fam, []):
                g_ = kg.loc[gname]
                near = np.flatnonzero(((snps.chrom == g_.chrom) & (snps.pos >= g_.start - 300_000)
                                       & (snps.pos <= g_.end + 300_000)).to_numpy() & (mac_["mac"][j] >= 30))
                k = near[np.argmax(main_["chi2"][j][near])]
                if main_["chi2"][j][k] > thr:
                    cond.setdefault(t, []).append(int(k))
        print("conditioning SNPs:", {t: [snps.id[k] for k in v] for t, v in cond.items()}, flush=True)
    if a.condition_snps:                                      # given SNPs as fixed effects for given traits
        main_ = np.load(ROOT / f"results/gwas/emmax_{a.traitset}.npz", allow_pickle=True)
        assert [str(x) for x in main_["traits"]] == traits, "run the main analysis (with --add_traits) first"
        snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
        idx_ = {s_: i for i, s_ in enumerate(snps.id)}
        tl_, sl_ = a.condition_snps.split(":")
        for t in tl_.split(","):
            assert t in traits, t
            cond[t] = [idx_[s_] for s_ in sl_.split(",")]
        print("conditioning SNPs:", {t: [snps.id[k] for k in v] for t, v in cond.items()}, flush=True)
    for j, t in enumerate(traits):
        if add and t not in add:                              # traits of the existing file: copied unchanged
            k_ = oi[t]
            chi2[j], beta[j] = old["chi2"][k_], old["beta"][k_]
            ns.append(int(old["n"][k_])), deltas.append(float(old["delta"][k_])), qs.append(int(old["q"][k_]) if "q" in old else 1)
            continue
        if (a.condition != "none" or a.condition_snps) and t not in cond:   # unchanged traits: main analysis
            chi2[j], beta[j] = main_["chi2"][j], main_["beta"][j]
            ns.append(int(main_["n"][j])), deltas.append(float(main_["delta"][j])), qs.append(1)
            continue
        ft = data.fold_targets(long[long.trait == t], t, every, 1)
        rows = np.array([pos[x] for x in ft.train_acc])
        o = np.argsort(rows)
        rows, y = rows[o], ft.y_train[o]
        accs = np.asarray(ft.train_acc)[o]
        y = (y - y.mean()) / (y.std() + 1e-12)
        n = len(y)
        K = torch.from_numpy(np.asarray(Kmm[np.ix_(rows, rows)], dtype=np.float64)).to(dev)
        s, U = torch.linalg.eigh(K)
        del K
        s = s.clamp_min(0)
        yt = torch.tensor(y, dtype=torch.float64, device=dev)
        C = None
        if a.covar == "taxon_pcs":
            cols = [U[:, -a.pcs:]]                                   # eigenvectors of the largest eigenvalues
            soja = torch.tensor((pd.Series(accs).map(taxon) == "Glycine soja").to_numpy(), dtype=torch.float64, device=dev)
            if 0 < float(soja.sum()) < n:
                cols.append(soja[:, None])
            C = torch.cat(cols, 1)
        if t in cond:                                            # lead SNPs near major genes (or given SNPs) as fixed effects
            Xc = torch.from_numpy(np.asarray(G[np.ix_(rows, cond[t])], dtype=np.float64)).to(dev)
            Xc = Xc - Xc.mean(0)
            C = Xc if C is None else torch.cat([C, Xc], 1)
        m = SpectralREML().fit(torch.zeros(1, device=dev), yt, C=C, eig=(s, U))
        d = 1.0 / (s / m.delta + 1.0)
        dh = torch.sqrt(d).float()
        Uf = U.float()
        W = torch.ones(n, 1, dtype=torch.float64, device=dev) if C is None else torch.cat([torch.ones(n, 1, dtype=torch.float64, device=dev), C], 1)
        Wt = torch.sqrt(d)[:, None] * (U.T @ W)
        Q = torch.linalg.qr(Wt)[0].float()                           # orthonormal basis of the transformed fixed effects
        q = Q.shape[1]
        del U, W, Wt
        yv = dh * (Uf.T @ yt.float())
        yc = yv - Q @ (Q.T @ yv)
        yy = float(yc @ yc)
        for c0 in range(0, p, a.chunk):
            X = torch.from_numpy(np.asarray(G[rows, c0:c0 + a.chunk], dtype=np.float32)).to(dev)
            X = X - X.mean(0)
            Xt = dh[:, None] * (Uf.T @ X)
            Xt = Xt - Q @ (Q.T @ Xt)
            xx = (Xt * Xt).sum(0).clamp_min(1e-8)
            xy = Xt.T @ yc
            b = xy / xx
            s2 = (yy - b * b * xx) / (n - q - 1)
            chi2[j, c0:c0 + a.chunk] = (b * b * xx / s2.clamp_min(1e-12)).cpu().numpy()
            beta[j, c0:c0 + a.chunk] = b.cpu().numpy()
        if t in cond:                                            # SNPs fitted as fixed effects are not tested
            chi2[j, cond[t]] = 0.0
            beta[j, cond[t]] = 0.0
        ns.append(n)
        deltas.append(m.delta)
        qs.append(q)
        del Uf
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{time.time() - t0:6.0f}s] {t}: n {n}, fixed effects {q}, delta {m.delta:.3f}, max chi2 {chi2[j].max():.1f}", flush=True)
    out = ROOT / "results/gwas"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"emmax_{a.traitset}{a.tag}.npz", chi2=chi2, beta=beta, traits=np.array(traits),
                        n=np.array(ns), delta=np.array(deltas), q=np.array(qs))
    print("written", out / f"emmax_{a.traitset}{a.tag}.npz")


if __name__ == "__main__":
    main()
