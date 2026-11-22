"""Calibration of the locus scores with 12 genes of known position, leave-one-gene-out.

usage: python -I calibrate_readouts.py
Scores (all-data models, data/processed/known_genes.tsv for the genes):
  mtboost   split gain of every SNP for every trait in the final model (runs/all/mtboost_final*/importance.npz, key
            "model": the rounds of shared and trait-specific trees that the trait's prediction uses, mean over the
            four refits), normalised per trait to sum 1
  emmax     test statistic of the mixed-model association analysis (results/gwas/emmax_<set>.npz); SNPs with fewer
            than 30 accessions homozygous for the less frequent allele among the trait's accessions are not used
            (results/gwas/mac_<set>.npz, as in locus_map.py)
  combined  mean of the two per-SNP genome-wide ranks
A gene with several traits (Rhg1: SCN races 1/3/5/14, Rhg4: 1/3/5, Rps1: Phytophthora 1/3/4/5/7/10/17, E1-E4: maturity)
uses the sum of the trait scores (each trait normalised to sum 1; for emmax the sum of chi-square / mean chi-square).
Windows: block (the 16-SNP block of the gene, or of its nearest SNP), +-20 kb, +-50 kb around the gene (score = max
over the window's SNPs; the gene's percentile is taken among all windows of the same kind centred on every SNP /
every block). Leave-one-gene-out: for each gene, the read-out x window with most other genes in the top 1% (ties:
lower mean percentile) is applied to the held-out gene. Writes results/calibration/{percentiles.tsv,logo.tsv};
logo.tsv also gives the traits of the gene for which its block holds a SNP significant after Bonferroni correction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scipy.stats import chi2 as chi2_dist

from soymultires import data  # noqa: E402

MIN_MAC = 30

GENE_TRAITS = {
    "W1": ["flower_purple"], "T": ["pubescence_tawny"], "Dt1": ["indeterminate"],
    "E1": ["maturity_group"], "E2": ["maturity_group"], "E3": ["maturity_group"], "E4": ["maturity_group"],
    "I": ["seedcoat_pigmented"],
    "Rhg1": ["NEMATCYST_1", "NEMATCYST_3", "NEMATCYST_5", "NEMATCYST_14"],
    "Rhg4": ["NEMATCYST_1", "NEMATCYST_3", "NEMATCYST_5"],
    "Rps1": ["PHYTOROT_1", "PHYTOROT_3", "PHYTOROT_4", "PHYTOROT_5", "PHYTOROT_7", "PHYTOROT_10", "PHYTOROT_17"],
    "GmSALT3": ["SALTREACT"],
}


def load_scores():
    """Per-SNP score vectors per read-out and trait: {readout: {trait: array(p)}}."""
    out = {"mtboost": {}, "emmax": {}, "chi2": {}}
    for sub, gw in (("mtboost_final", "emmax_stress"), ("mtboost_final_mendel", "emmax_mendel")):
        imp = np.load(data.RUNS / "all" / sub / "importance.npz", allow_pickle=True)
        g = imp["model"]
        for j, t in enumerate(imp["traits"]):
            v = np.clip(g[j], 0, None)
            out["mtboost"][str(t)] = v / max(v.sum(), 1e-12)
        e = np.load(ROOT / "results/gwas" / f"{gw}.npz", allow_pickle=True)
        mac = np.load(ROOT / "results/gwas" / f"{gw.replace('emmax', 'mac')}.npz", allow_pickle=True)
        assert [str(t) for t in mac["traits"]] == [str(t) for t in e["traits"]]
        for j, t in enumerate(e["traits"]):
            c = np.where(mac["mac"][j] >= MIN_MAC, np.nan_to_num(e["chi2"][j]), 0.0)
            out["emmax"][str(t)] = c / max(c.mean(), 1e-12)
            out["chi2"][str(t)] = c
    return out


def rank01(v):
    """Genome-wide rank in [0, 1] (1 = highest score); equal scores share their mean rank."""
    return rankdata(v, method="average") / len(v)


def main():
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    kg = pd.read_csv(data.P / "known_genes.tsv", sep="\t")
    kg = kg[kg.set == "calibration"].set_index("locus")
    sc = load_scores()
    readouts = {}
    for gene, traits in GENE_TRAITS.items():
        m = sum(sc["mtboost"][t] for t in traits)
        e = sum(sc["emmax"][t] for t in traits)
        readouts[gene] = {"mtboost": m, "emmax": e, "combined": (rank01(m) + rank01(e)) / 2}
    chrom, pos, blk = snps.chrom.to_numpy(), snps.pos.to_numpy(), snps.block16.to_numpy()
    rows = []
    for gene in GENE_TRAITS:
        g = kg.loc[gene]
        on = chrom == g.chrom
        mid = (g.start + g.end) / 2
        nearest = np.flatnonzero(on)[np.argmin(np.abs(pos[on] - mid))]
        for ro, v in readouts[gene].items():
            # block windows: max over each block
            bmax = pd.Series(v).groupby(blk).max()
            gb = bmax.loc[blk[nearest]]
            rows.append({"gene": gene, "readout": ro, "window": "block", "percentile": float((bmax > gb).mean() * 100)})
            for kb in (20, 50):
                w = on & (pos >= g.start - kb * 1000) & (pos <= g.end + kb * 1000)
                if not w.any():
                    w = np.zeros(len(pos), bool)
                    w[nearest] = True
                gs = v[w].max()
                # null: windows of the same width centred on every SNP (same chromosome scan), max inside
                sv = pd.Series(v)
                null = []
                for ch, idx in snps.groupby("chrom").indices.items():
                    p_ = pos[idx]
                    vv = sv.iloc[idx].to_numpy()
                    left = np.searchsorted(p_, p_ - kb * 1000 - (g.end - g.start) / 2)
                    right = np.searchsorted(p_, p_ + kb * 1000 + (g.end - g.start) / 2, side="right")
                    cm = np.maximum.accumulate
                    # window max by a sparse table would be faster; the chromosomes are short enough for a loop
                    null.extend(vv[a_:b_].max() for a_, b_ in zip(left, right))
                null = np.array(null)
                rows.append({"gene": gene, "readout": ro, "window": f"+-{kb}kb",
                             "percentile": float((null > gs).mean() * 100)})
    d = pd.DataFrame(rows)
    out = ROOT / "results/calibration"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "percentiles.tsv", sep="\t", index=False, float_format="%.5f")
    wide = d.pivot_table(index="gene", columns=["readout", "window"], values="percentile")
    pd.set_option("display.width", 220)
    print(wide.round(2).to_string())
    combos = [(r, w) for r in ("mtboost", "emmax", "combined") for w in ("block", "+-20kb", "+-50kb")]
    # a block is a locus of a trait at the stricter of the Bonferroni threshold and the trait's simulated threshold
    # (results/gwas/null_threshold{,_mendel}.tsv, gwas_null_threshold.py), as in locus_map.py
    nt = pd.concat([pd.read_csv(ROOT / f"results/gwas/null_threshold{x}.tsv", sep="\t") for x in ("", "_mendel")]).set_index("trait")
    thr_b = float(chi2_dist.isf(0.05 / len(snps), 1))
    thr = {t: max(thr_b, float(chi2_dist.isf(nt.threshold_p[t], 1))) for t in nt.index}
    logo = []
    for gene in GENE_TRAITS:
        others = [x for x in GENE_TRAITS if x != gene]
        best = min(combos, key=lambda c: (-int((wide.loc[others, c] <= 1).sum()), wide.loc[others, c].mean()))
        g = kg.loc[gene]
        on = chrom == g.chrom
        b = blk[np.flatnonzero(on)[np.argmin(np.abs(pos[on] - (g.start + g.end) / 2))]]
        sig = [t for t in GENE_TRAITS[gene] if sc["chi2"][t][blk == b].max() >= thr[t]]
        tied = [c for c in combos if int((wide.loc[others, c] <= 1).sum()) == int((wide.loc[others, best] <= 1).sum())]
        logo.append({"held_out": gene, "readout": best[0], "window": best[1],
                     "percentile": float(wide.loc[gene, best]), "top1pct": bool(wide.loc[gene, best] <= 1),
                     "others_in_top1pct": int((wide.loc[others, best] <= 1).sum()),
                     "tied_combinations": ";".join(f"{r}/{w}" for r, w in tied),
                     "block": int(b), "significant_traits": ",".join(sig)})
    lg = pd.DataFrame(logo)
    lg.to_csv(out / "logo.tsv", sep="\t", index=False, float_format="%.5f")
    print(lg.to_string(index=False))
    print(f"leave-one-gene-out: {int(lg.top1pct.sum())} of {len(lg)} known genes in the top 1%")
    for c in combos:
        print(f"  {c[0]:8s} {c[1]:7s}: {int((wide[c] <= 1).sum())}/12 in top 1%, {int((wide[c] <= 5).sum())}/12 in top 5%")


if __name__ == "__main__":
    main()
