"""Step 9: shared / specific / antagonistic resistance loci across the 32 stress traits, and candidate genes.

usage: python -I locus_map.py [--readout combined|mtboost|emmax] [--window_kb 20] [--top 0.01]
Inputs: final all-data MT-Boost per-trait SNP gains (runs/all/mtboost_final/importance.npz), mixed-model GWAS
(results/gwas/emmax_stress.npz: chi2 and beta per alternative allele on the standardised trait; every trait is coded
so that a higher value means more resistant / less damage), 16-SNP blocks (geno/snps.tsv), gene models and functional
annotation of Wm82.a2 (LIS Wm82.gnm2.ann1.RVB6).
Per trait the block score = max over the block's SNPs of the read-out (calibrated in step 8, calibrate_readouts.py);
the trait's top blocks = top `top` share of the 2,478 blocks. For every block in the top of >= 1 trait: the traits,
their stress categories, the direction of the alternative allele at the block's strongest SNP for each trait
(sign of beta, counted only where |z| >= 3), and the class
  specific            one trait
  unresolved          >= 2 traits, fewer than 2 clear directions
  concordant          all clear directions equal
  race-differential   opposite directions only among races of one pathogen (SCN, Phytophthora or frogeye races)
  trade-off           opposite directions only between different pathogens / stresses
  mixed               opposite directions both within a pathogen and between pathogens
Candidate genes: gene models overlapping the block span +- window_kb, with Pfam-based flags for resistance gene
families. Writes results/loci/{block_traits, loci, candidate_genes}_<readout>.tsv and summary_<readout>.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

ANN = ROOT / "data/raw/wm82_gnm2_annotation"   # LIS Wm82.gnm2.ann1.RVB6 (symbolic link to the downloaded files)
PFAM_FLAGS = {"NB-ARC": {"PF00931"}, "LRR": {"PF00560", "PF07723", "PF07725", "PF12799", "PF13516", "PF13855",
                                            "PF08263"},
              "TIR": {"PF01582", "PF13676"}, "kinase": {"PF00069", "PF07714"}, "WRKY": {"PF03106"},
              "lectin": {"PF00139", "PF01453"}, "PR/chitinase": {"PF00187", "PF00182"}}
CAT = {row[0]: row[4] for row in T.STRESS}
# races of one pathogen; every other trait is its own pathogen / stress (stress categories such as fungi_oomycetes
# group different pathogens, so they cannot define race differences)
PATHOGEN = {t: "SCN" if t.startswith("NEMATCYST") else "Phytophthora" if t.startswith("PHYTOROT")
            else "frogeye" if t.startswith("FROGEYE") else t for t in T.TRAITS}


def rank01(v):
    return (np.argsort(np.argsort(v)) + 1) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", default="combined", choices=["combined", "mtboost", "emmax"])
    ap.add_argument("--window_kb", type=float, default=20)
    ap.add_argument("--top", type=float, default=0.01)
    a = ap.parse_args()
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    blk = snps.block16.to_numpy()
    imp = np.load(data.RUNS / "all/mtboost_final/importance.npz", allow_pickle=True)
    gw = np.load(ROOT / "results/gwas/emmax_stress.npz", allow_pickle=True)
    traits = [str(t) for t in gw["traits"]]
    it = {str(t): j for j, t in enumerate(imp["traits"])}
    gain = imp["shared"] + imp["own"]
    nb = int(blk.max()) + 1
    score = np.zeros((len(traits), nb))
    best_snp = np.zeros((len(traits), nb), int)
    for j, t in enumerate(traits):
        m = np.clip(gain[it[t]], 0, None)
        m = m / max(m.sum(), 1e-12)
        e = np.nan_to_num(gw["chi2"][j])
        v = {"mtboost": m, "emmax": e, "combined": (rank01(m) + rank01(e)) / 2}[a.readout]
        s = pd.Series(v).groupby(blk)
        score[j] = s.max().reindex(range(nb)).fillna(0).to_numpy()
        ec = pd.Series(e).groupby(blk).idxmax()                                   # strongest GWAS SNP per block
        best_snp[j] = ec.reindex(range(nb)).fillna(0).astype(int).to_numpy()
    k = max(1, int(round(a.top * nb)))
    top = np.zeros_like(score, bool)
    for j in range(len(traits)):
        top[j, np.argsort(-score[j])[:k]] = True
    rows = []
    for j, t in enumerate(traits):
        for b in np.flatnonzero(top[j]):
            sn = best_snp[j, b]
            beta = float(gw["beta"][j, sn])
            z = float(np.sign(beta) * np.sqrt(max(gw["chi2"][j, sn], 0)))
            rows.append({"block": int(b), "trait": t, "short": T.SHORT[t], "category": CAT[t],
                         "score": float(score[j, b]), "rank_in_trait": int((score[j] > score[j, b]).sum()) + 1,
                         "snp": snps["id"].iloc[sn],
                         "chrom": snps.chrom.iloc[sn], "pos": int(snps.pos.iloc[sn]), "beta_alt": beta, "z": z,
                         "direction": ("+" if z >= 3 else "-" if z <= -3 else "0")})
    bt = pd.DataFrame(rows)
    out = ROOT / "results/loci"
    out.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out / f"block_traits_{a.readout}.tsv", sep="\t", index=False, float_format="%.4g")
    loci = []
    tix = {t: j for j, t in enumerate(traits)}
    for b, d in bt.groupby("block"):
        bs = snps[blk == b]
        # one common SNP per block for every trait's direction: the SNP with the highest summed chi-square over the
        # block's traits (different strongest SNPs per trait could tag one haplotype with opposite phase)
        idx = np.flatnonzero(blk == b)
        js = [tix[t] for t in d.trait]
        sn = idx[np.argmax(np.nan_to_num(gw["chi2"][np.ix_(js, idx)]).sum(0))]
        z = np.sign(gw["beta"][js, sn]) * np.sqrt(np.clip(gw["chi2"][js, sn], 0, None))
        d = d.assign(direction=np.where(z >= 3, "+", np.where(z <= -3, "-", "0")), z=z,
                     snp=snps["id"].iloc[sn], pos=int(snps.pos.iloc[sn]))
        clear = d[d.direction != "0"]
        if len(d) == 1:
            cls = "specific"
        elif len(clear) < 2:
            cls = "unresolved"
        elif clear.direction.nunique() == 1:
            cls = "concordant"
        else:
            pg = clear.trait.map(PATHOGEN)
            within = any(g.nunique() > 1 for _, g in clear.direction.groupby(pg))
            if pg.nunique() == 1:
                cls = "race-differential"                      # opposite signs among races of one pathogen
            elif not within:
                cls = "trade-off"                              # opposite signs only between pathogens / stresses
            else:
                cls = "mixed"                                  # opposite signs within and between pathogens
        loci.append({"block": int(b), "chrom": bs.chrom.iloc[0], "start": int(bs.pos.min()), "end": int(bs.pos.max()),
                     "n_traits": len(d), "traits": ",".join(d.short), "categories": ",".join(sorted(set(d.category))),
                     "n_categories": d.category.nunique(), "directions": ",".join(f"{s}{x}" for s, x in zip(d.short, d.direction)),
                     "lead_snp": d.snp.iloc[0], "lead_pos": int(d.pos.iloc[0]), "class": cls})
    lc = pd.DataFrame(loci).sort_values(["n_traits", "n_categories"], ascending=False)
    # candidate genes
    bed = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.gene_models_main.bed.gz", sep="\t", header=None,
                      usecols=[0, 1, 2, 6], names=["chrom", "start", "end", "gene"])     # col 6 = gene (5 = strand)
    bed["chrom"] = bed.chrom.str.replace("glyma.Wm82.gnm2.", "", regex=False)
    bed["gene"] = bed.gene.str.replace("glyma.Wm82.gnm2.ann1.", "", regex=False)
    # the bed file has one row per transcript: one span per gene
    bed = bed.groupby(["chrom", "gene"], as_index=False).agg(start=("start", "min"), end=("end", "max"))
    info = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.info_annot.txt.gz", sep="\t", dtype=str)
    info = info.drop_duplicates("locusName").set_index("locusName")
    cand = []
    w = a.window_kb * 1000
    for _, L in lc[lc.n_traits >= 2].iterrows():
        g = bed[(bed.chrom == L.chrom) & (bed.end >= L.start - w) & (bed.start <= L.end + w)]
        for _, G in g.iterrows():
            pf = set(str(info.Pfam.get(G.gene, "")).replace(" ", "").split(",")) if G.gene in info.index else set()
            flags = [f for f, ids in PFAM_FLAGS.items() if pf & ids]
            cand.append({"block": L.block, "class": L["class"], "traits": L.traits, "gene": G.gene,
                         "chrom": G.chrom, "start": int(G.start), "end": int(G.end), "flags": ",".join(flags),
                         "arabidopsis": info["arabi-symbol"].get(G.gene, "") if G.gene in info.index else "",
                         "description": str(info["arabi-defline"].get(G.gene, ""))[:120] if G.gene in info.index else ""})
    cg = pd.DataFrame(cand)
    lc.to_csv(out / f"loci_{a.readout}.tsv", sep="\t", index=False)
    cg.to_csv(out / f"candidate_genes_{a.readout}.tsv", sep="\t", index=False)
    s = [f"read-out {a.readout}, top {a.top:.0%} of {nb} blocks per trait ({k} blocks), window +-{a.window_kb:g} kb",
         f"blocks in the top of >= 1 trait: {len(lc)}; >= 2 traits: {int((lc.n_traits >= 2).sum())}; "
         f">= 2 stress categories: {int((lc.n_categories >= 2).sum())}",
         f"classes among multi-trait blocks: {lc[lc.n_traits >= 2]['class'].value_counts().to_dict()}",
         f"candidate genes in multi-trait loci: {len(cg)}; with resistance-family flags: "
         f"{int((cg['flags'] != '').sum()) if len(cg) else 0}"]
    (out / f"summary_{a.readout}.txt").write_text("\n".join(s) + "\n")
    print("\n".join(s))
    pd.set_option("display.width", 220)
    print(lc.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
