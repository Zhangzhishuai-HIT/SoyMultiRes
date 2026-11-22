"""Genes with an NB-ARC domain in the shared loci, compared with the number expected when the shared loci are shifted
together along the genome.

usage: python -I nbarc_shift.py
Blocks of 16 SNPs (geno/snps.tsv, block16) span from their first to their last SNP; a gene of the Wm82.a2.v1
annotation (LIS Wm82.gnm2.ann1.RVB6, gene features of the main gene models) lies in a block when it overlaps this span,
and a gene has an NB-ARC domain when its Pfam domains include PF00931 (as in locus_map.py). The shared loci (blocks that
are loci of two or more traits, results/loci/loci_emmax.tsv) are shifted together along the concatenated chromosomes
by every possible number of blocks (circularly, 0 to nb - 1, the observed placement included; the arrangement of the
shared loci is kept), and the genes and the genes with an NB-ARC domain in the shifted blocks are counted. P = share of the shifts with at least the observed number. Writes results/loci/nbarc_shift.tsv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

ANN = ROOT / "data/raw/wm82_gnm2_annotation"


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    blk = snps.groupby("block16").agg(chrom=("chrom", "first"), start=("pos", "min"), end=("pos", "max")).sort_index()
    nb = len(blk)
    assert (blk.index.to_numpy() == np.arange(nb)).all()
    gff = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.gene_models_main.gff3.gz", sep="\t", header=None, comment="#",
                      usecols=[0, 2, 3, 4, 8], names=["chrom", "type", "start", "end", "attr"])
    genes = gff[gff.type == "gene"].copy()
    genes["chrom"] = genes.chrom.str.replace("glyma.Wm82.gnm2.", "", regex=False)
    genes["gene"] = genes.attr.str.extract(r"ID=glyma\.Wm82\.gnm2\.ann1\.([^;]+)")[0]
    info = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.info_annot.txt.gz", sep="\t", dtype=str)
    info = info.drop_duplicates("locusName").set_index("locusName")
    pf = info.Pfam.reindex(genes.gene).fillna("").str.replace(" ", "")
    genes["nbarc"] = pf.str.split(",").apply(lambda x: "PF00931" in x).to_numpy()
    genes = genes.reset_index(drop=True)
    in_block = []                                                   # gene indices of every block
    for chrom, b in blk.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        for i, r in b.iterrows():
            in_block.append((i, g.index[(g.end >= r.start) & (g.start <= r.end)].to_numpy()))
    in_block = dict(in_block)
    nbarc = genes.nbarc.to_numpy()
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    shared = np.sort(loci.block[loci.n_traits >= 2].to_numpy())

    def count(bs):
        u = np.unique(np.concatenate([in_block[int(b)] for b in bs]))
        return len(u), int(nbarc[u].sum())

    n_obs, k_obs = count(shared)
    sim = np.array([count((shared + k) % nb) for k in range(nb)])                 # every shift once
    out = pd.DataFrame([{"shared_blocks": len(shared), "genes": n_obs, "nbarc_genes": k_obs,
                         "expected_genes": sim[:, 0].mean(), "expected_nbarc_genes": sim[:, 1].mean(),
                         "expected_nbarc_q975": float(np.quantile(sim[:, 1], 0.975)),
                         "p_perm": float((sim[:, 1] >= k_obs).mean()),
                         "nbarc_genes_genome": int(nbarc[genes.chrom.str.match(r"Gm\d\d$").to_numpy()].sum()),
                         "genes_genome": int(genes.chrom.str.match(r"Gm\d\d$").sum()), "shifts": nb}])
    out.to_csv(ROOT / "results/loci/nbarc_shift.tsv", sep="\t", index=False, float_format="%.6g")
    print(out.T.to_string())


if __name__ == "__main__":
    main()
