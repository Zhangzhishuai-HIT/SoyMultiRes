"""Genes and loci of known position used with the locus map: data/processed/known_genes.tsv.

usage: python -I known_loci.py
set = calibration  12 genes used to choose the locus score (calibrate_readouts.py)
      control      nine loci reported for a single stress and not used for calibration; used to check the map (a locus
                   counts for the traits of its stress, e.g. all races of soybean cyst nematode)
      annotation   further genes and loci (most from the tables of Lin et al. 2022, Wm82.a2 positions) used to
                   annotate the regions of the map
Columns: set, locus, trait (trait or trait family the locus was reported for), where (gene model or interval), chrom,
start, end, SNPs inside, distance of the nearest SNP to the midpoint, blocks of 8, 16 and 32 SNPs, source.
"""
import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GFF = ROOT / "data/raw/wm82_gnm2_annotation/glyma.Wm82.gnm2.ann1.RVB6.gene_models_main.gff3.gz"

CAL = [  # intervals for loci without a single gene
    ("W1", "flower_purple", "Glyma.13G072100", "Zabala and Vodkin 2007"),
    ("T", "pubescence_tawny", "Glyma.06G202300", "Toda et al. 2002"),
    ("Dt1", "indeterminate", "Glyma.19G194300", "Liu et al. 2010"),
    ("E1", "maturity_group", "Glyma.06G207800", "Xia et al. 2012"),
    ("E2", "maturity_group", "Glyma.10G221500", "Watanabe et al. 2011"),
    ("E3", "maturity_group", "Glyma.19G224200", "Watanabe et al. 2009"),
    ("E4", "maturity_group", "Glyma.20G090000", "Liu et al. 2008"),
    ("I", "seedcoat_pigmented", "Glyma.08G110400", "Tuteja et al. 2004"),
    ("Rhg1", "NEMATCYST", "Gm18:1636622-1653524", "Cook et al. 2012"),
    ("Rhg4", "NEMATCYST", "Glyma.08G108900", "Liu et al. 2012"),
    ("Rps1", "PHYTOROT", "Gm03:4026212-4208647", "Gao et al. 2005; Lin et al. 2022"),
    ("GmSALT3", "SALTREACT", "Glyma.03G171600", "Guan et al. 2014"),
]
CONTROL = [
    ("Rfs2/GmRLK18-1", "SDS", "Glyma.18G023500", "Srour et al. 2012"),
    ("GmSNAP11", "NEMATCYST", "Glyma.11G234500", "Lakhssassi et al. 2017; Shaibu et al. 2022"),
    ("IDC QTL", "CHLOROSIS", "Gm03:34241291-34883065", "Kohlhase et al. 2024"),
    ("QTL-M/GmF3GlcT", "chewing_insects", "Glyma.07G110300", "Prabhakar et al. 2026"),
    ("Rcs(PI 594891)", "FROGEYE", "Gm13:29775320-29834027", "Pham et al. 2015"),
    ("Rsv1", "PMV", "Gm13:29815195-29912667", "Gore et al. 2002; Ma et al. 2016"),   # Rpv1 (peanut mottle virus) is linked to Rsv1
    ("Rag2", "APHID", "Gm13:30412426-30466537", "Kim et al. 2010; Yang et al. 2022"),
    ("Rag1", "APHID", "Gm07:5529532-5770718", "Neupane et al. 2019"),
    ("Rpp1", "RUST_TAN", "Gm18:56182230-56333803", "Pedley et al. 2019; Aoyagi et al. 2024"),
]
LIN = "Lin et al. 2022"
ANNOTATION = [
    ("NSF-RAN07", "NEMATCYST", "Glyma.07G195900", "Bayless et al. 2018"),
    ("qSCN10", "NEMATCYST", "Gm10:42430713-42809800", "Zhou et al. 2021; Lakhssassi et al. 2025"),   # fine-mapped interval
    ("Rps8", "PHYTOROT", "Gm13:24361239-28912878", LIN),
    ("qRps13-02", "PHYTOROT", "Gm13:27874365-30776191", LIN),
    ("Rps3a", "PHYTOROT", "Gm13:31802616-32721481", LIN),
    ("Rps2", "PHYTOROT", "Gm16:34035215-34035215", LIN),
    ("Rps5", "PHYTOROT", "Gm18:53866606-53866606", LIN),
    ("Rps6", "PHYTOROT", "Gm18:54450956-54450956", LIN),
    ("Rps12", "PHYTOROT", "Gm18:55962037-56333703", LIN),
    ("Rps4", "PHYTOROT", "Gm18:56333740-56333740", LIN),
    ("Rps13", "PHYTOROT", "Gm18:56333703-56341167", LIN),
    ("Rcs3", "FROGEYE", "Gm16:33818897-34035180", LIN),
    ("Rbs1-Rbs2", "BRNSTEMROT", "Gm16:28944536-36221397", LIN),
    # locus of pubescence form and leafhopper resistance: span of the two long non-coding RNA genes Glyma.12G213800
    # and Glyma.12G213900 (lncRG1, lncRG2 of Wang et al. 2024, Supplementary Table 6)
    ("qDRT12.3", "POTATOLHOP", "Gm12:37288662-37312496", "Wang et al. 2024"),
    # loci for resistance to southern stem canker (Diaporthe aspalathi); compared with northern stem canker
    ("Southern stem canker loci", "NSCANKER", "Gm14:1581712-1986224", LIN),
]


def main():
    snps = pd.read_csv(ROOT / "data/processed/geno/snps.tsv", sep="\t")
    gene_pos = {}
    with gzip.open(GFF, "rt") as f:
        for line in f:
            x = line.split("\t")
            if len(x) > 8 and x[2] == "gene":
                gid = x[8].split("ID=")[1].split(";")[0].replace("glyma.Wm82.gnm2.ann1.", "")
                gene_pos[gid] = (x[0].replace("glyma.Wm82.gnm2.", ""), int(x[3]), int(x[4]))
    rows = []
    for set_name, items in (("calibration", CAL), ("control", CONTROL), ("annotation", ANNOTATION)):
        for locus, trait, where, source in items:
            if where.startswith("Glyma"):
                c, a, b = gene_pos[where]
            else:
                c, rng = where.split(":")
                a, b = map(int, rng.split("-"))
            on = snps[(snps.chrom == c)]
            inside = on[(on.pos >= a) & (on.pos <= b)]
            nearest = on.iloc[(on.pos - (a + b) // 2).abs().argsort()[:1]]
            r = dict(set=set_name, locus=locus, trait=trait, where=where, chrom=c, start=a, end=b,
                     snps_inside=len(inside), nearest_snp_bp=int((nearest.pos - (a + b) // 2).abs().iloc[0]))
            for k in (8, 16, 32):
                bl = sorted(set(inside[f"block{k}"])) or [int(nearest[f"block{k}"].iloc[0])]
                r[f"blocks{k}"] = ",".join(map(str, bl))
            r["source"] = source
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data/processed/known_genes.tsv", sep="\t", index=False)
    print(d[["set", "locus", "trait", "chrom", "start", "end", "snps_inside", "blocks16"]].to_string(index=False))


if __name__ == "__main__":
    main()
