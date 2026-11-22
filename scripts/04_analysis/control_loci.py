"""Comparison of the locus map with resistance genes and loci of known position that were not used for calibration.

usage: python -I control_loci.py
For every locus of data/processed/known_genes.tsv with set == control (nine loci reported for one stress, fixed before
the map was made) or set == annotation (further known genes and loci): the loci of the map (results/loci/
loci_emmax.tsv) that overlap the known locus +- 300 kb, and among their traits those of the stress the known locus was
reported for. A known locus is recovered when there is such a trait.
Writes results/loci/control_loci_check.tsv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import traits as T  # noqa: E402

FLANK = 300_000
FAMILY = {"NEMATCYST": [t for t in T.TRAITS if t.startswith("NEMATCYST")],
          "PHYTOROT": [t for t in T.TRAITS if t.startswith("PHYTOROT")],
          "FROGEYE": ["FROGEYE2", "FROGEYE3"], "virus": ["PMV", "BPMV"],
          "chewing_insects": ["MEXBEANBTL", "LOOPER", "VBC"]}


def traits_of(family):
    """Abbreviations of the stress traits a known locus was reported for."""
    return [T.SHORT[t] for t in FAMILY.get(family, [family])]


def near(loci, chrom, start, end):
    return loci[(loci.chrom == chrom) & (loci.end >= start - FLANK) & (loci.start <= end + FLANK)]


def main():
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    kg = pd.read_csv(ROOT / "data/processed/known_genes.tsv", sep="\t")
    order = [T.SHORT[t] for t in T.TRAITS]
    rows = []
    for _, k in kg[kg["set"].isin(["control", "annotation"])].iterrows():
        nb = near(loci, k.chrom, k.start, k.end)
        found = sorted({t for ts in nb.traits for t in ts.split(",")}, key=order.index)
        hit = [t for t in found if t in traits_of(k.trait)]
        rows.append({"set": k["set"], "locus": k.locus, "reported_for": k.trait, "chrom": k.chrom, "start": k.start,
                     "end": k.end, "blocks_within_300kb": len(nb), "found": bool(hit), "traits_found": ",".join(hit),
                     "all_traits_within_300kb": ",".join(found), "source": k.source})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/loci/control_loci_check.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(d.drop(columns=["source"]).to_string(index=False))
    c = d[d["set"] == "control"]
    print(f"control loci recovered: {int(c.found.sum())} of {len(c)}: {', '.join(c.locus[c.found])}")


if __name__ == "__main__":
    main()
