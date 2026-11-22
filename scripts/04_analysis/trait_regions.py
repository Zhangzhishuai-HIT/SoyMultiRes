"""Loci of every trait merged into regions: blocks of a trait less than three blocks apart on the same chromosome form
one region (the rule used for the shared loci in locus_map.py).

usage: python -I trait_regions.py [--tag ""]
Reads results/loci/block_traits_emmax<tag>.tsv (trait-block combinations) and geno/snps.tsv (block16, chrom) and
writes results/loci/trait_regions<tag>.tsv (per trait: loci (blocks) and regions; last row: all traits).
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

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
a = ap.parse_args()
snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
bchrom = snps.groupby("block16").chrom.first()
bt = pd.read_csv(ROOT / f"results/loci/block_traits_emmax{a.tag}.tsv", sep="\t")
rows = []
for t in T.TRAITS:
    b = np.sort(bt.block[bt.trait == t].unique())
    if len(b):
        new = np.ones(len(b), bool)
        new[1:] = (np.diff(b) > 2) | (bchrom.reindex(b[1:]).to_numpy() != bchrom.reindex(b[:-1]).to_numpy())
        nreg = int(new.sum())
    else:
        nreg = 0
    rows.append({"trait": t, "short": T.SHORT[t], "loci": len(b), "regions": nreg})
out = pd.DataFrame(rows)
out = pd.concat([out, pd.DataFrame([{"trait": "all", "short": "all", "loci": int(out.loci.sum()),
                                     "regions": int(out.regions.sum())}])], ignore_index=True)
out.to_csv(ROOT / f"results/loci/trait_regions{a.tag}.tsv", sep="\t", index=False)
print(out.to_string(index=False))
