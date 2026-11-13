"""Race-differential locus next to Rhg1 (block 2059, Gm18 1.19-1.39 Mb): SCN race reactions by allele at the
block's lead SNP, and the genotypes of the SCN race differentials.

usage: python -I race_differential_check.py [--block 2059]
Accessions scored resistant to race 3 are split by the allele at the lead SNP (alternative homozygote vs reference
homozygote); for races 1, 5 and 14 the share of resistant records is compared (Fisher's exact test). Race 3 resistance
comes from Peking-type (rhg1-a + Rhg4; effective against races 1, 3 and 5) or PI 88788-type sources (rhg1-b; races 3
and 14), so an allele that tags one source type separates the two race spectra.
Writes results/loci/race_differential_<block>.tsv and prints the genotypes of the differentials.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

DIFFERENTIALS = {"Peking": "PI548402", "Pickett": "PI548988", "PI 90763": "PI90763", "PI 437654": "PI437654",
                 "PI 88788": "PI88788", "PI 209332": "PI209332", "PI 89772": "PI89772", "Cloud": "PI548316",
                 "Williams 82": "PI518671"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2059)
    a = ap.parse_args()
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t").set_index("block")
    L = loci.loc[a.block]
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    j = int(np.flatnonzero(snps["id"].to_numpy() == L.lead_snp)[0])
    ids = data.samples()
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    g = pd.Series(np.asarray(D[:, j]), index=ids.values, name="allele")
    print(f"block {a.block} {L.chrom}:{L.start:,}-{L.end:,}, lead SNP {L.lead_snp} at {L.lead_pos:,}, "
          f"alternative allele frequency {snps.alt_freq[j]:.3f}; directions {L.directions}")
    print("differentials (alternative-allele dosage):", {k: int(g[v]) for k, v in DIFFERENTIALS.items()})
    long = data.pheno_long()
    r = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.mean().unstack()
    r = (r > 0.5).where(r.notna()).join(g, how="inner")
    rows = []
    for subset, sub in (("all scored", r), ("race-3 resistant", r[r.NEMATCYST_3 == True])):  # noqa: E712
        for t in ("NEMATCYST_1", "NEMATCYST_3", "NEMATCYST_4", "NEMATCYST_5", "NEMATCYST_14"):
            s = sub[sub[t].notna()]
            c, n = s[s.allele == 2][t].astype(bool), s[s.allele == 0][t].astype(bool)
            if subset == "race-3 resistant" and t in ("NEMATCYST_3", "NEMATCYST_4"):
                continue
            p = fisher_exact([[c.sum(), (~c).sum()], [n.sum(), (~n).sum()]])[1]
            rows.append({"accessions": subset, "race": t.split("_")[1], "carriers": len(c),
                         "carriers_resistant": c.mean(), "noncarriers": len(n), "noncarriers_resistant": n.mean(),
                         "fisher_p": p})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/loci/race_differential_{a.block}.tsv", sep="\t", index=False, float_format="%.4g")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
