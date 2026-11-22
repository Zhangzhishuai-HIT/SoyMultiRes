"""Separation of SCN resistance types by the two race-differential lead SNPs when only resistant (R) reactions are
counted as resistant (Supplementary Note 4).

usage: python -I race_differential_r_only.py
In the main analysis (race_differential_check.py, Table S10), an accession is resistant to an SCN race when it was resistant
or moderately resistant (GRIN codes R or MR) in at least one experiment. Here, among the accessions resistant to race 3
and homozygous at the lead SNP of each race-differential locus (results/loci/loci_emmax.tsv), the percentage resistant
to races 1, 5 and 14 of the carriers (homozygous for the alternative allele) and of the non-carriers is given with
resistance to these races coded as R or MR (as in the main analysis) and as R only; the race-3-resistant subset is
taken with race 3 coded as R or MR (as in the main analysis) and as R only. The carriers resistant to race 14 under
the main coding are listed.
Writes results/loci/race_differential_r_only.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402


def main():
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    lead = loci[loci["class"] == "race-differential"].sort_values(["chrom", "start"])
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    ids = data.samples()
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    long = data.pheno_long()
    scn = long[long.trait.str.startswith("NEMATCYST") & long.acc.isin(set(ids))]
    assert scn.code.notna().all()
    coded = {"R or MR": scn.groupby(["acc", "trait"]).resistant.max().unstack(),
             "R only": scn.assign(r=(scn.code == "R").astype(int)).groupby(["acc", "trait"]).r.max().unstack()}
    rows = []
    for _, L in lead.iterrows():
        j = int(np.flatnonzero(snps["id"].to_numpy() == L.lead_snp)[0])
        g = pd.Series(np.asarray(D[:, j]), index=ids)
        for c3, M3 in coded.items():
            sub = M3.index[(M3.NEMATCYST_3 == 1).to_numpy()]
            sub = sub[g.reindex(sub).isin([0, 2]).to_numpy()]
            for race in (1, 5, 14):
                t = f"NEMATCYST_{race}"
                for cr, M in coded.items():
                    y = M.reindex(sub)[t].dropna()
                    car = (g.reindex(y.index) == 2).to_numpy()
                    rows.append({"block": int(L.block), "chrom": L.chrom, "lead_snp": L.lead_snp, "race3_coding": c3,
                                 "race3_resistant": len(sub), "race": race, "coding": cr, "carriers": int(car.sum()),
                                 "carriers_resistant": int(y[car].sum()), "noncarriers": int((~car).sum()),
                                 "noncarriers_resistant": int(y[~car].sum()),
                                 "pct_carriers": round(100 * float(y[car].mean()), 1),
                                 "pct_noncarriers": round(100 * float(y[~car].mean()), 1),
                                 "resistant_carriers": ",".join(sorted(y[car][y[car] == 1].index))
                                 if (race == 14 and c3 == cr == "R or MR") else ""})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/loci/race_differential_r_only.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["chrom"]).to_string(index=False))


if __name__ == "__main__":
    main()
