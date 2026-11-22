"""Pubescence form at the chromosome 12 locus of potato leafhopper (qDRT12.3).

usage: python -I pubescence_check.py
The lead SNP of the locus of potato leafhopper on chromosome 12 that contains qDRT12.3 (37.29-37.31 Mb; results/loci/loci_emmax.tsv)
is compared with the pubescence form recorded in GRIN-Global (descriptor PUBFORM, id 51052; observation table
downloaded on 2026-11-15 to data/raw/grin_pubescence/GRIN_crop51_51052_PUBFORM.csv). Classes: E erect, Sa
semi-appressed, A appressed, Va very appressed, C curly, I irregular, - glabrous; appressed = A or Va. Genotyped
accessions with one recorded form are used (accessions with two different forms are left out).
Writes results/loci/pubescence_form_chr12.tsv: accessions by genotype of the lead SNP (homozygous for the reference
allele, heterozygous, homozygous for the alternative allele) and pubescence form, with the odds ratio and Fisher's
exact test of appressed against erect pubescence between the two homozygous genotypes, and the potato leafhopper
scores (GRIN classes 1-5, 1 = least injury) by pubescence form.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

FORM = {"E": "erect", "Sa": "semi-appressed", "A": "appressed", "Va": "appressed"}


def main():
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    b = loci[(loci.chrom == "Gm12") & loci.traits.str.split(",").apply(lambda x: "PLH" in x)
             & (loci.start <= 37_300_000) & (loci.end >= 37_300_000)]
    assert len(b) == 1, "one locus of potato leafhopper at 37.3 Mb on chromosome 12 expected"
    lead = b.lead_snp.iloc[0]
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    j = int(np.flatnonzero(snps.id.to_numpy() == lead)[0])
    samples = list(data.samples())
    g = pd.Series(np.asarray(np.load(data.P / "geno/dosage.npy", mmap_mode="r")[:, j]), index=samples)
    d = pd.read_csv(ROOT / "data/raw/grin_pubescence/GRIN_crop51_51052_PUBFORM.csv", skiprows=2, dtype=str,
                    usecols=["accession_prefix", "accession_number", "accession_suffix", "observation_value"])
    d["acc"] = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
    d = d[d.acc.isin(set(samples))]
    v = d.groupby("acc").observation_value.agg(lambda x: sorted(set(x)))
    one = v[v.apply(len) == 1].apply(lambda x: x[0])
    x = pd.DataFrame({"code": one}).join(g.rename("genotype"), how="inner")
    x["form"] = x.code.map(FORM).fillna("other")
    tab = pd.crosstab(x.genotype, x.form).reindex(index=[0, 1, 2], columns=["erect", "semi-appressed", "appressed", "other"]).fillna(0).astype(int)
    odds, p = fisher_exact([[tab.at[2, "appressed"], tab.at[2, "erect"]], [tab.at[0, "appressed"], tab.at[0, "erect"]]])
    out = tab.rename(index={0: "homozygous reference", 1: "heterozygous", 2: "homozygous alternative"})
    out.insert(0, "genotype", out.index)
    out["accessions"] = tab.sum(axis=1).to_numpy()
    out["snp"], out["chrom"], out["pos"] = lead, snps.chrom[j], int(snps.pos[j])
    out["odds_ratio_appressed_vs_erect"], out["fisher_p"] = odds, p
    out.to_csv(ROOT / "results/loci/pubescence_form_chr12.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"{lead} ({snps.chrom[j]}:{int(snps.pos[j])}, {snps.ref[j]}/{snps.alt[j]}); genotyped accessions with one recorded form: {len(x):,} "
          f"(two forms: {int((v.apply(len) > 1).sum())})")
    print(out.drop(columns=["snp", "chrom", "pos"]).to_string(index=False))
    # potato leafhopper score by pubescence form
    long = data.pheno_long()
    ph = long[long.trait == "POTATOLHOP"].groupby("acc").code.first()
    y = x.join(ph.rename("leafhopper_class"), how="inner")
    print("potato leafhopper class (1 = least injury) by pubescence form and by genotype:")
    print(pd.crosstab(y.form, y.leafhopper_class).to_string())
    print(pd.crosstab(y.genotype, y.leafhopper_class).to_string())


if __name__ == "__main__":
    main()
