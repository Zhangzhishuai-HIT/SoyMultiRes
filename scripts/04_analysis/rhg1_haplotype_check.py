"""SNP haplotypes around Rhg1 and the allele near Rhg4 of the accessions with and without the alternative allele of the
chromosome 10 lead SNP ss715606985, which detects the retrotransposon RAC in the alpha-SNAP gene of the low-copy rhg1-a
haplotype of Rhg1 (Bayless et al. 2019) (Supplementary Note 4).

usage: python -I rhg1_haplotype_check.py
Calls: homozygous for the alternative allele (carrier) or for the reference allele (non-carrier) at ss715606985.
Rhg1 haplotype: the alleles at the SNPs of chromosome 18 between 1.56 and 1.72 Mb (around Rhg1, 1.64-1.65 Mb; Table S2);
an accession has the haplotype of Peking (PI 548402) or of PI 88788 when its calls equal those of that line at all
these SNPs. Rhg4 allele: at the SNP near Rhg4 most significant for SCN race 3 (ss715602753, fitted in the conditional
analyses), the allele associated with higher resistance to race 3 (sign of z of race 3 in the main analysis); an
accession has it when homozygous for it. Resistance to race 3: resistant in at least one experiment.
Writes results/loci/rhg1_haplotypes.tsv (carriers and non-carriers by Rhg1 haplotype, with r2 between the indicators of
the alternative allele (homozygous) and of the Peking haplotype over all genotyped accessions; G. max carriers by origin
with the Rhg4 allele and their resistance to race 3) and results/loci/rhg1_named_accessions.tsv (the genotyped accessions
named Peking in GRIN, the differential lines and resistance sources of Table S9, Jack, PI 567516C and Zi Hua No. 4:
call, identity with Peking and with PI 88788 at the Rhg1 SNPs, Rhg4 allele and resistance to SCN races 1, 3, 5 and 14).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

SNP, RHG4_SNP = "ss715606985", "ss715602753"
RHG1_WIN = ("Gm18", 1_560_000, 1_720_000)
CHR10_WIN = ("Gm10", 39_600_000, 41_800_000)
NAMED = {"PI297543": "Peking (GRIN name)", "PI438496A": "Peking (GRIN name)", "PI438496B": "Peking (GRIN name)",
         "PI438496C": "Peking (GRIN name)", "PI438497": "Peking (GRIN name)", "PI548402": "Peking (differential line)",
         "PI548402S": "Peking (GRIN name)", "PI90763": "Peking type", "PI437654": "Peking type", "PI89772": "Peking type",
         "PI548655": "Forrest (Peking type)", "PI88788": "PI 88788 type", "PI209332": "PI 88788 type",
         "PI548316": "Cloud (PI 88788 type)", "PI540556": "Jack", "PI567516C": "source of qSCN10",
         "PI561354": "Zi Hua No. 4"}


def main():
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    ids = data.samples()
    pos = {a: i for i, a in enumerate(ids)}
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    acc = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id").reindex(ids)
    long = data.pheno_long()
    res = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.max().unstack()
    col = lambda s: int(np.flatnonzero(snps.id.to_numpy() == s)[0])  # noqa: E731
    win = lambda c, a, b: np.flatnonzero((snps.chrom == c).to_numpy() & (snps.pos >= a).to_numpy() & (snps.pos <= b).to_numpy())  # noqa: E731
    j, j4 = col(SNP), col(RHG4_SNP)
    w1 = win(*RHG1_WIN)
    call = np.asarray(D[:, j])
    X1 = np.asarray(D[:, w1])
    g4 = np.asarray(D[:, j4])
    em = np.load(ROOT / "results/gwas/emmax_stress.npz", allow_pickle=True)
    k3 = [str(x) for x in em["traits"]].index("NEMATCYST_3")
    r4 = 2 if em["beta"][k3][j4] > 0 else 0                  # homozygous state of the allele associated with resistance to race 3
    hap_pk = (X1 == X1[pos["PI548402"]]).all(1)
    hap_88 = (X1 == X1[pos["PI88788"]]).all(1)
    c2, c0 = call == 2, call == 0
    r2 = float(np.corrcoef(c2.astype(float), hap_pk.astype(float))[0, 1] ** 2)
    rows = [{"group": "all genotyped accessions", "accessions": len(ids), "carriers": int(c2.sum()), "noncarriers": int(c0.sum()),
             "r2_with_peking_haplotype": round(r2, 4)}]
    for name, sel in (("Peking haplotype at Rhg1", hap_pk), ("PI 88788 haplotype at Rhg1", hap_88), ("other Rhg1 haplotypes", ~hap_pk & ~hap_88)):
        rows.append({"group": name, "accessions": int(sel.sum()), "carriers": int((c2 & sel).sum()), "noncarriers": int((c0 & sel).sum())})
    r3 = res.NEMATCYST_3
    for origin in ("Korea, South", "United States", "China"):
        sel = c2 & (acc.origin.to_numpy() == origin) & (acc.taxon.to_numpy() == "Glycine max")
        a_ = pd.Series(ids)[sel]
        has4 = pd.Series(g4[sel] == r4, index=a_.to_numpy())
        sc = a_[a_.isin(r3.dropna().index)].to_numpy()
        rows.append({"group": f"G. max carriers from {origin}", "accessions": int(sel.sum()), "carriers": int(sel.sum()),
                     "with_peking_rhg1": int((hap_pk[sel]).sum()), "with_rhg4_allele": int(has4.sum()),
                     "scored_race3": len(sc), "resistant_race3": int(r3.reindex(sc).sum()),
                     "scored_race3_with_rhg4_allele": int(has4.reindex(sc).sum()),
                     "resistant_race3_with_rhg4_allele": int(r3.reindex(sc[has4.reindex(sc).to_numpy()]).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/loci/rhg1_haplotypes.tsv", sep="\t", index=False)
    ident = lambda a, b, w: (np.asarray(D[pos[a], w]) == np.asarray(D[pos[b], w])).mean()  # noqa: E731
    w10 = np.setdiff1d(win(*CHR10_WIN), [j])
    nrows = []
    for a, lab in NAMED.items():
        if a not in pos:
            continue
        r = res.loc[a] if a in res.index else pd.Series(dtype=float)
        nrows.append({"accession": a, "label": lab, "call": int(call[pos[a]]), "identity_rhg1_peking": round(float(ident(a, "PI548402", w1)), 3),
                      "identity_rhg1_pi88788": round(float(ident(a, "PI88788", w1)), 3), "rhg4_allele": bool(g4[pos[a]] == r4),
                      **{f"resistant_SCN{k}": r.get(f"NEMATCYST_{k}", np.nan) for k in (1, 3, 5, 14)}})
    nm = pd.DataFrame(nrows)
    nm.to_csv(ROOT / "results/loci/rhg1_named_accessions.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(f"Rhg1 SNPs: {len(w1)}; allele near Rhg4 associated with resistance to race 3: {'alternative' if r4 == 2 else 'reference'}")
    print(out.to_string(index=False))
    print(nm.to_string(index=False))
    a_, b_ = "PI438496A", "PI438496B"
    print(f"{a_} vs {b_}: chromosome 10 ({CHR10_WIN[1] / 1e6:.1f}-{CHR10_WIN[2] / 1e6:.1f} Mb, without the lead SNP) "
          f"{int((np.asarray(D[pos[a_], w10]) == np.asarray(D[pos[b_], w10])).sum())} of {len(w10)} SNPs identical; "
          f"Rhg1 SNPs {int((np.asarray(D[pos[a_], w1]) == np.asarray(D[pos[b_], w1])).sum())} of {len(w1)} identical")


if __name__ == "__main__":
    main()
