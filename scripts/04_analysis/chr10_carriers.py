"""Origin, GRIN source type and resistance to SCN race 3 of the accessions carrying the alternative allele of the
chromosome 10 lead SNP (ss715606985) of the race-differential locus (Supplementary Note 4).

usage: python -I chr10_carriers.py
Carriers are the accessions homozygous for the alternative allele (dosage 2), non-carriers those homozygous for the
reference allele (dosage 0), as in the comparisons of the race-differential loci. Source type: GRIN field
source_type_code (COLLECTED, DEVELOPED, DONATED) from the downloaded descriptor tables. Resistance to SCN race 3: an
accession resistant in at least one experiment (data/processed/resistant_matrix.tsv.gz). Counts are given for the
G. max accessions by origin (countries with at least ten carriers) and for all G. max accessions.
Writes results/loci/chr10_carriers.tsv, and results/loci/chr10_conditional.tsv: z (signed square root of the
association statistic) of SCN races 1, 3, 5 and 14 at the lead SNP in the main analysis and with the five SNPs of
jobs/r11_cond_chr10.sh as fixed effects (results/gwas/emmax_stress_condscn.npz; Supplementary Note 4).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

SNP = "ss715606985"


def main():
    g = data.P / "geno"
    snps = pd.read_csv(g / "snps.tsv", sep="\t")
    sid = pd.read_csv(g / "samples.tsv", sep="\t", dtype=str).vcf_id.to_numpy()
    j = int(np.flatnonzero(snps.id.astype(str).to_numpy() == SNP)[0])
    dos = np.load(g / "dosage.npy", mmap_mode="r")
    assert dos.shape == (len(sid), len(snps))
    df = pd.DataFrame({"acc": sid, "dosage": np.asarray(dos[:, j])})
    acc = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id")
    df = df.join(acc[["taxon", "origin"]], on="acc")
    st = {}
    for f in sorted((ROOT / "data/raw/grin_descriptors").glob("GRIN_crop51_*.csv")) + sorted((ROOT / "data/raw/grin_pubescence").glob("GRIN_crop51_*.csv")):
        d = pd.read_csv(f, skiprows=2, dtype=str, usecols=["accession_prefix", "accession_number", "accession_suffix", "source_type_code"])
        a_ = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
        for k, v in zip(a_, d.source_type_code):
            if isinstance(v, str):
                assert st.setdefault(k, v) == v, (k, st[k], v)        # one source type per accession across the tables
    df["source_type"] = df.acc.map(st).fillna("not given")
    res = pd.read_csv(data.P / "resistant_matrix.tsv.gz", sep="\t", index_col=0)
    res.index = res.index.astype(str)
    df["scn3_resistant"] = res["NEMATCYST_3"].reindex(df.acc).to_numpy()
    print(f"{SNP}: {int((df.dosage == 2).sum())} of {len(df)} genotyped accessions homozygous for the alternative allele "
          f"({100 * (df.dosage == 2).mean():.1f}%), {int((df.dosage == 1).sum())} heterozygous")
    gm = df[df.taxon == "Glycine max"]
    rows = []
    origins = [o for o, n in gm[gm.dosage == 2].origin.value_counts().items() if n >= 10]
    for name, sel in [("all G. max", gm)] + [(o, gm[gm.origin == o]) for o in origins]:
        c, n = sel[sel.dosage == 2], sel[sel.dosage == 0]
        cs, ns = c[c.scn3_resistant.notna()], n[n.scn3_resistant.notna()]
        rows.append({"group": name, "accessions": len(sel), "carriers": len(c), "carrier_share": len(c) / len(sel),
                     "carriers_developed": int((c.source_type == "DEVELOPED").sum()),
                     "carriers_collected": int((c.source_type == "COLLECTED").sum()),
                     "carriers_scored_scn3": len(cs), "carriers_resistant_scn3": int(cs.scn3_resistant.sum()),
                     "noncarriers_scored_scn3": len(ns), "noncarriers_resistant_scn3": int(ns.scn3_resistant.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/loci/chr10_carriers.tsv", sep="\t", index=False, float_format="%.6f")
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    conditional(snps, j)


def conditional(snps, j):
    """z of SCN races 1, 3, 5 and 14 at the lead SNP without and with the five SNPs of jobs/r11_cond_chr10.sh as fixed
    effects (the SNPs are read from the job file, so that the table names what was fitted)."""
    job = (ROOT / "jobs/r11_cond_chr10.sh").read_text()
    fixed = job.split("--condition_snps")[1].split('"')[1].split(":")[1].split(",")
    rows = []
    for tag, f in (("main analysis", "emmax_stress.npz"), ("five SNPs fitted", "emmax_stress_condscn.npz")):
        z = np.load(ROOT / "results/gwas" / f, allow_pickle=True)
        tr = [str(x) for x in z["traits"]]
        for race in (1, 3, 5, 14):
            k = tr.index(f"NEMATCYST_{race}")
            rows.append({"analysis": tag, "race": race, "fixed_snps": ",".join(fixed) if tag != "main analysis" else "",
                         "fixed_effects": int(z["q"][k]), "z": float(np.sign(z["beta"][k][j]) * np.sqrt(z["chi2"][k][j]))})
    out = pd.DataFrame(rows)
    assert (out[out.analysis != "main analysis"].fixed_effects == 1 + len(fixed)).all()      # intercept + the five SNPs
    out.to_csv(ROOT / "results/loci/chr10_conditional.tsv", sep="\t", index=False, float_format="%.6f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
