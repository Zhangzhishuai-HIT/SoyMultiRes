"""Minor allele counts of every SNP among the accessions that enter the association analysis of each trait.

usage: python -I gwas_mac.py --traitset stress|mendel [--taxon gmax --tag _gmax]
For each trait: the accessions used by gwas_emmax.py (accessions with folds and a value for the trait); per SNP the
number of accessions homozygous for the less frequent allele among them. Associations of SNPs with fewer than 30 such
accessions are not used for the locus scores (locus_map.py, calibrate_readouts.py).
Writes results/gwas/mac_<traitset>.npz (mac: traits x SNPs, traits, n).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/03_model"))
from soymultires import data  # noqa: E402
import train_mtboost as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traitset", default="stress", choices=["stress", "mendel"])
    ap.add_argument("--taxon", default="all", choices=["all", "gmax"])
    ap.add_argument("--tag", default="", help="suffix of the association file and of the output (analysis of a subset)")
    a = ap.parse_args()
    long = tb.use_traitset(a.traitset)
    traits = list(tb.TRAITS)
    fo = data.folds()
    long = long[long.acc.isin(set(fo.index))]
    if a.taxon == "gmax":
        taxon = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id").taxon
        long = long[long.acc.map(taxon) == "Glycine max"]
    pos = {v: i for i, v in enumerate(data.samples())}
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    gw = np.load(ROOT / f"results/gwas/emmax_{a.traitset}{a.tag}.npz", allow_pickle=True)
    assert [str(t) for t in gw["traits"]] == traits
    mac = np.zeros((len(traits), G.shape[1]), np.int32)
    ns = []
    for j, t in enumerate(traits):
        rows = np.sort(np.array([pos[x] for x in long.acc[long.trait == t].unique()]))
        assert len(rows) == int(gw["n"][j]), (t, len(rows), int(gw["n"][j]))
        X = np.asarray(G[rows])
        alt, ref = (X == 2).sum(0), (X == 0).sum(0)
        mac[j] = np.minimum(alt, ref)
        ns.append(len(rows))
        print(f"{t}: n {len(rows)}, SNPs with fewer than 30 minor homozygotes: {int((mac[j] < 30).sum())}", flush=True)
    np.savez_compressed(ROOT / f"results/gwas/mac_{a.traitset}{a.tag}.npz", mac=mac, traits=np.array(traits), n=np.array(ns))


if __name__ == "__main__":
    main()
