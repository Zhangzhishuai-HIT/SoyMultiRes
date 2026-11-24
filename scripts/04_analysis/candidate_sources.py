"""GRIN source type of the G. max accessions with a multi-resistance index of four or more (the candidates), and of
all genotyped G. max accessions for comparison.

usage: python -I candidate_sources.py
Source type: GRIN field source_type_code (COLLECTED, landraces and other material collected in the field; DEVELOPED,
cultivars and breeding lines; DONATED, material donated by another collection) from the downloaded descriptor tables;
accessions without a source type in these tables are "not given". Index: results/germplasm/ranking.tsv
(multi_resistance_index_within_taxon, TraitBoost fitted to all accessions). Writes
results/germplasm/candidate_sources.tsv (counts for all candidates, the never-scored candidates and all G. max
accessions) and results/germplasm/source_types.tsv (source type of every genotyped accession).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def source_types():
    st = {}
    for f in sorted((ROOT / "data/raw/grin_descriptors").glob("GRIN_crop51_*.csv")) + sorted((ROOT / "data/raw/grin_pubescence").glob("GRIN_crop51_*.csv")):
        d = pd.read_csv(f, skiprows=2, dtype=str, usecols=["accession_prefix", "accession_number", "accession_suffix", "source_type_code"])
        a_ = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
        for k, v in zip(a_, d.source_type_code):
            if isinstance(v, str):
                assert st.setdefault(k, v) == v, (k, st[k], v)        # one source type per accession across the tables
    return st


def main():
    rk = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    rk.index = rk.index.astype(str)
    st = source_types()
    rk["source_type"] = [st.get(a, "not given") for a in rk.index]
    rk[["source_type"]].to_csv(ROOT / "results/germplasm/source_types.tsv", sep="\t")
    gm = rk[rk.taxon == "Glycine max"]
    cand = gm[gm.multi_resistance_index_within_taxon >= 4]
    never = cand[cand.n_stress_traits.fillna(0) == 0]
    rows = []
    for name, sel in (("candidates", cand), ("never-scored candidates", never), ("all G. max", gm)):
        vc = sel.source_type.value_counts()
        rows.append({"group": name, "accessions": len(sel), **{k: int(vc.get(k, 0)) for k in ("COLLECTED", "DEVELOPED", "DONATED", "not given")}})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/germplasm/candidate_sources.tsv", sep="\t", index=False)
    print(out.to_string(index=False))
    print("developed candidates by origin:", cand[cand.source_type == "DEVELOPED"].origin.value_counts().head(6).to_dict())
    print("collected candidates by origin:", cand[cand.source_type == "COLLECTED"].origin.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
