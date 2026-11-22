"""Never-scored candidate accessions: genetic identity with scored accessions and GRIN source type.

usage: python -I candidate_relatives.py [threads]
Candidates: Glycine max accessions with a multi-resistance index of four or more and no stress record
(results/germplasm/ranking.tsv). Identity with every scored accession (17,150, splits/folds.tsv) is computed as for
the groups of near-identical accessions (scripts/01_data/near_identical_groups.py): 1 - sum((g_i - g_j)^2) / (4 m) on
the 0/1/2 dosages of the 41,439 SNPs (missing calls set to the allele mean). Source type: GRIN field source_type_code
(COLLECTED, DEVELOPED, ...) as given in the downloaded descriptor tables (data/raw/grin_descriptors, data/raw/grin_pubescence).
Writes results/germplasm/candidate_relatives.tsv: per candidate the highest identity with a scored accession, that
accession, its taxon and origin, and the source type of the candidate; and candidate_relatives_reference.tsv: for
2,000 randomly drawn scored G. max accessions, the highest identity with the training accessions of their fold of the
cross-validation (the accessions of the other four folds).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

DOS = ROOT.parent / "HapGAM/data/interim/soysnp50k/dosage.npy"        # 41,439 SNPs, missing = -1 (read only)
SAM = ROOT.parent / "HapGAM/data/interim/soysnp50k/samples.tsv"


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    rk = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    cand = rk[(rk.taxon == "Glycine max") & (rk.multi_resistance_index_within_taxon >= 4)
              & (rk.n_stress_traits.fillna(0) == 0)].index.astype(str).tolist()
    cand_set = set(cand)
    fo = data.folds()
    scored = fo.index.astype(str).tolist()
    # reference: 2,000 scored G. max accessions (random, fixed seed), each against the training accessions of its fold
    acc0 = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id")
    gmax_scored = [a for a in scored if acc0.taxon.get(a) == "Glycine max"]
    ref = sorted(np.random.default_rng(20261116).choice(gmax_scored, 2000, replace=False).tolist())
    query = cand + ref
    sam = pd.read_csv(SAM, sep="\t", dtype=str)
    row_of = {v: i for i, v in enumerate(sam.vcf_id)}
    assert all(a in row_of for a in cand + scored)
    D = np.load(DOS, mmap_mode="r")
    G8s = np.asarray(D[np.array([row_of[a] for a in scored])])                  # int8, missing = -1
    G8c = np.asarray(D[np.array([row_of[a] for a in query])])
    m = G8s.shape[1]
    # identity = 1 - sum((c - s)^2) / (4 m), accumulated over chunks of SNPs (missing calls set to the allele mean
    # of the scored accessions)
    dot = torch.zeros(len(query), len(scored), dtype=torch.float64)
    nc, ns = torch.zeros(len(query), dtype=torch.float64), torch.zeros(len(scored), dtype=torch.float64)
    for j0 in range(0, m, 2048):
        Gs = G8s[:, j0:j0 + 2048].astype(np.float32)
        Gc = G8c[:, j0:j0 + 2048].astype(np.float32)
        miss_s, miss_c = Gs < 0, Gc < 0
        mu = np.where(miss_s, 0, Gs).sum(0) / np.maximum((~miss_s).sum(0), 1)
        Gs[miss_s] = mu[np.where(miss_s)[1]]
        Gc[miss_c] = mu[np.where(miss_c)[1]]
        S, C = torch.from_numpy(Gs), torch.from_numpy(Gc)
        dot += (C @ S.T).double()
        nc += (C * C).sum(1).double()
        ns += (S * S).sum(1).double()
    ident = 1 - (nc[:, None] + ns[None, :] - 2 * dot) / (4 * m)
    fold_s = torch.tensor(fo.cv1_fold.reindex(scored).to_numpy())
    ir = ident[len(cand):].clone()                                   # reference: only the training accessions of its fold
    ir[torch.tensor(fo.cv1_fold.reindex(ref).to_numpy())[:, None] == fold_s[None, :]] = -1.0
    rbest = ir.max(1).values.numpy()
    pd.DataFrame({"accession": ref, "fold": fo.cv1_fold.reindex(ref).to_numpy(), "max_identity_with_training": rbest}).to_csv(
        ROOT / "results/germplasm/candidate_relatives_reference.tsv", sep="\t", index=False)
    print(f"reference: {len(ref)} scored G. max accessions against the training accessions of their fold: median highest "
          f"identity {np.median(rbest):.3f}; identity >= 0.99: {int((rbest >= 0.99).sum())}")
    best, arg = ident[:len(cand)].max(1)
    acc = pd.read_csv(data.P / "accessions.tsv", sep="\t", dtype={"vcf_id": str}).set_index("vcf_id")
    # GRIN source type (COLLECTED, DEVELOPED, DONATED, ...) of every accession, from all downloaded descriptor tables
    st = {}
    for f in sorted((ROOT / "data/raw/grin_descriptors").glob("GRIN_crop51_*.csv")) + sorted((ROOT / "data/raw/grin_pubescence").glob("GRIN_crop51_*.csv")):
        d = pd.read_csv(f, skiprows=2, dtype=str, usecols=["accession_prefix", "accession_number", "accession_suffix", "source_type_code"])
        a_ = (d.accession_prefix.fillna("") + d.accession_number.fillna("") + d.accession_suffix.fillna("")).str.replace(" ", "")
        for k, v in zip(a_, d.source_type_code):
            if isinstance(v, str) and k in cand_set:
                assert st.setdefault(k, v) == v, (k, st[k], v)        # one source type per accession across the tables
    out = pd.DataFrame({"candidate": cand, "origin": acc.origin.reindex(cand).to_numpy(),
                        "source_type": [st.get(a, "") for a in cand],
                        "max_identity_with_scored": best.numpy(),
                        "closest_scored_accession": [scored[i] for i in arg.numpy()]})
    out["closest_taxon"] = acc.taxon.reindex(out.closest_scored_accession).to_numpy()
    out["closest_origin"] = acc.origin.reindex(out.closest_scored_accession).to_numpy()
    out.to_csv(ROOT / "results/germplasm/candidate_relatives.tsv", sep="\t", index=False)
    print(f"{len(cand)} never-scored candidates; identity >= 0.99 with a scored accession: "
          f"{int((out.max_identity_with_scored >= 0.99).sum())}; median highest identity {out.max_identity_with_scored.median():.3f}")
    print("source type:", out.source_type.replace("", "not given").value_counts().to_dict())
    print("origin:", out.origin.value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
