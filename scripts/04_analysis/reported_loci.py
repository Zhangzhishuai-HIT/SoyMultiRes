"""Loci of this study that were reported before for the same trait.

usage: python -I reported_loci.py
results/literature/reported_associations.tsv lists the SNP-trait associations that earlier association studies of
the USDA Soybean Germplasm Collection reported for the stress traits of this study (one row per association: source,
table, trait as reported, trait code of this study, SoySNP50K identifier, position, P value; compiled from the
published tables, see work/literature_compilation/README.md). For Wang et al. (2024), whose supplementary table lists
all tested SNPs, the SNPs of the chromosome 12 locus described in that article are used. A reported SNP is assigned
to the block that holds it (by its identifier; SNPs that are not among the 39,627 SNPs of this study by their
position on Wm82.a2). A trait-block combination of this study (results/loci/block_traits_emmax.tsv) counts as
reported before when a SNP reported for the same trait lies in the block or in an adjacent block of the same
chromosome.
Writes results/loci/reported_snps.tsv (study, trait, SNP, chromosome and position on Wm82.a2, block) and
results/loci/reported_before.tsv: one row per trait-block combination with the sources that reported a SNP in the
same block and in the same or an adjacent block (author and year, in order of the year). The number of combinations
counted as reported is compared with the number expected when the loci of every trait are shifted circularly along
the genome by a random number of blocks (10,000 shifts; results/loci/reported_before_null.tsv).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402


def author_year(source):
    """'Shook et al. 2021 G3 11:jkab117' -> 'Shook et al. 2021'; 'Chang & Hartman 2017 ...' -> 'Chang and Hartman 2017'."""
    return re.match(r"(.*? \d{4})", source.strip()).group(1).replace(" & ", " and ")


def main():
    ra = pd.read_csv(ROOT / "results/literature/reported_associations.tsv", sep="\t", low_memory=False)
    ra = ra[ra.trait_code.isin(set(T.TRAITS))].copy()
    ra = ra[~(ra.source.str.startswith("Wang et al. 2024") & (ra.chromosome != "Gm12"))]   # their table lists all tested SNPs: keep the locus they reported
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    rng = snps.groupby("block16").agg(chrom=("chrom", "first"), start=("pos", "min"), end=("pos", "max"))
    ra["block"] = ra.snp_id.map(snps.set_index("id").block16)
    for i in ra.index[ra.block.isna()]:                     # SNP not among the SNPs of this study: block from the position
        r = ra.loc[i]
        if pd.isna(r.position) or not str(r.assembly).startswith("Wm82.a2") and "equal Wm82.a2" not in str(r.assembly):
            continue
        x = rng[rng.chrom == r.chromosome]
        inside = x[(x.start <= r.position) & (x.end >= r.position)]
        ra.at[i, "block"] = inside.index[0] if len(inside) else np.minimum((x.start - r.position).abs(), (x.end - r.position).abs()).idxmin()
    print(f"{len(ra):,} reported associations of the {ra.trait_code.nunique()} traits; without a block: {int(ra.block.isna().sum())}")
    ra["src"] = ra.source.map(author_year)
    known = ra.snp_id.isin(set(snps.id))
    a2 = ra.assembly.astype(str).str.startswith("Wm82.a2") | ra.assembly.astype(str).str.contains("equal Wm82.a2")
    by_id = snps.set_index("id")
    lst = pd.DataFrame({"study": ra.src, "trait": ra.trait_code, "short": ra.trait_code.map(T.SHORT), "snp": ra.snp_id,
                        "chromosome": ra.snp_id.map(by_id.chrom).where(known, ra.chromosome.where(a2)),
                        "position": ra.snp_id.map(by_id.pos).where(known, ra.position.where(a2)).astype("Int64"),
                        "block": ra.block.astype("Int64")})
    lst = lst.drop_duplicates().sort_values(["study", "trait", "chromosome", "position"])
    lst.to_csv(ROOT / "results/loci/reported_snps.tsv", sep="\t", index=False)
    print(f"{len(lst):,} SNP-trait pairs of {lst.study.nunique()} studies -> results/loci/reported_snps.tsv")
    ra = ra[ra.block.notna()]
    ra["block"] = ra.block.astype(int)
    bt = pd.read_csv(ROOT / "results/loci/block_traits_emmax.tsv", sep="\t")
    year = lambda s: (int(re.search(r"(\d{4})$", s).group(1)), s)  # noqa: E731
    rows = []
    for r in bt.itertuples():
        x = ra[ra.trait_code == r.trait]
        same = x.block == r.block
        near = same | (x.block.isin([r.block - 1, r.block + 1]) & (x.block.map(rng.chrom) == rng.chrom[r.block]))
        rows.append({"block": r.block, "trait": r.trait, "short": r.short, "chrom": rng.chrom[r.block],
                     "reported_same_block": bool(same.any()), "reported_same_or_adjacent_block": bool(near.any()),
                     "sources_same_block": "; ".join(sorted(set(x.src[same]), key=year)),
                     "sources_same_or_adjacent_block": "; ".join(sorted(set(x.src[near]), key=year)),
                     "top_snp_reported": bool((x.snp_id == r.top_snp).any())})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/loci/reported_before.tsv", sep="\t", index=False)
    n = d.groupby("short").agg(loci=("block", "size"), reported=("reported_same_or_adjacent_block", "sum"))
    n = n.reindex([T.SHORT[t] for t in T.TRAITS if T.SHORT[t] in n.index])
    print(f"trait-block combinations: {len(d)}; reported in the same block: {int(d.reported_same_block.sum())}; in the same or an "
          f"adjacent block: {int(d.reported_same_or_adjacent_block.sum())}; not reported: {int((~d.reported_same_or_adjacent_block).sum())}")
    print("sources:", pd.Series([s for v in d.sources_same_or_adjacent_block for s in v.split("; ") if s]).value_counts().to_dict())
    print(n.assign(not_reported=n.loci - n.reported).T.to_string())
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    sh = d[d.block.isin(loci.block[loci.n_traits >= 2])].groupby("block").reported_same_or_adjacent_block.agg(["size", "sum"])
    print(f"shared loci: all traits reported {int((sh['sum'] == sh['size']).sum())}, two or more {int((sh['sum'] >= 2).sum())}, "
          f"one {int((sh['sum'] == 1).sum())}, none {int((sh['sum'] == 0).sum())}")
    # expected number of reported combinations when the loci of every trait are shifted circularly
    nb = int(rng.index.max()) + 1
    bchrom = rng.chrom.reindex(range(nb)).to_numpy()
    rep_blocks = {t: set(x.block) for t, x in ra.groupby("trait_code")}
    by_trait = {t: x.block.to_numpy() for t, x in bt.groupby("trait")}

    def n_reported(blocks_of):
        k = 0
        for t, b in blocks_of.items():
            R_ = rep_blocks.get(t, set())
            for b_ in b:
                if b_ in R_ or any(0 <= c < nb and c in R_ and bchrom[c] == bchrom[b_] for c in (b_ - 1, b_ + 1)):
                    k += 1
        return k

    obs = n_reported(by_trait)
    g = np.random.default_rng(20261116)
    ex = np.array([n_reported({t: (b + int(g.integers(nb))) % nb for t, b in by_trait.items()}) for _ in range(10000)])
    pd.DataFrame({"observed": [obs], "expected": [ex.mean()], "expected_q975": [np.quantile(ex, 0.975)],
                  "p_perm": [((ex >= obs).sum() + 1) / (len(ex) + 1)]}).to_csv(
        ROOT / "results/loci/reported_before_null.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"reported before: {obs} combinations, {ex.mean():.1f} expected under circular shifts")


if __name__ == "__main__":
    main()
