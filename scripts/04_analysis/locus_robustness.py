"""Robustness of the loci: genomic inflation factors and heritabilities of the association analyses, and the locus
maps obtained with other association models, block sizes, direction limits and without SCN race 14.

usage: python -I locus_robustness.py
Reads the association analyses results/gwas/emmax_stress{,_taxonpc,_gmax}.npz (gwas_emmax.py) with the minor-homozygote
counts mac_stress{,_gmax}.npz, and the locus maps results/loci/*_emmax<tag>.tsv written by locus_map.py for the tags
  (none)     main analysis: all accessions, relationship matrix only, blocks of 16 SNPs, |z| >= 3
  _taxonpc   Glycine soja indicator and five principal components as fixed effects
  _gmax      Glycine max accessions only (bean pod mottle virus left out: its variance ratio is at the boundary)
  _block8, _block32   blocks of 8 or 32 SNPs
  _z25, _z35          allele directions counted from |z| >= 2.5 or 3.5
  _noSCN14            without SCN race 14
  _min2snps           a locus needs at least two significant SNPs of the trait
  _res100             without the traits with fewer than 100 resistant accessions
  _cond               lead SNPs near Rhg1 and Rhg4 (SCN races) and Rps1 (P. sojae races) as fixed effects
                      (gwas_emmax.py --condition major_genes)
  _bonf               Bonferroni threshold for every trait (without the thresholds from simulated traits)
  _locusreading       allele directions read only for the traits of the block (not for the other races of a pathogen)
Writes results/gwas/diagnostics.tsv (per trait: accessions, tested SNPs, genomic heritability 1 / (1 + delta) and
inflation factor lambda = median statistic / 0.455 of the main analysis; accessions and lambda of the two other
analyses; loci in the three analyses) and results/loci/robustness.tsv (per analysis: trait-block combinations,
blocks, shared blocks with their expectation, classes, shared blocks by type, and what becomes of the shared blocks
of the main analysis, including the two race-differential loci; shared regions with their expectation) and
results/loci/race_differential_z.tsv (z of SCN races 1, 3, 5 and 14 at the lead SNPs of the two race-differential loci
in the three association analyses). Two checks of the main analysis:
  results/loci/direction_reading_snp.tsv   classes of the shared loci when the allele directions are read at the most
                                            significant SNP of each trait of the locus in turn instead of the lead SNP
  results/loci/locus_support.tsv           for every trait-block combination, the allele of the trait's most
                                            significant SNP that is associated with higher resistance, the scored and
                                            the resistant accessions homozygous for it, and the groups of
                                            near-identical accessions among the latter
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

GW, LOCI = ROOT / "results/gwas", ROOT / "results/loci"
ANALYSES = [("", "main analysis", ""), ("_taxonpc", "taxon and five principal components as fixed effects", ""),
            ("_gmax", "Glycine max accessions only", "_gmax"),
            ("_cond", "lead SNPs near Rhg1, Rhg4 and Rps1 as fixed effects", "")]
VARIANTS = [("", "Main analysis"), ("_taxonpc", "Taxon and five principal components as fixed effects"),
            ("_gmax", "Glycine max accessions only"), ("_block8", "Blocks of 8 SNPs"), ("_block32", "Blocks of 32 SNPs"),
            ("_z25", "Allele directions from |z| ≥ 2.5"), ("_z35", "Allele directions from |z| ≥ 3.5"),
            ("_noSCN14", "Without SCN race 14"), ("_min2snps", "Loci with at least two significant SNPs"),
            ("_res100", "Without the traits with fewer than 100 resistant accessions"),
            ("_cond", "Lead SNPs near Rhg1, Rhg4 and Rps1 as fixed effects"),
            ("_bonf", "Bonferroni threshold for all traits"),
            ("_locusreading", "Allele directions read for the traits of the block only")]
MED = float(chi2.ppf(0.5, 1))


def main():
    # ---- per trait
    cols = {}
    for tag, _, mtag in ANALYSES:
        g = np.load(GW / f"emmax_stress{tag}.npz", allow_pickle=True)
        m = np.load(GW / f"mac_stress{mtag}.npz", allow_pickle=True)
        traits = [str(t) for t in g["traits"]]
        assert traits == list(T.TRAITS)
        lam = [float(np.median(g["chi2"][j][m["mac"][j] >= 30]) / MED) for j in range(len(traits))]
        bt = pd.read_csv(LOCI / f"block_traits_emmax{tag}.tsv", sep="\t")
        k = tag or "_main"
        cols[f"accessions{k}"] = [int(x) for x in g["n"]]
        cols[f"lambda{k}"] = lam
        cols[f"delta{k}"] = [float(x) for x in g["delta"]]
        cols[f"loci{k}"] = bt.groupby("trait").size().reindex(traits).fillna(0).astype(int).tolist()
        if not tag:
            cols["tested_snps"] = [int((m["mac"][j] >= 30).sum()) for j in range(len(traits))]
    d = pd.DataFrame(cols, index=pd.Index(list(T.TRAITS), name="trait"))
    d.insert(0, "short", [T.SHORT[t] for t in d.index])
    d["h2_main"] = 1 / (1 + d["delta_main"])
    d["boundary_gmax"] = d["delta_gmax"] < 1e-4                 # variance ratio at the boundary: analysis not usable
    d.to_csv(GW / "diagnostics.tsv", sep="\t", float_format="%.6g")
    print(f"lambda, main analysis: {d.lambda_main.min():.2f}-{d.lambda_main.max():.2f} (median {d.lambda_main.median():.2f}); "
          f"h2 {d.h2_main.min():.2f}-{d.h2_main.max():.2f}; boundary in G. max only: {list(d.short[d.boundary_gmax])}")
    # ---- per analysis
    main_loci = pd.read_csv(LOCI / "loci_emmax.tsv", sep="\t")
    main_sh = main_loci[main_loci.n_traits >= 2]
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    rd = main_sh[main_sh["class"] == "race-differential"]
    rows = []
    for tag, label in VARIANTS:
        loci = pd.read_csv(LOCI / f"loci_emmax{tag}.tsv", sep="\t")
        bt = pd.read_csv(LOCI / f"block_traits_emmax{tag}.tsv", sep="\t")
        null = pd.read_csv(LOCI / f"sharing_null{tag}.tsv", sep="\t").set_index("min_traits")
        typ = pd.read_csv(LOCI / f"sharing_by_type{tag}.tsv", sep="\t").set_index("type")
        sh = loci[loci.n_traits >= 2]
        cl = sh["class"].value_counts()
        r = {"tag": tag or "main", "analysis": label, "trait_block_combinations": len(bt), "blocks": len(loci),
             "shared_blocks": len(sh), "expected_shared_blocks": float(null.expected_blocks[2]),
             "concordant": int(cl.get("concordant", 0)), "unresolved": int(cl.get("unresolved", 0)),
             "race_differential": int(cl.get("race-differential", 0)), "opposite_between_stresses": int(cl.get("trade-off", 0)),
             "mixed": int(cl.get("mixed", 0)),
             "shared_races_of_one_pathogen": int(typ.observed_blocks["races of one pathogen"]),
             "shared_different_pathogens": int(typ.observed_blocks["different pathogens or stresses"]),
             "expected_different_pathogens": float(typ.expected_pathogens_shifted_as_units["different pathogens or stresses"]),
             "p_different_pathogens": float(typ.p_pathogens_shifted_as_units["different pathogens or stresses"])}
        reg = pd.read_csv(LOCI / f"sharing_regions{tag}.tsv", sep="\t").iloc[0]
        r["shared_regions"], r["expected_shared_regions"] = int(reg.observed_regions), float(reg.expected_regions)
        # shared blocks of the main analysis: still loci of >= 2 traits (same SNP positions; for other block sizes the
        # block that holds the lead SNP of the main block)
        if tag in ("_block8", "_block32"):
            col = "block8" if tag == "_block8" else "block32"
            lead_block = snps.set_index("id")[col].reindex(main_sh.lead_snp).to_numpy()
            kept = np.isin(lead_block, sh.block.to_numpy())
        else:
            kept = main_sh.block.isin(sh.block).to_numpy()
        r["main_shared_blocks_kept"] = int(kept.sum())
        for _, b in rd.iterrows():                           # the two race-differential loci of the main analysis
            if tag in ("_block8", "_block32"):
                blk = int(snps.set_index("id")[col][b.lead_snp])
            else:
                blk = int(b.block)
            x = loci[loci.block == blk]
            r[f"class_{b.chrom}_{b.lead_pos}"] = "no locus" if x.empty else ("single trait" if x.n_traits.iloc[0] < 2 else x["class"].iloc[0])
            r[f"directions_{b.chrom}_{b.lead_pos}"] = "" if x.empty else x.directions.iloc[0]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(LOCI / "robustness.tsv", sep="\t", index=False, float_format="%.6g")
    # z of the SCN races at the lead SNPs of the two race-differential loci in the three association analyses
    zr = []
    for tag, _, mtag in ANALYSES:
        g = np.load(GW / f"emmax_stress{tag}.npz", allow_pickle=True)
        names = [str(t) for t in g["traits"]]
        r = {"tag": tag or "main"}
        for _, b in rd.sort_values(["chrom", "start"]).iterrows():
            j = int(np.flatnonzero(snps.id.to_numpy() == b.lead_snp)[0])
            z = [float(np.sign(g["beta"][names.index(f"NEMATCYST_{k}"), j]) * np.sqrt(g["chi2"][names.index(f"NEMATCYST_{k}"), j]))
                 for k in (1, 3, 5, 14)]
            r[f"z at {b.lead_snp} (chromosome {int(b.chrom[2:])}), SCN races 1, 3, 5 and 14"] = ", ".join(f"{v:.2f}".replace("-", "−") for v in z)
            r[f"zvals_{b.chrom}"] = z
        zr.append(r)
    zt = pd.DataFrame(zr)
    zt[[c for c in zt.columns if not c.startswith("zvals_")]].to_csv(LOCI / "race_differential_z.tsv", sep="\t", index=False)
    print(zt[[c for c in zt.columns if not c.startswith("zvals_")]].to_string(index=False))
    pd.set_option("display.width", 250)
    print(out.drop(columns=[c for c in out.columns if c.startswith("directions_")]).to_string(index=False))
    print(out[[c for c in out.columns if c.startswith(("tag", "directions_"))]].to_string(index=False))
    reading_snp(snps, main_loci)
    support(snps)


PATHOGEN = {t: "SCN" if t.startswith("NEMATCYST") else "Phytophthora" if t.startswith("PHYTOROT")
            else "frogeye" if t.startswith("FROGEYE") else t for t in T.TRAITS}


def classify(traits, dirs, other_traits=(), other_dirs=()):
    """Class of a block from its traits and allele directions, and the directions of the other races of their
    pathogens (as in locus_map.py)."""
    pathogen = PATHOGEN
    d = pd.DataFrame({"trait": traits, "direction": dirs})
    o = pd.DataFrame({"trait": list(other_traits), "direction": list(other_dirs)})
    clear = pd.concat([d[d.direction != "0"], o[o.direction != "0"]], ignore_index=True)
    if len(d) == 1:
        return "specific"
    if len(clear) < 2:
        return "unresolved"
    if clear.direction.nunique() == 1:
        return "concordant"
    pg = clear.trait.map(pathogen)
    within = any(g.nunique() > 1 for _, g in clear.direction.groupby(pg))
    return "race-differential" if pg.nunique() == 1 else ("trade-off" if not within else "mixed")


def reading_snp(snps, loci, zlim=3.0):
    """Classes of the shared loci of the main analysis when the directions are read at the most significant SNP of
    each trait of the locus in turn (an untested SNP of a trait gives no direction), with the other races of the
    pathogens of the locus read at the same SNP."""
    g = np.load(GW / "emmax_stress.npz", allow_pickle=True)
    m = np.load(GW / "mac_stress.npz", allow_pickle=True)
    traits = [str(t) for t in g["traits"]]
    chi = np.where(m["mac"] >= 30, np.nan_to_num(g["chi2"]), 0.0)
    sid = {s: i for i, s in enumerate(snps.id)}
    bt = pd.read_csv(LOCI / "block_traits_emmax.tsv", sep="\t")
    rows = []
    for _, L in loci[loci.n_traits >= 2].iterrows():
        b = bt[bt.block == L.block]
        js = [traits.index(t) for t in b.trait]
        pg = {PATHOGEN[t] for t in b.trait}
        os_ = [j for j, t in enumerate(traits) if j not in js and PATHOGEN[t] != t and PATHOGEN[t] in pg]
        cl = {}
        for s in dict.fromkeys(list(b.top_snp)):
            k = sid[s]
            z = np.sign(g["beta"][js, k]) * np.sqrt(chi[js, k])
            ot = [j for j in os_ if m["mac"][j, k] >= 30]
            zo = np.sign(g["beta"][ot, k]) * np.sqrt(chi[ot, k])
            cl[s] = classify(list(b.trait), np.where(z >= zlim, "+", np.where(z <= -zlim, "-", "0")),
                             [traits[j] for j in ot], np.where(zo >= zlim, "+", np.where(zo <= -zlim, "-", "0")))
        rows.append({"block": int(L.block), "chrom": L.chrom, "traits": L.traits, "class_lead_snp": L["class"],
                     "classes_at_top_snps": ";".join(f"{s}:{c}" for s, c in cl.items()),
                     "same_class_at_all_top_snps": all(c == L["class"] for c in cl.values())})
    d = pd.DataFrame(rows)
    d.to_csv(LOCI / "direction_reading_snp.tsv", sep="\t", index=False)
    print("shared loci with the same class at the most significant SNP of every trait:",
          d.groupby("class_lead_snp").same_class_at_all_top_snps.agg(["size", "sum"]).to_dict("index"))


def support(snps):
    """Scored and resistant accessions homozygous for the allele associated with higher resistance at the most
    significant SNP of every trait-block combination of the main analysis."""
    g = np.load(GW / "emmax_stress.npz", allow_pickle=True)
    traits = [str(t) for t in g["traits"]]
    bt = pd.read_csv(LOCI / "block_traits_emmax.tsv", sep="\t")
    sid = {s: i for i, s in enumerate(snps.id)}
    samples = list(data.samples())
    pos = {a: i for i, a in enumerate(samples)}
    cols = sorted({sid[s] for s in bt.top_snp})
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    X = pd.DataFrame(np.asarray(D[:, cols]), index=samples, columns=[snps.id[k] for k in cols])
    res = pd.read_csv(data.P / "resistant_matrix.tsv.gz", sep="\t", index_col=0)
    grp = data.folds().dup_group
    rows = []
    for r in bt.itertuples():
        k = sid[r.top_snp]
        alt = g["beta"][traits.index(r.trait), k] > 0                 # alternative allele associated with resistance
        sc = res[r.trait].dropna()
        sc = sc[sc.index.isin(set(pos))]
        x = X.loc[sc.index, r.top_snp]
        car = x == (2 if alt else 0)
        rc = sc.index[car & (sc == 1)]
        rows.append({"block": r.block, "trait": r.trait, "short": r.short, "top_snp": r.top_snp,
                     "resistance_allele": "alternative" if alt else "reference",
                     "scored_carriers": int(car.sum()), "resistant_carriers": len(rc),
                     "groups_of_resistant_carriers": int(grp.reindex(rc).nunique())})
    d = pd.DataFrame(rows)
    d.to_csv(LOCI / "locus_support.tsv", sep="\t", index=False)
    few = d.groups_of_resistant_carriers < 5
    print(f"trait-block combinations with resistant carriers in fewer than 5 groups of near-identical accessions: "
          f"{int(few.sum())} of {len(d)}; by trait: {d[few].short.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
