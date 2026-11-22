"""Loci of the 33 stress traits from the mixed-model association analysis, loci shared among traits, and candidate genes.

usage: python -I locus_map.py [--alpha 0.05] [--min_mac 30] [--perm 10000]
       python -I locus_map.py --gwas _taxonpc --tag _taxonpc        loci of another association analysis
       python -I locus_map.py --gwas _gmax --mac _gmax --drop BPMV --tag _gmax   (REML ratio of BPMV at its bound)
       python -I locus_map.py --block 8 --tag _block8 | --zlim 2.5 --tag _z25 | --drop NEMATCYST_14 --tag _noSCN14
       python -I locus_map.py --null none --tag _bonf | --read_races 0 --tag _locusreading
Inputs: mixed-model association analysis (results/gwas/emmax_stress<gwas>.npz: squared t statistic and beta per
alternative allele on the standardised trait; every trait is coded so that a higher value means more resistant / less
damage), blocks of 16 (8, 32) consecutive SNPs (geno/snps.tsv), gene models and functional annotation of Wm82.a2 (LIS
Wm82.gnm2.ann1.RVB6).
A block is a locus of a trait when it contains at least one SNP that is significant for the trait at the stricter of
two thresholds: the Bonferroni threshold for the number of SNPs (P < alpha / 39,627; the statistic is referred to the
chi-square distribution with one degree of freedom) and the trait's genome-wide threshold at a family-wise error rate of
5% from simulated traits without SNP effects (results/gwas/null_threshold.tsv, gwas_null_threshold.py; --null none:
Bonferroni threshold only, tag _bonf). The analyses other than the main one use the thresholds of the main analysis. SNPs with fewer than min_mac accessions homozygous for the less frequent allele among the
accessions scored for the trait are not tested (results/gwas/mac_stress<mac>.npz). For every block that is a locus of
>= 1 trait: the traits, their stress categories, the direction of the alternative allele for each trait at one SNP of
the block, its lead SNP (the SNP with the highest sum of the test statistics of the block's traits; sign of beta,
counted where |z| >= zlim; for the traits of a pathogen with several races or isolates (SCN, Phytophthora sojae,
Cercospora sojina) the directions of the other races of the pathogen tested at the lead SNP are read as well and enter
the class (column other_races; --read_races 0: traits of the block only, tag _locusreading)), and the class
  specific            one trait
  unresolved          >= 2 traits, fewer than 2 counted directions
  concordant          all counted directions equal
  race-differential   opposite directions only among races of one pathogen (SCN, Phytophthora or frogeye races)
  trade-off           opposite directions only between different pathogens / stresses
  mixed               opposite directions both within a pathogen and between pathogens
The number of blocks shared by >= 2, 3, 4 traits is compared with the numbers obtained when the loci of every trait
are shifted along the genome by a random number of blocks (circular shifts, which keep the clustering of the loci of
a trait); the expected numbers are exact (at every block a trait is a locus with probability k_t / nb, independently
of the other traits), the P values from the shifts. Shared blocks are also counted by type: all traits are races of one pathogen, or the traits belong to
different pathogens or stresses; the second count is compared with shifts in which the races of a pathogen are moved
together, so that the loci shared among races stay shared (sharing_by_type<tag>.tsv). Shared blocks less than three
blocks apart on a chromosome form regions; the number of regions is compared with the same shifts
(sharing_regions<tag>.tsv). --min_snps 2 requires two significant SNPs of a trait for a locus. Candidate genes (main analysis
only): gene models overlapping the shared blocks, with Pfam-based flags for resistance-gene families.
Writes results/loci/{block_traits, loci, candidate_genes}_emmax<tag>.tsv, sharing_null<tag>.tsv,
sharing_by_type<tag>.tsv and summary_emmax<tag>.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

ANN = ROOT / "data/raw/wm82_gnm2_annotation"   # LIS Wm82.gnm2.ann1.RVB6 (symbolic link to the downloaded files)
PFAM_FLAGS = {"NB-ARC": {"PF00931"}, "LRR": {"PF00560", "PF07723", "PF07725", "PF12799", "PF13516", "PF13855",
                                            "PF08263"},
              "TIR": {"PF01582", "PF13676"}, "kinase": {"PF00069", "PF07714"}, "WRKY": {"PF03106"},
              "lectin": {"PF00139", "PF01453"}, "PR/chitinase": {"PF00187", "PF00182"}}
CAT = {row[0]: row[4] for row in T.STRESS}
# races of one pathogen; every other trait is its own pathogen / stress (stress categories such as fungi_oomycetes
# group different pathogens, so they cannot define race differences)
PATHOGEN = {t: "SCN" if t.startswith("NEMATCYST") else "Phytophthora" if t.startswith("PHYTOROT")
            else "frogeye" if t.startswith("FROGEYE") else t for t in T.TRAITS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min_mac", type=int, default=30)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--window_kb", type=float, default=0)
    ap.add_argument("--gwas", default="", help="suffix of the association file (emmax_stress<gwas>.npz)")
    ap.add_argument("--mac", default="", help="suffix of the minor-homozygote file (mac_stress<mac>.npz)")
    ap.add_argument("--block", type=int, default=16, choices=[8, 16, 32], help="SNPs per block")
    ap.add_argument("--zlim", type=float, default=3.0, help="|z| from which an allele direction is counted")
    ap.add_argument("--drop", default="", help="traits left out, comma-separated")
    ap.add_argument("--min_snps", type=int, default=1, help="significant SNPs of a trait needed for a locus")
    ap.add_argument("--tag", default="", help="suffix of the output files (analyses other than the main one)")
    ap.add_argument("--null", default="results/gwas/null_threshold.tsv",
                    help="per-trait thresholds from simulated traits (gwas_null_threshold.py); 'none': Bonferroni only")
    ap.add_argument("--read_races", type=int, default=1, help="0: read the directions of the traits of the block only")
    a = ap.parse_args()
    main_run = not (a.gwas or a.mac or a.drop or a.block != 16 or a.zlim != 3.0 or a.min_snps != 1 or a.null == "none"
                    or a.read_races != 1)
    assert main_run == (a.tag == ""), "give --tag for (and only for) an analysis other than the main one"
    readout = "emmax" + a.tag
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    blk = snps[f"block{a.block}"].to_numpy()
    gw = np.load(ROOT / f"results/gwas/emmax_stress{a.gwas}.npz", allow_pickle=True)
    traits = [str(t) for t in gw["traits"]]
    chi = np.nan_to_num(gw["chi2"]).astype(float)
    mac = np.load(ROOT / f"results/gwas/mac_stress{a.mac}.npz", allow_pickle=True)
    assert [str(t) for t in mac["traits"]] == traits and [int(x) for x in mac["n"]] == [int(x) for x in gw["n"]]
    tested = mac["mac"] >= a.min_mac
    chi = np.where(tested, chi, 0.0)
    for t in filter(None, a.drop.split(",")):
        chi[traits.index(t)] = 0.0
    nb = int(blk.max()) + 1
    thr_b = float(chi2.isf(a.alpha / chi.shape[1], 1))
    thr = np.full(len(traits), thr_b)
    if a.null != "none":                                  # the stricter of Bonferroni and the simulated threshold
        nt = pd.read_csv(ROOT / a.null, sep="\t").set_index("trait")
        assert set(nt.index) == set(traits), "simulated thresholds missing for some traits"
        thr = np.maximum(thr, chi2.isf(nt.threshold_p.reindex(traits).to_numpy(float), 1))
    score = np.zeros((len(traits), nb))
    best_snp = np.zeros((len(traits), nb), int)
    nsig = np.zeros((len(traits), nb), int)
    for j in range(len(traits)):
        s = pd.Series(chi[j]).groupby(blk)
        score[j] = s.max().reindex(range(nb)).fillna(0).to_numpy()
        best_snp[j] = s.idxmax().reindex(range(nb)).fillna(0).astype(int).to_numpy()
        nsig[j] = pd.Series(chi[j] >= thr[j]).groupby(blk).sum().reindex(range(nb)).fillna(0).astype(int).to_numpy()
    locus = (score >= thr[:, None]) & (nsig >= a.min_snps)                        # traits x blocks
    tix = {t: j for j, t in enumerate(traits)}
    rows, loci = [], []
    for b in np.flatnonzero(locus.any(0)):
        js = np.flatnonzero(locus[:, b])
        idx = np.flatnonzero(blk == b)
        # one common SNP per block for every trait's direction: the SNP with the highest summed chi-square over the
        # block's traits (different strongest SNPs per trait could tag one haplotype with opposite phase)
        sn = idx[np.argmax(chi[np.ix_(js, idx)].sum(0))]
        z = np.sign(gw["beta"][js, sn]) * np.sqrt(chi[js, sn])
        direction = np.where(z >= a.zlim, "+", np.where(z <= -a.zlim, "-", "0"))
        for j, zj, dj in zip(js, z, direction):
            t = traits[j]
            ts = best_snp[j, b]
            rows.append({"block": int(b), "trait": t, "short": T.SHORT[t], "category": CAT[t],
                         "significant_snps": int(nsig[j, b]), "top_snp": snps["id"].iloc[ts], "top_pos": int(snps.pos.iloc[ts]),
                         "top_chi2": float(score[j, b]), "top_p": float(chi2.sf(score[j, b], 1)),
                         "rank_in_trait": int((score[j] > score[j, b]).sum()) + 1,
                         "snp": snps["id"].iloc[sn], "chrom": snps.chrom.iloc[sn], "pos": int(snps.pos.iloc[sn]),
                         "beta_alt": float(gw["beta"][j, sn]), "z": float(zj), "direction": str(dj)})
        d = pd.DataFrame(rows[-len(js):])
        clear = d[d.direction != "0"][["trait", "direction"]]
        other = []
        if a.read_races and len(js) >= 2:                  # the other races of the pathogens of the locus
            dropped = set(filter(None, a.drop.split(",")))
            for j in range(len(traits)):
                t = traits[j]
                if (js == j).any() or PATHOGEN[t] == t or PATHOGEN[t] not in set(d.trait.map(PATHOGEN)) \
                        or not tested[j, sn] or t in dropped:
                    continue
                zj = float(np.sign(gw["beta"][j, sn]) * np.sqrt(chi[j, sn]))
                other.append((t, zj, "+" if zj >= a.zlim else "-" if zj <= -a.zlim else "0"))
            ox = pd.DataFrame(other, columns=["trait", "z", "direction"])
            clear = pd.concat([clear, ox[ox.direction != "0"][["trait", "direction"]]], ignore_index=True)
        if len(d) == 1:
            cls = "specific"
        elif len(clear) < 2:
            cls = "unresolved"
        elif clear.direction.nunique() == 1:
            cls = "concordant"
        else:
            pg = clear.trait.map(PATHOGEN)
            within = any(g.nunique() > 1 for _, g in clear.direction.groupby(pg))
            if pg.nunique() == 1:
                cls = "race-differential"                      # opposite signs among races of one pathogen
            elif not within:
                cls = "trade-off"                              # opposite signs only between pathogens / stresses
            else:
                cls = "mixed"                                  # opposite signs within and between pathogens
        bs = snps[blk == b]
        loci.append({"block": int(b), "chrom": bs.chrom.iloc[0], "start": int(bs.pos.min()), "end": int(bs.pos.max()),
                     "n_traits": len(d), "traits": ",".join(d.short), "categories": ",".join(sorted(set(d.category))),
                     "n_categories": d.category.nunique(), "n_pathogens": d.trait.map(PATHOGEN).nunique(),
                     "directions": ",".join(f"{s_}:{x}" for s_, x in zip(d.short, d.direction)),
                     "other_races": ",".join(f"{T.SHORT[t_]}:{x_}({z_:.2f})" for t_, z_, x_ in other),
                     "lead_snp": d.snp.iloc[0], "lead_pos": int(d.pos.iloc[0]), "class": cls,
                     "same_top_snp": bool(d.top_snp.nunique() == 1),
                     "top_snp_span_kb": round((d.top_pos.max() - d.top_pos.min()) / 1e3, 1)})
    bt = pd.DataFrame(rows)
    lc = pd.DataFrame(loci).sort_values(["n_traits", "n_categories", "block"], ascending=[False, False, True])
    out = ROOT / "results/loci"
    out.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out / f"block_traits_{readout}.tsv", sep="\t", index=False, float_format="%.6g")
    # sharing expected when the loci of every trait are shifted along the genome (circular shifts)
    ntr = locus.sum(0)
    levels = (1, 2, 3, 4)
    obs = np.array([(ntr >= c).sum() for c in levels])
    groups = sorted(set(PATHOGEN.values()))
    gix = np.array([groups.index(PATHOGEN[t]) for t in traits])

    def by_type(shifted):
        """Blocks shared only among races of one pathogen, and blocks with traits of >= 2 pathogens or stresses."""
        cnt = shifted.sum(0)
        gcnt = np.zeros(nb, int)
        for g in range(len(groups)):
            if (gix == g).any():
                gcnt += shifted[gix == g].any(0)
        return int(((cnt >= 2) & (gcnt == 1)).sum()), int((gcnt >= 2).sum())

    obs_type = by_type(locus)
    bchrom = pd.Series(snps.chrom.to_numpy()).groupby(blk).first().reindex(range(nb)).to_numpy()

    def n_regions(shared):
        """Shared blocks merged into regions when less than three blocks apart on the same chromosome."""
        b_ = np.flatnonzero(shared)
        if len(b_) == 0:
            return 0
        new = np.ones(len(b_), bool)
        new[1:] = (np.diff(b_) > 2) | (bchrom[b_[1:]] != bchrom[b_[:-1]])
        return int(new.sum())

    obs_reg = n_regions(ntr >= 2)
    rng = np.random.default_rng(20261115)
    ex = np.zeros((a.perm, len(levels)), int)
    ex_type = np.zeros((a.perm, 2), int)
    ex_reg = np.zeros(a.perm, int)
    for r in range(a.perm):                              # every trait shifted independently
        sh = np.stack([np.roll(locus[j], int(rng.integers(nb))) for j in range(len(traits))])
        cnt = sh.sum(0)
        ex[r] = [(cnt >= c).sum() for c in levels]
        ex_type[r] = by_type(sh)
        ex_reg[r] = n_regions(cnt >= 2)
    pd.DataFrame({"observed_regions": [obs_reg], "expected_regions": [ex_reg.mean()],
                  "expected_q975": [np.quantile(ex_reg, 0.975)],
                  "p_perm": [((ex_reg >= obs_reg).sum() + 1) / (a.perm + 1)]}).to_csv(
        out / f"sharing_regions{a.tag}.tsv", sep="\t", index=False, float_format="%.6g")
    # exact expectation under independent uniform shifts: at every block, trait t is a locus with probability
    # k_t / nb, independently of the other traits (Poisson-binomial over the traits)
    pk = locus.sum(1) / nb
    dist = np.zeros(len(pk) + 1)
    dist[0] = 1.0
    for q_ in pk:
        dist[1:] = dist[1:] * (1 - q_) + dist[:-1] * q_
        dist[0] *= 1 - q_
    exact = [nb * float(dist[c:].sum()) for c in levels]
    null = pd.DataFrame({"min_traits": levels, "observed_blocks": obs, "expected_blocks": exact,
                         "expected_blocks_simulated": ex.mean(0), "expected_q975": np.quantile(ex, 0.975, axis=0),
                         "p_perm": ((ex >= obs).sum(0) + 1) / (a.perm + 1)})
    null.to_csv(out / f"sharing_null{a.tag}.tsv", sep="\t", index=False, float_format="%.6g")
    rng2 = np.random.default_rng(20261116)
    ex_unit = np.zeros(a.perm, int)
    for r in range(a.perm):                              # the races of a pathogen shifted together
        off = rng2.integers(nb, size=len(groups))
        ex_unit[r] = by_type(np.stack([np.roll(locus[j], int(off[gix[j]])) for j in range(len(traits))]))[1]
    typ = pd.DataFrame({
        "type": ["races of one pathogen", "different pathogens or stresses"], "observed_blocks": obs_type,
        "expected_traits_shifted_independently": ex_type.mean(0),
        "p_traits_shifted_independently": ((ex_type >= np.array(obs_type)).sum(0) + 1) / (a.perm + 1),
        "expected_pathogens_shifted_as_units": [np.nan, ex_unit.mean()],
        "p_pathogens_shifted_as_units": [np.nan, ((ex_unit >= obs_type[1]).sum() + 1) / (a.perm + 1)]})
    typ.to_csv(out / f"sharing_by_type{a.tag}.tsv", sep="\t", index=False, float_format="%.6g")
    cg, uniq = pd.DataFrame(columns=["gene", "flags"]), pd.DataFrame(columns=["gene", "flags"])
    if main_run:
        # candidate genes
        # gene models: the gene features of the annotation (one span per gene)
        gff = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.gene_models_main.gff3.gz", sep="\t", header=None, comment="#",
                          usecols=[0, 2, 3, 4, 8], names=["chrom", "type", "start", "end", "attr"])
        bed = gff[gff.type == "gene"].copy()
        bed["chrom"] = bed.chrom.str.replace("glyma.Wm82.gnm2.", "", regex=False)
        bed["gene"] = bed.attr.str.extract(r"ID=glyma\.Wm82\.gnm2\.ann1\.([^;]+)")[0]
        bed = bed[["chrom", "gene", "start", "end"]].reset_index(drop=True)
        assert bed.gene.notna().all() and bed.gene.is_unique
        info = pd.read_csv(ANN / "glyma.Wm82.gnm2.ann1.RVB6.info_annot.txt.gz", sep="\t", dtype=str)
        info = info.drop_duplicates("locusName").set_index("locusName")

        def flags_of(gene):
            pf = set(str(info.Pfam.get(gene, "")).replace(" ", "").split(",")) if gene in info.index else set()
            return [f for f, ids in PFAM_FLAGS.items() if pf & ids]

        cand = []
        w = a.window_kb * 1000
        for _, L in lc[lc.n_traits >= 2].iterrows():
            g = bed[(bed.chrom == L.chrom) & (bed.end >= L.start - w) & (bed.start <= L.end + w)]
            for _, G in g.iterrows():
                cand.append({"block": L.block, "class": L["class"], "traits": L.traits, "gene": G.gene,
                             "chrom": G.chrom, "start": int(G.start), "end": int(G.end), "flags": ",".join(flags_of(G.gene)),
                             "arabidopsis_hit": info["Best-hit-arabi-name"].get(G.gene, "") if G.gene in info.index else "",
                             "arabidopsis": info["arabi-symbol"].get(G.gene, "") if G.gene in info.index else "",
                             "description": str(info["arabi-defline"].get(G.gene, ""))[:120] if G.gene in info.index else ""})
        cg = pd.DataFrame(cand)
        # genome background of the gene families (genes on the 20 chromosomes)
        chrom_genes = bed[bed.chrom.str.match(r"Gm\d\d$")]
        fam = chrom_genes.gene.map(lambda g_: ",".join(flags_of(g_)))
        bg = {f: int(fam.str.split(",").apply(lambda x: f in x).sum()) for f in PFAM_FLAGS}
        uniq = cg.drop_duplicates("gene") if len(cg) else cg
        fg = {f: int(uniq["flags"].str.split(",").apply(lambda x: f in x).sum()) if len(uniq) else 0 for f in PFAM_FLAGS}
        pd.DataFrame({"family": list(PFAM_FLAGS), "genes_in_shared_blocks": [fg[f] for f in PFAM_FLAGS],
                      "all_genes_in_shared_blocks": len(uniq), "genes_on_chromosomes": [bg[f] for f in PFAM_FLAGS],
                      "all_genes_on_chromosomes": len(chrom_genes)}).to_csv(out / "gene_family_background.tsv", sep="\t", index=False)
    lc.to_csv(out / f"loci_{readout}.tsv", sep="\t", index=False)
    if main_run:
        cg.to_csv(out / f"candidate_genes_{readout}.tsv", sep="\t", index=False)
    per_trait = pd.Series(locus.sum(1), index=[T.SHORT[t] for t in traits])
    s = [f"association statistic, P < {a.alpha:g} / {chi.shape[1]} (chi2 >= {thr_b:.2f}) and the simulated threshold "
         f"({a.null}; chi2 {thr.min():.2f}-{thr.max():.2f}), SNPs with >= {a.min_mac} minor "
         f"homozygotes among the trait's accessions ({int(tested.sum(1).min())}-{int(tested.sum(1).max())} SNPs per trait); {nb} blocks",
         f"trait-locus pairs: {len(bt)}; blocks that are a locus of >= 1 trait: {len(lc)}; >= 2 traits: "
         f"{int((lc.n_traits >= 2).sum())}; >= 2 stress categories: {int((lc.n_categories >= 2).sum())}",
         f"loci per trait: {per_trait.to_dict()}",
         f"classes among shared blocks: {lc[lc.n_traits >= 2]['class'].value_counts().to_dict()}",
         "sharing (observed / expected under circular shifts): " + "; ".join(
             f">= {r_.min_traits} traits {r_.observed_blocks} / {r_.expected_blocks:.1f} (P = {r_.p_perm:.2g})" for r_ in null.itertuples()),
         "shared blocks by type (observed / expected with traits shifted independently / with pathogens shifted as units): "
         + "; ".join(f"{r_.type} {r_.observed_blocks} / {r_.expected_traits_shifted_independently:.1f} / "
                     f"{r_.expected_pathogens_shifted_as_units:.1f} (P = {r_.p_pathogens_shifted_as_units:.2g})" for r_ in typ.itertuples()),
         f"genes in shared blocks: {cg.gene.nunique() if len(cg) else 0}; with resistance-family domains: "
         f"{int((uniq['flags'] != '').sum()) if len(uniq) else 0}"]
    (out / f"summary_{readout}.txt").write_text("\n".join(s) + "\n")
    print("\n".join(s))
    pd.set_option("display.width", 250)
    print(lc[lc.n_traits >= 2].head(60).to_string(index=False))


if __name__ == "__main__":
    main()
