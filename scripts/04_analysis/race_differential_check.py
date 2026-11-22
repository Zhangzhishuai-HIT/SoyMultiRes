"""Race-differential loci of soybean cyst nematode (default: block 2059 next to Rhg1, Gm18 1.19-1.39 Mb): SCN race
reactions by allele at the block's lead SNP, the genotypes of the SCN race differentials, and the linkage
disequilibrium of the lead SNP with the Rhg1 block and with the other race-differential blocks.

usage: python -I race_differential_check.py [--block 2059]
       python -I race_differential_check.py --genomewide
Accessions scored resistant to race 3 are split by the allele at the lead SNP (alternative homozygote vs reference
homozygote); for races 1, 5 and 14 the share of resistant records is compared (Fisher's exact test). Race 3 resistance
comes from Peking-type (rhg1-a + Rhg4; effective against races 1, 3 and 5) or PI 88788-type sources (rhg1-b; races 3
and 14), so an allele that tags one source type separates the two race spectra.
The comparison is repeated with one accession per near-identical group (the first resistant accession of a group).
Writes results/loci/race_differential_<block>.tsv, race_differential_<block>_ld.tsv (also: SNPs within 500 kb and
300 kb of the lead SNP, the largest r2 among the former and the number of the latter with D' >= 0.8) and
race_differential_<block>_ldpartners.tsv (the ten SNPs of the genome with the highest r2 with the lead SNP), and prints
the genotypes of the differentials.
--genomewide: the same comparison among the race-3-resistant accessions for every SNP with at least ten carriers and
ten non-carriers scored for race 1 and for race 14. The statistic of a SNP is the sum of the chi-square statistics of
the two 2 x 2 tables (allele x resistant) when the allele is associated with resistance to race 1 and susceptibility
to race 14 or the reverse, and zero otherwise. Writes results/loci/race_differential_genomewide.tsv (the 300 SNPs
with the highest statistic and the lead SNPs of the race-differential loci, with their ranks),
race_differential_genomewide_n.tsv (numbers of accessions and of SNPs compared) and
race_differential_lines.tsv (alleles of the SCN resistance sources at the SNPs of the race-differential blocks, with
the z values of the SCN races). The calibration of the race-14 statistics at the lead SNPs against traits simulated
without SNP effects is in results/gwas/null_threshold_watch.tsv (gwas_null_threshold.py --watch).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

DIFFERENTIALS = {"Peking": "PI548402", "Pickett": "PI548988", "PI 90763": "PI90763", "PI 437654": "PI437654",
                 "PI 88788": "PI88788", "PI 209332": "PI209332", "PI 89772": "PI89772", "Cloud": "PI548316",
                 "Williams 82": "PI518671"}
# sources of SCN resistance: Peking type (Peking, Forrest, PI 90763, PI 437654, PI 89772) and PI 88788 type (PI 88788,
# PI 209332, Cloud) after Liu et al. 2017; Pickett (Peking in its pedigree); PI 567516C (source of qSCN10); Williams 82
LINES = {"Peking": "PI548402", "Forrest": "PI548655", "PI 90763": "PI90763", "PI 437654": "PI437654",
         "PI 89772": "PI89772", "Pickett": "PI548988", "PI 88788": "PI88788", "PI 209332": "PI209332",
         "Cloud": "PI548316", "PI 567516C": "PI567516C", "Williams 82": "PI518671"}


def genomewide():
    """Separation of race 1 and race 14 resistance among the race-3-resistant accessions, for every SNP."""
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    ids = data.samples()
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    long = data.pheno_long()
    r = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.max().unstack()
    r = (r == 1).where(r.notna())
    sub = r[r.NEMATCYST_3 == True]  # noqa: E712
    sub = sub[sub.index.isin(set(ids.values))]
    pos = {v: i for i, v in enumerate(ids.values)}
    G = np.asarray(D[np.sort([pos[x] for x in sub.index])])
    sub = sub.loc[ids.values[np.sort([pos[x] for x in sub.index])]]
    car, non = (G == 2), (G == 0)
    stat = {}
    for race in (1, 14):
        y = sub[f"NEMATCYST_{race}"]
        ok = y.notna().to_numpy()
        res = (y[ok] == True).to_numpy()[:, None]  # noqa: E712
        a_ = (car[ok] & res).sum(0).astype(float)
        b_ = (car[ok] & ~res).sum(0).astype(float)
        c_ = (non[ok] & res).sum(0).astype(float)
        d_ = (non[ok] & ~res).sum(0).astype(float)
        n = a_ + b_ + c_ + d_
        with np.errstate(invalid="ignore", divide="ignore"):
            chi = n * (a_ * d_ - b_ * c_) ** 2 / ((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_))
            diff = a_ / (a_ + b_) - c_ / (c_ + d_)
        stat[race] = (np.nan_to_num(chi), np.nan_to_num(diff), a_ + b_, c_ + d_)
    ok = (stat[1][2] >= 10) & (stat[1][3] >= 10) & (stat[14][2] >= 10) & (stat[14][3] >= 10)
    opposite = np.sign(stat[1][1]) * np.sign(stat[14][1]) < 0
    S = np.where(ok & opposite, stat[1][0] + stat[14][0], 0.0)
    out = snps[["chrom", "pos", "id", "block16"]].assign(
        statistic=S, tested=ok, carriers_race1=stat[1][2].astype(int), noncarriers_race1=stat[1][3].astype(int),
        difference_race1=stat[1][1], carriers_race14=stat[14][2].astype(int), noncarriers_race14=stat[14][3].astype(int),
        difference_race14=stat[14][1])
    out = out[out.tested].copy()
    out["rank"] = out.statistic.rank(ascending=False, method="min").astype(int)
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    lead = loci[loci["class"] == "race-differential"]
    out["lead_snp_of_race_differential_locus"] = out.id.isin(set(lead.lead_snp))
    keep = out[(out["rank"] <= 300) | out.lead_snp_of_race_differential_locus].sort_values("rank")
    keep.drop(columns="tested").to_csv(ROOT / "results/loci/race_differential_genomewide.tsv", sep="\t", index=False,
                                       float_format="%.6g")
    pd.DataFrame({"race3_resistant_accessions": [len(sub)], "snps_compared": [len(out)]}).to_csv(
        ROOT / "results/loci/race_differential_genomewide_n.tsv", sep="\t", index=False)
    print(f"{len(sub)} race-3-resistant accessions; {len(out)} SNPs with at least ten carriers and ten non-carriers "
          f"scored for races 1 and 14")
    for _, x in out[out.lead_snp_of_race_differential_locus].iterrows():
        better = out[out.statistic > x.statistic]
        print(f"  {x.id} {x.chrom}:{x.pos:,}: statistic {x.statistic:.1f}, rank {x['rank']} of {len(out)}; SNPs with a higher "
              f"statistic: {len(better)} in {better.block16.nunique()} blocks on {better.chrom.nunique()} chromosomes "
              f"({better.chrom.value_counts().to_dict()})")
    print(out.sort_values("rank").head(25).drop(columns="tested").to_string(index=False))
    # alleles of the resistance sources at the SNPs of the race-differential blocks, with z of the SCN races
    gw = np.load(ROOT / "results/gwas/emmax_stress.npz", allow_pickle=True)
    mac = np.load(ROOT / "results/gwas/mac_stress.npz", allow_pickle=True)
    names = [str(t) for t in gw["traits"]]
    rows = []
    for _, L in lead.sort_values(["chrom", "start"]).iterrows():
        for j in np.flatnonzero(snps.block16.to_numpy() == L.block):
            col = np.asarray(D[:, j])
            row = {"block": int(L.block), "chrom": snps.chrom[j], "pos": int(snps.pos[j]), "snp": snps.id[j],
                   "lead_snp": snps.id[j] == L.lead_snp, "alt_homozygotes": int((col == 2).sum())}
            for race in (1, 3, 4, 5, 14):
                k = names.index(f"NEMATCYST_{race}")
                z = float(np.sign(gw["beta"][k, j]) * np.sqrt(gw["chi2"][k, j]))
                row[f"z_race{race}"] = z if mac["mac"][k, j] >= 30 else np.nan
            row.update({name: int(col[pos[a]]) for name, a in LINES.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "results/loci/race_differential_lines.tsv", sep="\t", index=False, float_format="%.10g")


def separation(sub, car, non):
    """Statistic of genomewide() for every SNP: chi-square of race 1 plus chi-square of race 14 resistance of carriers
    against non-carriers, for SNPs with at least ten carriers and ten non-carriers scored for each race and opposite
    differences for the two races (otherwise 0); also the resistant shares and counts."""
    stat = {}
    for race in (1, 14):
        y = sub[f"NEMATCYST_{race}"]
        ok = y.notna().to_numpy()
        res = (y[ok] == True).to_numpy()[:, None]  # noqa: E712
        a_ = (car[ok] & res).sum(0).astype(float)
        b_ = (car[ok] & ~res).sum(0).astype(float)
        c_ = (non[ok] & res).sum(0).astype(float)
        d_ = (non[ok] & ~res).sum(0).astype(float)
        n = a_ + b_ + c_ + d_
        with np.errstate(invalid="ignore", divide="ignore"):
            chi = n * (a_ * d_ - b_ * c_) ** 2 / ((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_))
            diff = a_ / (a_ + b_) - c_ / (c_ + d_)
        stat[race] = (np.nan_to_num(chi), np.nan_to_num(diff), a_ + b_, c_ + d_, a_, c_)
    ok = (stat[1][2] >= 10) & (stat[1][3] >= 10) & (stat[14][2] >= 10) & (stat[14][3] >= 10)
    opposite = np.sign(stat[1][1]) * np.sign(stat[14][1]) < 0
    return np.where(ok & opposite, stat[1][0] + stat[14][0], 0.0), stat


def crossval(k=5, seed=20261116):
    """Out-of-sample check of the genome-wide comparison: the race-3-resistant accessions are divided into k folds by
    groups of near-identical accessions (splits/folds.tsv, dup_group); for each fold, the SNP with the highest statistic
    on the other folds is chosen, and its carriers and non-carriers among the accessions of the fold are compared for
    resistance to races 1 and 14. Writes results/loci/race_differential_crossval.tsv (one row per fold and a pooled
    row with Fisher's exact test of the pooled held-out counts)."""
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    ids = data.samples()
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    long = data.pheno_long()
    r = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.max().unstack()
    r = (r == 1).where(r.notna())
    sub = r[r.NEMATCYST_3 == True]  # noqa: E712
    sub = sub[sub.index.isin(set(ids.values))]
    pos = {v: i for i, v in enumerate(ids.values)}
    rows_ = np.sort([pos[x] for x in sub.index])
    G = np.asarray(D[rows_])
    sub = sub.loc[ids.values[rows_]]
    grp = data.folds().dup_group.reindex(sub.index)
    assert grp.notna().all()
    ug = np.array(sorted(grp.unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k))
    fold = grp.map(fold_of).to_numpy()
    out, pooled = [], {1: np.zeros(4), 14: np.zeros(4)}
    for f in range(k):
        tr, te = fold != f, fold == f
        S, _ = separation(sub[tr], G[tr] == 2, G[tr] == 0)
        j = int(np.argmax(S))
        row = {"fold": f, "training_accessions": int(tr.sum()), "test_accessions": int(te.sum()), "snp": snps.id[j],
               "chrom": snps.chrom[j], "pos": int(snps.pos[j]), "block": int(snps.block16[j]), "training_statistic": float(S[j])}
        for race in (1, 14):
            y = sub[te][f"NEMATCYST_{race}"]
            g = G[te][:, j]
            m = y.notna().to_numpy()
            yc, yn = y[m & (g == 2)].astype(bool), y[m & (g == 0)].astype(bool)
            cnt = np.array([yc.sum(), len(yc), yn.sum(), len(yn)], dtype=float)
            pooled[race] += cnt
            row.update({f"carriers_race{race}": int(cnt[1]), f"carriers_resistant_race{race}": int(cnt[0]),
                        f"noncarriers_race{race}": int(cnt[3]), f"noncarriers_resistant_race{race}": int(cnt[2])})
        out.append(row)
    prow = {"fold": "pooled", "test_accessions": int(len(sub))}
    for race in (1, 14):
        a_, n1, c_, n0 = pooled[race]
        prow.update({f"carriers_race{race}": int(n1), f"carriers_resistant_race{race}": int(a_),
                     f"noncarriers_race{race}": int(n0), f"noncarriers_resistant_race{race}": int(c_),
                     f"fisher_p_race{race}": float(fisher_exact([[a_, n1 - a_], [c_, n0 - c_]])[1])})
    o = pd.DataFrame(out + [prow])
    o.to_csv(ROOT / "results/loci/race_differential_crossval.tsv", sep="\t", index=False, float_format="%.6g")
    print(o.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2059)
    ap.add_argument("--genomewide", action="store_true")
    ap.add_argument("--crossval", action="store_true", help="out-of-sample check of the genome-wide comparison")
    a = ap.parse_args()
    if a.crossval:
        return crossval()
    if a.genomewide:
        return genomewide()
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t").set_index("block")
    L = loci.loc[a.block]
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    j = int(np.flatnonzero(snps["id"].to_numpy() == L.lead_snp)[0])
    ids = data.samples()
    D = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    g = pd.Series(np.asarray(D[:, j]), index=ids.values, name="allele")
    print(f"block {a.block} {L.chrom}:{L.start:,}-{L.end:,}, lead SNP {L.lead_snp} at {L.lead_pos:,}, "
          f"alternative allele frequency {snps.alt_freq[j]:.3f}; directions {L.directions}")
    print("differentials (alternative-allele dosage):", {k: int(g[v]) for k, v in DIFFERENTIALS.items()})
    long = data.pheno_long()
    # resistant = scored resistant in at least one experiment (the definition used throughout)
    r = long[long.trait.str.startswith("NEMATCYST")].groupby(["acc", "trait"]).resistant.max().unstack()
    r = (r == 1).where(r.notna()).join(g, how="inner")
    rows = []
    for subset, sub in (("all scored", r), ("race-3 resistant", r[r.NEMATCYST_3 == True])):  # noqa: E712
        for t in ("NEMATCYST_1", "NEMATCYST_3", "NEMATCYST_4", "NEMATCYST_5", "NEMATCYST_14"):
            s = sub[sub[t].notna()]
            c, n = s[s.allele == 2][t].astype(bool), s[s.allele == 0][t].astype(bool)
            if subset == "race-3 resistant" and t in ("NEMATCYST_3", "NEMATCYST_4"):
                continue
            p = fisher_exact([[c.sum(), (~c).sum()], [n.sum(), (~n).sum()]])[1]
            rows.append({"accessions": subset, "race": t.split("_")[1], "carriers": len(c),
                         "carriers_resistant": c.mean(), "noncarriers": len(n), "noncarriers_resistant": n.mean(),
                         "fisher_p": p})
    # the same comparison with one accession per near-identical group (the first race-3-resistant accession of a group)
    fo = data.folds()
    sub3 = r[r.NEMATCYST_3 == True]  # noqa: E712
    grp = fo.dup_group.reindex(sub3.index)
    sub = sub3[~grp.duplicated() | grp.isna()]
    for t in ("NEMATCYST_1", "NEMATCYST_5", "NEMATCYST_14"):
        s_ = sub[sub[t].notna()]
        c, n = s_[s_.allele == 2][t].astype(bool), s_[s_.allele == 0][t].astype(bool)
        p = fisher_exact([[c.sum(), (~c).sum()], [n.sum(), (~n).sum()]])[1]
        rows.append({"accessions": "race-3 resistant, one per near-identical group", "race": t.split("_")[1],
                     "carriers": len(c), "carriers_resistant": c.mean(), "noncarriers": len(n),
                     "noncarriers_resistant": n.mean(), "fisher_p": p})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/loci/race_differential_{a.block}.tsv", sep="\t", index=False, float_format="%.6g")
    print(out.to_string(index=False))
    # linkage disequilibrium of the lead SNP with the SNPs of the Rhg1 block (same chromosome only) and with the lead
    # SNPs of the other race-differential blocks
    kg = pd.read_csv(data.P / "known_genes.tsv", sep="\t").set_index("locus")
    x = (np.asarray(D[:, j]) == 2).astype(float)

    def ld(k):
        y = (np.asarray(D[:, k]) == 2).astype(float)
        pa, pb, pab = x.mean(), y.mean(), (x * y).mean()
        d = pab - pa * pb
        dmax = min(pa * (1 - pb), (1 - pa) * pb) if d > 0 else min(pa * pb, (1 - pa) * (1 - pb))
        return d * d / (pa * (1 - pa) * pb * (1 - pb)), abs(d) / dmax if dmax > 0 else np.nan

    res = {"block": a.block, "lead_snp": L.lead_snp, "carriers": int(x.sum()), "carrier_share": float(x.mean())}
    if L.chrom == kg.chrom["Rhg1"]:
        b_rhg1 = snps.block16[(snps.chrom == "Gm18") & (snps.pos <= kg.end["Rhg1"])].iloc[-1]
        ks = np.flatnonzero(snps.block16.to_numpy() == b_rhg1)
        v = np.array([ld(k) for k in ks])
        res.update({"rhg1_block": int(b_rhg1), "max_r2_rhg1_block": float(np.nanmax(v[:, 0])),
                    "max_dprime_rhg1_block": float(np.nanmax(v[:, 1]))})
    others = loci[(loci["class"] == "race-differential") & (loci.index != a.block)]
    for b_, o in others.iterrows():
        k = int(np.flatnonzero(snps["id"].to_numpy() == o.lead_snp)[0])
        y = np.asarray(D[:, k]) == 2
        r2, dp = ld(k)
        res[f"block_{b_}_carriers"] = int(y.sum())
        res[f"block_{b_}_both"] = int(((x == 1) & y).sum())
        res[f"block_{b_}_r2"] = float(r2)
        res[f"block_{b_}_dprime"] = float(dp)
    # linkage disequilibrium of the lead SNP with every SNP of the genome (indicators of homozygosity for the
    # alternative allele): the strongest partners, and the SNPs near the lead SNP
    nacc = len(x)
    pa = x.mean()
    r2_all, dp_all = np.zeros(D.shape[1]), np.zeros(D.shape[1])
    for c0 in range(0, D.shape[1], 4096):
        Y = (np.asarray(D[:, c0:c0 + 4096]) == 2).astype(float)
        pb = Y.mean(0)
        d = (x @ Y) / nacc - pa * pb
        dmax = np.where(d > 0, np.minimum(pa * (1 - pb), (1 - pa) * pb), np.minimum(pa * pb, (1 - pa) * (1 - pb)))
        with np.errstate(invalid="ignore", divide="ignore"):
            r2_all[c0:c0 + 4096] = d * d / (pa * (1 - pa) * pb * (1 - pb))
            dp_all[c0:c0 + 4096] = np.abs(d) / dmax
    r2_all[j], dp_all[j] = np.nan, np.nan
    near5 = (snps.chrom == L.chrom).to_numpy() & (np.abs(snps.pos.to_numpy() - L.lead_pos) <= 500_000)
    near3 = (snps.chrom == L.chrom).to_numpy() & (np.abs(snps.pos.to_numpy() - L.lead_pos) <= 300_000)
    near5[j] = near3[j] = False
    res.update({"snps_within_500kb": int(near5.sum()), "max_r2_within_500kb": float(np.nanmax(r2_all[near5])),
                "snps_within_300kb": int(near3.sum()), "snps_within_300kb_dprime_0.8": int((dp_all[near3] >= 0.8).sum())})
    top = np.argsort(-np.nan_to_num(r2_all))[:10]
    pd.DataFrame({"snp": snps["id"].to_numpy()[top], "chrom": snps.chrom.to_numpy()[top], "pos": snps.pos.to_numpy()[top],
                  "r2": r2_all[top], "dprime": dp_all[top]}).to_csv(
        ROOT / f"results/loci/race_differential_{a.block}_ldpartners.tsv", sep="\t", index=False, float_format="%.6g")
    pd.Series(res).to_csv(ROOT / f"results/loci/race_differential_{a.block}_ld.tsv", sep="\t", header=False)
    print(res)


if __name__ == "__main__":
    main()
