"""Step 10: multi-resistant germplasm. Out-of-fold validation of resistance predictions and the ranking of all
genotyped accessions by the final all-data TraitBoost.

usage: python -I germplasm.py [--top 0.10]
1. Validation with CV1 out-of-fold predictions (runs/cv1/<model>/<trait>.tsv.gz; every accession predicted by the
   fold model that never saw it), for TraitBoost (mtboost_v2_final) and the best single-trait model (best_single_avg4):
   per trait, predictions standardised within fold, accessions ranked; precision = share of observed resistant
   accessions among the top 5% / 10% (resistant = resistant in at least one experiment: GRIN resistant classes, or the
   10% least-damaged accessions of an experiment for numeric traits), base rate and enrichment.
2. Multi-resistance consistency: accessions with out-of-fold predictions in the top 10% for >= k traits of
   different stress categories (among the traits they were scored for) -> share of those records observed resistant.
3. Ranking: final predictions (runs/all/mtboost_final/predictions.npz, "final") of the 20,087 accessions,
   standardised per trait within taxon; category score = mean z of the category's traits; multi-resistance index = number of the 8
   categories with a category score in the top 10%; accessions table with taxon, origin, maturity group.
4. The index out of fold: the predictions of all 33 traits for the test accessions of every fold
   (runs/cv1/mtboost_dev/v2/fold<f>_full.npz, train_mtboost.py reseed --dump_full) are standardised within fold and
   within taxon and averaged within stress categories, and the index is computed as in 3 among the 17,150 scored
   accessions. For accessions with an index of at least k, the scores of the traits of the categories that count
   towards the index are compared with the proportions of resistant accessions of these traits (index_out_of_fold.tsv;
   also for the categorical traits alone, without the numeric traits whose resistant class is the least-damaged 10%;
   and with the expected proportions computed within strata of maturity group class and taxon; and for the single
   records (accession, trait, experiment) of these traits, against the proportion of resistant records of the same
   trait in the same experiment). The same index is computed from the out-of-fold GBLUP predictions of all traits
   (GBLUP fitted to all training accessions of a trait; work/mtboost_start/cv1_fold<f>_evalall.npz, written by
   train_mtboost.py start --scenario cv1 --fold <f> --eval_all) as a baseline (index_out_of_fold_gblup.tsv, index_out_of_fold_multi_gblup.tsv).
   Multi-resistance of single accessions (index_out_of_fold_multi.tsv): among the accessions with an index of at least
   k (k = 2, 3, 4) that were scored in at least two of their counted categories, the proportion resistant in at least
   two of them (resistant in a category: resistant for at least one scored trait of the category); expected (a) if the
   categories were independent within these accessions (Poisson-binomial with the proportion resistant of every
   category among them) and (b) among accessions with an index of 0 scored in at least the same categories.
5. Share of accessions with an index of four or more (ranking of 3) by maturity group and by origin (index_by_group.tsv).
Writes results/germplasm/{precision.tsv, consistency.tsv, top10_by_trait.tsv, ranking.tsv, top_accessions.tsv,
index_out_of_fold.tsv, index_out_of_fold_multi.tsv, index_out_of_fold_gblup.tsv, index_out_of_fold_multi_gblup.tsv,
index_out_of_fold_refits.tsv, index_by_group.tsv, index_out_of_fold_by_accession.tsv,
index_out_of_fold_bootstrap.tsv}; index_out_of_fold_refits.tsv gives, by fold
and trait, the correlation between the repeated refits of part 4 and the predictions used for the accuracy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

CAT = {row[0]: row[4] for row in T.STRESS}


def per_acc(rec, trait):
    """Per accession: mean standardized prediction and the resistant status used throughout the article (resistant in
    at least one experiment; for numeric traits a record is resistant when it is among the 10% least-damaged
    accessions of its experiment)."""
    r = evaluate.within_fold_z(rec)
    g = r.groupby("acc").agg(pred=("pred", "mean"), score=("score", "mean"), resistant=("resistant", "max"))
    g["resistant"] = g.resistant == 1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=float, default=0.10)
    a = ap.parse_args()
    out = ROOT / "results/germplasm"
    out.mkdir(parents=True, exist_ok=True)
    prec, oof = [], {}
    for model in ("mtboost_v2_final", "best_single_avg4"):
        for t in T.TRAITS:
            g = per_acc(pd.read_csv(data.RUNS / "cv1" / model / f"{t}.tsv.gz", sep="\t", dtype={"acc": str}), t)
            if model == "mtboost_v2_final":
                oof[t] = g
            base = g.resistant.mean()
            row = {"model": model, "trait": t, "short": T.SHORT[t], "category": CAT[t], "n": len(g), "base_rate": base}
            for q in (0.05, 0.10):
                k = max(1, int(round(q * len(g))))
                p = g.sort_values("pred", ascending=False).head(k).resistant.mean()
                row[f"precision_top{int(q * 100)}"] = p
                row[f"enrichment_top{int(q * 100)}"] = p / base if base > 0 else np.nan
            prec.append(row)
    P = pd.DataFrame(prec)
    P.to_csv(out / "precision.tsv", sep="\t", index=False, float_format="%.10g")
    w = P.pivot_table(index="short", columns="model", values="enrichment_top10")
    print("enrichment of observed resistance among the top 10% predicted (out of fold), median over traits:",
          w.median().round(2).to_dict())
    print("TraitBoost enrichment higher than the best single-trait model for", int((w.mtboost_v2_final > w.best_single_avg4).sum()),
          "of", len(w), "traits")
    # 2. consistency across stress categories (out of fold)
    flags = []
    for t, g in oof.items():
        thr = g.pred.quantile(1 - a.top)
        flags.append(pd.DataFrame({"acc": g.index, "trait": t, "category": CAT[t], "top": g.pred >= thr,
                                   "resistant": g.resistant, "base": g.resistant.mean()}))
    F = pd.concat(flags, ignore_index=True)
    bt = F.groupby("trait", sort=False).agg(test_accessions=("acc", "size"), resistant_share=("resistant", "mean"))
    ft = F[F.top].groupby("trait", sort=False).agg(top10_predictions=("acc", "size"), confirmed_share=("resistant", "mean"))
    bt = bt.join(ft).reset_index()
    bt.insert(1, "short", bt.trait.map(T.SHORT))
    bt.insert(2, "category", bt.trait.map(CAT))
    bt.to_csv(out / "top10_by_trait.tsv", sep="\t", index=False, float_format="%.10g")
    ncat = F[F.top].groupby("acc").category.nunique()
    rows = []
    for k in (1, 2, 3, 4):
        accs = set(ncat[ncat >= k].index)
        sub = F[F.acc.isin(accs) & F.top]
        # expected share: mean resistant share of the traits of the flagged accession-trait pairs (what a ranking
        # without information would give for the same traits)
        rows.append({"min_categories_top": k, "accessions": len(accs), "records_predicted_top": len(sub),
                     "observed_resistant_share": sub.resistant.mean() if len(sub) else np.nan,
                     "expected_share_same_traits": sub.base.mean() if len(sub) else np.nan,
                     "base_rate_all_records": F.resistant.mean()})
    C = pd.DataFrame(rows)
    C.to_csv(out / "consistency.tsv", sep="\t", index=False, float_format="%.6f")
    print(C.to_string(index=False))
    # 3. ranking of all genotyped accessions by the final all-data model
    if not (data.RUNS / "all/mtboost_final/predictions.npz").exists():
        print("final all-data predictions not there yet: ranking skipped")
        return
    z = np.load(data.RUNS / "all/mtboost_final/predictions.npz", allow_pickle=True)
    pred = pd.DataFrame(z["final"], index=z["acc"], columns=[str(t) for t in z["traits"]])
    cats = sorted(set(CAT.values()))
    meta = pd.read_csv(data.P / "accessions.tsv", sep="\t").set_index("vcf_id")
    meta = meta[["taxon", "origin", "region", "mg", "n_stress_traits"]].reindex(pred.index)
    # wild and cultivated soybean are ranked separately: predictions standardised within taxon (accessions without
    # taxon information form a third group), averaged over the traits of a stress category; index = number of
    # categories in which the accession is in the top 10% of its taxon
    tax = meta.taxon.fillna("unknown")
    CS = pd.DataFrame(index=pred.index, columns=cats, dtype=float)
    within = pd.Series(0, index=pred.index)
    for t_, idx in tax.groupby(tax).groups.items():
        Zt = (pred.loc[idx] - pred.loc[idx].mean()) / pred.loc[idx].std()
        CSt = pd.DataFrame({c: Zt[[t for t in Zt.columns if CAT[t] == c]].mean(axis=1) for c in cats})
        CS.loc[idx] = CSt
        within.loc[idx] = (CSt >= CSt.quantile(1 - a.top)).sum(axis=1)
    R = pd.DataFrame({"multi_resistance_index_within_taxon": within, "mean_category_z": CS.mean(axis=1)}).join(CS.add_prefix("z_"))
    R = R.join(meta, how="left")
    R = R.sort_values(["multi_resistance_index_within_taxon", "mean_category_z"], ascending=False)
    R.to_csv(out / "ranking.tsv", sep="\t", float_format="%.3f")
    R.head(100).to_csv(out / "top_accessions.tsv", sep="\t", float_format="%.3f")
    for t_ in ("Glycine max", "Glycine soja"):
        v = R[R.taxon == t_].multi_resistance_index_within_taxon
        print(f"{t_}: within-taxon index distribution {v.value_counts().sort_index().to_dict()}; "
              f">= 4: {int((v >= 4).sum())} (never screened {int(((v >= 4) & (R[R.taxon == t_].n_stress_traits.fillna(0) == 0)).sum())})")
    print(R.head(15)[["multi_resistance_index_within_taxon", "mean_category_z", "taxon", "origin", "mg", "n_stress_traits"]].to_string())
    # 4. the index from out-of-fold predictions of all traits
    files = [data.RUNS / "cv1/mtboost_dev/v2" / f"fold{f}_full.npz" for f in range(5)]
    if all(f.exists() for f in files):
        parts = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            Pf = pd.DataFrame(z["final"], index=[str(x) for x in z["acc"]], columns=[str(t) for t in z["traits"]])
            parts.append((Pf - Pf.mean()) / Pf.std())                      # within fold, over the test accessions of the fold
        O = pd.concat(parts)
        assert O.index.is_unique and len(O) == len(data.folds())
        # the repeated refits against the predictions used for the accuracy (scored traits), by fold and trait
        raw = [pd.DataFrame(np.load(f, allow_pickle=True)["final"], index=[str(x) for x in np.load(f, allow_pickle=True)["acc"]],
                            columns=[str(t) for t in np.load(f, allow_pickle=True)["traits"]]) for f in files]
        chk = []
        for t in T.TRAITS:
            rr = pd.read_csv(data.RUNS / "cv1/mtboost_v2_final" / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
            for f_, d_ in rr.groupby("fold"):
                d_ = d_.drop_duplicates("acc")
                chk.append({"fold": int(f_), "trait": t, "accessions": len(d_),
                            "r": float(np.corrcoef(d_.pred, raw[int(f_)].loc[d_.acc, t].to_numpy())[0, 1])})
        chk = pd.DataFrame(chk)
        chk.to_csv(out / "index_out_of_fold_refits.tsv", sep="\t", index=False, float_format="%.6f")
        print(f"repeated refits against the stored predictions: r {chk.r.min():.3f}-{chk.r.max():.3f}, median {chk.r.median():.4f} "
              f"({len(chk)} combinations of fold and trait)")
        res = pd.read_csv(data.P / "resistant_matrix.tsv.gz", sep="\t", index_col=0).reindex(O.index)[list(T.TRAITS)]
        base = res.mean()                                                   # proportion of resistant accessions per trait
        # the same proportions within strata of maturity group class and taxon (as for the associations between traits)
        mgc = meta.mg.reindex(O.index).map(T.mg_group)               # maturity-group classes of traits.mg_group (000 = -2)
        stratum = mgc + "|" + meta.taxon.reindex(O.index).fillna("unknown")
        base_s = res.groupby(stratum).mean().reindex(stratum.to_numpy())
        base_s.index = res.index
        rc = pd.DataFrame({c: res[[t for t in T.TRAITS if CAT[t] == c]].max(axis=1, skipna=True) for c in cats})   # NaN = not scored
        # single records (accession, trait, experiment) and the proportion of resistant records of the same trait in the
        # same experiment
        long_r = data.pheno_long()
        long_r = long_r[long_r.acc.isin(set(O.index))][["acc", "trait", "batch", "resistant"]].copy()
        long_r["p_exp"] = long_r.groupby(["trait", "batch"]).resistant.transform("mean")

        def validate(Om):
            """Index from the out-of-fold predictions Om (accessions x traits, standardised within fold), resistant
            scores of the accessions with an index of at least k, and resistance in several counted categories."""
            tax_o = meta.taxon.reindex(Om.index).fillna("unknown")
            top = pd.DataFrame(False, index=Om.index, columns=cats)
            for t_, idx in tax_o.groupby(tax_o).groups.items():
                Zt = (Om.loc[idx] - Om.loc[idx].mean()) / Om.loc[idx].std()
                CSt = pd.DataFrame({c: Zt[[t for t in Zt.columns if CAT[t] == c]].mean(axis=1) for c in cats})
                top.loc[idx] = CSt >= CSt.quantile(1 - a.top)
            index_oof = top.sum(axis=1)
            counted = pd.DataFrame({t: top[CAT[t]] for t in T.TRAITS})         # trait of a category that counts for the index
            rows = []
            for k in (1, 2, 3, 4):
                sel = index_oof >= k
                m = res[sel].notna() & counted[sel]
                obs = float(res[sel].where(m).sum().sum() / m.sum().sum())
                exp = float((m * base).sum().sum() / m.sum().sum())
                allm = res[sel].notna()
                mc = m[[t for t in T.TRAITS if t not in T.NUMERIC]]              # categorical traits only (no 10% rule)
                cs = counted[sel].stack()
                lr = long_r.merge(pd.DataFrame(list(cs[cs].index), columns=["acc", "trait"]), on=["acc", "trait"])
                rows.append({"min_index": k, "accessions": int(sel.sum()), "accessions_with_scores_in_counted_categories": int(m.any(axis=1).sum()),
                             "scored_traits_in_counted_categories": int(m.sum().sum()), "observed_resistant_share": obs,
                             "expected_resistant_share": exp,
                             "expected_resistant_share_within_strata": float((m * base_s).sum().sum() / m.sum().sum()),
                             "scored_categorical_traits_in_counted_categories": int(mc.sum().sum()),
                             "observed_resistant_share_categorical": float(res[sel][mc.columns].where(mc).sum().sum() / mc.sum().sum()),
                             "expected_resistant_share_categorical": float((mc * base[mc.columns]).sum().sum() / mc.sum().sum()),
                             "observed_resistant_share_all_scored_traits": float(res[sel].sum().sum() / allm.sum().sum()),
                             "expected_resistant_share_all_scored_traits": float((allm * base).sum().sum() / allm.sum().sum()),
                             "records_in_counted_categories": len(lr),
                             "observed_resistant_share_records": float(lr.resistant.mean()),
                             "expected_resistant_share_records_same_experiment": float(lr.p_exp.mean())})
            V = pd.DataFrame(rows)
            # resistance in several categories of the same accession
            mrows = []
            zero = index_oof == 0
            for k in (2, 3, 4):
                sel = (index_oof >= k)
                obs_, ind_, base_ = [], [], []
                cnt = top[sel] & rc[sel].notna()                              # counted categories in which it was scored
                two = cnt.sum(axis=1) >= 2
                accs = cnt.index[two]
                rate = {c: float(rc.loc[accs, c][cnt.loc[accs, c]].mean()) for c in cats if cnt.loc[accs, c].any()}
                for a_ in accs:
                    C_ = [c for c in cats if cnt.at[a_, c]]
                    r_ = rc.loc[a_, C_]
                    obs_.append(float((r_ == 1).sum() >= 2))
                    p_ = np.array([rate[c] for c in C_])                      # Poisson-binomial: P(>= 2 resistant)
                    dist = np.zeros(len(p_) + 1)
                    dist[0] = 1
                    for q_ in p_:
                        dist[1:] = dist[1:] * (1 - q_) + dist[:-1] * q_
                        dist[0] *= 1 - q_
                    ind_.append(float(dist[2:].sum()))
                    pool = rc[zero & rc[C_].notna().all(axis=1)][C_]
                    if len(pool):
                        base_.append(float(((pool == 1).sum(axis=1) >= 2).mean()))
                mrows.append({"min_index": k, "accessions_scored_in_two_or_more_counted_categories": len(accs),
                              "resistant_in_two_or_more": float(np.mean(obs_)) if obs_ else np.nan,
                              "expected_independent_categories": float(np.mean(ind_)) if ind_ else np.nan,
                              "index_zero_same_categories": float(np.mean(base_)) if base_ else np.nan,
                              "accessions_with_index_zero_comparison": len(base_)})
            return index_oof, V, pd.DataFrame(mrows)

        index_oof, V, MV = validate(O)
        V.to_csv(out / "index_out_of_fold.tsv", sep="\t", index=False, float_format="%.6f")
        # 95% intervals of the observed and expected resistant shares and of their difference: 10,000 bootstrap
        # resamples of the groups of near-identical accessions among the accessions with an index of at least k
        grp = data.folds().dup_group.reindex(O.index)
        counted_b = pd.DataFrame({t: [False] * len(O) for t in T.TRAITS}, index=O.index)
        tax_o2 = meta.taxon.reindex(O.index).fillna("unknown")
        for t_, idx in tax_o2.groupby(tax_o2).groups.items():           # counted categories, as in validate()
            Zt = (O.loc[idx] - O.loc[idx].mean()) / O.loc[idx].std()
            CSt = pd.DataFrame({c: Zt[[t for t in Zt.columns if CAT[t] == c]].mean(axis=1) for c in cats})
            topt = CSt >= CSt.quantile(1 - a.top)
            for t in T.TRAITS:
                counted_b.loc[idx, t] = topt[CAT[t]].to_numpy()
        rng = np.random.default_rng(20261117)
        brows = []
        for k in (1, 2, 3, 4):
            sel = index_oof >= k
            m = res[sel].notna() & counted_b[sel]
            per = pd.DataFrame({"n": m.sum(axis=1), "r": res[sel].where(m).sum(axis=1), "e": (m * base).sum(axis=1),
                                "g": grp[sel]})
            per = per[per.n > 0].groupby("g")[["n", "r", "e"]].sum()
            G_ = per.to_numpy()
            obs0, exp0 = G_[:, 1].sum() / G_[:, 0].sum(), G_[:, 2].sum() / G_[:, 0].sum()
            assert abs(obs0 - float(V.loc[V.min_index == k, "observed_resistant_share"].iloc[0])) < 1e-9
            S_ = np.zeros((10000, 3))                                        # group sums of n, r, e per resample
            for c0 in range(0, 10000, 500):
                S_[c0:c0 + 500] = G_[rng.integers(0, len(G_), size=(500, len(G_)))].sum(axis=1)
            ob, ex = S_[:, 1] / S_[:, 0], S_[:, 2] / S_[:, 0]
            q = lambda v: np.quantile(v, [0.025, 0.975])  # noqa: E731
            brows.append({"min_index": k, "groups": len(G_), "observed": obs0, "observed_lo": q(ob)[0], "observed_hi": q(ob)[1],
                          "expected": exp0, "expected_lo": q(ex)[0], "expected_hi": q(ex)[1],
                          "difference": obs0 - exp0, "difference_lo": q(ob - ex)[0], "difference_hi": q(ob - ex)[1]})
        BV = pd.DataFrame(brows)
        BV.to_csv(out / "index_out_of_fold_bootstrap.tsv", sep="\t", index=False, float_format="%.6f")
        print("out-of-fold index, bootstrap over groups of near-identical accessions:")
        print(BV.round(4).to_string(index=False))
        MV.to_csv(out / "index_out_of_fold_multi.tsv", sep="\t", index=False, float_format="%.6f")
        print("resistance in two or more counted categories:")
        print(MV.to_string(index=False))
        # the same index from the out-of-fold GBLUP predictions of all traits (GBLUP fitted to all training accessions of
        # a trait, work/mtboost_start/cv1_fold<f>_evalall.npz of train_mtboost.py --dump_full)
        gfiles = [ROOT / "work/mtboost_start" / f"cv1_fold{f}_evalall.npz" for f in range(5)]
        if all(f.exists() for f in gfiles):
            gparts = []
            for f in gfiles:
                zg = np.load(f, allow_pickle=True)
                Pg = pd.DataFrame(np.asarray(zg["S0e"], float), index=[str(x) for x in zg["E"]], columns=list(T.TRAITS))
                assert np.isfinite(Pg.to_numpy()).all()
                gparts.append((Pg - Pg.mean()) / Pg.std())
            Og = pd.concat(gparts)
            assert Og.index.is_unique and set(Og.index) == set(O.index)
            Og = Og.reindex(O.index)
            cg_ = []
            for t in T.TRAITS:                      # against the GBLUP predictions of the article, within folds (Og is
                rr = pd.read_csv(data.RUNS / "cv1/gblup" / f"{t}.tsv.gz", sep="\t", dtype={"acc": str}).drop_duplicates("acc")
                for _, g_ in rr.groupby("fold"):     # standardized within folds)
                    if len(g_) >= 3:
                        cg_.append(float(np.corrcoef(g_.pred, Og.loc[g_.acc, t].to_numpy())[0, 1]))
            print(f"GBLUP of all test accessions against runs/cv1/gblup (scored accessions, within folds): r {min(cg_):.4f}-{max(cg_):.4f}")
            _, Vg, MVg = validate(Og)
            Vg.to_csv(out / "index_out_of_fold_gblup.tsv", sep="\t", index=False, float_format="%.6f")
            MVg.to_csv(out / "index_out_of_fold_multi_gblup.tsv", sep="\t", index=False, float_format="%.6f")
            print("index from out-of-fold GBLUP predictions:")
            print(Vg.to_string(index=False))
        both = pd.DataFrame({"oof": index_oof, "final": R.multi_resistance_index_within_taxon.reindex(O.index)})
        print("index from out-of-fold predictions:")
        print(V.to_string(index=False))
        print("rank correlation between the out-of-fold index and the index of the model fitted to all accessions "
              f"(scored accessions): {both.corr(method='spearman').iloc[0, 1]:.3f}; "
              f"index >= 4: out of fold {int((both.oof >= 4).sum())}, final model {int((both.final >= 4).sum())}, both {int(((both.oof >= 4) & (both.final >= 4)).sum())}")
        # out-of-fold index of every scored accession (index_out_of_fold_by_accession.tsv), and the candidates of the model fitted to all
        # accessions that also have an out-of-fold index of four or more, by taxon
        pd.DataFrame({"acc": index_oof.index, "index_out_of_fold": index_oof.to_numpy()}).to_csv(
            out / "index_out_of_fold_by_accession.tsv", sep="\t", index=False)
        tax_b = R.taxon.reindex(both.index)
        for tx in ("Glycine max", "Glycine soja"):
            cand = (both.final >= 4) & (tax_b == tx)
            print(f"{tx}: scored accessions with an index >= 4 (model fitted to all accessions) {int(cand.sum())}, of which "
                  f"{int((cand & (both.oof >= 4)).sum())} also >= 4 out of fold and {int((cand & (both.oof >= 2)).sum())} >= 2 out of fold")
    else:
        print("out-of-fold predictions of all traits not there yet: index validation skipped")
    # 5. accessions with an index of four or more by maturity group and by origin (Glycine max)
    gm = R[R.taxon == "Glycine max"].copy()
    gm["high"] = gm.multi_resistance_index_within_taxon >= 4
    gm["scored"] = gm.n_stress_traits.fillna(0) > 0
    mgc = np.floor(gm.mg)
    rows = []
    for lab, key in (("maturity group", mgc.map(lambda v: "not given" if pd.isna(v) else str(int(v)))),
                     ("origin", gm.origin.fillna("not given"))):
        for g_, d_ in gm.groupby(key, sort=False):
            if lab == "origin" and len(d_) < 200 and d_.high.sum() < 10:      # countries with >= 200 accessions or >= 10 high
                continue
            rows.append({"grouping": lab, "group": g_, "accessions": len(d_), "index_at_least_4": int(d_.high.sum()),
                         "share": float(d_.high.mean()), "never_scored": int((~d_.scored).sum()),
                         "never_scored_index_at_least_4": int((d_.high & ~d_.scored).sum())})
    rows.append({"grouping": "all", "group": "Glycine max", "accessions": len(gm), "index_at_least_4": int(gm.high.sum()),
                 "share": float(gm.high.mean()), "never_scored": int((~gm.scored).sum()),
                 "never_scored_index_at_least_4": int((gm.high & ~gm.scored).sum())})
    Gt = pd.DataFrame(rows)
    Gt.to_csv(out / "index_by_group.tsv", sep="\t", index=False, float_format="%.6f")
    print(Gt.to_string(index=False))


if __name__ == "__main__":
    main()
