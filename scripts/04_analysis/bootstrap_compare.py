"""Paired bootstrap comparisons of the models on the same test accessions.

usage: python -I bootstrap_compare.py run <cv1|cv2|cv3> --trait TRAIT [--reps 100000]
       python -I bootstrap_compare.py collect <cv1|cv2|cv3>

Models: the single-trait models, the multi-trait GBLUP, the best single-trait model, the three TraitBoost variants,
ens_gblup_lightgbm (mean of the standardised GBLUP and LightGBM predictions) and ens_lightgbm_noshare (tree ensemble:
mean of the standardised LightGBM and TraitBoost-without-shared-trees predictions; make_ensemble.py).
cv2 (measured traits): the predictions of TraitBoost and of the best single-trait model for the hidden target records,
without (<model>_cv1) and with (<model>_si) the selection index on the accession's other measured traits
(scripts/04_analysis/cv2_selection_index.py).

run      For one trait: the predictions of every model are standardised within fold (evaluate.within_fold_z) and
         pooled. Groups of near-identical test accessions (splits/folds.tsv, dup_group; an accession without a
         near-identical partner is a group of its own) are resampled with replacement, all records of a group
         together; a resample is a vector of counts per accession, and the accuracy of every model (the trait's
         measure in evaluate.metrics: AUPRC, Pearson r, or r within experiments) is computed with these counts as
         weights, which is identical to recomputing it on the resampled records. The same resamples are used for all
         models.
         Writes results/compare/boot/<scenario>/<TRAIT>.npz (accuracy of every model, and its bootstrap values).
collect  For every pair of models compared in the article: difference a - b, 95% interval of the bootstrap
         differences, two-sided P = 2 x the smaller share of bootstrap differences on either side of zero, and the Holm
         adjustment across the sparse traits and across the other traits. Resamples in which the accuracy of one of
         the two models is undefined are dropped. A trait for which the two accuracies agree in every resample (within
         1e-6, the precision of the stored values; cv2: the index changed no accuracy) is not tested (P empty) and does
         not enter the Holm adjustment. Writes results/compare/<scenario>_<a>_vs_<b>_<group>.tsv.
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

MODELS = {"cv1": ["gblup", "bayesb", "lightgbm_avg4", "bglr_mt", "best_single_avg4", "mtboost_v2_noshare4",
                  "mtboost_v2_share4", "mtboost_v2_final", "ens_gblup_lightgbm", "ens_lightgbm_noshare"],
          "cv3": ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4", "bglr_mt", "best_single_avg4", "mtboost_v2_noshare4",
                  "mtboost_v2_share4", "mtboost_v2_final", "ens_gblup_lightgbm", "ens_lightgbm_noshare"],
          "cv2": ["mtboost_v2_final_cv1", "mtboost_v2_final_si", "best_single_avg4_cv1", "best_single_avg4_si"]}
# every model against the reported model; the chosen option against the option without shared trees (four refits each);
# multi-trait GBLUP against the single-trait models
PAIRS = {sc: [("mtboost_v2_final", m) for m in MODELS[sc] if m != "mtboost_v2_final"]
             + [("mtboost_v2_share4", "mtboost_v2_noshare4"), ("bglr_mt", "best_single_avg4"),
                ("bglr_mt", "gblup" if sc == "cv1" else "mtboost_v2_gblup")] for sc in ("cv1", "cv3")}
# cv2: the index against no index for both models, and TraitBoost against the best single-trait model
PAIRS["cv2"] = [("mtboost_v2_final_si", "mtboost_v2_final_cv1"), ("best_single_avg4_si", "best_single_avg4_cv1"),
                ("mtboost_v2_final_cv1", "best_single_avg4_cv1"), ("mtboost_v2_final_si", "best_single_avg4_si")]
EXCLUDE = {"cv1": set(), "cv2": set(), "cv3": {"NSCANKER"}}       # cv3: one class left in the training set
SEED = 20261115


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def corr_w(W, P, y):
    """Weighted Pearson correlation of every row of P (models x n) with y (n) for every row of W (resamples x n)."""
    tot = W.sum(1)
    sy, syy = W @ y, W @ (y * y)
    sp, spp, spy = W @ P.T, W @ (P * P).T, W @ (P * y).T
    cov = spy - sp * (sy / tot)[:, None]
    vp = spp - sp * sp / tot[:, None]
    vy = syy - sy * sy / tot
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vp * vy[:, None])
    r[~((vp > 1e-12) & (vy[:, None] > 1e-12) & (tot[:, None] > 2))] = np.nan
    return r, tot


class Trait:
    """Weighted accuracy of all models for one trait; acc(W) -> resamples x models."""

    def __init__(self, scenario, trait, models, rare):
        self.trait, self.models = trait, models
        key = ["acc", "batch", "fold"]
        recs = {}
        for m in models:
            r = pd.read_csv(data.RUNS / scenario / m / f"{trait}.tsv.gz", sep="\t", dtype={"acc": str})
            recs[m] = evaluate.within_fold_z(r).sort_values(key).reset_index(drop=True)
        base = recs[models[0]]
        for m in models[1:]:
            assert (recs[m][key].to_numpy() == base[key].to_numpy()).all(), f"{trait}: test records differ ({m})"
        self.recs = recs
        self.accs, inv = np.unique(base.acc.to_numpy(), return_inverse=True)
        self.n = len(self.accs)
        grp = data.folds().dup_group.reindex(self.accs)
        assert grp.notna().all(), f"{trait}: test accession without a group"
        self.groups, self.group_of = np.unique(grp.to_numpy(), return_inverse=True)   # resampling units
        self.kind = "r_within_batch" if trait in T.NUMERIC else ("auprc" if rare else "r")
        if self.kind == "r_within_batch":
            self.batches = []
            for _, idx in base.groupby("batch").indices.items():
                a = inv[idx]
                assert len(np.unique(a)) == len(a), f"{trait}: an accession has two records in one experiment"
                self.batches.append((a, base.score.to_numpy(float)[idx],
                                     np.stack([recs[m].pred.to_numpy(float)[idx] for m in models])))
        else:
            g = base.groupby("acc").agg(y=("score", "mean"), res=("resistant", "max"))
            assert (g.index.to_numpy() == self.accs).all()
            self.y = g.y.to_numpy(float)
            self.lab = g.res.to_numpy() == 1
            self.P = np.stack([recs[m].groupby("acc").pred.first().to_numpy(float) for m in models])
            if self.kind == "auprc":
                self.order = [np.argsort(-p, kind="mergesort") for p in self.P]
                # last position of every group of tied predictions (one threshold per distinct value)
                self.last = [np.flatnonzero(np.r_[p[o][1:] != p[o][:-1], True]) for p, o in zip(self.P, self.order)]

    def acc(self, W):
        W = np.asarray(W, float)
        if self.kind == "r":
            return corr_w(W, self.P, self.y)[0]
        if self.kind == "r_within_batch":
            num = np.zeros((len(W), len(self.models)))
            den = np.zeros((len(W), len(self.models)))
            for a, y, P in self.batches:
                r, tot = corr_w(W[:, a], P, y)
                ok = np.isfinite(r) & (tot[:, None] >= 10)
                num += np.where(ok, r * tot[:, None], 0.0)
                den += np.where(ok, tot[:, None], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den
        out = np.full((len(W), len(self.models)), np.nan)
        for k, (o, last) in enumerate(zip(self.order, self.last)):
            Ws = W[:, o]
            lab = self.lab[o]
            tp = np.cumsum(Ws * lab, 1)[:, last]
            fp = np.cumsum(Ws * ~lab, 1)[:, last]
            d = np.diff(tp, axis=1, prepend=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                prec = np.where(d > 0, tp / (tp + fp), 0.0)
                out[:, k] = np.where((tp[:, -1] > 0) & (fp[:, -1] > 0), (d * prec).sum(1) / tp[:, -1], np.nan)
        return out

    def check(self, rng, rare):
        """The weighted accuracy equals evaluate.metrics on the records (all weights 1, and one resample)."""
        one = self.acc(np.ones((1, self.n)))[0]
        w = np.bincount(rng.integers(0, self.n, self.n), minlength=self.n)
        res = self.acc(w[None, :])[0]
        pos = {a: i for i, a in enumerate(self.accs)}
        for k, m in enumerate(self.models):
            r = self.recs[m]
            ref = evaluate.metrics(r, self.trait, rare)["primary"]
            assert np.isclose(one[k], ref, atol=1e-9, equal_nan=True), (self.trait, m, one[k], ref)
            rep = w[r.acc.map(pos).to_numpy()]
            s = r.loc[r.index.repeat(rep)].copy()
            s["acc"] = s.acc + "_" + s.groupby(level=0).cumcount().astype(str)        # copies stay distinct
            ref = evaluate.metrics(s.reset_index(drop=True), self.trait, rare)["primary"]
            assert np.isclose(res[k], ref, atol=1e-9, equal_nan=True), (self.trait, m, res[k], ref)
        return one


def run(a):
    rare = set(data.rare_traits())
    models = [m for m in MODELS[a.scenario] if (data.RUNS / a.scenario / m / f"{a.trait}.tsv.gz").exists()]
    missing = sorted(set(MODELS[a.scenario]) - set(models))
    assert not missing or a.allow_missing, f"{a.trait}: no predictions of {missing}"
    tr = Trait(a.scenario, a.trait, models, a.trait in rare)
    rng = np.random.default_rng([SEED, list(T.TRAITS).index(a.trait)])
    point = tr.check(rng, a.trait in rare)
    chunk = max(50, min(2000, int(4e6 // tr.n)))
    B = np.empty((a.reps, len(models)), np.float32)
    G = len(tr.groups)
    p = np.full(G, 1.0 / G)
    for c0 in range(0, a.reps, chunk):
        c = min(chunk, a.reps - c0)
        B[c0:c0 + c] = tr.acc(rng.multinomial(G, p, size=c)[:, tr.group_of])
    out = ROOT / "results/compare/boot" / a.scenario
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.trait}.npz", point=point, boot=B, models=np.array(models), measure=tr.kind,
                        accessions=tr.n, groups=G)
    print(f"{a.scenario} {a.trait}: {tr.n} test accessions in {G} groups, {tr.kind}, {a.reps} resamples;",
          {m: round(float(v), 4) for m, v in zip(models, point)}, flush=True)


def collect(a):
    sparse = set(data.sparse_traits())
    traits = [t for t in T.TRAITS if t not in EXCLUDE[a.scenario] and (ROOT / "results/compare/boot" / a.scenario / f"{t}.npz").exists()]
    Z = {t: np.load(ROOT / "results/compare/boot" / a.scenario / f"{t}.npz", allow_pickle=True) for t in traits}
    groups = {"sparse": [t for t in traits if t in sparse], "nonsparse": [t for t in traits if t not in sparse]}
    print(a.scenario, {g: len(v) for g, v in groups.items()}, "traits; resamples:", sorted({len(Z[t]["boot"]) for t in traits}))
    for ma, mb in PAIRS[a.scenario]:
        for g, ts in groups.items():
            rows = []
            for t in ts:
                z = Z[t]
                models = [str(m) for m in z["models"]]
                if ma not in models or mb not in models:
                    continue
                ia, ib = models.index(ma), models.index(mb)
                d = z["boot"][:, ia].astype(float) - z["boot"][:, ib].astype(float)
                d = d[~np.isnan(d)]
                same = bool(np.all(np.abs(d) < 1e-6)) and abs(float(z["point"][ia] - z["point"][ib])) < 1e-6
                p = np.nan if same else min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                rows.append(dict(trait=t, short=T.SHORT[t], metric=str(z["measure"]), a=float(z["point"][ia]),
                                 b=float(z["point"][ib]), diff=float(z["point"][ia] - z["point"][ib]),
                                 lo=np.quantile(d, 0.025), hi=np.quantile(d, 0.975), p=p, resamples=len(d),
                                 groups=int(z["groups"])))
            if not rows:
                continue
            assert len(rows) == len(ts), f"{ma} vs {mb}: predictions for only {len(rows)} of {len(ts)} traits"
            d = pd.DataFrame(rows)
            d["p_holm"] = np.nan
            tst = d.p.notna()
            if tst.any():
                d.loc[tst, "p_holm"] = holm(d.p[tst].to_numpy())
            d.to_csv(ROOT / f"results/compare/{a.scenario}_{ma}_vs_{mb}_{g}.tsv", sep="\t", index=False, float_format="%.6g")
            hi, lo = d["diff"] > 1e-9, d["diff"] < -1e-9                  # differences of stored values below 1e-9 count as ties
            sig = d.p_holm < 0.05
            print(f"{ma} vs {mb} [{g}, {len(d)} traits]: higher {int(hi.sum())} ({int((hi & sig).sum())}: "
                  f"{','.join(d.short[hi & sig])}), lower {int(lo.sum())} ({int((lo & sig).sum())}: {','.join(d.short[lo & sig])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect"])
    ap.add_argument("scenario", choices=["cv1", "cv2", "cv3"])
    ap.add_argument("--trait")
    ap.add_argument("--reps", type=int, default=100000)
    ap.add_argument("--allow_missing", action="store_true", help="run without the models that have no predictions")
    a = ap.parse_args()
    run(a) if a.mode == "run" else collect(a)


if __name__ == "__main__":
    main()
