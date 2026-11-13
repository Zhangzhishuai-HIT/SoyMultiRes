"""CV2 ("fill in untested resistances", plan 3.1 / 3.6): selection-index correction of a model's CV1 predictions with
the measured values of the accession's other traits.

usage: python -I cv2_selection_index.py <model> [<model> ...]      e.g. mtboost_v2_final lightgbm_avg4 gblup
Design (splits/cv2_masks.tsv, frozen 2026-11-11): every test accession of outer fold f that was scored for >= 2 traits
hides one target trait together with every trait scored in the same evaluation batch (Runcie & Cheng 2019); the
remaining "visible" traits come from other batches. Steps, per outer fold f and model:
  1. residuals of the model's out-of-fold predictions on cv1_inner split f (inside the training part of fold f):
     per accession and trait, r = y - (a + b * pred) (a, b fitted per trait), standardised; numeric traits are centred
     within batch first (as their evaluation);
  2. residual correlation of every trait pair from accessions scored for both traits in different batches only
     (no shared field error, as in the test setting); pairs with < 50 such accessions or |rho| < 2/sqrt(n) -> 0;
  3. for each test accession: target t, visible set V; e_V = standardised residuals of the outer CV1 predictions of the
     visible traits; correction = rho_tV (R_VV + 0.1 I)^-1 e_V, added to the calibrated target prediction;
  4. the target's records of the test accessions are scored (evaluate.within_fold_z + evaluate.metrics).
Single-trait models get the same correction with their own residuals; without correction their CV2 score is their
CV1 score on the same records (plan). Writes runs/cv2/<model>_si/<trait>.tsv.gz and runs/cv2/<model>_cv1/<trait>.tsv.gz
(records of the CV2 targets, corrected / uncorrected) and results/cv2/si_<model>.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

TRAITS = list(T.TRAITS)


def per_accession(rec, trait, batch_means=None):
    """Per-accession target and prediction: numeric traits centred within batch (batch means of scores and of
    predictions from `batch_means` if given, else from the records themselves)."""
    r = rec.copy()
    if trait in T.NUMERIC:
        if batch_means is None:
            bm = r.groupby("batch")[["score", "pred"]].transform("mean")
            r["score"], r["pred"] = r.score - bm.score, r.pred - bm.pred
        else:
            r["score"] = r.score - r.batch.map(batch_means["score"]).fillna(batch_means["score"].mean())
            r["pred"] = r.pred - r.batch.map(batch_means["pred"]).fillna(batch_means["pred"].mean())
    g = r.groupby("acc")
    return pd.DataFrame({"y": g.score.mean(), "pred": g.pred.mean(), "batches": g.batch.agg(lambda b: set(b))})


def main():
    models = sys.argv[1:]
    masks = pd.read_csv(ROOT / "splits/cv2_masks.tsv", sep="\t")
    rare = set(data.rare_traits())
    out_res = ROOT / "results/cv2"
    out_res.mkdir(parents=True, exist_ok=True)
    long = data.pheno_long()
    fo = data.folds()
    for model in models:
        si_parts, cv1_parts = {t: [] for t in TRAITS}, {t: [] for t in TRAITS}
        diag = []
        for f in range(5):
            tr_long = long[long.acc.map(fo.cv1_fold) != f]
            # 1. out-of-fold residuals inside the training part of fold f (cv1_inner split f)
            inner = {}
            calib = {}
            for t in TRAITS:
                fi = data.RUNS / "cv1_inner" / model / f"{t}.tsv.gz"
                r = pd.read_csv(fi, sep="\t", dtype={"acc": str})
                r = r[r.fold == f]
                pa = per_accession(r, t)
                if len(pa) < 10 or pa.pred.std() == 0:
                    continue
                b, a = np.polyfit(pa.pred, pa.y, 1)
                res = pa.y - (a + b * pa.pred)
                calib[t] = (a, b, res.std() + 1e-12)
                inner[t] = pd.DataFrame({"r": res / calib[t][2], "batches": pa.batches})
            # 2. residual correlations from accessions scored for both traits in different batches
            R = pd.DataFrame(0.0, index=TRAITS, columns=TRAITS)
            for i, t in enumerate(TRAITS):
                R.loc[t, t] = 1.0
                for u in TRAITS[i + 1:]:
                    if t not in inner or u not in inner:
                        continue
                    j = inner[t].join(inner[u], lsuffix="_t", rsuffix="_u", how="inner")
                    j = j[[len(bt & bu) == 0 for bt, bu in zip(j.batches_t, j.batches_u)]]
                    n = len(j)
                    if n < 50:
                        continue
                    rho = float(np.corrcoef(j.r_t, j.r_u)[0, 1])
                    if np.isfinite(rho) and abs(rho) >= 2 / np.sqrt(n):
                        R.loc[t, u] = R.loc[u, t] = rho
            # 3. test accessions of fold f: standardised residuals of the outer CV1 predictions, every trait at once
            mf = masks[masks.cv1_fold == f]
            outer, e = {}, {}
            for t in TRAITS:
                r = pd.read_csv(data.RUNS / "cv1" / model / f"{t}.tsv.gz", sep="\t", dtype={"acc": str})
                outer[t] = r[r.fold == f]
                if t not in calib:
                    continue
                if t in T.NUMERIC:
                    lt = tr_long[tr_long.trait == t]
                    bm = pd.DataFrame({"score": lt.groupby("batch").score.mean()})
                    pm = outer[t].groupby("batch").pred.mean()
                    bm = bm.reindex(bm.index.union(pm.index))
                    bm["pred"] = pm.reindex(bm.index)
                    bm["score"] = bm.score.fillna(lt.score.mean())
                    bm["pred"] = bm.pred.fillna(outer[t].pred.mean())
                    pa = per_accession(outer[t], t, bm)
                else:
                    pa = per_accession(outer[t], t)
                a_, b_, s_ = calib[t]
                e[t] = ((pa.y - (a_ + b_ * pa.pred)) / s_).to_dict()
            corr = {}
            for row in mf.itertuples(index=False):
                t, acc = row.target, row.acc
                if t not in calib:
                    continue
                hidden = set(row.hidden.split(","))
                vis = [u for u in e if u not in hidden and acc in e[u]]
                c = 0.0
                if vis:
                    Rv = R.loc[vis, vis].to_numpy()
                    w, V = np.linalg.eigh(Rv)
                    Rv = (V * np.clip(w, 1e-3, None)) @ V.T                      # nearest PSD
                    ev = np.array([e[u][acc] for u in vis])
                    c = float(R.loc[t, vis].to_numpy() @ np.linalg.solve(Rv + 0.1 * np.eye(len(vis)), ev))
                corr[(acc, t)] = c
                diag.append({"fold": f, "acc": acc, "target": t, "n_visible_used": len(vis), "abs_correction_sd": abs(c)})
            for t in TRAITS:
                accs = {acc for (acc, tt) in corr if tt == t}
                if not accs or t not in calib:
                    continue
                a_t, b_t, s_t = calib[t]
                rt = outer[t][outer[t].acc.isin(accs)].copy()
                base = a_t + b_t * rt.pred
                si_parts[t].append(rt.assign(pred=base + s_t * rt.acc.map(lambda x: corr[(x, t)])))
                cv1_parts[t].append(rt.assign(pred=base))
        rows = []
        for t in TRAITS:
            if not si_parts[t]:
                continue
            si = pd.concat(si_parts[t], ignore_index=True)
            c1 = pd.concat(cv1_parts[t], ignore_index=True)
            for name, d in ((f"{model}_si", si), (f"{model}_cv1", c1)):
                o = data.RUNS / "cv2" / name
                o.mkdir(parents=True, exist_ok=True)
                d.to_csv(o / f"{t}.tsv.gz", sep="\t", index=False)
            m_si = evaluate.metrics(evaluate.within_fold_z(si), t, t in rare)
            m_c1 = evaluate.metrics(evaluate.within_fold_z(c1), t, t in rare)
            rows.append({"trait": t, "short": T.SHORT[t], "n_records": len(si), "metric": m_si["primary_metric"],
                         "cv1": m_c1["primary"], "si": m_si["primary"], "gain": m_si["primary"] - m_c1["primary"]})
        res = pd.DataFrame(rows)
        res.to_csv(out_res / f"si_{model}.tsv", sep="\t", index=False, float_format="%.4f")
        pd.DataFrame(diag).to_csv(out_res / f"si_{model}_accessions.tsv", sep="\t", index=False, float_format="%.4f")
        prim = set(data.primary_traits())
        p = res[res.trait.isin(prim)]
        print(f"{model}: CV2 targets on 16 primary traits: uncorrected {p.cv1.mean():.4f}, selection index "
              f"{p.si.mean():.4f}; other traits {res[~res.trait.isin(prim)].cv1.mean():.4f} -> "
              f"{res[~res.trait.isin(prim)].si.mean():.4f}")


if __name__ == "__main__":
    main()
