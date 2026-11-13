"""Metrics shared by every model (plan section 5).

Input: record-level predictions (acc, batch, score, resistant, pred) for one trait, pooled over test folds.
  categorical traits: per accession y = mean score, resistant = any; pooled Pearson r, Spearman, AUROC, AUPRC
  numeric traits:     within-batch Pearson r (batches with >= 10 test accessions, weighted by size) and pooled r
                      on batch-centred scores
  primary metric:     AUPRC for categorical traits with < 5% resistant accessions, otherwise r
                      (categorical: pooled r; numeric: within-batch r)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import traits as T


def _r(a, b):
    return float(pearsonr(a, b)[0]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else np.nan


def metrics(rec: pd.DataFrame, trait: str, rare: bool) -> dict:
    out = {"trait": trait, "n_records": len(rec)}
    if trait in T.NUMERIC:
        acc_pred = rec.groupby("acc").pred.first()
        out["n"] = int(acc_pred.size)
        w, rs = [], []
        for _, d in rec.groupby("batch"):
            if d.acc.nunique() >= 10:
                rs.append(_r(d.pred.to_numpy(), d.score.to_numpy()))
                w.append(d.acc.nunique())
        rs, w = np.array(rs), np.array(w, float)
        ok = ~np.isnan(rs)
        out["r_within_batch"] = float(np.sum(rs[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
        cen = rec.score - rec.groupby("batch").score.transform("mean")
        y = cen.groupby(rec.acc).mean()
        out["r_pooled"] = _r(acc_pred.reindex(y.index).to_numpy(), y.to_numpy())
        out["primary"] = out["r_within_batch"]
        out["primary_metric"] = "r_within_batch"
        return out
    g = rec.groupby("acc").agg(y=("score", "mean"), res=("resistant", "max"), pred=("pred", "first"))
    out["n"] = len(g)
    out["r"] = _r(g.pred.to_numpy(), g.y.to_numpy())
    out["spearman"] = float(spearmanr(g.pred, g.y)[0]) if g.y.nunique() > 1 else np.nan
    lab = g.res.to_numpy() == 1
    out["prevalence"] = float(lab.mean())
    out["n_resistant"] = int(lab.sum())
    if 0 < lab.sum() < len(lab):
        out["auroc"] = float(roc_auc_score(lab, g.pred))
        out["auprc"] = float(average_precision_score(lab, g.pred))
        out["auprc_lift"] = out["auprc"] / out["prevalence"]
    out["primary"] = out.get("auprc", np.nan) if rare else out["r"]
    out["primary_metric"] = "auprc" if rare else "r"
    return out


def within_fold_z(rec, fold_col: str = "fold"):
    """Evaluation rule (docs/FREEZE_DEVIATIONS.md, 2026-11-12): each model's predictions are standardised within fold
    (records of one trait) before the folds are pooled, so that pooled metrics measure ranking within each fitted model,
    not calibration differences between the five fold models. Applied identically to every model."""
    r = rec.copy()
    keys = [fold_col] + (["trait"] if "trait" in r.columns else [])
    g = r.groupby(keys)["pred"]
    r["pred"] = (r["pred"] - g.transform("mean")) / (g.transform("std") + 1e-12)
    return r
