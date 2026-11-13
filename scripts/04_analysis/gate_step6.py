"""Step-6 gate (plan 3.6) and the choice of the strongest single-trait model per trait, both on cv1_inner.

usage: python -I gate_step6.py <mt_model_dir_name> [more ...]      e.g. hgmt_m128
  reads runs/cv1_inner/{gblup,bayesb,lightgbm,<mt>}/<trait>.tsv.gz (records of the five inner splits, each accession
  validated at most once), computes the primary metric per trait (predictions standardised within fold before pooling,
  evaluate.within_fold_z; value_raw_pooled = without), and writes
  results/gate/cv1_inner_metrics.tsv          per trait x model
  results/gate/cv1_inner_best_single.tsv      strongest single-trait model per trait ON INNER CV (used for the outer
                                               CV1 comparison, so the baseline is chosen without the test folds)
  results/gate/gate_<mt>.txt                  gate verdict: mean over the 16 primary traits, MT vs best single-trait
                                               model per trait (the per-trait maximum on the same accessions)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

import os
SINGLE = tuple(os.environ.get("SINGLE", "gblup,bayesb,lightgbm").split(","))   # e.g. SINGLE=gblup,bayesb,lightgbm_avg4
SUFFIX = os.environ.get("SUFFIX", "")                                            # output name suffix for another set
mts = sys.argv[1:]
rare = set(data.rare_traits())
prim = set(data.primary_traits())
out = ROOT / "results/gate"
out.mkdir(parents=True, exist_ok=True)
rows = []
for m in SINGLE + tuple(mts):
    for t in T.TRAITS:
        f = data.RUNS / "cv1_inner" / m / f"{t}.tsv.gz"
        if not f.exists():
            continue
        rec = pd.read_csv(f, sep="\t", dtype={"acc": str})
        mt = evaluate.metrics(evaluate.within_fold_z(rec), t, t in rare)      # evaluation rule of 2026-11-12
        raw = evaluate.metrics(rec, t, t in rare)
        rows.append({"model": m, "trait": t, "short": T.SHORT[t], "primary_trait": t in prim, "n": mt.get("n"),
                     "metric": mt.get("primary_metric"), "value": mt.get("primary"), "value_raw_pooled": raw.get("primary")})
d = pd.DataFrame(rows)
d.to_csv(out / f"cv1_inner_metrics{SUFFIX}.tsv", sep="\t", index=False, float_format="%.4f")
w = d.pivot_table(index="trait", columns="model", values="value")
have = [m for m in SINGLE if m in w]
missing = {m: int(w[m].isna().sum()) if m in w else 32 for m in SINGLE}
best = w[have].idxmax(axis=1).rename("best_model").to_frame()
best["best_value"] = w[have].max(axis=1)
best["short"] = [T.SHORT[t] for t in best.index]
best["primary_trait"] = best.index.isin(prim)
best.to_csv(out / f"cv1_inner_best_single{SUFFIX}.tsv", sep="\t", float_format="%.4f")
print("single-trait results missing per model:", missing)
print("strongest single-trait model on inner CV:", best.best_model.value_counts().to_dict())
for m in mts:
    if m not in w:
        print(m, "not found")
        continue
    c = pd.DataFrame({"short": best.short, "primary": best.primary_trait, "mt": w[m], "best_single": best.best_value,
                      "best_model": best.best_model, **{s: w[s] for s in have}})
    c["diff"] = c.mt - c.best_single
    if "gblup" not in c:
        c["gblup"] = np.nan
    p = c[c.primary]
    lines = [f"gate on cv1_inner, model {m} (single-trait models complete: {missing})",
             f"16 primary traits: mean MT {p.mt.mean():.4f} vs mean of the best single-trait model per trait "
             f"{p.best_single.mean():.4f} -> {'PASS' if p.mt.mean() >= p.best_single.mean() else 'NOT YET'}",
             f"MT higher than the best single-trait model on {int((p['diff'] > 0).sum())} of {len(p)} primary traits; "
             f"other traits: {int((c[~c.primary]['diff'] > 0).sum())} of {int((~c.primary).sum())}",
             f"MT vs GBLUP on primary traits: {int((p.mt > p.gblup).sum())} of {len(p)} higher; mean {p.mt.mean() - p.gblup.mean():+.4f}",
             "", c.round(4).to_string()]
    txt = "\n".join(lines)
    (out / f"gate_{m}{SUFFIX}.txt").write_text(txt + "\n")
    print(txt)
