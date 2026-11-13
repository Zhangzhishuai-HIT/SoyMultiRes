"""Metrics for every finished model x scenario run (plan section 5), plus the comparison with the plan-stage pilot.

usage: python -I summarize.py cv1 gblup [model ...]
Writes results/baselines/<scenario>_<model>_metrics.tsv; for gblup also results/baselines/<scenario>_gblup_vs_pilot.tsv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402

scenario, models = sys.argv[1], sys.argv[2:]
rare = set(data.rare_traits())
out = ROOT / "results/baselines"
out.mkdir(parents=True, exist_ok=True)
for model in models:
    rows = []
    for t in T.TRAITS:
        f = data.RUNS / scenario / model / f"{t}.tsv.gz"
        if not f.exists():
            continue
        rec = pd.read_csv(f, sep="\t", dtype={"acc": str})
        m = evaluate.metrics(rec, t, t in rare)
        m["short"] = T.SHORT[t]
        m["rare"] = t in rare
        rows.append(m)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"{scenario}_{model}_metrics.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"{model}: {len(d)} traits")
    cols = [c for c in ["short", "n", "primary_metric", "primary", "r", "r_within_batch", "auroc", "auprc_lift"] if c in d]
    print(d[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if model == "gblup":
        pil = pd.read_csv(ROOT / "results/pilot_gblup/pilot_gblup_summary.tsv", sep="\t").set_index("trait")
        cmp_ = d.set_index("trait")[["short", "n", "primary_metric", "primary"]].copy()
        cmp_["r_formal"] = d.set_index("trait").apply(lambda r: r.get("r_within_batch") if r.name in T.NUMERIC else r.get("r"), axis=1)
        cmp_["r_pilot"] = pil.r_cv.reindex(cmp_.index)
        cmp_["auprc_lift_formal"] = d.set_index("trait").get("auprc_lift")
        cmp_["auprc_lift_pilot"] = pil.auprc_over_prev.reindex(cmp_.index)
        cmp_["diff_r"] = cmp_.r_formal - cmp_.r_pilot
        cmp_.to_csv(out / f"{scenario}_gblup_vs_pilot.tsv", sep="\t", float_format="%.4f")
        print("formal - pilot r: median %.3f; |diff| > 0.05 for %d traits" % (cmp_.diff_r.median(), (cmp_.diff_r.abs() > 0.05).sum()))
