"""Fold inputs for BGLR (BayesB) and collection of its predictions (pattern of HapGAM's bglr_prepare.py).

usage: python -I bglr_prepare.py prepare --scenario cv1|cv1_inner [--traits ...] -> work/bglr/<scenario>/<trait>/<split>/
       python -I bglr_prepare.py lists   --scenario cv1                       -> jobs/lists/bglr_<scenario>_{big,small}.txt
       python -I bglr_prepare.py collect --scenario cv1 [--model BayesB]      -> runs/<scenario>/bayesb/<trait>.tsv.gz

Each fold directory holds train_rows.txt / test_rows.txt (rows of data/processed/geno/dosage.npy), y_train.txt,
test_acc.txt and test_records.tsv; bglr_fit.R reads the genotype rows straight from the .npy (header offset in
geno/dosage.json). Targets come from soymultires.data.fold_targets, so BayesB sees exactly what GBLUP sees.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

BIG = 4000      # training accessions above which a fold goes to the 16 GB list


def npy_offset(path: Path) -> int:
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        return f.tell()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "lists", "collect"])
    ap.add_argument("--scenario", default="cv1")
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--model", default="BayesB")
    a = ap.parse_args()
    traits = a.traits or T.TRAITS
    base = ROOT / "work/bglr" / a.scenario
    geno = data.P / "geno"
    meta = json.load(open(geno / "dosage.json"))
    if "offset" not in meta:
        meta["offset"] = npy_offset(geno / meta["file"])
        json.dump(meta, open(geno / "dosage.json", "w"))
    if a.action == "prepare":
        long = data.pheno_long()
        fo = data.folds()
        pos = {v: i for i, v in enumerate(data.samples())}
        for t in traits:
            lt = long[long.trait == t]
            for f in range(data.n_splits(a.scenario)):
                if a.scenario == "cv3" and t not in data.cv3_traits_of(f):
                    continue
                d = base / t / data.split_name(a.scenario, f)
                d.mkdir(parents=True, exist_ok=True)
                ft = data.split_targets(lt, t, a.scenario, f, fo)
                np.savetxt(d / "train_rows.txt", [pos[x] for x in ft.train_acc], fmt="%d")
                np.savetxt(d / "test_rows.txt", [pos[x] for x in ft.test_acc], fmt="%d")
                np.savetxt(d / "y_train.txt", ft.y_train, fmt="%.6f")
                (d / "test_acc.txt").write_text("\n".join(ft.test_acc) + "\n")
                ft.test_records.assign(fold=f).to_csv(d / "test_records.tsv", sep="\t", index=False)
            print(t, "prepared")
    elif a.action == "lists":
        big, small = [], []
        for t in traits:
            for f in range(data.n_splits(a.scenario)):
                if a.scenario == "cv3" and t not in data.cv3_traits_of(f):
                    continue
                d = base / t / data.split_name(a.scenario, f)
                if (d / f"{a.model}.pred.txt").exists():
                    continue
                n = sum(1 for _ in open(d / "train_rows.txt"))
                (big if n > BIG else small).append((n, str(d)))
        lists = ROOT / "jobs/lists"
        for name, L in (("big", big), ("small", small)):
            L.sort(reverse=True)                     # largest first, so the long ones start early in the window
            (lists / f"bglr_{a.scenario}_{name}.txt").write_text("".join(f"{p}\n" for _, p in L))
            print(name, len(L), "fold directories")
    else:
        out = data.RUNS / a.scenario / a.model.lower()
        out.mkdir(parents=True, exist_ok=True)
        for t in traits:
            parts, missing = [], 0
            for f in range(data.n_splits(a.scenario)):
                if a.scenario == "cv3" and t not in data.cv3_traits_of(f):
                    continue
                d = base / t / data.split_name(a.scenario, f)
                p = d / f"{a.model}.pred.txt"
                if not p.exists():
                    missing += 1
                    continue
                pm = dict(zip((d / "test_acc.txt").read_text().split(), np.loadtxt(p, ndmin=1)))
                rec = pd.read_csv(d / "test_records.tsv", sep="\t", dtype={"acc": str})
                parts.append(rec.assign(pred=rec.acc.map(pm)))
            if missing:
                print(t, f"{missing} folds missing")
                continue
            pd.concat(parts, ignore_index=True).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
            print(t, "collected")


if __name__ == "__main__":
    main()
