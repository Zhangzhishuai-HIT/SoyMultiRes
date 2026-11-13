"""Inputs for the MegaLMM timing trial (week 1): the five SCN races on N random accessions with >= 1 SCN record.

usage: python -I megalmm_trial_prep.py N [N ...]   -> work/megalmm/trial_<N>/{Y.tsv, K.bin, ids.txt}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

SCN = ["NEMATCYST_1", "NEMATCYST_3", "NEMATCYST_4", "NEMATCYST_5", "NEMATCYST_14"]
mat = pd.read_csv(data.P / "pheno_matrix.tsv.gz", sep="\t", index_col=0)[SCN].dropna(how="all")
pos = {v: i for i, v in enumerate(data.samples())}
K = data.grm(mmap=True)
rng = np.random.default_rng(20261111)
for n in map(int, sys.argv[1:]):
    ids = np.sort(rng.choice(mat.index.to_numpy(), size=min(n, len(mat)), replace=False))
    d = ROOT / "work/megalmm" / f"trial_{n}"
    d.mkdir(parents=True, exist_ok=True)
    mat.loc[ids].to_csv(d / "Y.tsv", sep="\t")
    rows = np.array([pos[x] for x in ids])
    np.asarray(K[np.ix_(rows, rows)], dtype=np.float64).tofile(d / "K.bin")
    (d / "ids.txt").write_text("\n".join(ids) + "\n")
    print(n, "written; observed cells", int(mat.loc[ids].notna().sum().sum()))
