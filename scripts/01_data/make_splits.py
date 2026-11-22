"""Step 3 of the plan: the frozen splits shared by every model.

usage: python -I make_splits.py

Writes splits/:
  folds.tsv         one row per phenotyped accession: acc, dup_group, stratum, cv1_fold (0-4), inner_fold (0-4)
                    cv1_fold = 考法一 "new accessions": the test fold's accessions lose all their records.
                    inner_fold is assigned independently at group level, so for every outer fold it partitions the
                    training accessions (used for hyper-parameters and model choice only).
  cv2_masks.tsv     考法二 "fill in a missing trait": for each test accession of a cv1 fold with >= 2 traits, one random
                    target trait; hidden = the target plus every other trait recorded in the same GRIN evaluation
                    batch as the target (shared trial errors would inflate this test; Runcie & Cheng 2019).
  cv3_batches.tsv   考法三 "new evaluation round": for each trait with >= 2 batches, its largest batch with more than one
                    class (SOYBEAN.GENERAL and DEVELOPER.DATA excluded); each distinct batch is one hold-out unit covering
                    all its traits.
  fold_summary.tsv  accessions and resistant accessions per cv1 fold x trait
  SHA256SUMS
Rules: near-identical groups (IBS >= 0.99, results/near_identical) never cross a cv1 or inner fold; resistant accessions
of the rare (<5%) traits are spread over folds first (>= 4 per fold); groups are then assigned within strata of
maturity group, taxon and number of scored traits (1, 2-4, 5-9, 10+; counted without P. sojae race 10, which was added
after the folds had been fixed); seed 20261111.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import traits as T  # noqa: E402

SEED = 20261111
rng = np.random.default_rng(SEED)
P = ROOT / "data/processed"
S = ROOT / "splits"
S.mkdir(exist_ok=True)

# The folds are those of the 32-trait analysis: P. sojae race 10 (PHYTOROT_10), whose records were added after the folds
# had been fixed (2026-11-17; its first download had returned another descriptor), is left out of everything that
# defines them (the accessions with records, the rare traits and the strata). All its accessions were scored for
# other traits, so that every accession with a record has a fold.
STRATA_EXCLUDE = ["PHYTOROT_10"]
pm_ = pd.read_csv(P / "pheno_matrix.tsv.gz", sep="\t", index_col=0)
n_fold_traits = pm_.drop(columns=STRATA_EXCLUDE).notna().sum(axis=1)
acc = pd.read_csv(P / "accessions.tsv", sep="\t", dtype={"vcf_id": str})
acc = acc[acc.vcf_id.map(n_fold_traits).fillna(0) > 0].copy()
assert set(pm_.index[pm_[STRATA_EXCLUDE].notna().any(axis=1)]) <= set(acc.vcf_id), "accessions scored only for race 10"
assert acc.dup_group.notna().all(), "every phenotyped accession needs a near-identical group"
acc["dup_group"] = acc.dup_group.astype(int)
n_strata = n_fold_traits.reindex(acc.vcf_id).fillna(0).astype(int).to_numpy()
band = pd.cut(pd.Series(n_strata, index=acc.index), [0, 1, 4, 9, 99], labels=["1", "2-4", "5-9", "10+"]).astype(str)
acc["stratum"] = acc.mg_group.fillna("MG_NA") + "|" + acc.taxon.fillna("NA").str.replace("Glycine ", "G.") + "|" + band

# group-level stratum = stratum of the group's first member (groups are near-identical, so mostly homogeneous)
grp = acc.groupby("dup_group").agg(size=("vcf_id", "size"), stratum=("stratum", "first")).reset_index()


res_all = pd.read_csv(P / "resistant_matrix.tsv.gz", sep="\t", index_col=0)
res_fold = res_all.drop(columns=STRATA_EXCLUDE)
RARE = list(res_fold.mean()[res_fold.mean() < 0.05].sort_values().index)      # rarest first (race 10: 36%, not rare)
res_ph = res_all.reindex(acc.vcf_id).set_index(acc.dup_group.to_numpy())


def assign(groups: pd.DataFrame, k: int, rng) -> dict:
    """Balanced group-to-fold assignment.

    First the groups holding resistant accessions of the rare (<5%) traits, rarest trait first, each to the fold with the
    fewest resistant accessions of that trait so far; then the remaining groups within strata (large groups first, each
    to the currently smallest fold of its stratum)."""
    fold_of = {}
    load = np.zeros(k)
    rcount = {t: np.zeros(k) for t in RARE}
    for t in RARE:
        hit = res_ph.index[(res_ph[t] == 1).to_numpy()]
        todo = [g for g in pd.unique(hit) if g not in fold_of]
        rng.shuffle(todo)
        for g in todo:
            n_r = int((res_ph.loc[[g], t] == 1).sum())
            f = int(np.lexsort((load, rcount[t]))[0])
            fold_of[g] = f
            sz = int(groups.loc[groups.dup_group == g, "size"].iloc[0])
            load[f] += sz
            for u in RARE:
                rcount[u][f] += int((res_ph.loc[[g], u] == 1).sum())
    for _, g in groups[~groups.dup_group.isin(list(fold_of))].groupby("stratum"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values("size", ascending=False, kind="stable")
        local = np.zeros(k)
        for r in g.itertuples():
            f = int(np.lexsort((load, local))[0])     # smallest within this stratum, ties by global load
            fold_of[r.dup_group] = f
            local[f] += r.size
            load[f] += r.size
    return fold_of


cv1 = assign(grp, 5, rng)
inner = assign(grp, 5, rng)
acc["cv1_fold"] = acc.dup_group.map(cv1)
acc["inner_fold"] = acc.dup_group.map(inner)
folds = acc[["vcf_id", "dup_group", "dup_group_size", "stratum", "cv1_fold", "inner_fold"]].rename(columns={"vcf_id": "acc"})
folds.to_csv(S / "folds.tsv", sep="\t", index=False)

# --- CV2 masks
long = pd.read_csv(P / "pheno_long.tsv.gz", sep="\t", dtype={"acc": str})
by_acc = long.groupby("acc")
rows = []
for a, f in zip(folds.acc, folds.cv1_fold):
    d = by_acc.get_group(a)
    tr = sorted(d.trait.unique())
    if len(tr) < 2:
        continue
    target = tr[int(rng.integers(len(tr)))]
    batches = set(d.loc[d.trait == target, "batch"])
    hidden = sorted(set(d.loc[d.batch.isin(batches), "trait"]) | {target})
    rows.append(dict(acc=a, cv1_fold=f, target=target, hidden=",".join(hidden), n_hidden=len(hidden),
                     n_visible=len(tr) - len(hidden)))
cv2 = pd.DataFrame(rows)
cv2.to_csv(S / "cv2_masks.tsv", sep="\t", index=False)

# --- CV3 batch hold-outs
size = long.groupby(["trait", "batch"]).agg(n=("acc", "nunique"), sd=("score", "std")).reset_index()
cand = size[~size.batch.isin(["SOYBEAN.GENERAL", "SOYBEAN.DEVELOPER.DATA"]) & (size.sd > 0)]     # single-class batches can't be scored
multi = size.groupby("trait").batch.nunique()
pick = (cand[cand.trait.isin(multi[multi >= 2].index)].sort_values(["trait", "n"], ascending=[True, False])
        .groupby("trait").head(1))
units = sorted(pick.batch.unique())
cv3 = []
for i, b in enumerate(units):
    d = size[size.batch == b]
    cv3.append(dict(unit=i, batch=b, traits=",".join(d.trait), accessions=int(long[long.batch == b].acc.nunique()),
                    chosen_for=",".join(pick[pick.batch == b].trait)))
cv3 = pd.DataFrame(cv3)
cv3.to_csv(S / "cv3_batches.tsv", sep="\t", index=False)

# --- summary and checks
res = pd.read_csv(P / "resistant_matrix.tsv.gz", sep="\t", index_col=0)
summ = []
for f in range(5):
    ids = folds.acc[folds.cv1_fold == f]
    r = res.reindex(ids)
    for t in T.TRAITS:
        summ.append(dict(fold=f, trait=t, n=int(r[t].notna().sum()), resistant=int((r[t] == 1).sum())))
summ = pd.DataFrame(summ)
summ.to_csv(S / "fold_summary.tsv", sep="\t", index=False)
cross = folds.groupby("dup_group").cv1_fold.nunique().max(), folds.groupby("dup_group").inner_fold.nunique().max()
print("fold sizes (accessions):", folds.cv1_fold.value_counts().sort_index().to_dict())
print("inner fold sizes:", folds.inner_fold.value_counts().sort_index().to_dict())
print("max folds per near-identical group (cv1, inner):", cross)
rare = res.mean()[res.mean() < 0.05].index
mins = summ[summ.trait.isin(rare)].groupby("trait").resistant.min()
print("min resistant accessions per fold, rare traits:", mins.to_dict())
print(f"CV2: {len(cv2)} test accessions; hidden traits per accession: {cv2.n_hidden.value_counts().sort_index().to_dict()}; "
      f"with no visible trait left: {(cv2.n_visible == 0).sum()}")
print(f"CV3: {len(cv3)} hold-out batches covering {pick.trait.nunique()} traits")
print(cv3[["unit", "batch", "accessions", "chosen_for"]].to_string(index=False))
ok = cross == (1, 1) and (mins >= 4).all()
with open(S / "SHA256SUMS", "w") as fh:
    for f in ("folds.tsv", "cv2_masks.tsv", "cv3_batches.tsv", "fold_summary.tsv"):
        fh.write(f"{hashlib.sha256((S / f).read_bytes()).hexdigest()}  {f}\n")
print("checks:", "PASS" if ok else "FAIL (see above)")
sys.exit(0 if ok else 1)
