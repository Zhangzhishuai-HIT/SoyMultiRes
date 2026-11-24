"""Data access: phenotypes, training and test sets, genotype kernel, per-fold targets.

Phenotype values:
  categorical traits  y = mean ordinal score of the accession over batches; no batch adjustment
  numeric traits      batch fixed effects: y = mean over the accession's batches of (score - training batch mean);
                      batches without training records use the training grand mean
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import traits as T

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
SPLITS = ROOT / "splits"
RUNS = ROOT / "runs"


def pheno_long() -> pd.DataFrame:
    return pd.read_csv(P / "pheno_long.tsv.gz", sep="\t", dtype={"acc": str})


def folds() -> pd.DataFrame:
    return pd.read_csv(SPLITS / "folds.tsv", sep="\t", dtype={"acc": str}).set_index("acc")


def samples() -> pd.Series:
    """vcf_id in genotype row order."""
    return pd.read_csv(P / "geno/samples.tsv", sep="\t", dtype=str).vcf_id


def grm(mmap: bool = True) -> np.ndarray:
    return np.load(P / "geno/grm.f32.npy", mmap_mode="r" if mmap else None)


def rare_traits(threshold: float = 0.05) -> list[str]:
    res = pd.read_csv(P / "resistant_matrix.tsv.gz", sep="\t", index_col=0)
    m = res.mean()
    return [t for t in T.TRAITS if t not in T.NUMERIC and m[t] < threshold]


@dataclass
class FoldTargets:
    trait: str
    fold: int
    train_acc: np.ndarray          # accession ids (one value each)
    y_train: np.ndarray
    test_acc: np.ndarray
    test_records: pd.DataFrame     # acc, batch, score, resistant (record level, for evaluation)


def fold_targets(long_t: pd.DataFrame, trait: str, fold_of: pd.Series, fold: int) -> FoldTargets:
    """Targets for one trait and one cv1 fold (long_t = pheno_long rows of this trait)."""
    f = long_t.acc.map(fold_of)
    tr = long_t[f != fold]
    te = long_t[f == fold]
    if trait in T.NUMERIC:
        bm = tr.groupby("batch").score.mean()
        gm = tr.score.mean()
        adj = tr.score - tr.batch.map(bm).fillna(gm)
        y = adj.groupby(tr.acc).mean()
    else:
        y = tr.groupby("acc").score.mean()
    test_records = te[["acc", "batch", "score", "resistant"]].reset_index(drop=True)
    return FoldTargets(trait, fold, y.index.to_numpy(), y.to_numpy(float), np.array(sorted(te.acc.unique())), test_records)


# ------------------------------------------------------------------ scenarios shared by every model
SCENARIOS = ("cv1", "cv1_inner", "cv3")


def n_splits(scenario: str) -> int:
    return len(cv3_units()) if scenario == "cv3" else 5


def cv3_units() -> pd.DataFrame:
    """Held-out experiments (splits/cv3_batches.tsv): one held-out evaluation batch per unit."""
    return pd.read_csv(SPLITS / "cv3_batches.tsv", sep="\t").set_index("unit")


_CV3_TEST = {}


def cv3_test_accessions(u: int) -> set:
    """Accessions scored in the batch of unit u (any trait): they leave the training data entirely."""
    if u not in _CV3_TEST:
        lg = pheno_long()
        _CV3_TEST[u] = set(lg.acc[lg.batch == cv3_units().loc[u, "batch"]])
    return _CV3_TEST[u]


def split_name(scenario: str, f: int) -> str:
    if scenario == "cv3":
        return f"unit{f}"
    return f"fold{f}" if scenario == "cv1" else f"fold{f}_inner{f}"


def split_sets(scenario: str, f: int, fo: pd.DataFrame | None = None):
    """Boolean Series over accessions (index of folds()): (fit, eval).

    cv1        fit = cv1_fold != f, eval = cv1_fold == f (the test fold).
    cv1_inner  inside the training part of outer fold f: eval = inner_fold == f, fit = the other inner folds. Over
               f = 0..4 every accession is evaluated at most once (never those with cv1_fold == inner_fold), and the
               outer test fold is never touched. Used for model settings and for the step-6 check.
    """
    fo = folds() if fo is None else fo
    if scenario == "cv1":
        return fo.cv1_fold != f, fo.cv1_fold == f
    if scenario == "cv1_inner":
        outer_train = fo.cv1_fold != f
        return outer_train & (fo.inner_fold != f), outer_train & (fo.inner_fold == f)
    if scenario == "cv3":
        # 考法三, strict for multi-trait models: the held-out batch's accessions leave training entirely (every
        # trait); test = the records of the unit's chosen traits in that batch (cv3_batches.tsv, chosen_for)
        ev = fo.index.isin(cv3_test_accessions(f))
        return pd.Series(~ev, index=fo.index), pd.Series(ev, index=fo.index)
    raise ValueError(scenario)


def split_targets(long_t: pd.DataFrame, trait: str, scenario: str, f: int, fo: pd.DataFrame | None = None) -> FoldTargets:
    """fold_targets for any scenario (cv1 reproduces fold_targets(long_t, trait, cv1_fold, f) exactly)."""
    fo = folds() if fo is None else fo
    if scenario == "cv1":
        return fold_targets(long_t, trait, fo.cv1_fold, f)
    if scenario == "cv3":
        unit = cv3_units().loc[f]
        test_acc = cv3_test_accessions(f)
        tr = long_t[~long_t.acc.isin(test_acc)]
        te = long_t[(long_t.batch == unit.batch)] if trait in unit.chosen_for.split(",") else long_t.iloc[0:0]
        lt = pd.concat([tr, te])
        split = pd.Series(0, index=pd.Index(sorted(set(lt.acc))))
        split[split.index.isin(set(te.acc))] = 1
        return fold_targets(lt, trait, split, 1)
    fit, ev = split_sets(scenario, f, fo)
    lt = long_t[long_t.acc.map(fit | ev).fillna(False).astype(bool).to_numpy()]
    return fold_targets(lt, trait, ev.astype(int), 1)


def tuning_inner_folds(scenario: str, f: int) -> list[int]:
    """Inner folds used as validation sets when a model tunes itself inside the fit set (LightGBM: three of them)."""
    return [0, 1, 2] if scenario in ("cv1", "cv3") else [k for k in range(5) if k != f][:3]


def cv3_traits_of(u: int) -> list:
    return cv3_units().loc[u, "chosen_for"].split(",")


def hard_traits() -> list[str]:
    """Traits with fewer than 5% resistant accessions or fewer than 1,000 genotyped accessions (18 of the 33 traits)."""
    inv = pd.read_csv(ROOT / "results/inventory/trait_inventory.tsv", sep="\t").set_index("trait")
    rare = set(rare_traits())
    return [t for t in T.TRAITS if t in rare or inv.loc[t, "genotyped"] < 1000]


def sparse_traits() -> list[str]:
    """The 18 sparse traits: fewer than 5% resistant accessions or fewer than 1,000 genotyped accessions."""
    return hard_traits()

