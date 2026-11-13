"""MT-Boost: multi-task gradient boosting for the 32 stress traits (plan step 6 as revised on 2026-11-12, see
docs/FREEZE_DEVIATIONS.md). GPU implementation (gpu10 card 0), written for this project.

For trait t (standardised targets, squared loss):

    F_t(x) = GBLUP_t(x) + eta_s * sum_{m <= S_t} v_{m,t}(x) + eta_p * sum_{m <= P_t} u_{m,t}(x)

  GBLUP_t   cross-fitted per-trait GBLUP (start of the boosting; the polygenic part)
  v_m       shared trees: one tree per trait group and round; the split is shared by the traits of the group, every
            trait has its own leaf value (vector leaves). Split gain = sum over the group's traits, each trait weighted
            so that all traits of the group count equally (c_t = mean n / n_t).
  u_m       trait-specific trees (one per trait and round)
  S_t, P_t  rounds of shared trees borrowed (0 = none) and of own trees, chosen per trait on inner validation

Trees are grown depth-wise with second-order statistics (leaf value -G / (H + lambda), split gain as in XGBoost).
Genotypes are inbred-line dosages (heterozygotes 0.01% of calls), so a split is "carries the alternative allele"
(x >= 1) on one SNP, and the gradient/Hessian histograms of a whole tree level for all units are one matrix product
B' [g, h] with B = 1[x >= 1] (n x p) -- the reason this runs fast on a GPU.
Both kinds of trees use the same batched grower: an ensemble is a set of "units" (a trait group, or a single trait),
each unit growing its own tree from the gradients of its traits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e30


@dataclass
class BoostConfig:
    groups: str = "category"     # category (traits.STRESS category) | all | none (no shared trees)
    start_scale: float = 1.0     # boosting starts from start_scale x GBLUP (0 = from zero, as LightGBM)
    depth_s: int = 4
    depth_p: int = 4
    eta_s: float = 0.05
    eta_p: float = 0.05
    lam: float = 1.0             # L2 on leaf values
    gamma: float = 0.0           # minimum split gain
    mcw_s: float = 20.0          # minimum child weight of shared trees (balanced Hessian, in record units)
    mcw_p: float = 20.0          # minimum child weight of own trees (records)
    colsample: float = 0.12      # share of SNPs considered per tree
    subsample: float = 0.8       # share of accessions per tree
    rounds_s: int = 600
    rounds_p: int = 600
    check_every: int = 10
    patience: int = 15           # checks in which no trait improved by more than stop_tol -> stop the phase
    stop_tol: float = 1e-3
    seed: int = 0
    extra: dict = field(default_factory=dict)


class Engine:
    """Genotypes of the training rows on the device: X (int8, for routing) and Bt = 1[x >= 1]' (p x n, float32, for
    the histograms; transposed so that SNP chunks are contiguous). chunk: SNPs per histogram block (memory bound)."""

    def __init__(self, X: torch.Tensor, chunk: int = 40000):
        self.X = X
        self.Bt = (X >= 1).T.contiguous().float()
        self.n, self.p = X.shape
        self.chunk = chunk


def grow(eng: Engine, g, h, unit_of, U, depth, lam, gamma, mcw, colmask, imp=None):
    """Grow one depth-wise tree per unit. g, h: (n, T) gradients / Hessians (zero = not used). unit_of: (T,) unit of
    every trait. Returns feats (U, 2^D - 1), ok (U, 2^D - 1) split flags, leaf (T, 2^D) values, lidx (n, T) leaf index
    of every training row for every trait. imp (T, p), if given, accumulates every trait's gain at the chosen splits.
    Histograms are computed per block of eng.chunk SNPs (bounded memory)."""
    n, T = g.shape
    dev = g.device
    node = torch.zeros(n, U, dtype=torch.long, device=dev)
    nodes = 2 ** depth - 1
    feats = torch.zeros(U, nodes, dtype=torch.long, device=dev)
    oks = torch.zeros(U, nodes, dtype=torch.bool, device=dev)
    Uoh = F.one_hot(unit_of, U).float()                                     # T x U
    ar_u = torch.arange(U, device=dev)[None, :]
    for d in range(depth):
        K = 2 ** d
        oh = F.one_hot(node[:, unit_of], K).float()                         # n x T x K
        GH = torch.cat([(oh * g[:, :, None]).reshape(n, T * K), (oh * h[:, :, None]).reshape(n, T * K)], 1)
        del oh
        Gt, Ht = GH[:, :T * K].sum(0).reshape(T, K), GH[:, T * K:].sum(0).reshape(T, K)
        base = (Gt ** 2 / (Ht + lam))[None]
        best = torch.full((U, K), NEG, device=dev)
        bestf = torch.zeros(U, K, dtype=torch.long, device=dev)
        for c0 in range(0, eng.p, eng.chunk):
            R = eng.Bt[c0:c0 + eng.chunk] @ GH                              # c x 2TK  (sums over x >= 1)
            GR, HR = R[:, :T * K].reshape(-1, T, K), R[:, T * K:].reshape(-1, T, K)
            GL, HL = Gt[None] - GR, Ht[None] - HR
            term = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - base       # c x T x K
            gain = torch.einsum("ptk,tu->puk", term, Uoh)
            valid = (torch.einsum("ptk,tu->puk", HL, Uoh) >= mcw) & (torch.einsum("ptk,tu->puk", HR, Uoh) >= mcw)
            valid &= colmask[c0:c0 + eng.chunk, None, None]
            gain = torch.where(valid, gain, torch.full_like(gain, NEG))
            cb, cf = gain.max(0)                                            # U x K
            upd = cb > best
            best = torch.where(upd, cb, best)
            bestf = torch.where(upd, cf + c0, bestf)
            del R, GR, HR, GL, HL, term, gain, valid
        ok = best > gamma
        f = torch.where(ok, bestf, torch.zeros_like(bestf))
        sl = slice(2 ** d - 1, 2 ** (d + 1) - 1)
        feats[:, sl], oks[:, sl] = f, ok
        if imp is not None:
            ft = f[unit_of]                                                 # T x K: chosen SNP of each trait's node
            Rs = eng.Bt[ft.reshape(-1)] @ GH                                # TK x 2TK
            GRs = Rs[:, :T * K].reshape(T, K, T, K)
            HRs = Rs[:, T * K:].reshape(T, K, T, K)
            ti = torch.arange(T, device=dev)[:, None].expand(T, K)
            ki = torch.arange(K, device=dev)[None, :].expand(T, K)
            gr, hr = GRs[ti, ki, ti, ki], HRs[ti, ki, ti, ki]               # own trait, own node
            gl, hl = Gt - gr, Ht - hr
            tv = gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - Gt ** 2 / (Ht + lam)
            tv = torch.where(ok[unit_of], tv, torch.zeros_like(tv))
            imp.scatter_add_(1, ft, tv)
        fi, oki = f[ar_u.expand(n, U), node], ok[ar_u.expand(n, U), node]   # n x U
        x = eng.X.gather(1, fi)
        node = 2 * node + ((x >= 1) & oki).long()
    L = 2 ** depth
    lidx = node[:, unit_of]                                                 # n x T
    oh = F.one_hot(lidx, L).float()
    G, H = (oh * g[:, :, None]).sum(0), (oh * h[:, :, None]).sum(0)         # T x L
    leaf = -G / (H + lam)
    return feats, oks, leaf, lidx


def route(Xq: torch.Tensor, feats, oks, unit_of, depth):
    """Leaf index (m, T) of new rows Xq (int8) in one stored tree per unit."""
    m = Xq.shape[0]
    U = feats.shape[0]
    node = torch.zeros(m, U, dtype=torch.long, device=Xq.device)
    ar_u = torch.arange(U, device=Xq.device)[None, :].expand(m, U)
    for d in range(depth):
        idx = (2 ** d - 1) + node
        x = Xq.gather(1, feats[ar_u, idx])
        node = 2 * node + ((x >= 1) & oks[ar_u, idx]).long()
    return node[:, unit_of]


def leaf_values(leaf, lidx):
    T = leaf.shape[0]
    return leaf[torch.arange(T, device=leaf.device)[None, :], lidx]


def pearson_cols(P, Y, M):
    """Per-column Pearson r between predictions P and targets Y over rows where M (all (m, T)); nan if < 3 rows."""
    Mf = M.float()
    n = Mf.sum(0)
    mp = (P * Mf).sum(0) / n.clamp_min(1)
    my = (Y * Mf).sum(0) / n.clamp_min(1)
    dp, dy = (P - mp) * Mf, (Y - my) * Mf
    r = (dp * dy).sum(0) / torch.sqrt((dp ** 2).sum(0) * (dy ** 2).sum(0)).clamp_min(1e-12)
    r[n < 3] = float("nan")
    return r


def run_phase(eng, F0, Y, W, unit_of, U, depth, eta, lam, gamma, mcw, colsample, subsample, rounds, gen,
              val_idx=None, Yv=None, Mv=None, Xe=None, Fe0=None, snaps=None, check_every=10, patience=15, imp=None,
              keep_trees=True, stop_tol=1e-3):
    """Boost one ensemble (shared units or own trees) from the predictions F0 of the n training-matrix rows.
    W (n, T): loss weights (0 = cell not used; validation rows have weight 0, so their predictions are updated for
    free by the leaf assignment). With val_idx (rows) and targets Yv / mask Mv, per-trait r is tracked every
    check_every rounds and training stops when no trait has improved for `patience` checks (or after `rounds`).
    Xe / Fe0: other rows (eval accessions) routed through every new tree; snaps {round: [traits]} records their
    predictions for those traits at that round (training then runs to the largest requested round).
    Returns dict(F, Fe, trees, snap, curves (checks x T, numpy) or None)."""
    F_ = F0.clone()
    Fe = None if Fe0 is None else Fe0.clone()
    snap = None if (snaps is None or Fe is None) else Fe.clone()
    n, T = F_.shape
    dev = F_.device
    trees, curves = [], []
    best, bad = None, 0
    if val_idx is not None:
        r0 = pearson_cols(F_[val_idx], Yv, Mv)
        curves.append(r0)
        best = torch.nan_to_num(r0, nan=-2.0)
    n_rounds = rounds if snaps is None else max(snaps)
    for r in range(1, n_rounds + 1):
        rowm = (torch.rand(n, generator=gen, device=dev) < subsample).float()[:, None]
        g = (F_ - Y) * W * rowm
        h = W * rowm
        colmask = torch.rand(eng.p, generator=gen, device=dev) < colsample
        feats, oks, leaf, lidx = grow(eng, g, h, unit_of, U, depth, lam, gamma, mcw, colmask, imp)
        F_ += eta * leaf_values(leaf, lidx)
        if Fe is not None:
            Fe += eta * leaf_values(leaf, route(Xe, feats, oks, unit_of, depth))
        if keep_trees:
            trees.append((feats, oks, leaf))
        if snap is not None and r in snaps:
            cols = torch.as_tensor(snaps[r], device=dev, dtype=torch.long)
            snap[:, cols] = Fe[:, cols]
        if val_idx is not None and r % check_every == 0:
            rv = pearson_cols(F_[val_idx], Yv, Mv)
            curves.append(rv)
            imp_ = (rv > best + stop_tol) & ~torch.isnan(rv)
            if bool(imp_.any()):
                best = torch.where(imp_, rv, best)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return {"F": F_, "Fe": Fe, "trees": trees, "snap": snap,
            "curves": torch.stack(curves).cpu().numpy() if curves else None}


def replay(trees, Xq, unit_of, depth, eta, upto, F0q):
    """Predictions of stored trees on Xq, each trait t summed over its first upto[t] rounds (upto: (T,) ints)."""
    Fq = F0q.clone()
    up = torch.as_tensor(upto, device=Fq.device)
    for m, (feats, oks, leaf) in enumerate(trees, start=1):
        if m > int(up.max()):
            break
        v = leaf_values(leaf, route(Xq, feats, oks, unit_of, depth))
        Fq += eta * v * (m <= up).float()[None, :]
    return Fq
