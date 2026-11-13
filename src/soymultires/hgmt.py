"""HapGAM-MT: the multi-trait extension of HapGAM (2080/HapGAM/src/hapgam/model.py), plan step 6 (section 3.6).

For each trait t, on its training accessions S_t (targets as for GBLUP: mean score; numeric traits batch-adjusted):

    y_t = mu_t + g_t + F_t(x) + e_t,   g_t ~ N(0, s2g_t K_t),  e_t ~ N(0, s2e_t I)
    F_t(x) = sum_{b in S} A[t, b, :] . phi_b(z_b)      phi_b: one small network per block, SHARED by all traits

The block terms are fitted by generalised least squares under each trait's genomic covariance V_t = K_t / delta_t + I
(delta_t by REML of the null model), the profiled form of fitting them jointly with the polygenic background (as in
HapGAM). With A = 0 the model is exactly per-trait GBLUP. Blocks are screened inside the training data by the share of
the genetic variance carried by their REML-rrBLUP contribution, summed over traits (HapGAM's screen). Loss = mean over
traits of the GLS loss + L2 on the loadings and the first layer (+ optional sparse group lasso by proximal steps).
Prediction for trait t: F_t(new) + m_t + K(new, S_t) alpha_t, alpha_t the BLUP of the residual y_t - F_t(S_t)
(delta_t re-estimated on the residual, as HapGAM does).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HGConfig:
    top_m: int = 64            # blocks receiving a shared network (0 = per-trait GBLUP)
    K: int = 4                 # size of the shared block representation
    hidden: int = 8
    lr: float = 3e-3
    steps: int = 3000
    eval_every: int = 25
    patience: int = 8          # checks without improvement of any trait before stopping
    min_improve: float = 1e-4  # held-out r must rise by this much to count
    min_improve_se: float = 0.0  # ... and by this many standard errors of r, (1 - r^2) / sqrt(n_heldout)
    lam_A2: float = 1e-3       # L2 on the loadings (HapGAM: 1e-3 on the block output weights)
    wd_first: float = 1e-4     # L2 on the first layer (mean of squares, as HapGAM)
    lam_group: float = 0.0     # sparse group lasso (proximal): whole block over all traits
    lam_elem: float = 0.0      # ... and one trait x one block
    blocks: str = "block16"
    screen: str = "sum"         # sum: top_m by the summed variance shares; gwas: by the summed, per-trait normalised
                                # mixed-model score statistic of the block's best SNP; union / gwas_union: each trait's
                                # top per_trait_k blocks by the share / by the score statistic
                                # blocks (filled up / cut to top_m by the summed shares)
    per_trait_k: int = 8
    rare_weight: float = 1.0    # > 1: residuals of resistant accessions of rare traits scaled by sqrt(weight) in
                                # the GLS loss (class weighting, capped as in the plan at 10)
    refit_delta: bool = True
    seed: int = 0
    extra: dict = field(default_factory=dict)


class SharedBlocks(nn.Module):
    """phi_b for the selected blocks (evaluated together with einsum) and per-trait loadings A (zero-initialised)."""

    def __init__(self, mask: torch.Tensor, T: int, cfg: HGConfig):
        super().__init__()
        mask = mask.detach().cpu()                       # build on the CPU (seeded generator); .to(device) afterwards
        M, kmax = mask.shape
        g = torch.Generator().manual_seed(cfg.seed)
        k_eff = mask.sum(1).clamp_min(1)
        self.W1 = nn.Parameter(torch.randn(M, kmax, cfg.hidden, generator=g) * torch.sqrt(2.0 / k_eff)[:, None, None])
        self.b1 = nn.Parameter(torch.randn(M, cfg.hidden, generator=g) * 0.1)
        self.W2 = nn.Parameter(torch.randn(M, cfg.hidden, cfg.K, generator=g) / math.sqrt(cfg.hidden))
        self.A = nn.Parameter(torch.zeros(T, M, cfg.K))
        self.register_buffer("mask", mask)
        self.cfg = cfg

    def phi(self, Zb):
        h = F.gelu(torch.einsum("nmk,mkh->nmh", Zb * self.mask, self.W1) + self.b1)
        return torch.einsum("nmh,mhk->nmk", h, self.W2)

    def forward(self, Zb):
        return torch.einsum("nmk,tmk->nt", self.phi(Zb), self.A)

    def penalty(self):
        c = self.cfg
        return c.lam_A2 * (self.A ** 2).sum() + c.wd_first * (self.W1 ** 2).mean()

    @torch.no_grad()
    def prox(self, step):
        c = self.cfg
        if c.lam_elem > 0:
            nrm = torch.sqrt((self.A ** 2).sum(2, keepdim=True) + 1e-12)
            self.A.mul_(torch.clamp(1 - step * c.lam_elem / nrm, min=0))
        if c.lam_group > 0:
            nrm = torch.sqrt((self.A ** 2).sum((0, 2), keepdim=True) + 1e-12)
            self.A.mul_(torch.clamp(1 - step * c.lam_group / nrm, min=0))


def reml_rotated(s: torch.Tensor, Ur: torch.Tensor, U1: torch.Tensor, grid=(-12.0, 12.0, 97)):
    """REML for y = 1 b + g + e on the eigenbasis of K (s = eigenvalues, Ur = U'y, U1 = U'1), float64 vectors.
    Same objective and search as reml.SpectralREML. Returns (delta, b)."""
    n = s.shape[0]
    s = s.clamp_min(0)

    def ll(ld):
        d = s + math.exp(ld)
        w = 1.0 / d
        a = float((U1 * U1 * w).sum())
        b = float((U1 * Ur * w).sum()) / a
        r = Ur - U1 * b
        s2 = float((r * r * w).sum()) / (n - 1)
        return -0.5 * ((n - 1) * math.log(2 * math.pi * s2) + float(torch.log(d).sum()) + math.log(a) - math.log(n)
                       + (n - 1)), b

    lo, hi, m = grid
    lds = np.linspace(lo, hi, int(m))
    vals = [ll(x)[0] for x in lds]
    i = int(np.nanargmax(vals))
    a_, b_ = lds[max(i - 1, 0)], lds[min(i + 1, len(lds) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    c, d = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    fc, fd = ll(c)[0], ll(d)[0]
    for _ in range(40):
        if fc > fd:
            b_, d, fd = d, c, fc
            c = b_ - gr * (b_ - a_)
            fc = ll(c)[0]
        else:
            a_, c, fc = c, d, fd
            d = a_ + gr * (b_ - a_)
            fd = ll(d)[0]
    ld = (a_ + b_) / 2
    return math.exp(ld), ll(ld)[1]


class TraitGLS:
    """Per-trait pieces: standardised training targets, eigenbasis of K_t, REML delta_t, GLS weights, held-out kernel.
    Large tensors live on the device of Kt (GPU or CPU); the REML searches use small float64 vectors on the CPU."""

    def __init__(self, Kt: torch.Tensor, y: np.ndarray, Kv: torch.Tensor | None, case_w: np.ndarray | None = None):
        dev = Kt.device
        y = np.asarray(y, float)
        self.y_mean, self.y_sd = float(y.mean()), float(y.std() + 1e-8)
        ys = torch.tensor((y - self.y_mean) / self.y_sd, dtype=torch.float64, device=dev)
        s, U = torch.linalg.eigh(Kt.to(torch.float64))
        s = s.clamp_min(0)
        self.n = len(y)
        self.s64 = s.cpu()
        self.U1 = (U.T @ torch.ones(self.n, dtype=torch.float64, device=dev)).cpu()
        self.delta, self.b0 = reml_rotated(self.s64, (U.T @ ys).cpu(), self.U1)
        self.U = U.float()
        del U
        self.s = s.float()
        self.ys = ys.float()
        w = 1.0 / (self.s / self.delta + 1.0)
        self.w, self.wsum = w, float(w.sum())
        self.Kv = None if Kv is None else Kv.to(dev).float()
        self.cw = None if case_w is None else torch.tensor(np.sqrt(case_w), dtype=torch.float32, device=dev)
        self.dev = dev

    def loss(self, F_t):
        r = self.ys - F_t
        if self.cw is not None:
            r = r * self.cw
        Ur = self.U.T @ r
        return (self.w * Ur * Ur).sum() / self.wsum

    @torch.no_grad()
    def blup_parts(self, F_t, refit=False):
        """(m, alpha, delta) for the residual ys - F_t: m the GLS mean, alpha = (K + delta I)^-1 (r - m)."""
        Ur = (self.U.T @ (self.ys - F_t.to(self.dev)).float()).double().cpu()
        if refit:
            delta, m = reml_rotated(self.s64, Ur, self.U1)
        else:
            delta = self.delta
            w = 1.0 / (self.s64 + delta)
            m = float((self.U1 * Ur * w).sum() / (self.U1 * self.U1 * w).sum())
        alpha = self.U @ ((Ur - self.U1 * m) / (self.s64 + delta)).float().to(self.dev)
        return m, alpha, delta

    @torch.no_grad()
    def predict(self, F_t_train, F_t_new, refit=False):
        """Predictions (original scale) for the held-out accessions of this trait."""
        m, alpha, _ = self.blup_parts(F_t_train, refit)
        return ((F_t_new + m + self.Kv @ alpha) * self.y_sd + self.y_mean).cpu().numpy()
