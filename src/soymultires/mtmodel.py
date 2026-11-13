"""Multi-task model of the plan (section 3.6, Fig. 7): shared haplotype-block representations + per-trait loadings.

eta[i, t] = mu[t] + M^-1/2 sum_b A[t, b, :] . phi_b(z_ib) + (psi_i U) V[:, t]   (+ batch term inside the likelihood)

  phi_b    one hidden layer (GELU) per block, block SNPs -> K-dim block representation shared by all traits
  A        per-trait loadings on each block; sparse group lasso (whole block, and trait x block) by proximal steps
           -> exact zeros: shared / specific / inactive blocks
  U, V     low-rank shared linear background on the top genomic principal components psi (gpcs_<q>.npy,
           psi psi' ~ Z Z' / p): a reduced-rank, kernel-type multi-trait background (the polygenic part)
  blocks   the top-M blocks by a marginal multi-trait screen inside the training data (M = 0: all blocks)
Likelihood, only on measured cells (decision 1):
  categorical traits  cumulative-logit (ordinal) with per-trait cut points; batch random intercept (L2-penalised);
                      classes weighted by frequency (resistant classes up to 10x)
  numeric traits      Gaussian with per-trait sigma; batch fixed effect (unpenalised)
Genotypes enter as standardised dosages z = (x - 2p) / sqrt(2p(1-p)) with allele frequencies of all 20,087 accessions
(phenotype-free). Runs on CPU or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MTConfig:
    K: int = 4                 # block representation size
    H: int = 8                 # hidden units per block network
    R: int = 32                # rank of the linear background
    lam_group: float = 1e-3    # sparse group lasso: whole block (all traits)
    lam_elem: float = 1e-3     # sparse group lasso: one trait x one block
    lam_bg: float = 1e-4       # ridge on the background U, V
    q_pcs: int = 1000          # genomic principal components feeding the background
    background: str = "gblup"  # gblup = per-trait GBLUP offset fitted in the training split (plan option a);
                               # pcs = low-rank shared background on genomic PCs; both = the two together
    clip: float = 10.0         # gradient-norm clipping
    block_scale: str = "none"  # none: block effects enter as A . phi; sqrtM: divided by sqrt(M)
    prox: bool = True          # sparse group lasso by proximal steps after each update (exact zeros), not in the loss
    lr_head: float = 1e-3      # learning rate of mu, beta, cut points, sigma, batch effects during training
    head_steps: int = 300      # full-batch calibration of those parameters before training (offset model only)
    head_lr: float = 0.05
    lam_A2: float = 0.0        # L2 on the loadings (HapGAM: 1e-3 on the block output weights)
    opt_A: str = "sgd"         # loadings: sgd = proximal gradient (steps proportional to the gradient; with prox =
                               # ISTA for the sparse group lasso); adam = same optimiser as the block networks
    lr_A: float = 1.0
    mom_A: float = 0.0
    min_improve: float = 1e-4  # per-trait held-out loss must drop by this much to count as an improvement
    eval_every: int = 25       # full batch (batch_size <= 0): steps between held-out checks; patience counts checks
    lam_batch: float = 1e-1    # ridge on categorical batch intercepts
    weight_decay: float = 1e-4 # AdamW on block networks
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 25
    class_weight_cap: float = 10.0
    seed: int = 0
    blocks: str = "block16"
    n_blocks: int = 1000       # 0 = all blocks; otherwise top-M by marginal multi-trait association
    extra: dict = field(default_factory=dict)


class TaskData:
    """Record-level targets for a set of accessions (rows of the genotype matrix)."""

    def __init__(self, recs, traits, kinds, n_levels):
        # recs: DataFrame with columns row (accession index in the model's accession list), t (trait index),
        #       y (ordinal class index or standardised score), batch_id (global id), w (record weight)
        self.traits, self.kinds, self.n_levels = traits, kinds, n_levels
        self.row = torch.tensor(recs.row.to_numpy(), dtype=torch.long)
        self.t = torch.tensor(recs.t.to_numpy(), dtype=torch.long)
        self.y = torch.tensor(recs.y.to_numpy(), dtype=torch.float32)
        self.b = torch.tensor(recs.batch_id.to_numpy(), dtype=torch.long)
        self.w = torch.tensor(recs.w.to_numpy(), dtype=torch.float32)


class MultiResNet(nn.Module):
    def __init__(self, q_pcs, block_idx, T, kinds, n_levels, n_batch, batch_is_random, cfg: MTConfig):
        super().__init__()
        g = torch.Generator().manual_seed(cfg.seed)
        M, kmax = block_idx.shape
        self.register_buffer("block_idx", torch.as_tensor(block_idx, dtype=torch.long))      # -1 = padding
        self.register_buffer("block_mask", (torch.as_tensor(block_idx) >= 0).float())
        k_eff = self.block_mask.sum(1).clamp_min(1)
        self.W1 = nn.Parameter(torch.randn(M, kmax, cfg.H, generator=g) * torch.sqrt(2.0 / k_eff)[:, None, None])
        self.b1 = nn.Parameter(torch.zeros(M, cfg.H))
        self.W2 = nn.Parameter(torch.randn(M, cfg.H, cfg.K, generator=g) / np.sqrt(cfg.H))
        self.A = nn.Parameter(torch.zeros(T, M, cfg.K))                                    # start at zero (HapGAM)
        self.U = nn.Parameter(torch.randn(q_pcs, cfg.R, generator=g) / np.sqrt(cfg.R))
        self.M_scale = 1.0 / np.sqrt(M) if cfg.block_scale == "sqrtM" else 1.0
        self.V = nn.Parameter(torch.zeros(cfg.R, T))
        self.mu = nn.Parameter(torch.zeros(T))
        self.beta = nn.Parameter(torch.ones(T))           # scale of the GBLUP offset on each trait's latent scale
        self.use_pcs = cfg.background in ("pcs", "both")
        self.use_off = cfg.background in ("gblup", "both")
        self.register_buffer("kinds", torch.tensor([1 if k == "numeric" else 0 for k in kinds]))
        self.Lmax = max(n_levels)
        cut0 = torch.zeros(T, max(self.Lmax - 1, 1))
        for t, L in enumerate(n_levels):                       # evenly spaced initial cut points
            if L > 1:
                cut0[t, : L - 1] = torch.linspace(-1, 1, L - 1) if L > 2 else torch.zeros(1)
        self.cut_first = nn.Parameter(cut0[:, :1].clone())
        self.cut_gaps = nn.Parameter(torch.full((T, max(self.Lmax - 2, 1)), -0.5))
        self.register_buffer("n_levels", torch.tensor(n_levels))
        self.log_sigma = nn.Parameter(torch.zeros(T))
        self.batch_eff = nn.Parameter(torch.zeros(n_batch))
        self.register_buffer("batch_random", torch.as_tensor(batch_is_random, dtype=torch.bool))
        self.cfg = cfg

    # ---------------------------------------------------------------- forward
    def block_repr(self, Z):
        """Z: (B, p) standardised dosages -> phi (B, M, K)."""
        idx = self.block_idx.clamp_min(0)
        Xb = Z[:, idx] * self.block_mask                                                    # (B, M, kmax)
        h = F.gelu(torch.einsum("bmk,mkh->bmh", Xb, self.W1) + self.b1)
        return torch.einsum("bmh,mhk->bmk", h, self.W2)

    def forward(self, Zsel, Psi, Off=None):
        """Zsel: standardised dosages of the selected blocks' SNPs (B, p_sel); Psi: genomic PCs (B, q);
        Off: per-trait GBLUP offsets standardised per trait (B, T)."""
        phi = self.block_repr(Zsel)
        eta = self.mu + torch.einsum("bmk,tmk->bt", phi, self.A) * self.M_scale
        if self.use_pcs:
            eta = eta + (Psi @ self.U) @ self.V
        if self.use_off and Off is not None:
            eta = eta + self.beta * Off
        return eta                                                                          # (B, T)

    def cuts(self):
        gaps = F.softplus(self.cut_gaps)
        return torch.cat([self.cut_first, self.cut_first + torch.cumsum(gaps, 1)], 1)[:, : max(self.Lmax - 1, 1)]

    # ---------------------------------------------------------------- likelihood
    def record_nll(self, eta_rec, t, y, b):
        """Negative log-likelihood of each record (eta_rec = eta[row, t] of the record)."""
        e = eta_rec + self.batch_eff[b]
        num = self.kinds[t] == 1
        nll = torch.zeros_like(e)
        if num.any():
            s = self.log_sigma[t[num]]
            nll[num] = 0.5 * ((y[num] - e[num]) / torch.exp(s)) ** 2 + s
        cat = ~num
        if cat.any():
            tc, yc, ec = t[cat], y[cat].long(), e[cat]
            cuts = self.cuts()[tc]                                                          # (n, Lmax-1)
            L = self.n_levels[tc]
            cdf = torch.sigmoid(cuts - ec[:, None])
            valid = torch.arange(cuts.shape[1], device=cuts.device)[None, :] < (L[:, None] - 1)
            cdf = torch.where(valid, cdf, torch.ones_like(cdf))
            lo = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
            hi = torch.cat([cdf, torch.ones_like(cdf[:, :1])], 1)
            p = (hi - lo).gather(1, yc[:, None]).squeeze(1)
            nll[cat] = -torch.log(p.clamp_min(1e-8))
        return nll

    def head_parameters(self):
        return [self.mu, self.beta, self.cut_first, self.cut_gaps, self.log_sigma, self.batch_eff]

    def penalty(self):
        c = self.cfg
        pen = torch.zeros((), device=self.A.device)
        if not c.prox:
            pen = pen + c.lam_group * torch.sqrt((self.A ** 2).sum((0, 2)) + 1e-12).sum()
            pen = pen + c.lam_elem * torch.sqrt((self.A ** 2).sum(2) + 1e-12).sum()
        pen = pen + c.lam_bg * ((self.U ** 2).sum() + (self.V ** 2).sum()) + c.lam_A2 * (self.A ** 2).sum()
        pen = pen + c.lam_batch * (self.batch_eff[self.batch_random] ** 2).sum()
        return pen

    @torch.no_grad()
    def prox_A(self, step):
        """Proximal step of the sparse group lasso on the loadings (Simon et al. 2013): soft-threshold each trait x
        block vector, then each block (all traits). Exact zeros = blocks / trait-block pairs switched off."""
        c = self.cfg
        if c.lam_elem > 0:
            nrm = torch.sqrt((self.A ** 2).sum(2, keepdim=True) + 1e-12)
            self.A.mul_(torch.clamp(1 - step * c.lam_elem / nrm, min=0))
        if c.lam_group > 0:
            nrm = torch.sqrt((self.A ** 2).sum((0, 2), keepdim=True) + 1e-12)
            self.A.mul_(torch.clamp(1 - step * c.lam_group / nrm, min=0))

    @torch.no_grad()
    def expected_score(self, eta):
        """Prediction on the response scale: E[class index] for categorical traits, eta for numeric traits."""
        out = eta.clone()
        cuts = self.cuts()
        for t in range(eta.shape[1]):
            L = int(self.n_levels[t])
            if int(self.kinds[t]) == 1 or L < 2:
                continue
            cdf = torch.sigmoid(cuts[t, : L - 1][None, :] - eta[:, t:t + 1])
            lo = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
            hi = torch.cat([cdf, torch.ones_like(cdf[:, :1])], 1)
            probs = hi - lo
            out[:, t] = (probs * torch.arange(L, device=eta.device)[None, :]).sum(1)
        return out


def block_contributions(model: MultiResNet, X_int8, mean, sd, device, chunk=2048):
    """Variance over accessions of each block's effect on each trait: (T, M). X_int8 = raw dosages of the selected
    blocks' SNPs (CPU tensor), mean/sd the matching standardisation vectors."""
    T, M, _ = model.A.shape
    s1 = torch.zeros(T, M, device=device, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    n = 0
    with torch.no_grad():
        for i in range(0, X_int8.shape[0], chunk):
            Z = (X_int8[i:i + chunk].to(device).float() - mean) / sd
            phi = model.block_repr(Z)
            eff = (torch.einsum("bmk,tmk->btm", phi, model.A) * model.M_scale).double()
            s1 += eff.sum(0)
            s2 += (eff ** 2).sum(0)
            n += eff.shape[0]
    return (s2 / n - (s1 / n) ** 2).cpu().numpy()
