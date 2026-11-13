"""Diagnostic for the proximal loadings update (stage-A data of cv1_inner split 0): gradient norms of the loss with
respect to the loadings at A = 0 (per trait x block and per block), and the largest curvature L of the loss along the
loadings (power iteration on Hessian-vector products), which sets the ISTA step 1/L.

usage: python diag_prox.py [n_blocks]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/03_model"))
import train_mt as tm  # noqa: E402
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.mtmodel import MTConfig, MultiResNet  # noqa: E402

torch.set_num_threads(8)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
cfg = MTConfig(batch_size=0, n_blocks=M, prox=True, opt_A="sgd")
t0 = time.time()
fo = data.folds()
fit, ev = data.split_sets("cv1_inner", 0, fo)
es = fit & (fo.inner_fold == tm.es_fold("cv1_inner", 0))
fd = tm.FoldData(fo.index[fit & ~es], fo.index[es], cfg)
Xfull, mean_f, sd_f, snps, Psi = tm.load_genotypes(fd.accs, cfg.q_pcs)
keep = tm.screen_blocks(fd, Xfull, mean_f, sd_f, snps, cfg.blocks, cfg.n_blocks)
bidx, cols = tm.block_index(snps, cfg.blocks, keep)
X = Xfull[:, torch.as_tensor(cols)].contiguous()
del Xfull
mean, sd = mean_f[cols], sd_f[cols]
Off = tm.gblup_offsets(fd, "cv1_inner_fold0_inner0_A", 8)
dev = torch.device("cpu")
model = MultiResNet(Psi.shape[1], bidx, len(fd.traits), fd.kinds, fd.n_levels, len(fd.batch_id) + 1,
                    fd.batch_random, cfg)
tm.calibrate_head(model, fd, Off, cfg, dev, print)
tr = fd.train
rows = torch.as_tensor(fd.train_rows)
Ztr = (X[rows].float() - mean) / sd
pos = torch.full((len(fd.accs),), -1, dtype=torch.long)
pos[rows] = torch.arange(len(rows))
rpos, rt, ry, rb, rw = pos[tr.row], tr.t, tr.y, tr.b, tr.w
Ptr, Otr = Psi[rows], Off[rows]


def loss_fn():
    eta = model(Ztr, Ptr, Otr)
    return (rw * model.record_nll(eta[rpos, rt], rt, ry, rb)).sum()


print(f"setup {time.time() - t0:.0f}s; {bidx.shape[0]} blocks")
with torch.no_grad():
    phi = model.block_repr(Ztr)
    print("phi sd per block (median, max):", float(phi.std(0).median()), float(phi.std(0).max()))
L0 = loss_fn()
gA, = torch.autograd.grad(L0, model.A)
tb = gA.norm(dim=2)                       # trait x block
bl = gA.norm(dim=(0, 2))                  # block
qs = [0.5, 0.9, 0.99, 1.0]
print("loss at A=0:", float(L0))
print("trait x block gradient norm quantiles", {q: round(float(torch.quantile(tb.flatten(), q)), 5) for q in qs})
print("block gradient norm quantiles", {q: round(float(torch.quantile(bl, q)), 5) for q in qs})
for t in ("NEMATCYST_3", "NEMATCYST_14", "PHYTOROT_1", "PHYTOROT_20", "SDS", "HIGHTEMP"):
    ti = fd.traits.index(t)
    v = tb[ti]
    print(f"  {T.SHORT[t]:6s} max {float(v.max()):.5f}  99% {float(torch.quantile(v, 0.99)):.5f}  median {float(v.median()):.5f}")
# largest curvature along A: power iteration with Hessian-vector products
v = torch.randn_like(model.A)
v /= v.norm()
lam = 0.0
for it in range(25):
    L = loss_fn()
    g, = torch.autograd.grad(L, model.A, create_graph=True)
    hv, = torch.autograd.grad((g * v).sum(), model.A)
    lam = float((hv * v).sum())
    v = hv / hv.norm()
print(f"largest curvature L ~ {lam:.3f} -> ISTA step 1/L = {1 / lam:.5f}; {time.time() - t0:.0f}s")
