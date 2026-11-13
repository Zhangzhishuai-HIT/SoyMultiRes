"""Step 6: the multi-task model (src/soymultires/mtmodel.py) on the shared scenarios (soymultires.data.split_sets).

usage:
  python train_mt.py fit --scenario cv1_inner --fold 0 --config '{"K": 4}' --tag k4 [--stage A|AB] [--seeds 0,1,2,3,4]
  python train_mt.py collect --scenario cv1_inner --tag k4        -> runs/<scenario>/mtnet_<tag>/<trait>.tsv.gz
  python train_mt.py report --scenario cv1_inner --tags k2,k4,k8  -> stage-A comparison of settings (pooled over splits)

Two stages per split, the way LightGBM is tuned and refitted:
  A  fit set minus one inner fold g (g = f for cv1, f + 1 mod 5 for cv1_inner); early stopping on g gives the number of
     epochs E. Settings are chosen on the stage-A predictions for g pooled over the five splits, never on the eval set.
  B  whole fit set, E epochs, one model per seed, predictions averaged, evaluated on the eval set (cv1: the test fold).
Outputs: runs/<scenario>/mtnet_dev/<tag>/<split>.A.{json,tsv.gz} and <split>.B.{json,tsv.gz,contrib.npy}.
Stage-A records carry pred_offset, the cross-fitted GBLUP offset, so GBLUP is scored on the same accessions.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data, evaluate  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.mtmodel import MTConfig, MultiResNet, TaskData, block_contributions  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402

CODE = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/soymultires/mtmodel.py").read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------ data assembly
class FoldData:
    def __init__(self, train_accs, eval_accs, cfg: MTConfig):
        long = data.pheno_long()
        self.traits = list(T.TRAITS)
        self.kinds = [T.kind(t) for t in self.traits]
        tr = long[long.acc.isin(set(train_accs))].copy()
        ev = long[long.acc.isin(set(eval_accs))].copy()
        self.accs = np.array(sorted(set(tr.acc) | set(ev.acc)))
        self.row_of = {a: i for i, a in enumerate(self.accs)}
        # targets: ordinal class index (categorical) or standardised score (numeric), from training records only
        self.levels, self.n_levels, self.scale = {}, [], {}
        for t in self.traits:
            if t in T.NUMERIC:
                s = tr.score[tr.trait == t]
                self.scale[t] = (float(s.mean()), float(s.std()))
                self.n_levels.append(1)
            else:
                lv = sorted(set(T.levels(t).values()))
                self.levels[t] = np.array(lv, float)
                self.n_levels.append(len(lv))
        # batch ids: 0 = unseen batch (effect stays 0)
        keys = sorted(set(zip(tr.trait, tr.batch)))
        self.batch_id = {k: i + 1 for i, k in enumerate(keys)}
        self.batch_random = [False] + [k[0] not in T.NUMERIC for k in keys]
        self.class_w = {}
        self.train = self._records(tr, cfg, fit_weights=True)
        self.eval_df = ev
        self.eval = self._records(ev, cfg, fit_weights=False)
        self.train_rows = np.array(sorted(set(self.train.row.tolist())))
        self.eval_rows = np.array(sorted(set(self.eval.row.tolist())))

    def _records(self, df, cfg, fit_weights):
        t_idx = df.trait.map({t: i for i, t in enumerate(self.traits)}).to_numpy()
        y = np.zeros(len(df))
        for t in self.traits:
            m = (df.trait == t).to_numpy()
            if not m.any():
                continue
            if t in T.NUMERIC:
                mu, sd = self.scale[t]
                y[m] = (df.score[m].to_numpy() - mu) / sd
            else:
                lv = self.levels[t]
                sc = df.score[m].to_numpy()
                y[m] = np.abs(sc[:, None] - lv[None, :]).argmin(1)          # nearest level (replicate means)
        rec = pd.DataFrame({"row": df.acc.map(self.row_of).to_numpy(), "t": t_idx, "y": y,
                            "batch_id": [self.batch_id.get(k, 0) for k in zip(df.trait, df.batch)]})
        if fit_weights:
            w = np.ones(len(rec))
            for ti, t in enumerate(self.traits):
                m = (rec.t == ti).to_numpy()
                n_t = m.sum()
                if not n_t:
                    continue
                if t not in T.NUMERIC:
                    cls = rec.y[m].astype(int).to_numpy()
                    freq = np.bincount(cls, minlength=self.n_levels[ti]) / n_t
                    present = freq > 0
                    cw = np.where(present, np.minimum(cfg.class_weight_cap,
                                                      (1.0 / present.sum()) / np.maximum(freq, 1e-12)), 0)
                    wt = cw[cls]
                    cw = cw / wt.mean()
                    self.class_w[ti] = np.where(present, cw, cw[present].max() if present.any() else 1.0)
                    wt = wt / wt.mean()
                else:
                    wt = np.ones(n_t)
                w[m] = wt / n_t                                     # each trait contributes its mean NLL
            rec["w"] = w
        else:                                                       # eval: the same class weights as in training
            w = np.ones(len(rec))
            for ti, cw in self.class_w.items():
                m = (rec.t == ti).to_numpy()
                w[m] = cw[np.minimum(rec.y[m].astype(int).to_numpy(), len(cw) - 1)]
            rec["w"] = w
        return TaskData(rec, self.traits, self.kinds, self.n_levels)


def load_genotypes(accs, q):
    pos = {v: i for i, v in enumerate(data.samples())}
    rows = np.array([pos[a] for a in accs])
    G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")
    o = np.argsort(rows)
    X = np.empty((len(rows), G.shape[1]), dtype=np.int8)
    X[o] = G[rows[o]]
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    p = snps.alt_freq.to_numpy()
    mean = torch.tensor(2 * p, dtype=torch.float32)
    sd = torch.tensor(np.sqrt(np.maximum(2 * p * (1 - p), 1e-6)), dtype=torch.float32)
    psi = torch.from_numpy(np.load(data.P / f"geno/gpcs_{q}.npy", mmap_mode="r")[rows].astype(np.float32))
    return torch.from_numpy(X), mean, sd, snps, psi


def block_index(snps, scheme, keep=None):
    """Selected blocks -> (index matrix into the selected-SNP columns, -1 = padding; selected SNP column list)."""
    b = snps[scheme].to_numpy()
    groups = [np.flatnonzero(b == k) for k in np.unique(b)]
    if keep is not None:
        groups = [groups[i] for i in keep]
    cols = np.concatenate(groups)
    kmax = max(len(g) for g in groups)
    idx = -np.ones((len(groups), kmax), dtype=np.int64)
    start = 0
    for i, g in enumerate(groups):
        idx[i, : len(g)] = np.arange(start, start + len(g))
        start += len(g)
    return idx, cols


def screen_blocks(fd, X, mean, sd, snps, scheme, M, chunk=4000):
    """Top-M blocks by the sum over traits of the best squared marginal correlation inside the block (training data
    only; SNPs processed in chunks to keep memory low)."""
    tr = fd.train
    targets = []
    for ti in range(len(fd.traits)):
        m = (tr.t == ti).numpy()
        if m.sum() < 20:
            continue
        y = pd.Series(tr.y.numpy()[m]).groupby(tr.row.numpy()[m]).mean()
        yv = torch.tensor(y.to_numpy() - y.mean(), dtype=torch.float32)
        targets.append((torch.as_tensor(y.index.to_numpy()), yv / yv.norm().clamp_min(1e-6)))
    score = torch.zeros(X.shape[1])
    for c0 in range(0, X.shape[1], chunk):
        sl = slice(c0, c0 + chunk)
        for rows, yv in targets:
            Zc = (X[rows, sl].float() - mean[sl]) / sd[sl]
            Zc = Zc - Zc.mean(0)
            r = (Zc.T @ yv) / torch.sqrt((Zc ** 2).sum(0)).clamp_min(1e-6)
            score[sl] += r ** 2
    b = snps[scheme].to_numpy()
    bs = pd.Series(score.numpy()).groupby(b).max().to_numpy()
    return np.sort(np.argsort(-bs)[:M])



def gblup_offsets(fd, cache_name, threads, log=print):
    """Per-trait GBLUP offsets (plan step 6, background option a; per-trait GBLUP stands in until the step-5
    multi-trait model is fitted), cross-fitted as in stacking so that training accessions never see their own
    records: the training accessions are split by inner_fold (near-identical groups never split); each part is
    predicted from a GBLUP fitted on the other parts (numeric batch means also from those parts only); the eval
    accessions are predicted from a GBLUP fitted on all training accessions. Standardised per trait with the mean and
    sd of the out-of-fold predictions of the measured training accessions. Cached in work/mt_offsets/."""
    cache = ROOT / "work/mt_offsets" / f"{cache_name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["accs"]) == list(fd.accs):
            log(f"GBLUP offsets from {cache.name}")
            return torch.from_numpy(z["off"])
    torch.set_num_threads(threads)
    long = data.pheno_long()
    inner_of = data.folds().inner_fold
    train_accs = fd.accs[fd.train_rows]
    eval_accs = fd.accs[fd.eval_rows]
    tr = long[long.acc.isin(set(train_accs))]
    part_of = inner_of.reindex(train_accs)
    parts = sorted(part_of.unique())
    pos = {v: i for i, v in enumerate(data.samples())}
    loc = {a: i for i, a in enumerate(fd.accs)}
    Kmm = data.grm(mmap=True)

    def fit_predict(ft, target_accs):
        trows = np.array([pos[x] for x in ft.train_acc])
        Ktr = torch.from_numpy(np.asarray(Kmm[np.ix_(trows, trows)], dtype=np.float64))
        m = SpectralREML().fit(Ktr, torch.tensor(ft.y_train, dtype=torch.float64))
        del Ktr
        rows = np.array([pos[x] for x in target_accs])
        Kx = torch.from_numpy(np.asarray(Kmm[np.ix_(rows, trows)], dtype=np.float64))
        return m.predict(Kx).numpy()

    off = np.zeros((len(fd.accs), len(fd.traits)), dtype=np.float32)
    for ti, t in enumerate(fd.traits):
        lt = tr[tr.trait == t]
        if lt.empty:
            continue
        pred = np.zeros(len(fd.accs))
        for c in parts:                                            # out-of-fold predictions for training accessions
            ft = data.fold_targets(lt, t, part_of, c)
            tgt = train_accs[part_of.to_numpy() == c]
            pred[[loc[x] for x in tgt]] = fit_predict(ft, tgt)
        if len(eval_accs):                                         # eval accessions: fit on all training accessions
            ft = data.fold_targets(lt, t, pd.Series(0, index=pd.Index(train_accs)), 1)
            pred[[loc[x] for x in eval_accs]] = fit_predict(ft, eval_accs)
        meas = np.array([loc[x] for x in lt.acc.unique()])
        mu_, sd_ = pred[meas].mean(), pred[meas].std() + 1e-8
        off[:, ti] = (pred - mu_) / sd_
        log(f"offsets {t}: {len(meas)} measured training accessions, {len(parts)} parts + full fit")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, accs=fd.accs, off=off)
    return torch.from_numpy(off)


# ------------------------------------------------------------------ training
def calibrate_head(model, fd, Off, cfg, device, log=None):
    """Full-batch fit of the head (mu, offset scale beta, cut points, sigma, batch effects) with the blocks and the
    background at zero, i.e. the calibrated offset-only model: training starts from it and early stopping can return
    it (epoch 0) when no block improves the held-out loss."""
    tr = fd.train
    row, t, y, b, w = (v.to(device) for v in (tr.row, tr.t, tr.y, tr.b, tr.w))
    off = Off.to(device)[row, t]
    opt = torch.optim.Adam(model.head_parameters(), lr=cfg.head_lr)
    for _ in range(cfg.head_steps):
        eta = model.mu[t] + (model.beta[t] * off if model.use_off else 0.0)
        loss = (w * model.record_nll(eta, t, y, b)).sum() + cfg.lam_batch * (model.batch_eff[model.batch_random] ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if log:
        log(f"head calibrated: training loss {float(loss):.4f}; beta median {float(model.beta.median()):.3f}")


def run_fullbatch(model, fd, X, mean, sd, Psi, Off, cfg, device, max_steps, early=True, snap_steps=(), log=None):
    """Full-batch training of all traits together (HapGAM's recipe).

    Loadings A: proximal gradient (cfg.opt_A = sgd; ISTA steps of the sparse group lasso) or Adam; block networks and
    background: Adam; head: Adam with cfg.lr_head. early=True: held-out check every cfg.eval_every steps; each trait
    keeps the predictions of its own best check (step 0 = calibrated offset-only model); training stops when no trait
    has improved for cfg.patience checks. early=False: train max_steps and take predictions at snap_steps.
    Returns (best step per trait, best held-out loss per trait, {step: expected scores on the eval rows}, history)."""
    head = {id(p) for p in model.head_parameters()}
    rest = [p for p in model.parameters() if id(p) not in head and (cfg.opt_A != "sgd" or p is not model.A)]
    opt = torch.optim.AdamW([{"params": rest, "lr": cfg.lr},
                             {"params": model.head_parameters(), "lr": cfg.lr_head, "weight_decay": 0.0}],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_A = torch.optim.SGD([model.A], lr=cfg.lr_A, momentum=cfg.mom_A) if cfg.opt_A == "sgd" else None
    prox_step = cfg.lr_A if cfg.opt_A == "sgd" else cfg.lr
    tr = fd.train
    rows = torch.as_tensor(fd.train_rows)
    Ztr = ((X[rows].to(device).float() - mean) / sd)
    Ptr, Otr = Psi[rows].to(device), Off[rows].to(device)
    pos = torch.full((len(fd.accs),), -1, dtype=torch.long)
    pos[rows] = torch.arange(len(rows))
    rpos, rt, ry, rb, rw = pos[tr.row].to(device), tr.t.to(device), tr.y.to(device), tr.b.to(device), tr.w.to(device)
    T_ = len(fd.traits)
    snaps, history = {}, []

    @torch.no_grad()
    def snapshot(step):
        eta = predict_eta(model, fd.eval_rows, X, mean, sd, Psi, Off, device)
        snaps[step] = model.expected_score(eta).cpu().numpy()

    best_t = np.full(T_, np.inf)
    best_step_t = np.zeros(T_, dtype=int)
    if early:
        best_t = eval_loss_vec(model, fd, X, mean, sd, Psi, Off, device)
        history.append((0, np.nan, best_t.tolist()))
        snapshot(0)
    elif 0 in snap_steps:
        snapshot(0)
    bad, step = 0, 0
    for step in range(1, max_steps + 1):
        model.train()
        eta = model(Ztr, Ptr, Otr)
        loss = (rw * model.record_nll(eta[rpos, rt], rt, ry, rb)).sum() + model.penalty()
        opt.zero_grad()
        if opt_A is not None:
            opt_A.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rest + model.head_parameters(), cfg.clip)
        opt.step()
        if opt_A is not None:
            opt_A.step()
        if cfg.prox:
            model.prox_A(prox_step)
        if early and step % cfg.eval_every == 0:
            vt = eval_loss_vec(model, fd, X, mean, sd, Psi, Off, device)
            history.append((step, float(loss), vt.tolist()))
            imp = vt < best_t - cfg.min_improve
            if imp.any():
                best_t[imp], best_step_t[imp] = vt[imp], step
                snapshot(step)
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
            if log and step % (cfg.eval_every * 8) == 0:
                with torch.no_grad():
                    nz = int((model.A.abs().sum((0, 2)) > 0).sum())
                log(f"step {step}: train {float(loss):.4f} held-out {np.nansum(vt):.4f}; {int(imp.sum())} traits "
                    f"improved; {nz} active blocks")
        elif not early and step in snap_steps:
            snapshot(step)
    if log and early:
        log(f"stopped at step {step}; best step per trait: "
            f"{dict(zip([T.SHORT[t] for t in fd.traits], best_step_t.tolist()))}")
    return best_step_t, best_t, snaps, history


@torch.no_grad()
def eval_loss_vec(model, fd, X, mean, sd, Psi, Off, device):
    """Class-weighted mean held-out NLL per trait (nan for traits without held-out records)."""
    ev = fd.eval
    rows = fd.eval_rows
    eta = predict_eta(model, rows, X, mean, sd, Psi, Off, device)
    pos = torch.full((len(fd.accs),), -1, dtype=torch.long)
    pos[torch.as_tensor(rows)] = torch.arange(len(rows))
    eta_rec = eta[pos[ev.row].to(device), ev.t.to(device)]
    nll = model.record_nll(eta_rec, ev.t.to(device), ev.y.to(device), ev.b.to(device))
    tt, w = ev.t.to(device), ev.w.to(device)
    per = torch.zeros(len(fd.traits), device=device).index_add_(0, tt, w * nll)
    cnt = torch.zeros(len(fd.traits), device=device).index_add_(0, tt, w)
    out = (per / cnt.clamp_min(1e-12)).cpu().numpy()
    out[cnt.cpu().numpy() == 0] = np.nan
    return out


@torch.no_grad()
def predict_eta(model, rows, X, mean, sd, Psi, Off, device, chunk=2048):
    model.eval()
    out = []
    for i in range(0, len(rows), chunk):
        r = torch.as_tensor(rows[i:i + chunk])
        Z = (X[r].to(device).float() - mean) / sd
        out.append(model(Z, Psi[r].to(device), Off[r].to(device)))
    return torch.cat(out)


def eval_records(fd, es, Off):
    """Record-level predictions for the eval accessions (baseline format: acc, batch, score, resistant, pred);
    es = expected scores (n_eval_rows x T); pred_offset = the GBLUP offset of the same accession and trait."""
    rows = fd.eval_rows
    acc_of = fd.accs[rows]
    out = []
    df = fd.eval_df
    for ti, t in enumerate(fd.traits):
        d = df[df.trait == t]
        if d.empty:
            continue
        pm = dict(zip(acc_of, es[:, ti]))
        om = dict(zip(acc_of, Off[torch.as_tensor(rows), ti].numpy()))
        out.append(d[["acc", "batch", "score", "resistant"]].assign(trait=t, pred=d.acc.map(pm).to_numpy(),
                                                                    pred_offset=d.acc.map(om).to_numpy()))
    return pd.concat(out, ignore_index=True)


def es_fold(scenario, f):
    """Inner fold used for early stopping inside the fit set (stage A)."""
    return f if scenario == "cv1" else (f + 1) % 5


def run_stage(train_accs, eval_accs, cfg, seeds, steps, cache_name, threads, device, log, contrib=False):
    """One model per seed. steps None: per-trait early stopping on the eval set; otherwise {trait: steps} (from stage
    A): train max(steps) and take each trait's predictions at its own step. Predictions are averaged over seeds."""
    fd = FoldData(train_accs, eval_accs, cfg)
    Xfull, mean_f, sd_f, snps, Psi = load_genotypes(fd.accs, cfg.q_pcs)
    keep = screen_blocks(fd, Xfull, mean_f, sd_f, snps, cfg.blocks, cfg.n_blocks) if cfg.n_blocks else None
    bidx, cols = block_index(snps, cfg.blocks, keep)
    X = Xfull[:, torch.as_tensor(cols)].contiguous()                  # selected blocks' SNPs only
    del Xfull
    mean, sd = mean_f[cols].to(device), sd_f[cols].to(device)
    Off = gblup_offsets(fd, cache_name, threads, log) if cfg.background in ("gblup", "both") \
        else torch.zeros(len(fd.accs), len(fd.traits))
    torch.set_num_threads(threads)
    log(f"{cache_name}: {len(fd.train_rows)} train / {len(fd.eval_rows)} eval accessions, {len(fd.train.row)} train "
        f"records, {bidx.shape[0]} blocks, device {device}")
    T_ = len(fd.traits)
    preds, info, contribs = [], [], []
    for s_ in seeds:
        cfg_s = MTConfig(**{**cfg.__dict__, "seed": s_})
        model = MultiResNet(Psi.shape[1], bidx, T_, fd.kinds, fd.n_levels, len(fd.batch_id) + 1,
                            fd.batch_random, cfg_s).to(device)
        calibrate_head(model, fd, Off, cfg_s, device, log)
        if steps is None:
            st, bl, snaps, hist = run_fullbatch(model, fd, X, mean, sd, Psi, Off, cfg_s, device, cfg.max_epochs, True,
                                                log=log)
            info.append({"seed": s_, "best_step": dict(zip(fd.traits, st.tolist())),
                         "eval_loss": dict(zip(fd.traits, bl.tolist())), "checks": len(hist)})
        else:
            st = np.array([int(steps.get(t, 0)) for t in fd.traits])
            _, _, snaps, _ = run_fullbatch(model, fd, X, mean, sd, Psi, Off, cfg_s, device, int(st.max()), False,
                                           snap_steps=set(st.tolist()), log=log)
            info.append({"seed": s_, "steps": dict(zip(fd.traits, st.tolist()))})
        es = np.stack([snaps[int(st[ti])][:, ti] for ti in range(T_)], 1)
        with torch.no_grad():
            nz = (model.A.abs().sum(2) > 0)
            info[-1].update({"active_blocks": int(nz.any(0).sum()), "active_trait_blocks": int(nz.sum()),
                             "beta": model.beta.detach().cpu().numpy().round(3).tolist()})
        preds.append(es)
        if contrib:
            contribs.append(block_contributions(model, X[torch.as_tensor(fd.train_rows)], mean, sd, device))
    rec = eval_records(fd, np.mean(preds, 0), Off)
    return rec, info, (np.mean(contribs, 0) if contribs else None), keep


def trait_metrics(rec, col="pred"):
    rare, hard = set(data.rare_traits()), set(data.primary_traits())
    rows = []
    for t, d in rec.groupby("trait"):
        m = evaluate.metrics(d.drop(columns="trait").assign(pred=d[col]), t, t in rare)
        rows.append({"trait": t, "n": m.get("n"), "primary_metric": m.get("primary_metric"),
                     "primary": m.get("primary"), "hard": t in hard})
    return pd.DataFrame(rows)


def summary(m):
    return {"hard_mean": float(m.primary[m.hard].mean()), "other_mean": float(m.primary[~m.hard].mean()),
            "all_mean": float(m.primary.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "collect", "report"])
    ap.add_argument("--scenario", default="cv1_inner", choices=list(data.SCENARIOS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--tags", default="")
    ap.add_argument("--stage", default="AB", choices=["A", "AB"])
    ap.add_argument("--seeds", default="0,1,2,3,4", help="stage-B seeds (stage A always uses the first one)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = data.RUNS / a.scenario / "mtnet_dev"
    if a.mode == "collect":
        fs = [dev / a.tag / f"{data.split_name(a.scenario, f)}.B.tsv.gz" for f in range(5)]
        allp = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in fs if f.exists()], ignore_index=True)
        out = data.RUNS / a.scenario / f"mtnet_{a.tag}"
        out.mkdir(parents=True, exist_ok=True)
        for t, d in allp.groupby("trait"):
            d.drop(columns=["trait", "pred_offset"]).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
        print(f"collected {allp.trait.nunique()} traits from {sum(f.exists() for f in fs)} of 5 splits -> {out}")
        return
    if a.mode == "report":
        rows = []
        for tag in a.tags.split(","):
            fs = [dev / tag / f"{data.split_name(a.scenario, f)}.A.tsv.gz" for f in range(5)]
            have = [f for f in fs if f.exists()]
            if not have:
                continue
            rec = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in have], ignore_index=True)
            js = [json.load(open(str(f).replace(".tsv.gz", ".json"))) for f in have]
            def med_step(j):
                bs = j["seeds"][0].get("best_step", j["seeds"][0].get("best_epoch"))
                return int(np.median(list(bs.values()))) if isinstance(bs, dict) else bs

            def n_moved(j):
                bs = j["seeds"][0].get("best_step", j["seeds"][0].get("best_epoch"))
                return sum(v > 0 for v in bs.values()) if isinstance(bs, dict) else int(bs > 0)
            recs = [pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in have]
            row = {"tag": tag, "splits": len(have), **summary(trait_metrics(rec)),
                   "split_mean": float(np.mean([summary(trait_metrics(r))["hard_mean"] for r in recs])),
                   "gblup_split_mean": float(np.mean([summary(trait_metrics(r, "pred_offset"))["hard_mean"] for r in recs])),
                   "median_step": "/".join(str(med_step(j)) for j in js),
                   "traits_moved": "/".join(str(n_moved(j)) for j in js),
                   "active_blocks": "/".join(str(j["seeds"][0].get("active_blocks", "")) for j in js)}
            ref = summary(trait_metrics(rec, "pred_offset"))
            row.update({f"gblup_{k}": v for k, v in ref.items()})
            rows.append(row)
        r = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    cfg = MTConfig(**json.loads(a.config))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
    fo = data.folds()
    fit, ev = data.split_sets(a.scenario, a.fold, fo)
    es = fit & (fo.inner_fold == es_fold(a.scenario, a.fold))
    sp = data.split_name(a.scenario, a.fold)
    out = dev / a.tag
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    base = {"scenario": a.scenario, "fold": a.fold, "split": sp, "tag": a.tag, "config": cfg.__dict__, "code": CODE,
            "device": str(device)}
    # stage A: early stopping on inner fold g inside the fit set
    recA, infoA, _, keepA = run_stage(fo.index[fit & ~es], fo.index[es], cfg, seeds[:1], None,
                                      f"{a.scenario}_{sp}_A", a.threads, device, log)
    mA = trait_metrics(recA)
    recA.to_csv(out / f"{sp}.A.tsv.gz", sep="\t", index=False)
    json.dump({**base, "stage": "A", "seeds": infoA, **summary(mA), "gblup": summary(trait_metrics(recA, "pred_offset")),
               "metrics": mA.to_dict("records"), "seconds": round(time.time() - t0, 1)},
              open(out / f"{sp}.A.json", "w"), indent=1, default=float)
    E = infoA[0]["best_step"]
    log(f"stage A done: primary-trait mean {summary(mA)['hard_mean']:.4f} "
        f"(GBLUP offset alone {summary(trait_metrics(recA, 'pred_offset'))['hard_mean']:.4f})")
    if a.stage == "A":
        return
    # stage B: whole fit set, E epochs, several seeds, eval set
    recB, infoB, contrib, keepB = run_stage(fo.index[fit], fo.index[ev], cfg, seeds, E,
                                            f"{a.scenario}_{sp}_B", a.threads, device, log, contrib=True)
    mB = trait_metrics(recB)
    recB.assign(fold=a.fold).to_csv(out / f"{sp}.B.tsv.gz", sep="\t", index=False)
    np.save(out / f"{sp}.B.contrib.npy", contrib)
    json.dump({**base, "stage": "B", "steps": E, "seeds": infoB, **summary(mB),
               "gblup": summary(trait_metrics(recB, "pred_offset")), "metrics": mB.to_dict("records"),
               "blocks_kept": None if keepB is None else keepB.tolist(), "seconds": round(time.time() - t0, 1)},
              open(out / f"{sp}.B.json", "w"), indent=1, default=float)
    log(f"stage B done: primary-trait mean {summary(mB)['hard_mean']:.4f} "
        f"(GBLUP offset alone {summary(trait_metrics(recB, 'pred_offset'))['hard_mean']:.4f})")


if __name__ == "__main__":
    main()
