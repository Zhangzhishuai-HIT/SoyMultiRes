"""Step 6: HapGAM-MT (src/soymultires/hgmt.py) on the shared scenarios (soymultires.data.split_sets).

usage:
  python train_hgmt.py fit --scenario cv1_inner --fold 0 --config '{"top_m": 64}' --tag m64 [--stage A|AB] [--seeds 0,1,2,3,4]
  python train_hgmt.py collect --scenario cv1_inner --tag m64      -> runs/<scenario>/hgmt_<tag>/<trait>.tsv.gz
  python train_hgmt.py report --scenario cv1_inner --tags m32,m64  -> stage-A comparison of settings (pooled over splits)

Stages as in train_mt.py: A = fit set minus inner fold g (g = f for cv1, f + 1 mod 5 for cv1_inner), per-trait early
stopping on g (held-out r; step 0 = per-trait GBLUP) -> number of steps per trait; settings are chosen on the stage-A
predictions pooled over the five splits. B = whole fit set, those steps, seeds averaged, predictions for the eval set.
Outputs: runs/<scenario>/hgmt_dev/<tag>/<split>.{A,B}.{json,tsv.gz}; B also <split>.B.contrib.npy (trait x block
variance of the block terms on the training accessions) with the kept block ids in the json.
Records carry pred_offset = the step-0 prediction (per-trait GBLUP fitted in the same stage) for paired comparisons.
"""
import argparse
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
sys.path.insert(0, str(ROOT / "scripts/03_model"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.hgmt import HGConfig, SharedBlocks, TraitGLS  # noqa: E402
import train_mt as tm  # noqa: E402  (load_genotypes, block_index, es_fold, trait_metrics, summary)

CODE = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/soymultires/hgmt.py").read_bytes()).hexdigest()[:16]
DEV = "hgmt_dev"


def stage_targets(train_accs, eval_accs):
    """Per trait: data.fold_targets (training targets: mean score, numeric traits adjusted by training batch means) and
    held-out targets per accession for early stopping (same adjustment)."""
    long = data.pheno_long()
    tr_set, ev_set = set(train_accs), set(eval_accs)
    split = pd.Series(np.r_[np.zeros(len(train_accs), int), np.ones(len(eval_accs), int)],
                      index=np.r_[np.asarray(train_accs), np.asarray(eval_accs)])
    out = {}
    for t in T.TRAITS:
        lt = long[(long.trait == t) & long.acc.isin(tr_set | ev_set)]
        ft = data.fold_targets(lt, t, split, 1)
        te = lt[lt.acc.isin(ev_set)]
        if t in T.NUMERIC:
            trr = lt[lt.acc.isin(tr_set)]
            adj = te.score - te.batch.map(trr.groupby("batch").score.mean()).fillna(trr.score.mean())
            yv = adj.groupby(te.acc).mean()
        else:
            yv = te.groupby("acc").score.mean()
        out[t] = (ft, yv.reindex(ft.test_acc).to_numpy(float))
    return out


def screen_blup(gls, tg, X, row_of, snps, scheme, M, log, chunk=4000, screen="sum", per_trait_k=8):
    """HapGAM's screen, summed over traits: share of each trait's GBLUP variance carried by the block's rrBLUP
    contribution (marker effects beta = W' alpha / c for the VanRaden kernel K = W W' / c), on the trait's training
    accessions. Returns the indices of the top-M blocks (in block-id order) and the score of every block."""
    p2 = 2 * snps.alt_freq.to_numpy()
    c = float(np.sum(p2 * (1 - p2 / 2)))
    dev = X.device
    two_p = torch.tensor(p2, dtype=torch.float32, device=dev)
    bid = torch.as_tensor(snps[scheme].to_numpy(), dtype=torch.long, device=dev)
    nb = int(bid.max()) + 1
    score = torch.zeros(nb, dtype=torch.float64)
    per_trait = []
    for t in T.TRAITS:
        g = gls[t]
        _, alpha, _ = g.blup_parts(torch.zeros(g.n, device=dev))
        rows = torch.as_tensor([row_of[a] for a in tg[t][0].train_acc], device=dev)
        if screen in ("gwas", "gwas_union"):
            # mixed-model score test per SNP (GRAMMAR-type: V^-1 (y - m) is proportional to alpha), block = best SNP,
            # normalised by the trait's mean block score so that traits count equally
            stat = torch.zeros(X.shape[1], device=dev)
            for s0 in range(0, X.shape[1], chunk):
                W = X[rows, s0:s0 + chunk].float() - two_p[s0:s0 + chunk]
                W = W - W.mean(0)
                stat[s0:s0 + chunk] = (W.T @ alpha) ** 2 / (W * W).sum(0).clamp_min(1e-6)
            bs = torch.zeros(nb, device=dev).scatter_reduce(0, bid, stat, reduce="amax", include_self=False)
            sh = (bs / bs.mean().clamp_min(1e-12)).double().cpu()
            score += sh
            per_trait.append(sh)
            continue
        contrib = torch.zeros(len(rows), nb, device=dev)
        for s0 in range(0, X.shape[1], chunk):
            W = X[rows, s0:s0 + chunk].float() - two_p[s0:s0 + chunk]
            beta = (W.T @ alpha) / c
            contrib.index_add_(1, bid[s0:s0 + chunk], W * beta)
        gv = contrib.sum(1)
        vg = float(gv.var())
        if vg > 1e-10:
            sh = (contrib.var(0) / vg).double().cpu()
            score += sh
            per_trait.append(sh)
    order = torch.argsort(score, descending=True).numpy()
    if screen in ("union", "gwas_union"):
        sel = []
        for sh in per_trait:
            for b in torch.argsort(sh, descending=True)[:per_trait_k].tolist():
                if b not in sel:
                    sel.append(b)
        sel = sorted(sel, key=lambda b: -float(score[b]))[:M]
        for b in order:
            if len(sel) >= M:
                break
            if b not in sel:
                sel.append(int(b))
        keep = np.sort(np.array(sel, dtype=int))
    else:
        keep = np.sort(order[:M])
    log(f"screen: top {M} blocks carry {float(score[keep].sum() / score.sum()):.3f} of the summed variance shares")
    return keep, score.numpy()


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_stage(train_accs, eval_accs, cfg, seeds, steps, threads, log, contrib=False, device=torch.device("cpu")):
    """steps None: per-trait early stopping on the eval accessions; otherwise {trait: steps}. Seeds averaged."""
    torch.set_num_threads(threads)
    tg = stage_targets(train_accs, eval_accs)
    traits = list(T.TRAITS)
    R = sorted(set().union(*[set(tg[t][0].train_acc) for t in traits]))
    E = sorted(set().union(*[set(tg[t][0].test_acc) for t in traits]))
    accs = np.array(R + E)
    nR = len(R)
    row_of = {a: i for i, a in enumerate(accs)}
    X, mean, sd, snps, _ = tm.load_genotypes(accs, 1000)
    X, mean, sd = X.to(device), mean.to(device), sd.to(device)
    Kmm = data.grm(mmap=True)
    gpos = {v: i for i, v in enumerate(data.samples())}
    gls = {}
    t0 = time.time()
    rare = set(data.rare_traits())
    long = data.pheno_long()
    long_t_res = {t: long[long.trait == t].groupby("acc").resistant.mean() for t in rare}
    for t in traits:
        ft, _ = tg[t]
        tr_g = np.array([gpos[a] for a in ft.train_acc])
        ev_g = np.array([gpos[a] for a in ft.test_acc])
        Kt = torch.from_numpy(np.asarray(Kmm[np.ix_(tr_g, tr_g)], dtype=np.float64)).to(device)
        Kv = torch.from_numpy(np.asarray(Kmm[np.ix_(ev_g, tr_g)], dtype=np.float32)) if len(ev_g) else None
        cw = None
        if cfg.rare_weight > 1 and t in rare:
            res = long_t_res[t].reindex(ft.train_acc).fillna(0).to_numpy() > 0.5
            cw = np.where(res, min(cfg.rare_weight, 10.0), 1.0)
            cw = cw / cw.mean()
        gls[t] = TraitGLS(Kt, ft.y_train, Kv, cw)
        del Kt
    log(f"{len(R)} training / {len(E)} held-out accessions; per-trait REML + eigenbases {time.time() - t0:.0f}s; "
        f"median h2-type ratio delta {np.median([gls[t].delta for t in traits]):.2f}")
    keep, _ = screen_blup(gls, tg, X, row_of, snps, cfg.blocks, cfg.top_m, log, screen=cfg.screen,
                          per_trait_k=cfg.per_trait_k) if cfg.top_m else (np.array([], int), None)
    if cfg.top_m:
        bidx, cols = tm.block_index(snps, cfg.blocks, keep)
        idx = torch.as_tensor(bidx)
        idx = idx.to(device)
        mask = (idx >= 0).float()
        cols_t = torch.as_tensor(cols, device=device)
        Zsel = (X[:, cols_t].float() - mean[cols_t]) / sd[cols_t]
        Zb = Zsel[:, idx.clamp_min(0)] * mask
        del Zsel
    else:
        mask = torch.zeros(0, 1, device=device)
        Zb = torch.zeros(len(accs), 0, 1, device=device)
    del X
    Zb_tr, Zb_ev = Zb[:nR], Zb[nR:]
    tr_idx = {t: torch.as_tensor([row_of[a] for a in tg[t][0].train_acc], device=device) for t in traits}
    ev_idx = {t: torch.as_tensor([row_of[a] - nR for a in tg[t][0].test_acc], dtype=torch.long, device=device)
              for t in traits}
    Ti = len(traits)

    def predict_all(net, which=None):
        with torch.no_grad():
            Ftr = net(Zb_tr) if cfg.top_m else torch.zeros(nR, Ti, device=device)
            Fev = net(Zb_ev) if cfg.top_m else torch.zeros(len(E), Ti, device=device)
        out = {}
        for ti, t in enumerate(traits):
            if which is not None and t not in which:
                continue
            if gls[t].Kv is None:
                continue
            out[t] = gls[t].predict(Ftr[tr_idx[t], ti], Fev[ev_idx[t], ti], refit=cfg.refit_delta)
        return out

    preds_seeds, info, contribs = [], [], []
    for s_ in seeds:
        c_s = HGConfig(**{**cfg.__dict__, "seed": s_})
        net = SharedBlocks(mask, Ti, c_s).to(device) if cfg.top_m else None
        snap = predict_all(net) if cfg.top_m else predict_all(None)
        gblup = dict(snap)
        best_r = {t: pearson(snap[t], tg[t][1]) for t in snap}
        best_step = {t: 0 for t in traits}
        want = {} if steps is None else {t: int(steps.get(t, 0)) for t in traits}
        max_steps = cfg.steps if steps is None else max(want.values(), default=0)
        if cfg.top_m and max_steps > 0:
            opt = torch.optim.Adam([{"params": [net.W1, net.b1, net.W2]}, {"params": [net.A]}], lr=cfg.lr)
            bad = 0
            for step in range(1, max_steps + 1):
                Ftr = net(Zb_tr)
                loss = sum(gls[t].loss(Ftr[tr_idx[t], ti]) for ti, t in enumerate(traits)) / Ti + net.penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if cfg.lam_group > 0 or cfg.lam_elem > 0:
                    net.prox(cfg.lr)
                if steps is None and step % cfg.eval_every == 0:
                    p = predict_all(net)
                    imp = []
                    for t in p:
                        r = pearson(p[t], tg[t][1])
                        br = best_r[t]
                        margin = cfg.min_improve
                        if cfg.min_improve_se > 0 and np.isfinite(br):
                            margin = max(margin, cfg.min_improve_se * (1 - br ** 2) / np.sqrt(max(len(tg[t][1]), 4)))
                        if np.isfinite(r) and (not np.isfinite(br) or r > br + margin):
                            best_r[t], best_step[t], snap[t] = r, step, p[t]
                            imp.append(t)
                    bad = 0 if imp else bad + 1
                    if step % (cfg.eval_every * 8) == 0:
                        log(f"step {step}: loss {float(loss):.5f}; {len(imp)} traits improved")
                    if bad >= cfg.patience:
                        break
                elif steps is not None:
                    hit = [t for t, s in want.items() if s == step]
                    if hit:
                        snap.update(predict_all(net, set(hit)))
            if steps is None:
                log(f"stopped at step {step}; steps per trait "
                    f"{ {T.SHORT[t]: best_step[t] for t in traits} }")
        if steps is not None:
            best_step = want
        info.append({"seed": s_, "best_step": best_step,
                     "best_r": {t: (None if not np.isfinite(v) else v) for t, v in best_r.items()} if steps is None else None})
        preds_seeds.append(snap)
        if contrib and cfg.top_m:
            with torch.no_grad():
                eff = torch.einsum("nmk,tmk->tnm", net.phi(Zb_tr), net.A)
                contribs.append(eff.var(1).cpu().numpy())
    rows = []
    for t in traits:
        ft, _ = tg[t]
        if not len(ft.test_acc):
            continue
        pm = dict(zip(ft.test_acc, np.mean([p[t] for p in preds_seeds], 0)))
        om = dict(zip(ft.test_acc, gblup[t]))
        rec = ft.test_records
        rows.append(rec.assign(trait=t, pred=rec.acc.map(pm).to_numpy(), pred_offset=rec.acc.map(om).to_numpy()))
    return pd.concat(rows, ignore_index=True), info, (np.mean(contribs, 0) if contribs else None), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "collect", "report"])
    ap.add_argument("--scenario", default="cv1_inner", choices=list(data.SCENARIOS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--tags", default="")
    ap.add_argument("--stage", default="AB", choices=["A", "AB"])
    ap.add_argument("--seeds", default="0,1,2,3,4", help="stage-B seeds (stage A uses the first one)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (gpu10 card 0 only, selected by UUID)")
    ap.add_argument("--gpu_mem_gb", type=float, default=7.0, help="per-process cap on the GPU")
    a = ap.parse_args()
    dev = data.RUNS / a.scenario / DEV
    if a.mode == "collect":
        fs = [dev / a.tag / f"{data.split_name(a.scenario, f)}.B.tsv.gz" for f in range(5)]
        allp = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in fs if f.exists()], ignore_index=True)
        out = data.RUNS / a.scenario / f"hgmt_{a.tag}"
        out.mkdir(parents=True, exist_ok=True)
        for t, d in allp.groupby("trait"):
            d.drop(columns=["trait", "pred_offset"]).to_csv(out / f"{t}.tsv.gz", sep="\t", index=False)
        print(f"collected {allp.trait.nunique()} traits from {sum(f.exists() for f in fs)} of 5 splits -> {out}")
        return
    if a.mode == "report":
        rows = []
        for tag in a.tags.split(","):
            have = [f for f in (dev / tag / f"{data.split_name(a.scenario, f)}.A.tsv.gz" for f in range(5)) if f.exists()]
            if not have:
                continue
            recs = [pd.read_csv(f, sep="\t", dtype={"acc": str}).assign(fold=i) for i, f in enumerate(have)]
            js = [json.load(open(str(f).replace(".tsv.gz", ".json"))) for f in have]
            rec = pd.concat(recs, ignore_index=True)
            moved = [sum(v > 0 for v in j["seeds"][0]["best_step"].values()) for j in js]
            med = [int(np.median(list(j["seeds"][0]["best_step"].values()))) for j in js]
            from soymultires.evaluate import within_fold_z
            zrec = within_fold_z(rec)
            zoff = within_fold_z(rec.assign(pred=rec.pred_offset))
            rows.append({"tag": tag, "splits": len(have),
                         "z_primary": tm.summary(tm.trait_metrics(zrec))["hard_mean"],
                         "z_gblup": tm.summary(tm.trait_metrics(zoff))["hard_mean"],
                         "z_other": tm.summary(tm.trait_metrics(zrec))["other_mean"],
                         "z_gblup_other": tm.summary(tm.trait_metrics(zoff))["other_mean"],
                         **tm.summary(tm.trait_metrics(rec)),
                         **{f"gblup_{k}": v for k, v in tm.summary(tm.trait_metrics(rec, "pred_offset")).items()},
                         "split_mean": float(np.mean([tm.summary(tm.trait_metrics(r))["hard_mean"] for r in recs])),
                         "gblup_split_mean": float(np.mean([tm.summary(tm.trait_metrics(r, "pred_offset"))["hard_mean"]
                                                            for r in recs])),
                         "traits_moved": "/".join(map(str, moved)), "median_step": "/".join(map(str, med))})
        pd.set_option("display.width", 220)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    cfg = HGConfig(**json.loads(a.config))
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    if device.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / tot), 0)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
    fo = data.folds()
    fit, ev = data.split_sets(a.scenario, a.fold, fo)
    es = fit & (fo.inner_fold == tm.es_fold(a.scenario, a.fold))
    sp = data.split_name(a.scenario, a.fold)
    out = dev / a.tag
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    base = {"scenario": a.scenario, "fold": a.fold, "split": sp, "tag": a.tag, "config": cfg.__dict__, "code": CODE,
            "device": str(device)}
    recA, infoA, _, keepA = run_stage(fo.index[fit & ~es], fo.index[es], cfg, seeds[:1], None, a.threads, log,
                                      device=device)
    mA = tm.trait_metrics(recA)
    recA.to_csv(out / f"{sp}.A.tsv.gz", sep="\t", index=False)
    json.dump({**base, "stage": "A", "seeds": infoA, **tm.summary(mA),
               "gblup": tm.summary(tm.trait_metrics(recA, "pred_offset")), "metrics": mA.to_dict("records"),
               "blocks_kept": keepA.tolist(), "seconds": round(time.time() - t0, 1)},
              open(out / f"{sp}.A.json", "w"), indent=1, default=float)
    log(f"stage A done: primary-trait mean {tm.summary(mA)['hard_mean']:.4f} "
        f"(per-trait GBLUP {tm.summary(tm.trait_metrics(recA, 'pred_offset'))['hard_mean']:.4f})")
    if a.stage == "A":
        return
    E = infoA[0]["best_step"]
    recB, infoB, contrib, keepB = run_stage(fo.index[fit], fo.index[ev], cfg, seeds, E, a.threads, log, contrib=True,
                                            device=device)
    mB = tm.trait_metrics(recB)
    recB.assign(fold=a.fold).to_csv(out / f"{sp}.B.tsv.gz", sep="\t", index=False)
    if contrib is not None:
        np.save(out / f"{sp}.B.contrib.npy", contrib)
    json.dump({**base, "stage": "B", "steps": E, "seeds": infoB, **tm.summary(mB),
               "gblup": tm.summary(tm.trait_metrics(recB, "pred_offset")), "metrics": mB.to_dict("records"),
               "blocks_kept": keepB.tolist(), "seconds": round(time.time() - t0, 1)},
              open(out / f"{sp}.B.json", "w"), indent=1, default=float)
    log(f"stage B done: primary-trait mean {tm.summary(mB)['hard_mean']:.4f} "
        f"(per-trait GBLUP {tm.summary(tm.trait_metrics(recB, 'pred_offset'))['hard_mean']:.4f})")


if __name__ == "__main__":
    main()
