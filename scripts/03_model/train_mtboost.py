"""TraitBoost (model in src/soymultires/mtboost.py): tuning, per-trait choice,
refits and split gains.

usage:
  python train_mtboost.py start   --scenario cv1 --fold 0                       GBLUP predictions used as a start (CPU)
  python train_mtboost.py fit     --scenario cv1 --fold 0 --tag v2 [--trials 15]   tuning, choice and refits (GPU)
  python train_mtboost.py reseed  --scenario cv1 --fold 0 --tag v2 --first_seed 2 --seeds 2    further refits
  python train_mtboost.py reseed  --scenario all --fold 0 --tag final --first_seed 0 --seeds 2  split gains of the
                                                                                 model fitted to all accessions
  python train_mtboost.py start   --scenario cv1 --fold 0 --eval_all          GBLUP predictions of every trait for all
                                                                            test accessions (CPU; used by --dump_full)
  python train_mtboost.py reseed  --scenario cv1 --fold 0 --tag v2 --first_seed 0 --seeds 2 --dump_full
                                                             predictions of all traits for the test accessions (<split>_full.npz)
  python train_mtboost.py collect --scenario cv1 --tag v2      -> runs/<scenario>/mtboost_<tag>{_final,_noshare4,_gblup,...}/
Scenarios: cv1_inner, cv1 (new accessions), cv3 (held-out experiments), all (all accessions).
Per split (fit):
  1. targets as for the other models (data.split_targets); rows = training accessions with any record; the test rows
     are used only at the end;
  2. GBLUP predictions of every trait: training rows cross-validated over the inner folds (each row predicted by a
     GBLUP fitted without its fold), test rows by a GBLUP fitted to all training rows; cached in work/mtboost_start/;
  3. tuning: Optuna (TPE, seed --optuna_seed, default 20261112), --trials settings, validation on the three inner folds
     data.tuning_inner_folds(scenario, f). For a setting, shared trees are grown from zero, and S* is the number of
     shared rounds with the highest fold-averaged validation correlation of a trait. Every trait has three options:
     trait-specific trees from zero; trait-specific trees after S* rounds of shared trees; trait-specific trees
     starting from the GBLUP predictions (no shared trees). The option and number of rounds with the highest
     fold-averaged validation correlation are chosen; the best option without shared trees is recorded as well.
     The value of a setting is the mean over the traits; every trait uses the setting with its highest correlation;
  4. refits on all training rows with the chosen numbers of rounds x 1.1, for the chosen option and for the best
     option without shared trees, each with --seeds random seeds; predictions for the test rows. The prediction of
     the model (collect: mtboost_<tag>_final) is the mean of the two options with two seeds each; mtboost_<tag>_gblup
     holds the GBLUP predictions of step 2;
  5. split gain of every SNP for every trait: the gain of the trait in the splits of the rounds that its prediction
     uses (shared: first S_t rounds; trait-specific: P_t rounds; option without shared trees: its rounds), mean over
     the refits (importance.npz, key "model").
Outputs: runs/<scenario>/mtboost_dev/<tag>/<split>.{json,tsv.gz}; runs/all/mtboost_<tag>/.
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
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402
from soymultires.mtboost import BoostConfig, Engine, pearson_cols, replay, run_phase  # noqa: E402
from soymultires.reml import SpectralREML  # noqa: E402

CODE = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/soymultires/mtboost.py").read_bytes()).hexdigest()[:16]
TRAITS = list(T.TRAITS)
CATEGORY = {row[0]: row[4] for row in T.STRESS}
def use_traitset(name):
    """The 33 stress traits (pheno_long)."""
    return data.pheno_long()


def units(groups):
    """(unit_of (T,), U) for the shared trees."""
    if groups in ("all", "none"):
        return np.zeros(len(TRAITS), int), 1
    cats = sorted(set(CATEGORY[t] for t in TRAITS))
    return np.array([cats.index(CATEGORY[t]) for t in TRAITS]), len(cats)


class SplitData:
    def __init__(self, scenario, f, device, log, long=None):
        self.scenario, self.f, self.dev = scenario, f, device
        fo = data.folds()
        long = data.pheno_long() if long is None else long
        if scenario == "all":                 # final model: every accession with folds is a fit row; predict all
            long = long[long.acc.isin(set(fo.index))]
            every = pd.Series(0, index=fo.index)
            self.tg = {t: data.fold_targets(long[long.trait == t], t, every, 1) for t in TRAITS}
        else:
            self.tg = {t: data.split_targets(long[long.trait == t], t, scenario, f, fo) for t in TRAITS}
        self.R = np.array(sorted(set().union(*[set(self.tg[t].train_acc) for t in TRAITS])))
        if scenario == "all":
            self.E = np.array(sorted(data.samples()))
        else:
            self.E = np.array(sorted(set().union(*[set(self.tg[t].test_acc) for t in TRAITS])))
        rr = {a: i for i, a in enumerate(self.R)}
        n, Tn = len(self.R), len(TRAITS)
        Y = np.full((n, Tn), np.nan)
        for j, t in enumerate(TRAITS):
            ft = self.tg[t]
            Y[[rr[a] for a in ft.train_acc], j] = ft.y_train
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        # a trait without training records (possible in a held-out experiment) gets mean 0: its start must be finite,
        # because a non-finite gradient would make the gains of every trait non-finite (mtboost.grow sums over traits)
        self.mu = np.where(np.isfinite(mu), mu, 0.0)
        self.sd = np.where(np.isfinite(sd), sd, 0.0) + 1e-8
        self.Yraw = Y
        Ys = (Y - self.mu) / self.sd
        self.M = torch.as_tensor(np.isfinite(Ys), device=device)
        self.Y = torch.as_tensor(np.nan_to_num(Ys), dtype=torch.float32, device=device)
        self.inner = fo.inner_fold.reindex(self.R).to_numpy()
        self.tune_folds = [0, 1, 2] if scenario in ("all", "cv3") else data.tuning_inner_folds(scenario, f)
        pos = {v: i for i, v in enumerate(data.samples())}
        G = np.load(data.P / "geno/dosage.npy", mmap_mode="r")

        def rows(accs):
            idx = np.array([pos[a] for a in accs])
            o = np.argsort(idx)
            X = np.empty((len(idx), G.shape[1]), dtype=np.int8)
            X[o] = G[idx[o]]
            return torch.from_numpy(X).to(device)
        self.XR, self.XE = rows(self.R), rows(self.E)
        log(f"{scenario} {data.split_name(scenario, f)}: {n} fit rows, {len(self.E)} eval rows, {int(self.M.sum())} cells")

    def gblup_start(self, log):
        """Cross-fitted GBLUP predictions (raw scale): S0 (fit rows, out-of-part) and S0e (eval rows, all fit rows)."""
        nm = "all" if self.scenario == "all" else data.split_name(self.scenario, self.f)
        cache = ROOT / "work/mtboost_start" / f"{self.scenario}_{nm}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if list(z["R"]) == list(self.R) and list(z["E"]) == list(self.E) and z["S0"].shape[1] == len(TRAITS):
                log(f"GBLUP start from {cache.name}")
                return self.finite(z["S0"]), self.finite(z["S0e"])
        t0 = time.time()
        Kmm = data.grm(mmap=True)
        pos = {v: i for i, v in enumerate(data.samples())}
        rr = {a: i for i, a in enumerate(self.R)}
        ee = {a: i for i, a in enumerate(self.E)}
        S0 = np.tile(self.mu, (len(self.R), 1))
        S0e = np.tile(self.mu, (len(self.E), 1))
        for j, t in enumerate(TRAITS):
            ft = self.tg[t]
            tr = np.array(ft.train_acc)
            y = ft.y_train
            part = self.inner[[rr[a] for a in tr]]

            def fit_pred(train_mask, target_accs):
                g_tr = np.array([pos[a] for a in tr[train_mask]])
                g_tg = np.array([pos[a] for a in target_accs])
                K = torch.from_numpy(np.asarray(Kmm[np.ix_(g_tr, g_tr)], dtype=np.float64)).to(self.dev)
                m = SpectralREML().fit(K, torch.tensor(y[train_mask], dtype=torch.float64, device=self.dev))
                del K
                Kx = torch.from_numpy(np.asarray(Kmm[np.ix_(g_tg, g_tr)], dtype=np.float64)).to(self.dev)
                out = m.predict(Kx).cpu().numpy()
                del m, Kx
                return out
            for c in np.unique(part):
                S0[[rr[a] for a in tr[part == c]], j] = fit_pred(part != c, tr[part == c])
            targets = self.E if self.scenario == "all" else np.array(ft.test_acc)
            if len(targets):
                S0e[[ee[a] for a in targets], j] = fit_pred(np.ones(len(tr), bool), targets)
            torch.cuda.empty_cache() if self.dev.type == "cuda" else None
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, R=self.R, E=self.E, S0=S0, S0e=S0e)
        log(f"GBLUP start computed in {time.time() - t0:.0f}s")
        return self.finite(S0), self.finite(S0e)

    def finite(self, S):
        """Starts of traits without training records (mean NaN in older caches) are set to the (finite) mean."""
        return np.where(np.isfinite(S), S, self.mu[None, :])

    def gblup_eval_all(self, log):
        """GBLUP predictions of every trait for ALL eval rows (raw scale), from a GBLUP fitted to all training
        accessions of the trait. gblup_start predicts only the eval rows scored for a trait (the others keep the
        training mean), which is all the evaluation needs; the predictions of all traits for all test accessions
        (--dump_full) need the GBLUP start of every accession. Cached in work/mtboost_start/<split>_evalall.npz."""
        nm = "all" if self.scenario == "all" else data.split_name(self.scenario, self.f)
        cache = ROOT / "work/mtboost_start" / f"{self.scenario}_{nm}_evalall.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if list(z["R"]) == list(self.R) and list(z["E"]) == list(self.E) and z["S0e"].shape[1] == len(TRAITS):
                log(f"GBLUP start of all eval rows from {cache.name}")
                return self.finite(z["S0e"])
        t0 = time.time()
        Kmm = data.grm(mmap=True)
        pos = {v: i for i, v in enumerate(data.samples())}
        g_e = np.array([pos[a] for a in self.E])
        S0e = np.tile(self.mu, (len(self.E), 1))
        for j, t in enumerate(TRAITS):
            ft = self.tg[t]
            tr = np.array(ft.train_acc)
            if len(tr) == 0:
                continue
            g_tr = np.array([pos[a] for a in tr])
            K = torch.from_numpy(np.asarray(Kmm[np.ix_(g_tr, g_tr)], dtype=np.float64)).to(self.dev)
            m = SpectralREML().fit(K, torch.tensor(ft.y_train, dtype=torch.float64, device=self.dev))
            del K
            Kx = torch.from_numpy(np.asarray(Kmm[np.ix_(g_e, g_tr)], dtype=np.float64)).to(self.dev)
            S0e[:, j] = m.predict(Kx).cpu().numpy()
            del m, Kx
            log(f"  {t}: {len(tr)} training accessions")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, R=self.R, E=self.E, S0e=S0e)
        log(f"GBLUP start of all eval rows computed in {time.time() - t0:.0f}s")
        return S0e


def avg_curves(curve_list):
    """Average per-trait curves of several folds on a common check grid (shorter runs padded with their last row)."""
    L = max(c.shape[0] for c in curve_list)
    pad = [np.vstack([c, np.repeat(c[-1:], L - c.shape[0], 0)]) for c in curve_list]
    return np.nanmean(np.stack(pad), 0)                                          # checks x T


def trial_eval(cfg: BoostConfig, sd: SplitData, eng: Engine, S0s, log):
    """Validation of one setting on the three tuning folds. Shared trees are grown from a zero start; every trait then
    has three options: trait-specific trees from the zero start after 0 or S* rounds of shared trees (S* = best of the
    fold-averaged validation curve of the shared trees), or trait-specific trees from the GBLUP start without shared
    trees. Returns (value, choice): per-trait best option (start, S_t, P_t, r) and the best option without shared
    trees (start0, P0_t, r0)."""
    dev = sd.dev
    unit_of_s, Us = units(cfg.groups)
    unit_of_s = torch.as_tensor(unit_of_s, device=dev)
    unit_of_p = torch.arange(len(TRAITS), device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    ce = cfg.check_every
    zero = torch.zeros_like(S0s)
    shared = {}
    for v in ([] if cfg.groups == "none" else sd.tune_folds):
        train = torch.as_tensor(sd.inner != v, device=dev)[:, None]
        val_idx = torch.as_tensor(np.flatnonzero(sd.inner == v), device=dev)
        Mtr = sd.M & train
        n_t = Mtr.sum(0).float()
        c = torch.zeros(len(TRAITS), device=dev)
        for u in range(Us):
            m = unit_of_s == u
            c[m] = n_t[m].mean() / n_t[m].clamp_min(1)
        shared[v] = run_phase(eng, zero, sd.Y, Mtr.float() * c[None, :], unit_of_s, Us, cfg.depth_s, cfg.eta_s,
                              cfg.lam, cfg.gamma, cfg.mcw_s, cfg.colsample, cfg.subsample, cfg.rounds_s, gen,
                              val_idx=val_idx, Yv=sd.Y[val_idx], Mv=sd.M[val_idx], check_every=ce,
                              patience=cfg.patience, stop_tol=cfg.stop_tol)
    if cfg.groups == "none":
        s_star = np.zeros(len(TRAITS), int)
        shared = {v: {"trees": []} for v in sd.tune_folds}
    else:
        cs = avg_curves([shared[v]["curves"] for v in sd.tune_folds])
        s_star = np.nanargmax(np.nan_to_num(cs, nan=-2), 0) * ce
    cands = [np.zeros_like(s_star), s_star]
    options = [("zero", 0), ("zero", 1), ("gblup", 0)]
    curves = {}
    for v in sd.tune_folds:
        train = torch.as_tensor(sd.inner != v, device=dev)[:, None]
        val_idx = torch.as_tensor(np.flatnonzero(sd.inner == v), device=dev)
        Wp = (sd.M & train).float()
        for oi, (start, ci) in enumerate(options):
            if start == "zero" and ci > 0 and np.array_equal(cands[ci], cands[ci - 1]):
                curves[(v, oi)] = curves[(v, oi - 1)]
                continue
            if start == "zero":
                F0 = replay(shared[v]["trees"], sd.XR, unit_of_s, cfg.depth_s, cfg.eta_s, cands[ci], zero)
            else:
                F0 = S0s
            out = run_phase(eng, F0, sd.Y, Wp, unit_of_p, len(TRAITS), cfg.depth_p, cfg.eta_p, cfg.lam, cfg.gamma,
                            cfg.mcw_p, cfg.colsample, cfg.subsample, cfg.rounds_p, gen, val_idx=val_idx,
                            Yv=sd.Y[val_idx], Mv=sd.M[val_idx], check_every=ce, patience=cfg.patience,
                            keep_trees=False, stop_tol=cfg.stop_tol)
            curves[(v, oi)] = out["curves"]
    Tn = len(TRAITS)
    res = {}
    for oi in range(len(options)):
        cc = np.nan_to_num(avg_curves([curves[(v, oi)] for v in sd.tune_folds]), nan=-2)
        res[oi] = (cc.max(0), cc.argmax(0) * ce)
    best = np.full(Tn, -2.0)
    opt = np.zeros(Tn, int)
    for oi in range(len(options)):
        better = res[oi][0] > best + 1e-6
        best = np.where(better, res[oi][0], best)
        opt = np.where(better, oi, opt)
    S_t = np.array([cands[options[o][1]][j] if options[o][0] == "zero" else 0 for j, o in enumerate(opt)])
    P_t = np.array([res[o][1][j] for j, o in enumerate(opt)])
    start = [options[o][0] for o in opt]
    ns = np.where(res[2][0] > res[0][0] + 1e-6, 2, 0)                            # best without sharing
    choice = {"start": start, "S_t": S_t.tolist(), "P_t": P_t.tolist(), "r_val": best.tolist(),
              "start0": [options[o][0] for o in ns], "P0_t": [int(res[o][1][j]) for j, o in enumerate(ns)],
              "r_val_noshare": [float(res[o][0][j]) for j, o in enumerate(ns)], "s_star": s_star.tolist()}
    return float(np.mean(best)), choice


def refit(cfg: BoostConfig, sd: SplitData, eng: Engine, S0s, S0es, choice, log, scale=1.1, seed_offset=0):
    """Refit on all fit rows with the chosen per-trait options (rounds x 1.1); predictions for the eval rows of the
    chosen option and of the best option without shared trees, and per-trait SNP gains: of the shared trees (first
    S_t rounds of trait t) and own trees (P_t rounds) of the chosen option, and their mean with the gains of the own
    trees (P0_t rounds) of the option without shared trees."""
    dev = sd.dev
    unit_of_s, Us = units(cfg.groups)
    unit_of_s = torch.as_tensor(unit_of_s, device=dev)
    unit_of_p = torch.arange(len(TRAITS), device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + seed_offset)
    sc = lambda v: (np.round(np.asarray(v) * scale)).astype(int)                   # noqa: E731
    S_t, P_t, P0_t = sc(choice["S_t"]), sc(choice["P_t"]), sc(choice["P0_t"])
    gb = torch.as_tensor([s_ == "gblup" for s_ in choice["start"]], device=dev)
    gb0 = torch.as_tensor([s_ == "gblup" for s_ in choice["start0"]], device=dev)
    zero, zeroe = torch.zeros_like(S0s), torch.zeros_like(S0es)
    n_t = sd.M.sum(0).float()
    c = torch.zeros(len(TRAITS), device=dev)
    for u in range(Us):
        m = unit_of_s == u
        c[m] = n_t[m].mean() / n_t[m].clamp_min(1)
    imp_s = torch.zeros(len(TRAITS), eng.p, device=dev)
    imp_p = torch.zeros(len(TRAITS), eng.p, device=dev)
    imp_0 = torch.zeros(len(TRAITS), eng.p, device=dev)
    lim = lambda v: torch.as_tensor(np.asarray(v), device=dev)                     # noqa: E731
    trees = []
    if cfg.groups != "none" and S_t.max() > 0:
        out = run_phase(eng, zero, sd.Y, sd.M.float() * c[None, :], unit_of_s, Us, cfg.depth_s, cfg.eta_s, cfg.lam,
                        cfg.gamma, cfg.mcw_s, cfg.colsample, cfg.subsample, int(S_t.max()), gen, imp=imp_s,
                        imp_rounds=lim(S_t))
        trees = out["trees"]
    FR = torch.where(gb[None, :], S0s, replay(trees, sd.XR, unit_of_s, cfg.depth_s, cfg.eta_s, S_t, zero))
    FE = torch.where(gb[None, :], S0es, replay(trees, sd.XE, unit_of_s, cfg.depth_s, cfg.eta_s, S_t, zeroe))

    def own(FR_, FE_, P, imp, seed):
        snaps = {}
        for j, p_ in enumerate(P):
            snaps.setdefault(int(p_), []).append(j)
        g_ = torch.Generator(device=dev).manual_seed(seed)
        o = run_phase(eng, FR_, sd.Y, sd.M.float(), unit_of_p, len(TRAITS), cfg.depth_p, cfg.eta_p, cfg.lam,
                      cfg.gamma, cfg.mcw_p, cfg.colsample, cfg.subsample, int(P.max()), g_, Xe=sd.XE, Fe0=FE_,
                      snaps=snaps, imp=imp, keep_trees=False, imp_rounds=lim(P))
        return o["snap"] if o["snap"] is not None else FE_
    pred = own(FR, FE, P_t, imp_p, cfg.seed + 11 + seed_offset)
    pred0 = own(torch.where(gb0[None, :], S0s, zero), torch.where(gb0[None, :], S0es, zeroe), P0_t, imp_0,
                cfg.seed + 12 + seed_offset)
    imp = 0.5 * (imp_s + imp_p) + 0.5 * imp_0            # the trait's model is the mean of the two refits
    return pred.cpu().numpy(), pred0.cpu().numpy(), imp_s.cpu().numpy(), imp_p.cpu().numpy(), S_t, P_t, P0_t, \
        imp.cpu().numpy()


SPACE = {
    "groups": ("cat", ["category", "all"]),
    "eta_s": ("float", 0.04, 0.15, True), "eta_p": ("float", 0.04, 0.15, True),
    "depth_s": ("int", 2, 5), "depth_p": ("int", 2, 5),
    "colsample": ("float", 0.02, 0.5, True), "subsample": ("float", 0.5, 1.0, False),
    "lam": ("float", 1e-3, 10.0, True), "mcw_p": ("float", 10.0, 200.0, True),
}


def suggest(trial, base):
    p = {}
    for k, spec in SPACE.items():
        if spec[0] == "cat":
            p[k] = trial.suggest_categorical(k, spec[1])
        elif spec[0] == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2])
        else:
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
    p["mcw_s"] = p["mcw_p"]
    return BoostConfig(**{**base, **p})


def refit_all(choices, pick, sd, eng, S0s, S0es, log, n_seeds, scale, first_seed=0):
    """One refit per distinct chosen trial and seed; every trait takes the predictions of its own trial."""
    Tn = len(TRAITS)
    preds = [(np.zeros((len(sd.E), Tn), np.float32), np.zeros((len(sd.E), Tn), np.float32)) for _ in range(n_seeds)]
    imp_s = np.zeros((Tn, eng.p), np.float32)
    imp_p = np.zeros((Tn, eng.p), np.float32)
    imp_m = np.zeros((Tn, eng.p), np.float32)             # mean over the refits that make up the final model
    per_trait = {}
    for k in sorted(set(pick)):
        cfg_, ch = choices[k]
        cols = [j for j in range(Tn) if pick[j] == k]
        for si in range(n_seeds):
            pr, pr0, i_s, i_p, S_t, P_t, P0_t, i_m = refit(cfg_, sd, eng, S0s, S0es, ch, log, scale=scale,
                                                           seed_offset=1000 * (si + first_seed))
            preds[si][0][:, cols], preds[si][1][:, cols] = pr[:, cols], pr0[:, cols]
            if si == 0:
                imp_s[cols], imp_p[cols] = i_s[cols], i_p[cols]
            imp_m[cols] += i_m[cols] / n_seeds
        for j in cols:
            per_trait[TRAITS[j]] = {"trial": k, "start": ch["start"][j], "S_t": int(S_t[j]), "P_t": int(P_t[j]),
                                    "start0": ch["start0"][j], "P0_t": int(P0_t[j]), "r_val": ch["r_val"][j],
                                    "r_val_noshare": ch["r_val_noshare"][j]}
        log(f"refit with trial {k} for {len(cols)} traits ({n_seeds} seeds)")
    return preds, per_trait, imp_s, imp_p, imp_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "collect", "start", "reseed"])
    ap.add_argument("--scenario", default="cv1_inner", choices=list(data.SCENARIOS) + ["all"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--config", default="{}", help="fixed settings (with --trials 0) or base settings for tuning")
    ap.add_argument("--gpu_mem_gb", type=float, default=4.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--chunk", type=int, default=40000, help="SNPs per histogram block (GPU memory)")
    ap.add_argument("--seeds", type=int, default=2, help="refit seeds (predictions of every seed are kept)")
    ap.add_argument("--first_seed", type=int, default=1, help="reseed: first extra seed index (1 -> column s2)")
    ap.add_argument("--optuna_seed", type=int, default=20261112,
                    help="seed of the Optuna sampler (repeated tuning runs: another seed, another --tag and "
                         "--config '{\"seed\": k}' for the random numbers of the trees)")
    ap.add_argument("--eval_all", action="store_true",
                    help="start: GBLUP predictions of every trait for all eval rows (cache used by --dump_full)")
    ap.add_argument("--dump_full", action="store_true",
                    help="reseed --first_seed 0 --seeds 2: recompute the four refits of the model and save its predictions "
                         "of all traits for all evaluation accessions (<split>_full.npz); the fold file is not changed")
    a = ap.parse_args()
    dev_dir = data.RUNS / a.scenario / "mtboost_dev" / a.tag
    if a.mode == "collect":
        fs = [dev_dir / f"{data.split_name(a.scenario, f)}.tsv.gz" for f in range(data.n_splits(a.scenario))]
        allp = pd.concat([pd.read_csv(f, sep="\t", dtype={"acc": str}) for f in fs if f.exists()], ignore_index=True)
        if "pred_s1" not in allp:                                   # fold 0 of cv1 / first inner runs: s1 = pred
            allp["pred_s1"] = allp["pred"]
        allp["pred_s1"] = allp["pred_s1"].fillna(allp["pred"])
        if "pred_noshare_s1" not in allp:
            allp["pred_noshare_s1"] = allp["pred_noshare"]
        allp["pred_noshare_s1"] = allp["pred_noshare_s1"].fillna(allp["pred_noshare"])
        sh = sorted(c for c in allp if c.startswith("pred_s") and allp[c].notna().all())
        ns = sorted(c for c in allp if c.startswith("pred_noshare_s") and allp[c].notna().all())
        variants = {"": ["pred_s1"], "_noshare": ["pred_noshare_s1"], "_gblup": ["pred_gblup"],
                    # TraitBoost: mean of the chosen option and the best option without shared trees, two seeds each
                    "_final": ["pred_s1", "pred_s2", "pred_noshare_s1", "pred_noshare_s2"],
                    f"_share{len(sh)}": sh, f"_noshare{len(ns)}": ns}
        for suffix, cols in variants.items():
            if not all(c in allp and allp[c].notna().all() for c in cols):
                print(f"mtboost_{a.tag}{suffix}: columns missing, skipped")
                continue
            out = data.RUNS / a.scenario / f"mtboost_{a.tag}{suffix}"
            out.mkdir(parents=True, exist_ok=True)
            for t, d in allp.groupby("trait"):
                d[["acc", "batch", "score", "resistant", "fold"]].assign(pred=d[cols].mean(1)).to_csv(
                    out / f"{t}.tsv.gz", sep="\t", index=False)
            print(f"mtboost_{a.tag}{suffix}: mean of {cols}")
        print(f"collected {allp.trait.nunique()} traits from {sum(f.exists() for f in fs)} of {len(fs)} splits")
        return
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
    if a.mode == "reseed":                    # extra refit seeds from the stored choices (no tuning); adds columns
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / tot), 0)
        t0 = time.time()
        log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
        out_all = data.RUNS / "all" / f"mtboost_{a.tag}"
        sp = data.split_name(a.scenario, a.fold) if a.scenario != "all" else "all"
        js = json.load(open(out_all / "run.json" if a.scenario == "all" else dev_dir / f"{sp}.json"))
        sd = SplitData(a.scenario, a.fold, device, log, long=use_traitset("stress"))
        S0, S0e = sd.gblup_start(log)
        if a.dump_full:                       # every test accession needs the GBLUP start of every trait
            S0e_all = sd.gblup_eval_all(log)
            scored = np.zeros(S0e.shape, bool)
            ee_ = {x: i for i, x in enumerate(sd.E)}
            for j, t in enumerate(TRAITS):
                scored[[ee_[x] for x in sd.tg[t].test_acc], j] = True
            log(f"GBLUP start of the scored eval rows: max difference to the cached start "
                f"{float(np.abs(S0e_all - S0e)[scored].max()):.2e}")
            S0e = S0e_all
        S0s = torch.as_tensor((S0 - sd.mu) / sd.sd, dtype=torch.float32, device=device)
        S0es = torch.as_tensor((S0e - sd.mu) / sd.sd, dtype=torch.float32, device=device)
        eng = Engine(sd.XR, chunk=a.chunk)
        choices, pick = {}, []
        for t in TRAITS:
            pick.append(int(js["per_trait"][t]["trial"]))
        for k in sorted(set(pick)):
            cfg_ = BoostConfig(**js["configs"][str(k)])
            ch = {key: [js["per_trait"][t][key] if pick[j] == k else (0 if key in ("S_t", "P_t", "P0_t") else "zero")
                        for j, t in enumerate(TRAITS)] for key in ("start", "S_t", "P_t", "start0", "P0_t")}
            ch["r_val"] = [0.0] * len(TRAITS)
            ch["r_val_noshare"] = [0.0] * len(TRAITS)
            choices[k] = (cfg_, ch)
        preds, _, imp_s, imp_p, imp_m = refit_all(choices, pick, sd, eng, S0s, S0es, log, a.seeds, scale=1.0,
                                                  first_seed=a.first_seed)
        if a.scenario == "all":               # split gains of the final model (same seeds as the fit: --first_seed 0)
            assert a.first_seed == 0 and a.seeds == 2
            z = np.load(out_all / "predictions.npz", allow_pickle=True)
            back = lambda v: v * sd.sd[None, :] + sd.mu[None, :]                     # noqa: E731
            dmax = max(float(np.abs(back(pr) - z[f"share_s{si}"]).max()) for si, (pr, _) in enumerate(preds, start=1))
            d0 = max(float(np.abs(back(pr0) - z[f"noshare_s{si}"]).max()) for si, (_, pr0) in enumerate(preds, start=1))
            log(f"refits reproduce the stored predictions: max difference {dmax:.2e} (shared), {d0:.2e} (not shared)")
            np.savez_compressed(out_all / "importance.npz", shared=imp_s.astype(np.float32), own=imp_p.astype(np.float32),
                                model=imp_m.astype(np.float32), traits=np.array(TRAITS))
            log(f"split gains of the final model -> {out_all / 'importance.npz'}")
            return
        R = pd.read_csv(dev_dir / f"{sp}.tsv.gz", sep="\t", dtype={"acc": str})
        ee = {x: i for i, x in enumerate(sd.E)}
        if a.dump_full:      # predictions of all traits for all evaluation accessions (the fold files hold scored traits only)
            assert a.first_seed == 0 and a.seeds == 2
            if "pred_s1" not in R:
                R["pred_s1"], R["pred_noshare_s1"] = R["pred"], R["pred_noshare"]
            back = lambda v: v * sd.sd[None, :] + sd.mu[None, :]                     # noqa: E731
            four = [back(x) for pr in preds for x in pr]          # s1 shared, s1 not shared, s2 shared, s2 not shared
            dmax = 0.0
            for j, t in enumerate(TRAITS):
                m = (R.trait == t).to_numpy()
                if not m.any():
                    continue
                ix = np.array([ee[x] for x in R.acc[m]], dtype=int)
                for k, col in enumerate(("pred_s1", "pred_noshare_s1", "pred_s2", "pred_noshare_s2")):
                    dmax = max(dmax, float(np.abs(four[k][ix, j] - R.loc[m, col].to_numpy()).max()))
            log(f"refits against the stored predictions of the scored traits: max difference {dmax:.2e}")
            np.savez_compressed(dev_dir / f"{sp}_full.npz", acc=sd.E, traits=np.array(TRAITS), final=np.mean(four, 0).astype(np.float32),
                                max_difference_to_stored=dmax)
            log(f"predictions of all traits for {len(sd.E)} accessions -> {dev_dir / (sp + '_full.npz')}")
            return
        for j, t in enumerate(TRAITS):
            m = (R.trait == t).to_numpy()
            if not m.any():                    # cv3: only the traits of the held-out experiment have test records
                continue
            ix = np.array([ee[x] for x in R.acc[m]], dtype=int)
            for si, (pr, pr0) in enumerate(preds, start=a.first_seed + 1):     # column s<k> = seed offset 1000(k-1)
                R.loc[m, f"pred_s{si}"] = pr[ix, j] * sd.sd[j] + sd.mu[j]
                R.loc[m, f"pred_noshare_s{si}"] = pr0[ix, j] * sd.sd[j] + sd.mu[j]
        R.to_csv(dev_dir / f"{sp}.tsv.gz", sep="\t", index=False)
        log(f"added {len(preds)} refit seed(s) to {sp}.tsv.gz")
        return
    if a.mode == "start":                     # GBLUP starts on the CPU, cached for the GPU runs
        t0 = time.time()
        log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
        sd = SplitData(a.scenario, a.fold, torch.device("cpu"), log, long=use_traitset("stress"))
        sd.gblup_eval_all(log) if a.eval_all else sd.gblup_start(log)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / tot), 0)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
    sd = SplitData(a.scenario, a.fold, device, log, long=use_traitset("stress"))
    S0, S0e = sd.gblup_start(log)
    S0s = torch.as_tensor((S0 - sd.mu) / sd.sd, dtype=torch.float32, device=device)
    S0es = torch.as_tensor((S0e - sd.mu) / sd.sd, dtype=torch.float32, device=device)
    eng = Engine(sd.XR, chunk=a.chunk)
    base = {**BoostConfig().__dict__, **json.loads(a.config)}
    trials, choices = [], {}
    if a.trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.optuna_seed))

        def objective(trial):
            cfg_ = suggest(trial, base)
            ts = time.time()
            val, ch = trial_eval(cfg_, sd, eng, S0s, log)
            choices[trial.number] = (cfg_, ch)
            trials.append({"trial": trial.number, "value": val, "seconds": round(time.time() - ts, 1),
                           **{k: getattr(cfg_, k) for k in SPACE}})
            log(f"trial {trial.number}: value {val:.4f} ({time.time() - ts:.0f}s) "
                f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in trial.params.items()} }")
            return val
        study.optimize(objective, n_trials=a.trials)
    else:
        cfg_ = BoostConfig(**base)
        val, ch = trial_eval(cfg_, sd, eng, S0s, log)
        choices[0] = (cfg_, ch)
        trials.append({"trial": 0, "value": val, **{k: getattr(cfg_, k) for k in SPACE}})
        log(f"fixed setting: value {val:.4f}")
    # every trait takes the trial (setting) with its best fold-averaged validation r (as LightGBM is tuned per trait)
    nums = sorted(choices)
    rv = np.array([choices[k][1]["r_val"] for k in nums])                       # trials x T
    pick = [nums[i] for i in rv.argmax(0)]
    preds, per_trait, imp_s, imp_p, imp_m = refit_all(choices, pick, sd, eng, S0s, S0es, log, a.seeds, scale=1.1)
    if a.scenario == "all":
        out = data.RUNS / "all" / f"mtboost_{a.tag}"
        out.mkdir(parents=True, exist_ok=True)
        back = lambda v: v * sd.sd[None, :] + sd.mu[None, :]                         # noqa: E731
        arrs = {"acc": sd.E, "traits": np.array(TRAITS), "gblup": back((S0e - sd.mu) / sd.sd)}
        for si, (pr, pr0) in enumerate(preds, start=1):
            arrs[f"share_s{si}"], arrs[f"noshare_s{si}"] = back(pr), back(pr0)
        arrs["final"] = np.mean([arrs[k] for k in arrs if k.startswith(("share_s", "noshare_s"))], 0)
        np.savez_compressed(out / "predictions.npz", **arrs)
        np.savez_compressed(out / "importance.npz", shared=imp_s.astype(np.float32), own=imp_p.astype(np.float32),
                            model=imp_m.astype(np.float32), traits=np.array(TRAITS))
        json.dump({"scenario": "all", "traitset": "stress", "tag": a.tag, "code": CODE, "trials": trials,
                   "configs": {k: choices[k][0].__dict__ for k in sorted(set(pick))}, "per_trait": per_trait,
                   "seconds": round(time.time() - t0, 1)}, open(out / "run.json", "w"), indent=1, default=float)
        log(f"done: predictions for {len(sd.E)} accessions -> {out}")
        return
    ee = {x: i for i, x in enumerate(sd.E)}
    recs = []
    for j, t in enumerate(TRAITS):
        ft = sd.tg[t]
        if not len(ft.test_acc):
            continue
        rec = ft.test_records
        ix = np.array([ee[x] for x in rec.acc])
        back = lambda v: v[ix, j] * sd.sd[j] + sd.mu[j]                              # noqa: E731
        cols = {"pred_gblup": back((S0e - sd.mu) / sd.sd)}
        for si, (pr, pr0) in enumerate(preds, start=1):
            cols[f"pred_s{si}"], cols[f"pred_noshare_s{si}"] = back(pr), back(pr0)
        cols["pred"], cols["pred_noshare"] = cols["pred_s1"], cols["pred_noshare_s1"]
        recs.append(rec.assign(trait=t, fold=a.fold, **cols))
    sp = data.split_name(a.scenario, a.fold)
    dev_dir.mkdir(parents=True, exist_ok=True)
    R = pd.concat(recs, ignore_index=True)
    R.to_csv(dev_dir / f"{sp}.tsv.gz", sep="\t", index=False)
    np.savez_compressed(dev_dir / f"{sp}.imp.npz", shared=imp_s.astype(np.float32), own=imp_p.astype(np.float32),
                        traits=np.array(TRAITS))
    json.dump({"scenario": a.scenario, "fold": a.fold, "split": sp, "tag": a.tag, "code": CODE, "trials": trials,
               "configs": {k: choices[k][0].__dict__ for k in sorted(set(pick))}, "per_trait": per_trait,
               "seconds": round(time.time() - t0, 1)}, open(dev_dir / f"{sp}.json", "w"), indent=1, default=float)
    borrow = [T.SHORT[t] for t, v in per_trait.items() if v["S_t"] > 0]
    log(f"done: {len(borrow)} traits borrow shared trees: {borrow}")


if __name__ == "__main__":
    main()
