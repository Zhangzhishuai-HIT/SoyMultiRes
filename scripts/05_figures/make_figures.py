"""Main figures (printed size, Times New Roman, PDF + 600-dpi PNG in figures/).

usage: python make_figures.py [fig1 ... fig6]
fig1  overview of the study (data, MT-Boost, evaluation, locus discovery and germplasm)
fig2  stress records of the collection: scored accession x trait cells and associations between traits
fig3  prediction: new accessions (model means, MT-Boost vs the best single-trait model, contribution of the shared
      trees) and held-out experiments (model means)
fig4  known-gene calibration of the locus read-outs
fig5  shared loci: genome-wide map, allele directions in the chromosome 13 region, race-differential locus at Rhg1
fig6  multi-resistant germplasm: out-of-fold validation and the multi-resistance index
Colours (figstyle.py): MT-Boost red, comparators navy/ochre/teal/greys; stress categories, locus classes and allele
directions each have a fixed palette checked with the dataviz palette validator.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
from figstyle import plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402
from soymultires import traits as T  # noqa: E402

PRIM = set(data.primary_traits())
CAT = {row[0]: row[4] for row in T.STRESS}
CATS = list(fs.CATEGORY)


def model_marker(ax, x, y, m, s=26, z=4):
    lab, col, mk, filled = fs.MODEL[m]
    ax.scatter(x, y, s=s, marker=mk, color=col if filled else "white", edgecolors=col,
               linewidths=0.4 if filled else 0.9, zorder=z)


# ---------------------------------------------------------------- Fig. 1 overview
def fig1():
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    W, H = 174, 104
    fig = plt.figure(figsize=(fs.W2, H * fs.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ink2, muted = fs.INK2, fs.MUTED

    def stage(x, y, w, h, letter, title, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fs.PANEL_BG,
                                    ec="#d5d9e0", lw=0.6, zorder=0))
        ax.add_patch(Rectangle((x + 2.5, y + h - 2.2), 9, 0.9, fc=colour, ec="none", zorder=1))
        ax.text(x + 2.4, y + h - 3.8, title, fontsize=7.6, fontweight="bold", va="top", ha="left", zorder=2)
        ax.text(x - 0.4, y + h + 0.6, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    def txt(x, y, s, size=6, colour=None, **kw):
        ax.text(x, y, s, fontsize=size, color=colour or ink2, va="top", ha=kw.pop("ha", "left"), linespacing=1.3, **kw)

    def arrow(p0, p1, colour="#7a8494"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=colour, zorder=3))

    # (a) genebank records
    stage(2, 3, 47, 96, "a", "Genebank records", fs.NAVY)
    txt(4.5, 91.5, "USDA Soybean Germplasm Collection")
    rng = np.random.default_rng(3)
    x0, y0, cw, ch, nr = 11.5, 41.5, 1.78, 1.55, 28
    cols = [c for c in CATS for _ in range({"SCN": 3, "Phytophthora": 4}.get(c, 2))]
    ax.add_patch(Rectangle((x0 - 0.4, y0 - 0.4), cw * len(cols) + 0.8, ch * nr + 0.8, fc="white", ec="#c8ccd4",
                           lw=0.5, zorder=1))
    for j, c in enumerate(cols):
        dens = {"SCN": 0.55, "Phytophthora": 0.35}.get(c, 0.16)
        for i in range(nr):
            if rng.random() < dens * (1.25 - i / nr):
                ax.add_patch(Rectangle((x0 + j * cw + 0.13, y0 + (nr - 1 - i) * ch + 0.15), cw - 0.26, ch - 0.3,
                                       fc=fs.CATEGORY[c], ec="none", zorder=2))
    for i in range(nr):                                  # genotype strip
        ax.add_patch(Rectangle((x0 - 5.0, y0 + i * ch + 0.15), 3.4, ch - 0.3,
                               fc=str(0.35 + 0.5 * rng.random()), ec="none", zorder=2))
    txt(x0 - 5.4, y0 - 1.4, "SNPs", 5.5, muted)
    ax.text(x0 - 6.9, y0 + nr * ch / 2, "accessions", rotation=90, fontsize=5.5, color=muted, va="center", ha="center")
    txt(x0 + cw * len(cols) / 2, y0 - 1.4, "stress traits", 5.5, muted, ha="center")
    txt(4.5, 35.5, "17,150 accessions with stress records\n32 stress traits in 8 categories\n"
                   "81% of accession × trait cells unscored\n39,627 SoySNP50K SNPs")
    for k, c in enumerate(CATS):
        cx, cy = 5 + (k % 2) * 22, 16.8 - (k // 2) * 3.4
        ax.add_patch(Rectangle((cx, cy - 1.7), 2.2, 1.6, fc=fs.CATEGORY[c], ec="none"))
        txt(cx + 3.0, cy, fs.CATEGORY_LABEL[c], 5.6)

    # (b) MT-Boost
    stage(56, 3, 56, 96, "b", "MT-Boost", fs.RED)
    txt(58.5, 91.5, "Multi-task gradient boosting on GPU")

    def tree(cx, top, colour, leaves_multi=None, scale=1.0):
        dx, dy = 5.2 * scale, 5.0 * scale
        nodes = [(cx, top), (cx - dx, top - dy), (cx + dx, top - dy)]
        leaves = [(cx - dx - dx / 2, top - 2 * dy), (cx - dx + dx / 2, top - 2 * dy),
                  (cx + dx - dx / 2, top - 2 * dy), (cx + dx + dx / 2, top - 2 * dy)]
        for (px, py), (qx, qy) in [(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], leaves[0]),
                                   (nodes[1], leaves[1]), (nodes[2], leaves[2]), (nodes[2], leaves[3])]:
            ax.plot([px, qx], [py, qy], color="#9aa3b2", lw=0.6, zorder=2)
        for (px, py) in nodes:
            ax.add_patch(Circle((px, py), 1.1 * scale, fc="white", ec=colour, lw=0.9, zorder=3))
        for k, (px, py) in enumerate(leaves):
            if leaves_multi:
                for m, lc in enumerate(leaves_multi):
                    h = (0.8 + 1.6 * ((k * 3 + m * 5) % 7) / 6) * scale
                    ax.add_patch(Rectangle((px - 1.5 * scale + m * 1.05 * scale, py - h), 0.9 * scale, h, fc=lc,
                                           ec="none", zorder=3))
            else:
                ax.add_patch(Rectangle((px - 0.9 * scale, py - 1.6 * scale), 1.8 * scale, 1.6 * scale, fc=colour,
                                       ec="none", zorder=3))

    tree(69.5, 85, fs.CATEGORY["SCN"], leaves_multi=["#2F4E9E", "#6f86c4", "#a9b8e0"])   # shared tree (SCN races)
    txt(58.5, 70.5, "Shared trees, one per stress\ncategory: one split for all its\ntraits, a leaf value per trait",
        5.7)
    for k, c in enumerate(["#2F4E9E", "#E64B35", "#00A087"]):                            # trait-specific trees
        tree(88.0 + k * 8.9, 85, c, scale=0.5)
    txt(85.5, 76.5, "Trait-specific trees, one\nper trait and round", 5.7)
    ax.text(81.2, 80.5, "+", fontsize=9, ha="center", va="center", color=muted)
    ax.add_patch(FancyBboxPatch((58.5, 50.0), 51, 10.5, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec="#d5d9e0", lw=0.5, zorder=1))
    ax.text(84, 57.6, "prediction  =  start  +  S shared rounds  +  P own rounds", fontsize=6.1,
            ha="center", va="center", zorder=2)
    ax.text(84, 53.0, "start: zero or cross-validated GBLUP", fontsize=5.6, color=muted, ha="center", va="center",
            zorder=2)
    txt(58.5, 46.0, "Per-trait choice on three inner folds", 6.0, fs.INK, fontweight="bold")
    opts = [("zero + own trees", False), ("zero + S shared + own", True), ("GBLUP + own trees", False)]
    for k, (lab, chosen) in enumerate(opts):
        yy = 39.5 - k * 5.0
        ax.add_patch(FancyBboxPatch((59.5, yy - 1.8), 30, 3.6, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="#fdecea" if chosen else "white", ec=fs.RED if chosen else "#c8ccd4", lw=0.6,
                                    zorder=2))
        ax.text(74.5, yy, lab, fontsize=5.6, ha="center", va="center", color=fs.INK if chosen else ink2, zorder=3)
        if chosen:
            ax.text(91.6, yy, "✓", fontsize=7, ha="left", va="center", color=fs.RED, zorder=3)
    txt(95.0, 41.0, "highest\ninner\naccuracy", 5.4, muted)
    txt(58.5, 21.0, "Final prediction: mean of the chosen model\nand the best model without shared trees\n"
                    "(two seeds each)", 5.8)
    txt(58.5, 10.6, "A tree level for all traits and SNPs is one\nmatrix product of allele carriers and gradients",
        5.6, muted)

    # (c) evaluation
    stage(118, 54, 54, 45, "c", "Evaluation", fs.NAVY)
    ys = [88.5, 78.8, 69.1]
    labels = [("New accessions", "5-fold cross-validation; near-identical\naccessions in the same fold"),
              ("Measured traits", "one trait hidden with all traits of\nits experiment"),
              ("New experiments", "the largest experiment of 22 traits\nheld out (16 experiments)")]
    for k, (y, (h1, h2)) in enumerate(zip(ys, labels)):
        for j in range(5):
            if k == 0:
                fc = fs.RED if j == 2 else "#cfd6e4"
            elif k == 1:
                fc = "#cfd6e4" if j != 3 else "white"
            else:
                fc = "#cfd6e4" if j != 1 else fs.RED
            ax.add_patch(Rectangle((121 + j * 2.6, y - 3.0), 2.2, 3.0, fc=fc,
                                   ec="#9aa3b2" if fc == "white" else "none", lw=0.4, zorder=2))
        if k == 1:
            ax.text(121 + 3 * 2.6 + 1.1, y - 1.5, "×", fontsize=6, ha="center", va="center", color=fs.RED, zorder=3)
        txt(135.5, y + 0.6, h1, 6.2, fs.INK, fontweight="bold")
        txt(135.5, y - 2.0, h2, 5.5)
    txt(121, 60.2, "Compared with GBLUP, BayesB, LightGBM\nand multi-trait GBLUP", 5.5, muted)

    # (d) loci and germplasm
    stage(118, 3, 54, 45, "d", "Loci and germplasm", fs.RED)
    for j, hgt in enumerate([9, 8.5, 8, 7.6, 7, 6.6, 6, 5.5, 5, 4.6, 2.2, 1.4]):      # calibration
        ax.add_patch(Rectangle((121 + j * 1.3, 29.0), 1.0, hgt * 0.62, fc=fs.NAVY if j < 10 else "#c8c8c8",
                               ec="none", zorder=2))
    ax.plot([120.6, 136.6], [32.0, 32.0], color=muted, lw=0.5, ls=(0, (2, 1.5)), zorder=3)
    txt(138.5, 38.4, "Calibration: 10 of 12 known\ngenes in the top 1% of blocks", 5.8)
    txt(121, 26.6, "133 loci shared by ≥ 2 traits; allele directions:", 5.8)
    glyphs = [("concordant", "+ + +", "59 concordant"), ("race-differential", "+ + −", "2 race-differential"),
              ("trade-off", "+ | −", "10 trade-off")]
    for k, (cls, sig, lab) in enumerate(glyphs):
        cx = 124 + k * 16
        ax.add_patch(FancyBboxPatch((cx - 3.5, 18.2), 12.5, 5.2, boxstyle="round,pad=0,rounding_size=1",
                                    fc="white", ec=fs.CLASS_COLOUR[cls], lw=0.8, zorder=2))
        ax.text(cx + 2.75, 20.8, sig, fontsize=7, ha="center", va="center", color=fs.CLASS_COLOUR[cls],
                fontweight="bold", zorder=3)
        ax.text(cx + 2.75, 16.6, lab, fontsize=5.4, ha="center", va="top", color=ink2)
    for k, c in enumerate(CATS):                                                       # germplasm
        ax.add_patch(Circle((123 + k * 2.6, 8.0), 1.0, fc=fs.CATEGORY[c] if k in (0, 1, 2, 5) else "white",
                            ec=fs.CATEGORY[c], lw=0.6, zorder=2))
    txt(144.0, 10.6, "Multi-resistance index: 347 cultivated\nand 9 wild accessions in the top 10%\nfor at least 4 of 8 categories",
        5.6)
    arrow((49.6, 51), (55.4, 51))
    arrow((112.6, 77), (117.4, 77))
    arrow((112.6, 26), (117.4, 26))
    fs.save(fig, "fig1_overview")


# ---------------------------------------------------------------- Fig. 2 data
def fig2():
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    long = data.pheno_long()
    geno = set(data.samples())
    lg = long[long.acc.isin(geno)]
    traits = list(T.TRAITS)
    M = lg.groupby(["acc", "trait"]).size().unstack().reindex(columns=traits).notna()
    pat = M.to_numpy()
    order = np.lexsort([pat[:, ::-1].T[k] for k in range(pat.shape[1])][::-1] + [-pat.sum(1)])
    pat = pat[order]
    img = np.ones(pat.shape + (3,))
    for j, t in enumerate(traits):
        img[pat[:, j], j] = to_rgb(fs.CATEGORY[CAT[t]])
    fig = plt.figure(figsize=(fs.W2, 92 * fs.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3, left=0.075, right=0.95, top=0.86, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=5.3)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_yticks([0, len(pat) - 1], ["1", f"{len(pat):,}"])
    ax.set_ylabel("Accessions (sorted by traits scored)")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#9aa3b2")
    starts = {}
    for j, t in enumerate(traits):
        starts.setdefault(CAT[t], []).append(j)
    nrow = len(pat)
    for k, (c, js) in enumerate(starts.items()):
        # category track above the matrix; labels in two staggered rows so that narrow categories do not collide
        ax.add_patch(plt.Rectangle((js[0] - 0.45, -nrow * 0.028), len(js) - 0.1, nrow * 0.018, color=fs.CATEGORY[c],
                                   lw=0, clip_on=False))
        ax.text((js[0] + js[-1]) / 2, -nrow * (0.036 if k % 2 == 0 else 0.075),
                fs.CATEGORY_LABEL[c].replace(" insects", ""), ha="center", va="bottom", fontsize=5.6,
                color=fs.CATEGORY[c], fontweight="bold", clip_on=False)
        if js[0] > 0:
            ax.axvline(js[0] - 0.5, color="white", lw=1.0)
    ax.set_ylim(nrow - 0.5, -0.5)
    fs.panel(ax, "a", x=-0.16, y=1.14)
    ax = fig.add_subplot(gs[1])
    pa = pd.read_csv(ROOT / "results/pheno/pair_associations.tsv", sep="\t")
    idx = {t: i for i, t in enumerate(traits)}
    A = np.full((len(traits), len(traits)), np.nan)
    tested = np.zeros_like(A, bool)
    for r in pa.itertuples():
        i, j = sorted((idx[r.a], idx[r.b]), reverse=True)
        tested[i, j] = True
        if r.q_mh < 0.05:
            A[i, j] = np.clip(np.log2(max(r.or_mh, 1e-3)), -5, 5)
    cmap = LinearSegmentedColormap.from_list("div", fs.DIVERGING)
    base = np.ones((len(traits), len(traits), 3))
    base[tested & np.isnan(A)] = to_rgb("#e3e3e3")
    ax.imshow(base, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-5, vmax=5, interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=4.6)
    ax.set_yticks(range(len(traits)), [T.SHORT[t] for t in traits], fontsize=4.6)
    ax.tick_params(length=0, pad=1)
    for lab_ in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab_.set_color(fs.CATEGORY[CAT[traits[[T.SHORT[t] for t in traits].index(lab_.get_text())]]])
    n_ = len(traits)
    for c, js in starts.items():
        k = js[0]
        if k > 0:
            ax.plot([-0.5, k - 0.5], [k - 0.5, k - 0.5], color="#b9bec8", lw=0.4)
            ax.plot([k - 0.5, k - 0.5], [k - 0.5, n_ - 0.5], color="#b9bec8", lw=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="both")
    cb.set_ticks([-4, -2, 0, 2, 4], labels=["1/16", "1/4", "1", "4", "16"])
    cb.set_label("Odds ratio of resistance", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.add_patch(plt.Rectangle((21.6, 0.6), 1.0, 1.0, fc="#e3e3e3", ec="none"))
    ax.text(23.0, 1.1, "tested, not significant", fontsize=5.4, va="center", color=fs.INK2)
    fs.panel(ax, "b", x=-0.14, y=1.03)
    fs.save(fig, "fig2_data")


# ---------------------------------------------------------------- Fig. 3 prediction
def means(fig, spec, frame, models, sets, letter, xlabel):
    """Mean accuracy of each model (rows) for two trait sets (two small axes)."""
    sub = spec.subgridspec(1, 2, wspace=0.14)
    for k, (sel, lab) in enumerate(sets):
        ax = fig.add_subplot(sub[k])
        y = np.arange(len(models))[::-1]
        for i, m in enumerate(models):
            model_marker(ax, frame[sel][m].mean(), y[i], m, s=24)
        short = {"MT-Boost without shared trees": "MT-Boost, no sharing", "Best single-trait model": "Best single-trait"}
        ax.set_yticks(y, [short.get(fs.MODEL[m][0], fs.MODEL[m][0]) for m in models] if k == 0 else [], fontsize=6.2)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.grid(axis="x", color=fs.GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(lab, fontsize=6.3, color=fs.INK2, pad=3)
        v = frame[sel][models].mean()
        lo, hi = v.min(), v.max()
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=5.6)
        if k == 0:
            pos = ax.get_position()
            fig.text(0.012, pos.y1 + 0.022, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
            ax.text(1.07, -0.15, xlabel, transform=ax.transAxes, ha="center", va="top", fontsize=6.6)


def fig3():
    pe = pd.read_csv(ROOT / "results/compare/cv1_point_estimates.tsv", sep="\t")
    pe3 = pd.read_csv(ROOT / "results/compare/cv3_point_estimates.tsv", sep="\t")
    cmp_ = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_best_single_avg4_{s}.tsv", sep="\t")
                      for s in ("primary", "other")])
    shr = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_mtboost_v2_noshare4_{s}.tsv", sep="\t")
                     for s in ("primary", "other")])
    fig = plt.figure(figsize=(fs.W2, 132 * fs.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1, 0.8], hspace=0.42, wspace=0.6,
                          left=0.13, right=0.985, top=0.95, bottom=0.09)
    means(fig, gs[0, 0], pe, ["gblup", "bglr_mt", "bayesb", "lightgbm_avg4", "best_single_avg4",
                              "mtboost_v2_noshare4", "mtboost_v2_final"],
          ((pe.prim, "Primary traits (16)"), (~pe.prim, "Other traits (16)")), "a", "Mean accuracy, new accessions")
    means(fig, gs[1, 0], pe3, ["mtboost_v2_gblup", "bayesb", "lightgbm_avg4", "mtboost_v2_noshare2", "mtboost_v2_final"],
          ((pe3.prim, f"Primary traits ({int(pe3.prim.sum())})"), (~pe3.prim, f"Other traits ({int((~pe3.prim).sum())})")),
          "d", "Mean accuracy, held-out experiments")
    for k, (df, title, letter) in enumerate(((cmp_, "MT-Boost − best single-trait model", "b"),
                                              (shr, "MT-Boost − MT-Boost without sharing", "c"))):
        ax = fig.add_subplot(gs[:, k + 1])
        d = df.assign(prim=df.trait.isin(PRIM)).sort_values(["prim", "diff"], ascending=[False, True])
        y = np.arange(len(d))
        sig = (d.p_holm < 0.05).to_numpy()
        np_ = int(d.prim.sum())
        ax.axhspan(-0.5, np_ - 0.5, color="#f7f4f3", lw=0, zorder=0)
        ax.axvline(0, color="#9aa3b2", lw=0.6, zorder=1)
        ax.hlines(y, d.lo, d.hi, color="#c3c8d0", lw=0.9, zorder=2)
        ax.scatter(d["diff"], y, s=14, color=np.where(sig, fs.RED, "white"), edgecolors=fs.RED, linewidths=0.7,
                   zorder=3)
        ax.set_yticks(y, d.short)
        ax.tick_params(axis="y", length=0, labelsize=5.8)
        for lab_, cat_ in zip(ax.get_yticklabels(), d.trait.map(CAT)):
            lab_.set_color(fs.CATEGORY[cat_])
        ax.set_xlabel("Difference in accuracy")
        ax.set_title(title, fontsize=6.6, pad=4)
        ax.set_ylim(len(d) - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        # y axis runs downwards: the primary traits (shaded) are at the top
        ax.text(0.99, 1 - (np_ / 2) / len(d), "primary\ntraits", transform=ax.transAxes, fontsize=5.6,
                color=fs.MUTED, ha="right", va="center")
        ax.text(0.99, (len(d) - np_) / 2 / len(d), "other\ntraits", transform=ax.transAxes, fontsize=5.6,
                color=fs.MUTED, ha="right", va="center")
        fs.panel(ax, letter, x=-0.27, y=1.015)
    fs.save(fig, "fig3_prediction")


# ---------------------------------------------------------------- Fig. 4 calibration
def fig4():
    p = pd.read_csv(ROOT / "results/calibration/percentiles.tsv", sep="\t")
    p = p[p.window == "block"]
    order = ["W1", "T", "Dt1", "I", "E1", "E2", "E3", "E4", "Rhg1", "Rhg4", "Rps1", "GmSALT3"]
    fig, ax = plt.subplots(figsize=(fs.W1, 64 * fs.MM))
    x = np.arange(len(order))
    ax.axhspan(0.008, 1, color="#eef2f9", lw=0, zorder=0)
    for off, (ro, lab, col, mk) in zip((-0.16, 0.16), (("emmax", "Association statistic", fs.NAVY, "s"),
                                                        ("mtboost", "MT-Boost split gain", fs.RED, "o"))):
        v = p[p.readout == ro].set_index("gene").percentile.reindex(order).clip(lower=0.01)
        ax.scatter(x + off, v, s=16, color=col, marker=mk, label=lab, zorder=3, linewidths=0.4, edgecolors="white")
    ax.axhline(1, color=fs.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_ylim(100, 0.008)
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["≤0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_ylabel("Rank of the gene's block (top %)")
    ax.set_xticks(x, order, rotation=45, ha="right", fontstyle="italic")
    ax.tick_params(axis="x", length=0)
    ax.text(len(order) - 0.5, 0.85, "top 1%", fontsize=6, color=fs.MUTED, ha="right", va="bottom")
    ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.3)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
    fs.save(fig, "fig4_known_genes")


# ---------------------------------------------------------------- Fig. 5 loci
def fig5():
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    chroms = sorted(snps.chrom.unique(), key=lambda c: int(c[2:]))
    length = snps.groupby("chrom").pos.max()
    off, acc = {}, 0
    for c in chroms:
        off[c] = acc
        acc += length[c] + 5e6
    loci["x"] = [off[c] + (s_ + e) / 2 for c, s_, e in zip(loci.chrom, loci.start, loci.end)]
    fig = plt.figure(figsize=(fs.W2, 128 * fs.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.75, 1], hspace=0.62, wspace=0.28,
                          left=0.07, right=0.98, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, :])
    for i, c in enumerate(chroms):
        if i % 2 == 0:
            ax.axvspan(off[c], off[c] + length[c], color=fs.BAND, lw=0, zorder=0)
    for k, cls in enumerate(("specific", "unresolved", "concordant", "trade-off", "race-differential")):
        d = loci[loci["class"] == cls]
        size = 6 + 3 * d.n_traits
        if cls == "race-differential":
            # next to another block with as many traits (Rhg1: a trade-off block 0.6 Mb away): drawn smaller on top
            others = loci[loci["class"] != cls]
            hidden = [((others.chrom == r.chrom) & (others.n_traits == r.n_traits)
                       & ((others.x - r.x).abs() < 5e6)).any() for r in d.itertuples()]
            size = np.where(hidden, size * 0.35, size)
        ax.scatter(d.x, d.n_traits, s=size, color=fs.CLASS_COLOUR[cls], label=cls.capitalize(), zorder=3 + k,
                   linewidths=0.35, edgecolors="white" if cls != "specific" else "#cfcfcf")
    h, l_ = ax.get_legend_handles_labels()
    order = [l_.index(c) for c in ("Concordant", "Race-differential", "Trade-off", "Unresolved", "Specific")]
    notes = [("Gm13", 29.1e6, 32.1e6, "Chr. 13 region", "normal"), ("Gm03", 3.0e6, 5.8e6, "Rps1", "italic"),
             ("Gm18", 1.1e6, 1.4e6, "Rhg1", "italic"), ("Gm18", 55.6e6, 56.9e6, "Rpp1", "italic"),
             ("Gm17", 6.7e6, 7.2e6, "SCN", "normal"), ("Gm11", 32.7e6, 33.1e6, "GmSNAP11", "italic")]
    for c, a_, b_, lab, style in notes:
        d = loci[(loci.chrom == c) & (loci.end >= a_) & (loci.start <= b_)]
        if len(d):
            ax.annotate(lab, (d.x.mean(), d.n_traits.max()), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6, color=fs.INK2, fontstyle=style)
    ax.set_xticks([off[c] + length[c] / 2 for c in chroms], [c[2:].lstrip("0") for c in chroms])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-2e6, acc)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Traits")
    ax.set_ylim(0.3, loci.n_traits.max() + 1.8)
    ax.legend([h[i] for i in order], [l_[i] for i in order], loc="upper left", ncol=5, handletextpad=0.1,
              columnspacing=0.9, bbox_to_anchor=(0, 1.2))
    fs.panel(ax, "a", x=-0.055, y=1.12)
    ax = fig.add_subplot(gs[1, 0])                    # (b) allele directions in the chromosome 13 region
    reg = loci[(loci.chrom == "Gm13") & (loci.start >= 29.1e6) & (loci.end <= 32.1e6)].sort_values("start")
    dirs = {}
    for _, r in reg.iterrows():
        for item in r.directions.split(","):
            m = item.rstrip("+-0")
            dirs[(r.block, m)] = item[len(m):]
    shorts = [T.SHORT[t] for t in T.TRAITS]
    rows = [t for t in shorts if any((b_, t) in dirs for b_ in reg.block)]
    for j, b_ in enumerate(reg.block):
        for i, t in enumerate(rows):
            v = dirs.get((b_, t))
            if v is None:
                continue
            ax.add_patch(plt.Rectangle((j - 0.44, i - 0.42), 0.88, 0.84, color=fs.DIRECTION[v], lw=0))
            if v in "+-":
                ax.text(j, i + 0.03, "+" if v == "+" else "−", ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
    ax.set_xlim(-0.6, len(reg) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    tlab = {T.SHORT[t]: CAT[t] for t in T.TRAITS}
    ax.set_yticks(range(len(rows)), rows, fontsize=5.8)
    for lab_ in ax.get_yticklabels():
        lab_.set_color(fs.CATEGORY[tlab[lab_.get_text()]])
    ax.set_xticks(range(len(reg)), [f"{v / 1e6:.2f}" for v in reg.start], rotation=90, fontsize=5.5)
    ax.set_xlabel("Block start on chromosome 13 (Mb)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    kg = pd.read_csv(data.P / "known_genes.tsv", sep="\t")
    for lab, loc in (("Rcs, Rsv1", "Rsv1"), ("Rag2", "Rag2")):
        g = kg[kg.locus == loc].iloc[0]
        j = [k for k, (_, r) in enumerate(reg.iterrows()) if r.start <= g.start <= r.end + 2e4]
        if j:
            ax.annotate(lab, (j[0], -0.6), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.8, color=fs.INK2, annotation_clip=False, fontstyle="italic")
    fs.panel(ax, "b", x=-0.1, y=1.08)
    ax = fig.add_subplot(gs[1, 1])                    # (c) race reactions at the race-differential block
    rd = pd.read_csv(ROOT / "results/loci/race_differential_2059.tsv", sep="\t")
    rd = rd[rd.accessions == "race-3 resistant"].set_index("race").loc[[1, 5, 14]]
    x = np.arange(len(rd))
    w = 0.36
    ax.bar(x - w / 2, rd.carriers_resistant * 100, w, color=fs.RED, label="with allele")
    ax.bar(x + w / 2, rd.noncarriers_resistant * 100, w, color="#c8ccd4", label="without allele")
    for k, r in enumerate(rd.itertuples()):
        ax.text(k - w / 2, r.carriers_resistant * 100 + 2, f"{r.carriers}", ha="center", fontsize=5.5, color=fs.INK2)
        ax.text(k + w / 2, r.noncarriers_resistant * 100 + 2, f"{r.noncarriers}", ha="center", fontsize=5.5,
                color=fs.INK2)
    ax.set_xticks(x, [f"Race {r}" for r in rd.index])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Resistant (%)")
    ax.set_title("Race-3-resistant accessions", fontsize=6.8, pad=4)
    ax.legend(loc="upper right", handlelength=1, borderaxespad=0)
    fs.panel(ax, "c", x=-0.28, y=1.08)
    fs.save(fig, "fig5_loci")


# ---------------------------------------------------------------- Fig. 6 germplasm
def fig6():
    c = pd.read_csv(ROOT / "results/germplasm/consistency.tsv", sep="\t")
    r = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, 58 * fs.MM), gridspec_kw={"wspace": 0.32})
    ax = axs[0]
    ax.bar(c.min_categories_top, c.observed_resistant_share * 100, color=fs.NAVY, width=0.56)
    ax.axhline(c.base_rate_all_records.iloc[0] * 100, color=fs.RED, lw=0.8, ls=(0, (3, 2)), label="all records")
    for x, y, n in zip(c.min_categories_top, c.observed_resistant_share * 100, c.accessions):
        ax.text(x, y + 1.2, f"n = {n:,}", ha="center", fontsize=5.8, color=fs.INK2)
    ax.set_xticks(c.min_categories_top, [f"≥{k}" for k in c.min_categories_top])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Stress categories with a top-10% prediction")
    ax.set_ylabel("Records observed resistant (%)")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper left", handlelength=2)
    fs.panel(ax, "a")
    ax = axs[1]
    for k, (tax, col) in enumerate((("Glycine max", fs.NAVY), ("Glycine soja", fs.TEAL))):
        v = r[r.taxon == tax].multi_resistance_index_within_taxon.value_counts(normalize=True).sort_index() * 100
        ax.bar(v.index + (k - 0.5) * 0.38, v.values, width=0.38, color=col, label=tax.replace("Glycine", "G."))
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Stress categories with a top-10% prediction")
    ax.set_ylabel("Accessions (%)")
    ax.legend(loc="upper right", prop={"style": "italic", "size": 6.5})
    fs.panel(ax, "b")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
    fs.save(fig, "fig6_germplasm")


if __name__ == "__main__":
    fs.setup()
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
    for f in todo:
        globals()[f]()
        print(f, "done")
