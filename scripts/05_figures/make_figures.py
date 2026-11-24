"""Main figures (printed size, Times New Roman, PDF + 600-dpi PNG in figures/).

usage: python make_figures.py [fig1 ... fig5]
fig1  overview of the study (data, TraitBoost, evaluation, loci and germplasm)
fig2  stress records of the collection: scored accession x trait cells and associations between traits
fig3  prediction: new accessions (model means, TraitBoost vs the best single-trait model, contribution of the shared
      trees) and held-out experiments (model means)
fig4  shared loci: genome-wide map, allele directions of the shared loci, reactions of race-3-resistant accessions
fig5  multi-resistant germplasm: out-of-fold validation and the multi-resistance index
figS1, figS2  supplementary figures: quantile-quantile and Manhattan plots of the association analyses
Colours (figstyle.py): TraitBoost red, comparators navy/ochre/teal/greys; stress categories, locus classes and allele
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

PRIM = set(data.sparse_traits())     # the sparse traits of the article
CAT = {row[0]: row[4] for row in T.STRESS}
CATS = list(fs.CATEGORY)


def model_marker(ax, x, y, m, s=26, z=4):
    lab, col, mk, filled = fs.MODEL[m]
    ax.scatter(x, y, s=s, marker=mk, color=col if filled else "white", edgecolors=col,
               linewidths=0.4 if filled else 0.9, zorder=z)


# ---------------------------------------------------------------- Fig. 1 overview
FIG1_CATEGORY = {"SCN": "SCN", "Phytophthora": r"$\it{P.\,sojae}$",
                 "fungi_oomycetes": "Other fungi and\noomycetes", "bacteria": "Bacteria", "virus": "Viruses",
                 "chewing_insects": "Chewing insects", "sucking_insects": "Sucking insects", "abiotic": "Abiotic stresses"}


def fig1():
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
    W, H = 174, 104                       # layout coordinates; the figure is drawn 1.25 times as high (130 mm)
    STRETCH = 1.25
    fig = plt.figure(figsize=(fs.W2, H * STRETCH * fs.MM))

    def Circle(xy, r, **kw):              # round on paper
        return Ellipse(xy, 2 * r, 2 * r / STRETCH, **kw)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ink2, muted = fs.INK2, fs.MUTED

    def stage(x, y, w, h, letter, title, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fs.PANEL_BG,
                                    ec="#d5d9e0", lw=0.6, zorder=0))
        ax.add_patch(Rectangle((x + 2.5, y + h - 2.2), 9, 0.9, fc=colour, ec="none", zorder=1))
        ax.text(x + 2.4, y + h - 3.4, title, fontsize=8, fontweight="bold", va="top", ha="left", zorder=2)
        ax.text(x - 0.4, y + h + 0.6, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    def txt(x, y, s, size=8, colour=None, **kw):
        ax.text(x, y, s, fontsize=max(size, 8), color=colour or ink2, va="top", ha=kw.pop("ha", "left"), linespacing=1.25, **kw)

    def arrow(p0, p1, colour="#7a8494"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=colour, zorder=3))

    # (a) genebank records
    stage(2, 3, 47, 96, "a", "Genebank records", fs.NAVY)
    txt(4.5, 92.0, "USDA Soybean\nGermplasm Collection")
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
    txt(x0 - 5.6, y0 - 1.0, "SNPs", 6.5)
    ax.text(x0 - 7.0, y0 + nr * ch / 2, "accessions", rotation=90, fontsize=8, color=ink2, va="center", ha="center")
    txt(x0 + cw * len(cols) / 2, y0 - 1.0, "stress traits", 6.5, ha="center")
    fo_ = data.folds()                                   # scored accessions; share of unscored accession–trait combinations
    lgp_ = data.pheno_long()
    unsc_ = 1 - lgp_[lgp_.acc.isin(set(fo_.index))].groupby(["acc", "trait"]).ngroups / (len(fo_) * len(T.TRAITS))
    txt(4.5, 36.3, f"{len(fo_):,} accessions with stress records\n{len(T.TRAITS)} stress traits in 8 categories\n"
                   f"{100 * unsc_:.0f}% of accession–trait combinations\nunscored; 39,627 SoySNP50K SNPs")
    for k, c in enumerate(["SCN", "Phytophthora", "bacteria", "virus", "chewing_insects", "sucking_insects",
                           "abiotic", "fungi_oomycetes"]):                      # names of Table 1, two columns
        cx, cy = 4.2 + (k // 4) * 17.5, 22.4 - (k % 4) * 4.5
        ax.add_patch(Rectangle((cx, cy - 1.9), 1.7, 1.5, fc=fs.CATEGORY[c], ec="none"))
        txt(cx + 2.2, cy, FIG1_CATEGORY[c], 6.5)

    # (b) TraitBoost
    stage(56, 3, 56, 96, "b", "TraitBoost", fs.RED)
    txt(58.5, 92.0, "Gradient boosting for several traits")

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
    txt(58.5, 71.0, "Shared trees: one split for\nall traits of a group (all\ntraits or a stress category),\na leaf value per trait")
    for k, c in enumerate(["#2F4E9E", "#E64B35", "#00A087"]):                            # trait-specific trees
        tree(88.0 + k * 8.9, 85, c, scale=0.5)
    txt(85.0, 77.0, "Trait-specific trees\n(one per trait)")
    ax.text(81.2, 80.5, "+", fontsize=9, ha="center", va="center", color=muted)
    ax.add_patch(FancyBboxPatch((58.5, 50.0), 51, 10.5, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec="#d5d9e0", lw=0.5, zorder=1))
    for k_, line_ in enumerate(("option = starting value", "+ S rounds of shared trees", "+ P rounds of trait-specific trees")):
        ax.text(84, 58.2 - k_ * 2.95, line_, fontsize=8, ha="center", va="center", zorder=2)
    txt(58.5, 46.6, "One of three options for every trait", 6.8, fs.INK, fontweight="bold")
    opts = [("trait-specific trees from zero", False), ("S shared + P trait-specific rounds", True),
            ("GBLUP + trait-specific trees", False)]
    for k, (lab, chosen) in enumerate(opts):
        yy = 39.5 - k * 5.0
        ax.add_patch(FancyBboxPatch((59.5, yy - 1.8), 44, 3.6, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="#fdecea" if chosen else "white", ec=fs.RED if chosen else "#c8ccd4", lw=0.6,
                                    zorder=2))
        ax.text(81.5, yy, lab, fontsize=8, ha="center", va="center", color=fs.INK if chosen else ink2, zorder=3)
        if chosen:
            ax.text(105.0, yy, "✓", fontsize=8, ha="left", va="center", color=fs.RED, zorder=3)
    txt(58.5, 25.6, "chosen by the validation correlation\non three inner folds")
    txt(58.5, 16.4, "Prediction: mean of the chosen option\nand the best option without shared\ntrees (two refits each)")

    # (c) evaluation
    stage(118, 54, 54, 45, "c", "Evaluation", fs.NAVY)
    ys = [90.4, 81.2, 72.0]
    labels = [("New accessions", "five folds; near-identical\naccessions in the same fold"),
              ("Held-out experiments", "largest experiment of a trait\n(21 traits, 15 experiments)"),
              ("Measured traits", "one trait hidden with all\ntraits of its experiments")]
    for k, (y, (h1, h2)) in enumerate(zip(ys, labels)):
        for j in range(5):
            if k == 0:
                fc = fs.RED if j == 2 else "#cfd6e4"
            elif k == 2:
                fc = "#cfd6e4" if j != 3 else "white"
            else:
                fc = "#cfd6e4" if j != 1 else fs.RED
            ax.add_patch(Rectangle((121 + j * 2.6, y - 3.0), 2.2, 3.0, fc=fc,
                                   ec="#9aa3b2" if fc == "white" else "none", lw=0.4, zorder=2))
        if k == 2:
            ax.text(121 + 3 * 2.6 + 1.1, y - 1.5, "×", fontsize=8, ha="center", va="center", color=fs.RED, zorder=3)
        txt(135.5, y + 1.0, h1, 6.8, fs.INK, fontweight="bold")
        txt(135.5, y - 1.5, h2, 6.5)
    txt(121, 65.0, "Compared with GBLUP, BayesB,\nLightGBM, multi-trait GBLUP,\nthe best single-trait model\nand a tree ensemble", 6.5)

    # (d) loci and germplasm
    stage(118, 3, 54, 45, "d", "Association loci and germplasm", fs.RED)
    # genome-wide thresholds calibrated by simulation (schematic): bars, blocks; dotted, Bonferroni; solid, calibrated
    bonf_, cal_ = 4.0, 6.2
    for j, hgt in enumerate([2.0, 3.1, 8.6, 2.4, 5.0, 1.6, 4.4, 9.2, 2.8, 3.6, 7.4, 1.9]):
        fc_ = fs.NAVY if hgt > cal_ else ("#9aa3b2" if hgt > bonf_ else "#d3d7de")
        ax.add_patch(Rectangle((121 + j * 1.3, 32.0), 1.0, hgt * 0.62, fc=fc_, ec="none", zorder=2))
    ax.plot([120.6, 136.6], [32.0 + bonf_ * 0.62] * 2, color=muted, lw=0.6, ls=(0, (1, 1.2)), zorder=3)
    ax.plot([120.6, 136.6], [32.0 + cal_ * 0.62] * 2, color=fs.RED, lw=0.8, zorder=3)
    loci_ = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    shared_ = loci_[loci_.n_traits >= 2]
    ncls = shared_["class"].value_counts()
    null_ = pd.read_csv(ROOT / "results/loci/sharing_null.tsv", sep="\t").set_index("min_traits")
    rk_ = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    top_ = rk_[rk_.multi_resistance_index_within_taxon >= 4].taxon.value_counts()
    txt(138.5, 39.2, f"Calibrated thresholds:\nloci in {len(loci_)} blocks")
    txt(121, 30.6, f"{len(shared_)} blocks are loci of ≥ 2 traits")
    glyphs = [("concordant", "+ + +", f"{ncls.get('concordant', 0)}\nconcordant"),
              ("race-differential", "+ + −", f"{ncls.get('race-differential', 0)} race-\ndifferential"),
              ("trade-off", "+ | −", f"{ncls.get('trade-off', 0)}\nopposite"),
              ("unresolved", "+ 0 0", f"{ncls.get('unresolved', 0)}\nunresolved")]
    glyphs = [g_ for g_ in glyphs if ncls.get(g_[0], 0) > 0]                     # classes that occur
    for k, (cls, sig, lab) in enumerate(glyphs):
        cx = 123.6 + k * 15.5
        ax.add_patch(FancyBboxPatch((cx - 2.6, 22.6), 10.6, 4.4, boxstyle="round,pad=0,rounding_size=1",
                                    fc="white", ec=fs.CLASS_COLOUR[cls], lw=0.8, zorder=2))
        ax.text(cx + 2.7, 24.8, sig, fontsize=8, ha="center", va="center", color=fs.CLASS_COLOUR[cls], zorder=3)
        ax.text(cx + 2.7, 21.6, lab, fontsize=8, ha="center", va="top", color=ink2, linespacing=1.1)
    for k, c in enumerate(CATS):                                                       # germplasm
        ax.add_patch(Circle((122.4 + k * 2.3, 13.4), 0.9, fc=fs.CATEGORY[c] if k in (0, 1, 2, 5) else "white",
                            ec=fs.CATEGORY[c], lw=0.6, zorder=2))
    txt(141.2, 14.8, "Multi-resistance index:")
    txt(121, 11.4, f"{top_['Glycine max']} $\\it{{G.\\,max}}$ and {top_['Glycine soja']} $\\it{{G.\\,soja}}$ accessions\nin the top 10% for ≥ 4 of 8 categories")
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
    fig = plt.figure(figsize=(fs.W2, 190 * fs.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.37, 0.63], hspace=0.30, left=0.085, right=0.97, top=0.925, bottom=0.075)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=8)
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
        ax.text((js[0] + js[-1]) / 2, -nrow * (0.036 if k % 2 == 0 else 0.10),
                FIG2_CATEGORY[c], ha="center", va="bottom", fontsize=8, color=fs.INK, clip_on=False)
        if js[0] > 0:
            ax.axvline(js[0] - 0.5, color="white", lw=1.0)
    ax.set_ylim(nrow - 0.5, -0.5)
    fig.text(0.008, 0.955, "a", fontsize=10, fontweight="bold", va="bottom", ha="left")
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
    base[tested & np.isnan(A)] = to_rgb("#c9c9c9")
    ax.imshow(base, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-5, vmax=5, interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=8, ha="right", va="center",
                  rotation_mode="anchor")                        # labels hang from just below the category strip
    ax.set_yticks(range(len(traits)), [T.SHORT[t] for t in traits], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=3.2)
    ax.tick_params(axis="x", length=0, pad=1.5)
    n_ = len(traits)
    for j, t in enumerate(traits):                       # stress category of every trait (colours as in a)
        ax.add_patch(plt.Rectangle((-1.25, j - 0.42), 0.6, 0.84, color=fs.CATEGORY[CAT[t]], lw=0, clip_on=False))
        ax.add_patch(plt.Rectangle((j - 0.42, n_ - 0.5 + 0.15), 0.84, 0.6, color=fs.CATEGORY[CAT[t]], lw=0, clip_on=False))
    for c, js in starts.items():
        k = js[0]
        if k > 0:
            ax.plot([-0.5, k - 0.5], [k - 0.5, k - 0.5], color="#b9bec8", lw=0.4)
            ax.plot([k - 0.5, k - 0.5], [k - 0.5, n_ - 0.5], color="#b9bec8", lw=0.4)
    ax.set_xlim(-1.3, n_ - 0.5)                         # limits fixed at the category strips (no autoscale margins)
    ax.set_ylim(n_ - 0.5 + 0.8, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_anchor("W")                                   # heatmap at the left, colour scale next to it
    ax.apply_aspect()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.06, pos.y0 + 0.30 * pos.height, 0.016, 0.62 * pos.height])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([-4, -2, 0, 2, 4], labels=["1/16", "1/4", "1", "4", "16"])
    cb.set_label("Odds ratio of resistance (logarithmic scale)", fontsize=8)
    cb.ax.tick_params(labelsize=8, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.add_patch(plt.Rectangle((16.6, 0.6), 1.2, 1.2, fc="#c9c9c9", ec="none"))
    ax.text(18.4, 1.2, "tested, not significant", fontsize=8, va="center", color=fs.INK2)
    fig.text(0.008, ax.get_position().y1 + 0.01, "b", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fs.save(fig, "fig2_data")


# category names of Table 1 above the columns of Fig. 2a (SCN and P. sojae abbreviated)
FIG2_CATEGORY = {"SCN": "SCN", "Phytophthora": r"$\it{P.\,sojae}$", "fungi_oomycetes": "Other fungi and oomycetes",
                 "bacteria": "Bacteria", "virus": "Viruses", "chewing_insects": "Chewing insects",
                 "sucking_insects": "Sucking insects", "abiotic": "Abiotic stresses"}


# ---------------------------------------------------------------- Fig. 3 prediction
def means(fig, spec, frame, models, sets, letter, xlabel):
    """Mean accuracy of each model (rows) for two trait sets (two small axes)."""
    sub = spec.subgridspec(1, 2, wspace=0.22)
    for k, (sel, lab) in enumerate(sets):
        ax = fig.add_subplot(sub[k])
        y = np.arange(len(models))[::-1]
        for i, m in enumerate(models):
            model_marker(ax, frame[sel][m].mean(), y[i], m, s=24)
        short = {"TraitBoost without shared trees": "TraitBoost without\nshared trees", "Best single-trait model": "Best single-\ntrait model",
                 "Multi-trait GBLUP": "Multi-trait\nGBLUP"}
        ax.set_yticks(y, [short.get(fs.MODEL[m][0], fs.MODEL[m][0]) for m in models] if k == 0 else [], fontsize=8)
        for lab_ in ax.get_yticklabels():
            lab_.set_linespacing(0.95)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.grid(axis="x", color=fs.GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(lab, fontsize=8, color=fs.INK2, pad=3)
        v = frame[sel][models].mean()
        lo, hi = v.min(), v.max()
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        x0_, x1_ = ax.get_xlim()                         # two ticks inside the narrow axis, at round values, far enough
        tk, one = None, None                             # apart that their labels do not touch (else one tick)
        for margin in (0.1, 0.03):
            for step in (0.01, 0.02, 0.03, 0.04, 0.05, 0.1):
                tk_ = np.arange(np.ceil((x0_ + margin * (x1_ - x0_)) / step) * step, x1_ - margin * (x1_ - x0_) + 1e-9, step)
                if len(tk_) == 2 and (tk_[1] - tk_[0]) / (x1_ - x0_) >= 0.4:
                    tk = tk_
                    break
                if len(tk_) == 1 and one is None:
                    one = tk_
            if tk is not None:
                break
        tk = tk if tk is not None else one
        ax.set_xticks(tk, [f"{v:.2f}" for v in tk])
        ax.tick_params(axis="x", labelsize=8)
        if k == 0:
            pos = ax.get_position()
            fig.text(0.012, pos.y1 + 0.022, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
            ax.text(1.07, -0.16, xlabel, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def fig3():
    pe = pd.read_csv(ROOT / "results/compare/cv1_point_estimates.tsv", sep="\t")
    pe3 = pd.read_csv(ROOT / "results/compare/cv3_point_estimates.tsv", sep="\t")
    cmp_ = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_best_single_avg4_{s}.tsv", sep="\t")
                      for s in ("sparse", "nonsparse")])
    shr = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_mtboost_v2_noshare4_{s}.tsv", sep="\t")
                     for s in ("sparse", "nonsparse")])
    sp, sp3 = pe.trait.isin(PRIM), pe3.trait.isin(PRIM)
    # traits for which no training set used shared trees: the two models of panel c differ only in the random seeds
    import json
    used = {t: 0 for t in T.TRAITS}
    for f in range(5):
        js = json.load(open(data.RUNS / "cv1/mtboost_dev/v2" / f"fold{f}.json"))
        for t in T.TRAITS:
            used[t] += js["per_trait"][t]["S_t"] > 0
    seeds_only = {t for t, n in used.items() if n == 0}
    fig = plt.figure(figsize=(fs.W2, 165 * fs.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1, 1], hspace=0.36, wspace=0.62,
                          left=0.135, right=0.975, top=0.953, bottom=0.085)
    order_m = ["gblup", "bayesb", "lightgbm_avg4", "bglr_mt", "best_single_avg4", "ens_lightgbm_noshare",
               "mtboost_v2_noshare4", "mtboost_v2_final"]
    means(fig, gs[0, 0], pe, order_m,
          ((sp, f"Sparse ({int(sp.sum())})"), (~sp, f"Other ({int((~sp).sum())})")), "a",
          "Mean accuracy, new accessions")
    means(fig, gs[1, 0], pe3, [("mtboost_v2_gblup" if m == "gblup" else m) for m in order_m if m in pe3 or m == "gblup"],
          ((sp3, f"Sparse ({int(sp3.sum())})"), (~sp3, f"Other ({int((~sp3).sum())})")),
          "d", "Mean accuracy, held-out experiments")
    from matplotlib.patches import Rectangle
    for k, (df, title, letter) in enumerate(((cmp_, "TraitBoost − best single-trait model", "b"),
                                              (shr, "TraitBoost − TraitBoost without shared trees", "c"))):
        ax = fig.add_subplot(gs[:, k + 1])
        d = df.assign(prim=df.trait.isin(PRIM)).sort_values(["prim", "diff"], ascending=[False, True])
        y = np.arange(len(d))
        sig = (d.p_holm < 0.05).to_numpy()
        np_ = int(d.prim.sum())
        ax.axhspan(-0.5, np_ - 0.5, color="#f7f4f3", lw=0, zorder=0)
        ax.axvline(0, color="#9aa3b2", lw=0.6, zorder=1)
        ax.hlines(y, d.lo, d.hi, color="#c3c8d0", lw=0.9, zorder=2)
        grey = d.trait.isin(seeds_only).to_numpy() if letter == "c" else np.zeros(len(d), bool)
        edge = np.where(grey, "#8c8c8c", fs.RED)
        ax.scatter(d["diff"], y, s=14, color=np.where(sig, fs.RED, "white"), edgecolors=edge, linewidths=0.7,
                   zorder=3)
        ax.set_yticks(y, d.short)
        ax.tick_params(axis="y", length=0, labelsize=8, pad=5.5)
        for yy, cat_ in zip(y, d.trait.map(CAT)):                       # stress category of the trait
            ax.add_patch(Rectangle((-0.045, yy - 0.36), 0.028, 0.72, transform=ax.get_yaxis_transform(),
                                   color=fs.CATEGORY[cat_], lw=0, clip_on=False))
        ax.set_xlabel("Difference in accuracy\n(" + title.replace(" − TraitBoost without", " −\nTraitBoost without") + ")", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(len(d) - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        # y axis runs downwards: the sparse traits (shaded) are at the top
        ax.text(0.99, 1 - (np_ / 2) / len(d), "sparse\ntraits", transform=ax.transAxes, fontsize=8,
                color=fs.INK2, ha="right", va="center")
        ax.text(0.99, (len(d) - np_) / 2 / len(d), "other\ntraits", transform=ax.transAxes, fontsize=8,
                color=fs.INK2, ha="right", va="center")
        x_letter = fig.transFigure.inverted().transform(ax.transAxes.transform((-0.3, 0)))[0]     # level with a and d
        fig.text(x_letter, ax.get_position().y1 + 0.022, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fs.save(fig, "fig3_prediction")


# ---------------------------------------------------------------- Fig. 4 loci
def fig4():
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    reg = pd.read_csv(ROOT / "results/loci/regions_emmax.tsv", sep="\t")
    chroms = sorted(snps.chrom.unique(), key=lambda c: int(c[2:]))
    length = snps.groupby("chrom").pos.max()
    off, acc = {}, 0
    for c in chroms:
        off[c] = acc
        acc += length[c] + 5e6
    loci["x"] = [off[c] + (s_ + e) / 2 for c, s_, e in zip(loci.chrom, loci.start, loci.end)]
    fig = plt.figure(figsize=(fs.W2, 150 * fs.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], width_ratios=[2.35, 1], hspace=0.52, wspace=0.2,
                          left=0.075, right=0.985, top=0.905, bottom=0.1)
    # (a) loci along the genome
    ax = fig.add_subplot(gs[0, :])
    for i, c in enumerate(chroms):
        if i % 2 == 0:
            ax.axvspan(off[c], off[c] + length[c], color=fs.BAND, lw=0, zorder=0)
    # blocks close to each other with the same number of traits are drawn side by side (shifted along the y axis)
    loci = loci.sort_values("x").reset_index(drop=True)
    yv = loci.n_traits.astype(float).to_numpy().copy()
    for n_ in sorted(loci.n_traits.unique()):
        idx = np.flatnonzero((loci.n_traits == n_).to_numpy() & (loci.n_traits >= 2).to_numpy())
        run = 0
        for a_, b_ in zip(idx[:-1], idx[1:]):
            run = run + 1 if loci.x[b_] - loci.x[a_] < 6e6 else 0
            yv[b_] = n_ + (0.26 * ((run + 1) // 2) * (1 if run % 2 else -1) if run else 0.0)
    loci["y"] = yv
    for k, cls in enumerate(("specific", "unresolved", "concordant", "trade-off", "race-differential")):
        d = loci[loci["class"] == cls]
        ax.scatter(d.x, d.y, s=13 if cls == "specific" else 17, color=fs.CLASS_COLOUR[cls], label=fs.CLASS_LABEL[cls],
                   zorder=3 + k, linewidths=0.45, edgecolors="#8c8c8c" if cls == "specific" else "white")
    h, l_ = ax.get_legend_handles_labels()
    order = [l_.index(fs.CLASS_LABEL[c]) for c in ("concordant", "race-differential", "trade-off", "unresolved", "specific")
             if (loci["class"] == c).any()]                                       # classes that occur
    notes = {("Gm03", 3.4e6): "Rps1", ("Gm07", 35.7e6): r"$\mathit{NSF}_{\mathrm{RAN07}}$", ("Gm10", 40.6e6): "Rhg1|(RAC)", ("Gm11", 32.7e6): "GmSNAP11",
             ("Gm12", 37.0e6): "qDRT12.3", ("Gm13", 29.7e6): "Rsv1", ("Gm16", 32.7e6): "Rbs1–Rbs2", ("Gm18", 1.1e6): "Rhg1",
             ("Gm18", 56.1e6): "Rps4/12/13"}
    for (c, a_), lab in notes.items():
        d = loci[(loci.chrom == c) & (loci.end >= a_) & (loci.start <= a_ + 2.2e6) & (loci.n_traits >= 2)]
        if len(d) and "|" in lab:                         # gene name (italic) followed by a plain-text note
            g_, n_ = lab.split("|")
            ax.annotate(g_, (d.x.mean(), d.y.max()), xytext=(-1.2, 6), textcoords="offset points", ha="right",
                        fontsize=8, color=fs.INK, fontstyle="italic")
            ax.annotate(n_, (d.x.mean(), d.y.max()), xytext=(1.2, 6), textcoords="offset points", ha="left",
                        fontsize=8, color=fs.INK)
        elif len(d) and lab:
            if lab == "Rps4/12/13":                         # end of chromosome 18: label raised, right-aligned, leader line
                ax.annotate(lab, (d.x.mean(), d.y.max()), xytext=(4, 17), textcoords="offset points", ha="right",
                            fontsize=8, color=fs.INK, fontstyle="italic",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2, shrinkA=1, shrinkB=3, relpos=(0.93, 0.0)))
            else:
                ax.annotate(lab, (d.x.mean(), d.y.max()), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=fs.INK, fontstyle="italic")
    ax.set_xticks([off[c] + length[c] / 2 for c in chroms], [c[2:].lstrip("0") for c in chroms])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-2e6, acc)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of traits")
    ax.set_yticks(range(1, int(loci.n_traits.max()) + 1))
    ax.set_ylim(0.4, loci.n_traits.max() + 1.6)
    ax.legend([h[i] for i in order], [l_[i] for i in order], loc="upper left", ncol=5, handletextpad=0.1,
              columnspacing=0.9, bbox_to_anchor=(0, 1.22))
    fs.panel(ax, "a", x=-0.06, y=1.13)
    # (b) allele directions in the shared blocks
    ax = fig.add_subplot(gs[1, 0])
    sh = loci[loci.n_traits >= 2].assign(chrn=lambda d: d.chrom.str[2:].astype(int)).sort_values(["chrn", "start"])
    reg_of = {int(b_): int(r.region) for _, r in reg.iterrows() for b_ in str(r.blocks).split(",")}
    dirs, other = {}, {}
    flipv = {"+": "-", "-": "+", "0": "0"}
    for _, r in sh.iterrows():
        d_ = dict(item.split(":") for item in r.directions.split(","))
        # shown for the allele of the common SNP that is associated with higher resistance for most traits of the block
        flip = sum(v == "-" for v in d_.values()) > sum(v == "+" for v in d_.values())
        for t_, v in d_.items():
            dirs[(r.block, t_)] = flipv[v] if flip else v
        # other races of the pathogen read at the lead SNP (not loci of the block): shown where |z| >= 3
        for item in filter(None, str(r.other_races if isinstance(r.other_races, str) else "").split(",")):
            t_, rest = item.split(":")
            v = rest[0]
            if v in "+-":
                other[(r.block, t_)] = flipv[v] if flip else v
    shorts = [T.SHORT[t] for t in T.TRAITS]
    rows = [t for t in shorts if any((b_, t) in dirs or (b_, t) in other for b_ in sh.block)]
    xs, x, last = [], 0.0, None
    for b_ in sh.block:                                   # a gap between regions
        r_ = reg_of[int(b_)]
        if last is not None and r_ != last:
            x += 0.7
        xs.append(x)
        x += 1.0
        last = r_
    xs = np.array(xs)
    for xj, b_ in zip(xs, sh.block):
        for i, t in enumerate(rows):
            v = dirs.get((b_, t))
            if v is None:
                v = other.get((b_, t))
                if v is not None:                         # another race of the pathogen, not a locus of the block
                    ax.add_patch(plt.Rectangle((xj - 0.40, i - 0.37), 0.8, 0.74, facecolor="white",
                                               edgecolor=fs.DIRECTION[v], lw=0.9))
                    ax.text(xj, i + 0.03, "+" if v == "+" else "−", ha="center", va="center", fontsize=8,
                            color=fs.DIRECTION[v], fontweight="bold")
                continue
            ax.add_patch(plt.Rectangle((xj - 0.45, i - 0.42), 0.9, 0.84, color=fs.DIRECTION[v], lw=0))
            if v in "+-":
                ax.text(xj, i + 0.03, "+" if v == "+" else "−", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
    ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    tlab = {T.SHORT[t]: CAT[t] for t in T.TRAITS}
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    from matplotlib.transforms import blended_transform_factory
    trb = blended_transform_factory(ax.transAxes, ax.transData)
    for i, t in enumerate(rows):                          # stress category of the trait (left of the panel)
        ax.add_patch(plt.Rectangle((-0.032, i - 0.36), 0.016, 0.72, color=fs.CATEGORY[tlab[t]], lw=0, clip_on=False,
                                   transform=trb))
    span_ = xs.max() - xs.min() + 1.6
    for kx, (v, lab) in enumerate((("+", "higher resistance"), ("-", "lower resistance"), ("0", "$|z|$ < 3"),
                                   ("o", "other race, no locus"))):
        x0_ = xs.min() - 0.6 + (kx % 2) * span_ / 2
        y0_ = -3.05 + (kx // 2) * 1.05
        if v == "o":                                      # open squares of either sign
            for dx_, sg_ in ((0.0, "+"), (1.0, "-")):
                ax.add_patch(plt.Rectangle((x0_ + 0.05 + dx_, y0_ + 0.05), 0.8, 0.74, facecolor="white",
                                           edgecolor=fs.DIRECTION[sg_], lw=0.9, clip_on=False))
                ax.text(x0_ + 0.45 + dx_, y0_ + 0.45, "+" if sg_ == "+" else "−", ha="center", va="center", fontsize=8,
                        color=fs.DIRECTION[sg_], fontweight="bold", clip_on=False)
        else:
            ax.add_patch(plt.Rectangle((x0_, y0_), 0.9, 0.84, color=fs.DIRECTION[v], lw=0, clip_on=False))
            if v in "+-":
                ax.text(x0_ + 0.45, y0_ + 0.45, "+" if v == "+" else "−", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", clip_on=False)
        ax.text(x0_ + (2.4 if v == "o" else 1.4), y0_ + 0.45, lab, ha="left", va="center", fontsize=8, color=fs.INK2, clip_on=False)
    ticks, labels = [], []
    for r_, g in pd.Series(xs, index=[reg_of[int(b_)] for b_ in sh.block]).groupby(level=0, sort=False):
        R_ = reg[reg.region == r_].iloc[0]
        ticks.append(g.mean())
        labels.append(f"{int(R_.chrom[2:])}: {R_.start / 1e6:.1f}")
        ax.plot([g.min() - 0.45, g.max() + 0.45], [len(rows) - 0.45] * 2, color="#8c8c8c", lw=0.6, clip_on=False)
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
    ax.set_xlabel("Region (chromosome: start, Mb)")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fs.panel(ax, "b", x=-0.1, y=1.07)
    y_letter = fig.transFigure.inverted().transform(ax.transAxes.transform((0, 1.07)))[1]     # panel letters b and c level
    # (c) race reactions at the two race-differential loci
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.62)
    rdb = loci[loci["class"] == "race-differential"].assign(chrn=lambda d: d.chrom.str[2:].astype(int)).sort_values("chrn")
    for k, (_, L) in enumerate(rdb.iterrows()):
        ax = fig.add_subplot(sub[k])
        rd = pd.read_csv(ROOT / f"results/loci/race_differential_{L.block}.tsv", sep="\t")
        rd = rd[rd.accessions == "race-3 resistant"].set_index("race").loc[[1, 5, 14]]
        x = np.arange(len(rd))
        w = 0.36
        ax.bar(x - w / 2, rd.carriers_resistant * 100, w, color=fs.TEAL, label="Carriers")
        ax.bar(x + w / 2, rd.noncarriers_resistant * 100, w, color="#c8ccd4", label="Non-carriers")
        for j, r in enumerate(rd.itertuples()):
            ax.text(j - w / 2, r.carriers_resistant * 100 + 2.5, f"{r.carriers}", ha="center", fontsize=8, color=fs.INK2)
            ax.text(j + w / 2, r.noncarriers_resistant * 100 + 2.5, f"{r.noncarriers}", ha="center", fontsize=8,
                    color=fs.INK2)
        ax.set_xticks(x, [f"Race {r}" for r in rd.index])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("Resistant (%)")
        ax.set_title(f"Chromosome {int(L.chrom[2:])}, {L.lead_pos / 1e6:.2f} Mb" + (" (RAC)" if L.lead_snp == "ss715606985" else ""),
                     fontsize=8, pad=3)
        if k == 0:
            ax.legend(loc="upper center", handlelength=1, borderaxespad=0, ncol=2, columnspacing=0.8,
                      bbox_to_anchor=(0.5, 1.52), fontsize=8)
            x_letter = fig.transFigure.inverted().transform(ax.transAxes.transform((-0.3, 0)))[0]
            fig.text(x_letter, y_letter, "c", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fs.save(fig, "fig4_loci")


# ---------------------------------------------------------------- Fig. 5 germplasm
MG_NAMES = {-2: "000", -1: "00", 0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}


def fig5():
    v = pd.read_csv(ROOT / "results/germplasm/index_out_of_fold.tsv", sep="\t")
    r = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    g = pd.read_csv(ROOT / "results/germplasm/index_by_group.tsv", sep="\t")
    fig, axs = plt.subplots(1, 3, figsize=(fs.W2, 62 * fs.MM), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.15]})
    ax = axs[0]
    w = 0.36
    ax.bar(v.min_index - w / 2, v.observed_resistant_share * 100, color=fs.NAVY, width=w, label="Observed")
    ax.bar(v.min_index + w / 2, v.expected_resistant_share * 100, color="#c8ccd4", width=w, label="Expected")
    top = max(v.observed_resistant_share.max(), v.expected_resistant_share.max()) * 100
    for x, y, n in zip(v.min_index, v.observed_resistant_share * 100, v.accessions_with_scores_in_counted_categories):
        ax.text(x - w / 2, y + top * 0.02, f"{n:,}", ha="center", fontsize=8, color=fs.INK2)
    ax.set_xticks(v.min_index, [f"≥{k}" for k in v.min_index])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Index from out-of-fold predictions")
    ax.set_ylabel("Resistant scores (%)")
    ax.set_ylim(0, top * 1.32)
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2, ncol=2, columnspacing=0.8)
    fs.panel(ax, "a", x=-0.2, y=1.04)
    ax = axs[1]
    for tax, col, mk in (("Glycine max", fs.NAVY, "o"), ("Glycine soja", fs.TEAL, "s")):
        d = r[r.taxon == tax].multi_resistance_index_within_taxon.value_counts(normalize=True).sort_index() * 100
        ax.plot(d.index, d.values, color=col, marker=mk, ms=4, lw=1, mec="white", mew=0.5, label=tax.replace("Glycine", "G."))
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_ylim(0.003, 150)
    ax.minorticks_off()
    ax.set_xticks(range(0, int(r.multi_resistance_index_within_taxon.max()) + 1))
    ax.set_xlabel("Multi-resistance index")
    ax.set_ylabel("Accessions (%)")
    ax.legend(loc="upper right", prop={"style": "italic", "size": 8})
    fs.panel(ax, "b", x=-0.2, y=1.04)
    ax = axs[2]
    mg = g[(g.grouping == "maturity group") & (g.group != "not given")].copy()
    mg["k"] = mg.group.astype(float)
    mg = mg[mg.accessions >= 100].sort_values("k")                     # groups with at least 100 accessions
    y = np.arange(len(mg))
    ax.barh(y, mg.share * 100, color=fs.NAVY, height=0.7)
    for yy, v, n in zip(y, mg.share * 100, mg.index_at_least_4):
        ax.text(v + 0.15, yy, f"{int(n)}", va="center", fontsize=8, color=fs.INK2)
    ax.set_yticks(y, [MG_NAMES[int(k)] for k in mg.k], fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, mg.share.max() * 100 * 1.15)
    ax.set_ylabel("Maturity group")
    ax.set_xlabel(r"$\it{G.\,max}$ accessions with an index ≥ 4 (%)")
    fs.panel(ax, "c", x=-0.2, y=1.04)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.2)
    fig.align_xlabels(axs)
    fs.save(fig, "fig5_germplasm")


# ---------------------------------------------------------------- Supplementary figures: association analyses
def _assoc():
    """-log10 P of the tested SNPs (at least 30 minor-allele homozygotes) for every trait, and lambda; thresholds:
    Bonferroni and, per trait, the stricter of Bonferroni and the simulated threshold (results/gwas/null_threshold.tsv)."""
    from scipy.stats import chi2
    gw = np.load(ROOT / "results/gwas/emmax_stress.npz", allow_pickle=True)
    mac = np.load(ROOT / "results/gwas/mac_stress.npz", allow_pickle=True)
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    out = {}
    for j, t in enumerate(gw["traits"]):
        ok = mac["mac"][j] >= 30
        c = np.nan_to_num(gw["chi2"][j][ok]).astype(float)
        out[str(t)] = (np.flatnonzero(ok), -chi2.logsf(c, 1) / np.log(10), float(np.median(c) / chi2.ppf(0.5, 1)))
    nt = pd.read_csv(ROOT / "results/gwas/null_threshold.tsv", sep="\t").set_index("trait")
    bonf = 0.05 / len(snps)
    thr_t = {t: -np.log10(min(bonf, float(nt.threshold_p[t]))) for t in out}
    return snps, out, (-np.log10(bonf), thr_t)


def figS1():
    """Quantile-quantile plots of the association analyses of the stress traits."""
    snps, res, thr = _assoc()
    fig, axes = plt.subplots(6, 6, figsize=(fs.W2, 178 * fs.MM))
    for ax, t in zip(axes.flat, T.TRAITS):
        idx, lp, lam = res[t]
        obs = np.sort(lp)[::-1]
        exp = -np.log10((np.arange(1, len(obs) + 1) - 0.5) / len(obs))
        hi = max(exp.max(), 5)
        ax.plot([0, hi], [0, hi], color=fs.MUTED, lw=0.5, zorder=1)
        ax.scatter(exp, obs, s=1.5, color=fs.NAVY, linewidths=0, rasterized=True, zorder=2)
        ax.text(0.05, 0.95, f"{T.SHORT[t]}\n$\\lambda$ = {lam:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlim(0, hi * 1.03)
        ax.set_ylim(0, max(obs.max() * 1.08, 6))
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(T.TRAITS):]:
        ax.axis("off")
    fig.supxlabel("Expected $-\\log_{10}P$", fontsize=8, y=0.012)
    fig.supylabel("Observed $-\\log_{10}P$", fontsize=8, x=0.012)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.06, wspace=0.42, hspace=0.36)
    fs.save(fig, "figS1_qq")


def figS2():
    """Manhattan plots of the association analyses of the stress traits."""
    snps, res, thr = _assoc()
    chrom = snps.chrom.str[2:].astype(int).to_numpy()
    off, start = np.zeros(21), 0.0
    for c in range(1, 21):
        off[c] = start
        start += snps.pos[chrom == c].max() + 3e6
    x_all = snps.pos.to_numpy() + off[chrom]
    mid = [x_all[chrom == c].mean() for c in range(1, 21)]
    nr = int(np.ceil(len(T.TRAITS) / 2))                   # two columns of traits
    fig, axes = plt.subplots(nr, 2, figsize=(fs.W2, 232 * fs.MM), sharex=True)
    for ax in axes.ravel()[len(T.TRAITS):]:                  # empty panel when the number of traits is odd
        ax.set_visible(False)
    for k, t in enumerate(T.TRAITS):
        ax = axes[k % nr, k // nr]
        idx, lp, lam = res[t]
        col = np.where(chrom[idx] % 2 == 1, fs.NAVY, "#9aa7c7")
        ax.scatter(x_all[idx], lp, s=0.6, c=col, linewidths=0, rasterized=True)
        thr_b, thr_t = thr
        ax.axhline(thr_t[t], color=fs.RED, lw=0.4, ls=(0, (3, 2)))
        if thr_t[t] > thr_b + 0.05:
            ax.axhline(thr_b, color=fs.MUTED, lw=0.4, ls=(0, (1, 1.5)))
        ax.text(0.006, 0.93, T.SHORT[t], transform=ax.transAxes, va="top", ha="left", fontsize=8, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=5)
        ax.set_ylim(0, max(lp.max() * 1.15, thr_t[t] * 1.3, thr_b * 1.75))
        ax.set_xlim(x_all.min() - 2e6, x_all.max() + 2e6)
        ax.tick_params(labelsize=8, length=1.5)
        ax.locator_params(axis="y", nbins=3)
    for c in range(2):                                      # chromosome axis under the last panel of each column
        last = max(k for k in range(len(T.TRAITS)) if k // nr == c) % nr
        ax = axes[last, c]
        ax.set_xticks(mid, [str(c_) for c_ in range(1, 21)], fontsize=8)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Chromosome", fontsize=8)
    fig.supylabel("$-\\log_{10}P$", fontsize=8, x=0.008)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.997, bottom=0.035, wspace=0.09, hspace=0.18)
    fs.save(fig, "figS2_manhattan")


if __name__ == "__main__":
    fs.setup()
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for f in todo:
        globals()[f]()
        print(f, "done")
