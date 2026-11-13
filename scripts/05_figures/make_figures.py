"""Main figures (printed size, Times New Roman, PDF + 600-dpi PNG in figures/).

usage: python make_figures.py [fig2 fig3 fig4 fig5]
fig2  prediction of new accessions (CV1 test sets): model means, MT-Boost vs the best single-trait model per trait,
      contribution of the shared trees
fig3  known-gene calibration of the locus read-outs
fig4  genome-wide map of multi-trait loci
fig5  multi-resistant germplasm: out-of-fold validation and the multi-resistance index
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
CATEGORY_LABEL = {"SCN": "SCN", "Phytophthora": "Phytophthora", "fungi_oomycetes": "Fungi/oomycetes",
                  "bacteria": "Bacteria", "virus": "Viruses", "chewing_insects": "Chewing insects",
                  "sucking_insects": "Sucking insects", "abiotic": "Abiotic"}
CAT = {row[0]: row[4] for row in T.STRESS}
CATS = ["SCN", "Phytophthora", "fungi_oomycetes", "bacteria", "virus", "chewing_insects", "sucking_insects", "abiotic"]
# categorical order checked with the dataviz palette validator (adjacent pairs, white surface): CVD dE >= 9.1,
# normal-vision dE >= 19.6; three slots below 3:1 contrast, so every category is also labelled in the figures
CAT_COLOUR = dict(zip(CATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
CAT_SHORT = {"SCN": "SCN", "Phytophthora": "Phytophthora", "fungi_oomycetes": "Fungi", "bacteria": "Bact.",
             "virus": "Virus", "chewing_insects": "Chewing", "sucking_insects": "Sucking", "abiotic": "Abiotic"}


def fig1():
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
        img[pat[:, j], j] = to_rgb(CAT_COLOUR[CAT[t]])
    fig = plt.figure(figsize=(fs.W2, 135 * fs.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 0.62], width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28,
                          left=0.075, right=0.95, top=0.92, bottom=0.01)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=5.3)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_yticks([0, len(pat) - 1], ["1", f"{len(pat):,}"])
    ax.set_ylabel("Accessions (sorted by traits scored)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    starts = {}
    for j, t in enumerate(traits):
        starts.setdefault(CAT[t], []).append(j)
    for c, js in starts.items():
        ax.text((js[0] + js[-1]) / 2, -len(pat) * 0.02, CAT_SHORT[c], ha="center", va="bottom", fontsize=5.5,
                rotation=0 if len(js) > 2 else 90, color=fs.INK2)
        if js[0] > 0:
            ax.axvline(js[0] - 0.5, color="white", lw=0.8)
    fs.panel(ax, "a", x=-0.16, y=1.1)
    # (b) associations between traits
    ax = fig.add_subplot(gs[0, 1])
    pa = pd.read_csv(ROOT / "results/pheno/pair_associations.tsv", sep="\t")
    idx = {t: i for i, t in enumerate(traits)}
    A = np.full((len(traits), len(traits)), np.nan)
    tested = np.zeros_like(A, bool)
    for r in pa.itertuples():
        i, j = sorted((idx[r.a], idx[r.b]), reverse=True)
        tested[i, j] = True
        if r.q_mh < 0.05:
            A[i, j] = np.clip(np.log2(max(r.or_mh, 1e-3)), -5, 5)
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8201f"])
    base = np.ones((len(traits), len(traits), 3))
    base[tested & np.isnan(A)] = to_rgb("#d9d9d9")
    ax.imshow(base, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-5, vmax=5, interpolation="nearest")
    ax.set_xticks(range(len(traits)), [T.SHORT[t] for t in traits], rotation=90, fontsize=4.6)
    ax.set_yticks(range(len(traits)), [T.SHORT[t] for t in traits], fontsize=4.6)
    ax.tick_params(length=0, pad=1)
    n_ = len(traits)
    for c, js in starts.items():             # category separators in the lower triangle only
        k = js[0]
        if k > 0:
            ax.plot([-0.5, k - 0.5], [k - 0.5, k - 0.5], color=fs.LIGHT, lw=0.4)
            ax.plot([k - 0.5, k - 0.5], [k - 0.5, n_ - 0.5], color=fs.LIGHT, lw=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_ticks([-4, -2, 0, 2, 4], labels=["1/16", "1/4", "1", "4", "16"])
    cb.set_label("Odds ratio of resistance", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.text(len(traits) - 1, 1, "grey: tested,\nnot significant", fontsize=5.3, ha="right", va="top", color=fs.INK2)
    fs.panel(ax, "b", x=-0.16, y=1.02)
    # (c) MT-Boost
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=fs.INK2,
                                    lw=0.5))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=5.8, color=fs.INK2, linespacing=1.25)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6, lw=0.6, color=fs.INK2))

    box(0.5, 0.5, 17, 15.5, "Genotypes", "39,627 SNPs; a split separates\ncarriers and non-carriers\nof the alternative allele", "#f4f4f2")
    box(21, 0.5, 20, 15.5, "Start", "zero, or cross-validated\nGBLUP prediction", "#f4f4f2")
    box(44.5, 0.5, 25, 15.5, "Shared trees", "one tree per stress category and\nround; a split serves all traits of\nthe category, leaves differ by trait", "#e6eef8")
    box(73, 0.5, 26, 15.5, "Own trees", "trees of one trait; start and rounds\nchosen per trait on inner folds;\nfinal: mean with and without\nshared trees", "#f4f4f2")
    arrow(18.2, 20.6, 8)
    arrow(41.6, 44.1, 8)
    arrow(70.2, 72.6, 8)
    ax.text(50, 20.5, r"$F_t(\mathbf{x}) = s_t(\mathbf{x}) + \eta_\mathrm{s} \sum_{m \leq S_t} v_{m,t}(\mathbf{x})"
            r" + \eta_\mathrm{p} \sum_{m \leq P_t} u_{m,t}(\mathbf{x})$", ha="center", va="center", fontsize=7)
    fs.panel(ax, "c", x=-0.02, y=0.93)
    fs.save(fig, "fig1_data_model")


def fig2():
    pe = pd.read_csv(ROOT / "results/compare/cv1_point_estimates.tsv", sep="\t")
    cmp_ = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_best_single_avg4_{s}.tsv", sep="\t")
                      for s in ("primary", "other")])
    shr = pd.concat([pd.read_csv(ROOT / f"results/compare/cv1_mtboost_v2_final_vs_mtboost_v2_noshare4_{s}.tsv", sep="\t")
                     for s in ("primary", "other")])
    fig = plt.figure(figsize=(fs.W2, 125 * fs.MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.42, 0.42, 1.15, 1.15], wspace=0.6, left=0.085, right=0.98, top=0.9,
                          bottom=0.08)
    models = ["gblup", "bglr_mt", "bayesb", "lightgbm_avg4", "best_single_avg4", "mtboost_v2_final"]
    handles = []
    for i, (sel, lab) in enumerate(((True, "Primary traits"), (False, "Other traits"))):
        ax = fig.add_subplot(gs[i])
        sub = pe[pe.prim == sel]
        for j, m in enumerate(models):
            lab_m, col, mk = fs.MODEL[m]
            h = ax.scatter(j, sub[m].mean(), s=18, color=col, marker=mk, zorder=3, linewidths=0, label=lab_m)
            if i == 0:
                handles.append(h)
        ax.set_xticks([])
        ax.set_xlim(-0.8, len(models) - 0.2)
        ax.set_title(lab, fontsize=7)
        if i == 0:
            ax.set_ylabel("Mean accuracy (test sets)")
            fs.panel(ax, "a", x=-0.62)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=6, handletextpad=0.1,
               columnspacing=1.0, fontsize=6.5)
    # (b) MT-Boost minus best single-trait model, per trait
    for k, (df, title, letter) in enumerate(((cmp_, "MT-Boost − best single-trait model", "b"),
                                              (shr, "Contribution of the shared trees", "c"))):
        ax = fig.add_subplot(gs[k + 2])
        d = df.assign(prim=df.trait.isin(PRIM)).sort_values(["prim", "diff"], ascending=[False, True])
        y = np.arange(len(d))
        sig = (d.p_holm < 0.05).to_numpy()
        ax.axvline(0, color=fs.MUTED, lw=0.5)
        ax.hlines(y, d.lo, d.hi, color=fs.LIGHT, lw=0.8)
        ax.scatter(d["diff"], y, s=10, color=np.where(sig, "#1d4f91", "white"), edgecolors="#1d4f91",
                   linewidths=0.6, zorder=3)
        ax.set_yticks(y, d.short)
        ax.tick_params(axis="y", length=0, labelsize=5.8)
        np_ = int(d.prim.sum())
        ax.axhline(len(d) - np_ - 0.5, color=fs.LIGHT, lw=0.5, ls=(0, (2, 2)))
        ax.set_xlabel("Difference in accuracy")
        ax.set_title(title, fontsize=7)
        ax.invert_yaxis()
        ax.text(1.0, 1 - (np_ / 2) / len(d), "primary\ntraits", transform=ax.transAxes, fontsize=5.5,
                color=fs.MUTED, ha="right", va="center")
        ax.text(1.0, (len(d) - np_) / 2 / len(d), "other\ntraits", transform=ax.transAxes, fontsize=5.5,
                color=fs.MUTED, ha="right", va="center")
        fs.panel(ax, letter, x=-0.25)
    fs.save(fig, "fig2_prediction")


def fig3():
    p = pd.read_csv(ROOT / "results/calibration/percentiles.tsv", sep="\t")
    p = p[p.window == "block"]
    order = ["W1", "T", "Dt1", "I", "E1", "E2", "E3", "E4", "Rhg1", "Rhg4", "Rps1", "GmSALT3"]
    fig, ax = plt.subplots(figsize=(fs.W1, 62 * fs.MM))
    x = np.arange(len(order))
    for off, (ro, lab, col, mk) in zip((-0.15, 0.15), (("emmax", "Association statistic", "#6b6b6b", "s"),
                                                        ("mtboost", "MT-Boost split gain", "#1d4f91", "o"))):
        v = p[p.readout == ro].set_index("gene").percentile.reindex(order).clip(lower=0.01)
        ax.scatter(x + off, v, s=14, color=col, marker=mk, label=lab, zorder=3, linewidths=0)
    ax.axhline(1, color=fs.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_ylim(100, 0.008)
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["≤0.01", "0.1", "1", "10", "100"])
    ax.set_ylabel("Rank of the gene's block (top %)")
    ax.set_xticks(x, order, rotation=45, ha="right", fontstyle="italic")
    ax.text(len(order) - 0.6, 1.15, "top 1%", fontsize=6, color=fs.MUTED, ha="right", va="top")
    ax.legend(loc="lower left", handletextpad=0.2)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
    fs.save(fig, "fig3_known_genes")


def fig4():
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    loci = pd.read_csv(ROOT / "results/loci/loci_emmax.tsv", sep="\t")
    bt = pd.read_csv(ROOT / "results/loci/block_traits_emmax.tsv", sep="\t")
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
    # rarest classes drawn last; a race-differential block next to another block with as many traits (Rhg1: a
    # trade-off block 0.6 Mb away) is drawn smaller on top of it, so both stay visible
    for k, cls in enumerate(("specific", "unresolved", "concordant", "trade-off", "race-differential")):
        d = loci[loci["class"] == cls]
        size = 6 + 3 * d.n_traits
        if cls == "race-differential":
            others = loci[loci["class"] != cls]
            hidden = [((others.chrom == r.chrom) & (others.n_traits == r.n_traits)
                       & ((others.x - r.x).abs() < 5e6)).any() for r in d.itertuples()]
            size = np.where(hidden, size * 0.35, size)
        ax.scatter(d.x, d.n_traits, s=size, color=fs.CLASS_COLOUR[cls], label=cls.capitalize(),
                   zorder=3 + k, linewidths=0.3, edgecolors="white" if cls != "specific" else fs.LIGHT)
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
    ax.set_xlim(-2e6, acc)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Traits")
    ax.set_ylim(0.3, loci.n_traits.max() + 1.8)
    ax.legend([h[i] for i in order], [l_[i] for i in order], loc="upper left", ncol=5, handletextpad=0.1,
              columnspacing=0.8, bbox_to_anchor=(0, 1.2))
    fs.panel(ax, "a", x=-0.055, y=1.12)
    # (b) allele directions in the chromosome 13 region
    ax = fig.add_subplot(gs[1, 0])
    reg = loci[(loci.chrom == "Gm13") & (loci.start >= 29.1e6) & (loci.end <= 32.1e6)].sort_values("start")
    dirs = {}
    for _, r in reg.iterrows():
        for item in r.directions.split(","):
            m = item.rstrip("+-0")
            dirs[(r.block, m)] = item[len(m):]
    shorts = [T.SHORT[t] for t in T.TRAITS]
    rows = [t for t in shorts if any((b_, t) in dirs for b_ in reg.block)]
    colour = {"+": "#1d4f91", "-": "#e34948", "0": "#d9d9d9"}
    for j, b_ in enumerate(reg.block):
        for i, t in enumerate(rows):
            v = dirs.get((b_, t))
            if v is None:
                continue
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=colour[v], lw=0))
            if v in "+-":
                ax.text(j, i + 0.03, "+" if v == "+" else "\u2212", ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
    ax.set_xlim(-0.6, len(reg) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_yticks(range(len(rows)), rows, fontsize=5.8)
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
    # (c) race reactions by allele at the race-differential block next to Rhg1
    ax = fig.add_subplot(gs[1, 1])
    rd = pd.read_csv(ROOT / "results/loci/race_differential_2059.tsv", sep="\t")
    rd = rd[rd.accessions == "race-3 resistant"].set_index("race").loc[[1, 5, 14]]
    x = np.arange(len(rd))
    w = 0.36
    ax.bar(x - w / 2, rd.carriers_resistant * 100, w, color="#1d4f91", label="with allele")
    ax.bar(x + w / 2, rd.noncarriers_resistant * 100, w, color=fs.LIGHT, label="without allele")
    for k, r in enumerate(rd.itertuples()):
        ax.text(k - w / 2, r.carriers_resistant * 100 + 2, f"{r.carriers}", ha="center", fontsize=5.5, color=fs.INK2)
        ax.text(k + w / 2, r.noncarriers_resistant * 100 + 2, f"{r.noncarriers}", ha="center", fontsize=5.5,
                color=fs.INK2)
    ax.set_xticks(x, [f"Race {r}" for r in rd.index])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Resistant (%)")
    ax.set_title("Race-3-resistant accessions", fontsize=7)
    ax.legend(loc="upper right", handlelength=1, borderaxespad=0)
    fs.panel(ax, "c", x=-0.28, y=1.08)
    fs.save(fig, "fig4_loci")


def fig5():
    c = pd.read_csv(ROOT / "results/germplasm/consistency.tsv", sep="\t")
    r = pd.read_csv(ROOT / "results/germplasm/ranking.tsv", sep="\t", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, 55 * fs.MM), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    ax.bar(c.min_categories_top, c.observed_resistant_share * 100, color="#1d4f91", width=0.6)
    ax.axhline(c.base_rate_all_records.iloc[0] * 100, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)), label="all records")
    for x, y, n in zip(c.min_categories_top, c.observed_resistant_share * 100, c.accessions):
        ax.text(x, y + 1, f"n = {n:,}", ha="center", fontsize=5.8)
    ax.legend(loc="upper left", handlelength=2)
    ax.set_xticks(c.min_categories_top, [f"≥{k}" for k in c.min_categories_top])
    ax.set_xlabel("Stress categories with a top-10% prediction")
    ax.set_ylabel("Records observed resistant (%)")
    fs.panel(ax, "a")
    ax = axs[1]
    for k, (tax, col) in enumerate((("Glycine max", "#1d4f91"), ("Glycine soja", "#e66101"))):
        v = r[r.taxon == tax].multi_resistance_index_within_taxon.value_counts(normalize=True).sort_index() * 100
        ax.bar(v.index + (k - 0.5) * 0.38, v.values, width=0.38, color=col, label=tax.replace("Glycine", "G."))
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Stress categories with a top-10% prediction")
    ax.set_ylabel("Accessions (%)")
    ax.legend(loc="upper right", prop={"style": "italic", "size": 6.5})
    fs.panel(ax, "b")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
    fs.save(fig, "fig5_germplasm")


if __name__ == "__main__":
    fs.setup()
    todo = sys.argv[1:] or ["fig2", "fig3", "fig4", "fig5"]
    for f in todo:
        globals()[f]()
        print(f, "done")
