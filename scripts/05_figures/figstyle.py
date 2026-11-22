# Adapted 2026-11-13 from 2080/HapGAM/scripts/04_analysis/figstyle.py (sha256 379cff0ca477...): print sizes, Times New Roman, palette; models of this paper below.
"""Shared figure style: journal print sizes, Times New Roman and one fixed colour per model.

Figures are drawn at their final printed size (Springer double column 174 mm, single column 84 mm)
with 7-pt text and 0.5-pt axes, saved as vector PDF (fonts embedded) and as 600-dpi PNG.
The key-model colours were checked for separation
under simulated protanopia/deuteranopia (worst pair OKLab dE >= 11); within a model family the
hue is shared and the marker shape differs, so identity never depends on colour alone.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
W2 = 174 * MM   # double column
W1 = 84 * MM    # single column
FONT = "Times New Roman"
# register Times New Roman when it is installed outside matplotlib's font cache (e.g. in ~/.local/share/fonts)
for _d in ("~/.local/share/fonts", "~/.fonts"):
    for _f in glob.glob(os.path.join(os.path.expanduser(_d), "**", "[Tt]imes*.[Tt][Tt][Ff]"), recursive=True):
        font_manager.fontManager.addfont(_f)

INK, INK2, MUTED, LIGHT, GRID, BAND = "#222222", "#4a4a4a", "#8c8c8c", "#c8c8c8", "#ebebeb", "#f5f5f3"
PANEL_BG = "#f7f8fa"

# Palette (2026-11-15, user: figure colours at the level of Cell/Nature/Science). Nature-style hues adjusted to pass
# the dataviz palette validator (lightness band, chroma floor, CVD and normal-vision separation of adjacent / all
# pairs). One meaning per colour family across all figures.
RED, NAVY, TEAL, OCHRE = "#E64B35", "#2F4E9E", "#00A087", "#D9A400"
# model -> (label, colour, marker, filled); the proposed model in red, the comparators in navy/ochre/teal and greys
MODEL = {
    "mtboost_v2_final": ("TraitBoost", RED, "o", True),
    "mtboost_v2_noshare4": ("TraitBoost without shared trees", RED, "o", False),
    "mtboost_v2_share4": ("Chosen option alone", RED, "v", False),
    "best_single_avg4": ("Best single-trait model", NAVY, "D", True),
    "ens_gblup_lightgbm": ("Mean of GBLUP and LightGBM", OCHRE, "s", False),
    "ens_lightgbm_noshare": ("Tree ensemble", OCHRE, "D", False),          # mean of LightGBM and TraitBoost without shared trees (defined in the legend)
    "lightgbm_avg4": ("LightGBM", OCHRE, "^", True),
    "bglr_mt": ("Multi-trait GBLUP", TEAL, "h", True),
    "bayesb": ("BayesB", "#9a9a9a", "P", True),
    "gblup": ("GBLUP", "#5e5e5e", "s", True),
    "mtboost_v2_gblup": ("GBLUP", "#5e5e5e", "s", True),
}
# stress categories (adjacent-pair validated in this order: CVD dE >= 11.6, normal-vision dE >= 20.3)
CATEGORY = {"SCN": "#2F4E9E", "Phytophthora": "#E64B35", "fungi_oomycetes": "#00A087", "bacteria": "#D9A400",
            "virus": "#C2185B", "chewing_insects": "#7B6FC4", "sucking_insects": "#4DBBD5", "abiotic": "#EE8866"}
# short labels for tight places (the genus name in italics) and the names of Table 1
CATEGORY_LABEL = {"SCN": "SCN", "Phytophthora": r"$\it{P.\ sojae}$", "fungi_oomycetes": "Other fungi",
                  "bacteria": "Bacteria", "virus": "Viruses", "chewing_insects": "Chewing insects",
                  "sucking_insects": "Sucking insects", "abiotic": "Abiotic"}
CATEGORY_FULL = {"SCN": "Soybean cyst nematode", "Phytophthora": r"$\it{Phytophthora\ sojae}$",
                 "fungi_oomycetes": "Other fungi and oomycetes", "bacteria": "Bacteria", "virus": "Viruses",
                 "chewing_insects": "Chewing insects", "sucking_insects": "Sucking insects", "abiotic": "Abiotic stresses"}
CLASS_COLOUR = {"concordant": NAVY, "race-differential": RED, "trade-off": OCHRE, "unresolved": "#9a9a9a",
                "specific": "white"}
CLASS_LABEL = {"concordant": "Concordant", "race-differential": "Race-differential",
               "trade-off": "Opposite between stresses", "unresolved": "Unresolved", "specific": "One trait"}
DIRECTION = {"+": NAVY, "-": RED, "0": "#d9d9d9"}
DIVERGING = ["#1b2f63", NAVY, "#f4f4f4", RED, "#9e2a17"]


def setup():
    plt.rcParams.update({
        "font.family": [FONT, "STIXGeneral", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3, "lines.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.edgecolor": "#333333", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        # formulas in Times New Roman as well
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.cal": f"{FONT}:italic", "mathtext.sf": FONT, "mathtext.tt": FONT})


def save(fig, name):
    out = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    plt.close(fig)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
