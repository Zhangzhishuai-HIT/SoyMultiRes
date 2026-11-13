# Adapted 2026-11-13 from 2080/HapGAM/scripts/04_analysis/figstyle.py (sha256 379cff0ca477...): print sizes, Times New Roman, palette; models of this paper below.
"""Shared figure style: journal print sizes, Times New Roman and one fixed colour per model.

Figures are drawn at their final printed size (Springer double column 174 mm, single column 84 mm)
with 7-pt text and 0.5-pt axes, saved as vector PDF (fonts embedded) and as 600-dpi PNG.
The key-model colours (HapGAM-E, EGBLUP, BayesB, LightGBM, GBLUP) were checked for separation
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

INK, INK2, MUTED, LIGHT, GRID, BAND = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#bdbdbd", "#e8e8e8", "#f4f4f2"

# model -> (label, colour, marker); hue = model family, every model has its own marker shape
# model -> (label, colour, marker)
MODEL = {
    "mtboost_v2_final": ("MT-Boost", "#1d4f91", "o"),
    "best_single_avg4": ("Best single-trait model", "#111111", "D"),
    "lightgbm_avg4": ("LightGBM", "#e66101", "^"),
    "bayesb": ("BayesB", "#8e6bbf", "P"),
    "bglr_mt": ("Multi-trait GBLUP", "#1b9e77", "h"),
    "gblup": ("GBLUP", "#6b6b6b", "s"),
    "mtboost_v2_noshare4": ("MT-Boost without shared trees", "#7fa7dc", "o"),
}
CLASS_COLOUR = {"concordant": "#1d4f91", "trade-off": "#e66101", "race-differential": "#8e6bbf",
                "unresolved": "#bdbdbd", "specific": "#e8e8e8"}


def setup():
    plt.rcParams.update({
        "font.family": [FONT, "STIXGeneral", "DejaVu Serif"], "font.size": 7, "axes.titlesize": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        # formulas in Times New Roman as well
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold"})


def save(fig, name):
    out = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    plt.close(fig)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
