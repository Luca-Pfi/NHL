"""Shared chart styling (validated categorical palette: slots 1-3, light surface)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_T6 = "#2a78d6"   # Top 6 forwards  (slot 1, blue)
C_B6 = "#eb6834"   # Bottom 6 forwards (slot 2, orange)
C_D = "#1baf7a"    # Defense (slot 3, aqua) - always paired with a visible label
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#2a78d6", "#86b6ef", "#f0efec", "#f1a2a1", "#e34948"]  # blue <-> gray <-> red

LABEL = {"T6": "Top 6 forwards", "B6": "Bottom 6 forwards", "D": "Defense"}
COLOR = {"T6": C_T6, "B6": C_B6, "D": C_D}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 10,
        "lines.linewidth": 2, "lines.markersize": 8,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def subtitle(ax, text, y=1.02):
    ax.text(0, y, text, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom", ha="left")


def footnote(fig, text):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8.5, ha="left", va="top", wrap=True)
