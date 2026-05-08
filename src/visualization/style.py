"""Global style configuration for all analysis figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seaborn theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

# Refined rcParams for presentation quality
plt.rcParams.update({
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",

    # Grid
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,

    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Lines
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "lines.markeredgewidth": 0.5,
    "lines.markeredgecolor": "white",

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#dddddd",
    "legend.fancybox": True,
    "legend.loc": "best",

    # Patches
    "patch.edgecolor": "white",
    "patch.linewidth": 0.5,

    # Error bars
    "errorbar.capsize": 3,
})

# ---------------------------------------------------------------------------
# Color palette — colorblind-friendly professional palette
# ---------------------------------------------------------------------------
BLUE = "#2B7BBD"
PURPLE = "#8E44AD"
GREEN = "#27AE60"
RED = "#C0392B"
ORANGE = "#E67E22"
GRAY = "#7F8C8D"
DARK = "#2C3E50"

METHOD_COLORS = {
    "fixed": BLUE,
    "semantic": PURPLE,
    "density": GREEN,
    "recursive": RED,
}

METHOD_FAMILY = {
    "fixed-5": "fixed", "fixed-10": "fixed", "fixed-40": "fixed",
    "semantic-p3": "semantic", "semantic-p5": "semantic", "semantic-p10": "semantic",
    "density": "density",
    "recursive-512-100": "recursive",
}

METHOD_LABEL = {
    "fixed-5": "Fixed-5",
    "fixed-10": "Fixed-10",
    "fixed-40": "Fixed-40",
    "semantic-p3": "Semantic-p3",
    "semantic-p5": "Semantic-p5",
    "semantic-p10": "Semantic-p10",
    "density": "Density",
    "recursive-512-100": "Recursive",
}

DIM_MARKER = {512: "o", 1024: "D"}
DIM_LABEL = {512: "512d", 1024: "1024d"}
DIM_SIZE = {512: 8, 1024: 9}


def family(method: str) -> str:
    return METHOD_FAMILY.get(method, "other")


def color_for(method: str) -> str:
    return METHOD_COLORS.get(METHOD_FAMILY.get(method, ""), GRAY)


def label_for(method: str) -> str:
    return METHOD_LABEL.get(method, method)


def marker_for(dim: int) -> str:
    return DIM_MARKER.get(dim, "x")


def save(fig, name: str) -> Path:
    """Save figure with consistent settings."""
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
