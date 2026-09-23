"""
Publication-quality plotting helpers for the SIDN-CRO paper.

The defaults below are tuned for top-decile production / OR / IE journals
(IJPE / IJPR / EJOR / Omega / CIE), which generally require:

  * Single-column figure widths around 3.4 in, double around 7.0 in.
  * A serif font (Times-like) at 9 pt for axis labels and 8 pt for ticks.
  * 300+ dpi output for both raster and vector formats.
  * Print-safe colour palettes (Wong / Tol).

Each save() call writes both a PDF (vector, preferred by Elsevier / Springer)
and a PNG (300 dpi raster, useful for slides and Word drafts).
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
COLORS = {           # Wong / Tol colour-blind safe palette (8-class)
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "cyan":   "#56B4E9",
    "grey":   "#666666",
}

RETAILER_COLOR = {
    "Consumer":    COLORS["blue"],
    "Corporate":   COLORS["orange"],
    "Home Office": COLORS["green"],
}

PRODUCT_COLOR = {
    "Fan Shop": COLORS["blue"],
    "Apparel":  COLORS["orange"],
    "Golf":     COLORS["green"],
    "Footwear": COLORS["red"],
    "Outdoors": COLORS["purple"],
}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,         # embed TrueType fonts (Elsevier)
        "ps.fonttype":        42,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset":   "stix",
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "axes.linewidth":     0.6,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.frameon":     False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.4,
        "lines.linewidth":    1.2,
        "lines.markersize":   3.0,
    })


def figsize(kind: str = "single") -> tuple[float, float]:
    """Return (width, height) in inches for the requested figure size."""
    return {
        "single":  (3.4, 2.4),
        "single_tall": (3.4, 4.5),
        "double":  (7.0, 2.6),
        "double_tall": (7.0, 4.6),
        "panel4":  (7.0, 5.2),
    }[kind]


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save a figure to both PDF (vector) and PNG (300 dpi)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"  wrote {pdf.name} and {png.name}")
