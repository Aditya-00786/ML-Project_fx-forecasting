"""One consistent figure style for notebooks and the report.

Colours come from a validated categorical palette, assigned to each pair in a fixed order so a
pair keeps its colour in every figure. Charts with many pairs use small multiples rather than
many lines on one axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from fxml.config import load_config
from fxml.events import SHORT_NAMES, STRESS_PERIODS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SHADE = "#f0efec"  # neutral wash for stress periods

CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def pair_colors(pairs: tuple[str, ...] | None = None) -> dict[str, str]:
    """Fixed colour per pair, in configuration order."""
    pairs = pairs or load_config().pairs
    return {pair: CATEGORICAL[i] for i, pair in enumerate(pairs)}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def shade_stress(ax: plt.Axes, label: bool = False, since=None) -> None:
    """Shade the documented stress periods; optionally name them just above the axes.

    Periods ending before ``since`` are skipped, for charts that start later than 2006, and a
    period already under way at ``since`` is labelled from there.
    """
    for name, (start, end) in STRESS_PERIODS.items():
        if since is not None and pd.Timestamp(end) < pd.Timestamp(since):
            continue
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=SHADE, zorder=0, lw=0)
        if label:
            at = pd.Timestamp(start)
            if since is not None:
                at = max(at, pd.Timestamp(since))
            ax.annotate(SHORT_NAMES[name], xy=(at, 1.02),
                        xycoords=("data", "axes fraction"), rotation=45, va="bottom",
                        ha="left", fontsize=7, color=INK_MUTED, annotation_clip=False)


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG under reports/figures and return its path."""
    out = load_config().paths.reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
