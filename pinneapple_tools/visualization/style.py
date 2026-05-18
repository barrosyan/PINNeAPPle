"""CFD-style colour maps, themes and defaults for PINNeAPPle visualisation."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# CFD colormaps
# ---------------------------------------------------------------------------

_RAINBOW_DATA = [
    (0.0,   (0.000, 0.000, 0.550)),
    (0.125, (0.000, 0.000, 1.000)),
    (0.250, (0.000, 0.500, 1.000)),
    (0.375, (0.000, 1.000, 0.500)),
    (0.500, (0.000, 1.000, 0.000)),
    (0.625, (0.500, 1.000, 0.000)),
    (0.750, (1.000, 1.000, 0.000)),
    (0.875, (1.000, 0.500, 0.000)),
    (1.000, (1.000, 0.000, 0.000)),
]

_cfd_rainbow = mcolors.LinearSegmentedColormap.from_list(
    "cfd_rainbow",
    [(v[0], v[1]) for v in _RAINBOW_DATA],
    N=256,
)

_COOL_WARM_DATA = [
    (0.0, (0.085, 0.532, 0.201)),
    (0.25,(0.017, 0.427, 0.969)),
    (0.5, (0.929, 0.929, 0.929)),
    (0.75,(0.969, 0.266, 0.105)),
    (1.0, (0.655, 0.047, 0.047)),
]
_cfd_coolwarm = mcolors.LinearSegmentedColormap.from_list(
    "cfd_coolwarm", [(v[0], v[1]) for v in _COOL_WARM_DATA], N=256,
)

# Paraview-esque blue→white→red (diverging, for pressure/vorticity)
CMAPS: dict = {
    "rainbow":   _cfd_rainbow,
    "coolwarm":  _cfd_coolwarm,
    "viridis":   plt.cm.viridis,
    "plasma":    plt.cm.plasma,
    "jet":       plt.cm.jet,
    "inferno":   plt.cm.inferno,
    "turbo":     plt.cm.turbo,
    "grayscale": plt.cm.gray,
    "hot":       plt.cm.hot,
    "RdBu_r":    plt.cm.RdBu_r,
}

DEFAULT_CMAP = "rainbow"


# ---------------------------------------------------------------------------
# Axes theme
# ---------------------------------------------------------------------------

CFD_RC = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#0f3460",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "figure.titlesize": 13,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "font.family":      "monospace",
    "image.cmap":       "rainbow",
}

LIGHT_RC = {
    "figure.facecolor": "#f5f5f5",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#222222",
    "axes.titlecolor":  "#111111",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#111111",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "figure.titlesize": 13,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
}


def use_cfd_style(dark: bool = True) -> None:
    """Apply CFD dark/light theme globally."""
    mpl.rcParams.update(CFD_RC if dark else LIGHT_RC)
    mpl.colormaps.register(_cfd_rainbow,  name="cfd_rainbow",  force=True)
    mpl.colormaps.register(_cfd_coolwarm, name="cfd_coolwarm", force=True)


def get_cmap(name: str = DEFAULT_CMAP):
    """Return a matplotlib colormap by PINNeAPPle alias or standard name."""
    if name in CMAPS:
        return CMAPS[name]
    return plt.get_cmap(name)


def make_figure(nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a styled figure."""
    figsize = kwargs.pop("figsize", (6 * ncols, 5 * nrows))
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
