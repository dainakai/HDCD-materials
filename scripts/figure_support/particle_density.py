"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
SOURCE=OUTPUT=Path('.')
TRUTH_COLOR='#202020'
MEASURED_COLOR='#0072B2'

def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.cal": "Liberation Sans:italic",
            "mathtext.sf": "Liberation Sans",
            "mathtext.tt": "Liberation Mono",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.linewidth": 1.35,
            "axes.grid": False,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 7.0,
            "ytick.major.size": 7.0,
            "xtick.minor.size": 3.8,
            "ytick.minor.size": 3.8,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "legend.fontsize": 15,
            "legend.frameon": False,
            "legend.handlelength": 3.0,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

def main() -> None:
    data = pd.read_csv(SOURCE)
    data = data[
        data["diameter_bin_low_um"].ge(25.0)
        & data["diameter_bin_high_um"].le(300.0)
    ].copy()
    data = data.sort_values("diameter_bin_low_um").reset_index(drop=True)

    low = data["diameter_bin_low_um"].to_numpy(float)
    high = data["diameter_bin_high_um"].to_numpy(float)
    if not np.allclose(low[1:], high[:-1]):
        raise RuntimeError("diameter bins are not contiguous")
    edges = np.r_[low[0], high]
    bin_width_um = high - low
    truth = (
        data["truth_number_density_cm3_per_5um_bin"].to_numpy(float)
        / bin_width_um
    )
    measured = (
        data["measured_number_density_cm3_per_5um_bin"].to_numpy(float)
        / bin_width_um
    )

    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    ax.stairs(
        truth,
        edges,
        color=TRUTH_COLOR,
        linewidth=2.5,
        linestyle="-",
        label="Ground truth",
        zorder=2,
    )
    ax.stairs(
        measured,
        edges,
        color=MEASURED_COLOR,
        linewidth=2.5,
        linestyle="-",
        label="Measurement",
        zorder=3,
    )

    ax.set_xlim(25.0, 300.0)
    ax.set_ylim(0.0, 0.37)
    ax.set_xlabel(r"Particle diameter, $D$ [$\mathrm{\mu m}$]", labelpad=8)
    ax.set_ylabel(
        r"Number density [$\mathrm{cm}^{-3}\,\mathrm{\mu m}^{-1}$]",
        labelpad=9,
    )
    ax.xaxis.set_major_locator(MultipleLocator(50.0))
    ax.xaxis.set_minor_locator(MultipleLocator(10.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.35)

    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False, which="both")
    ax.legend(loc="upper right", borderaxespad=0.8)

    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.17, top=0.975)
    stem = OUTPUT / "paper_number_density_distribution_25_300um_seed260745"
    fig.savefig(f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(f"{stem}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
