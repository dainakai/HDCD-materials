"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_velocity(
    tracks: pd.DataFrame,
    terminal_velocity,
    out_pdf: Path,
    out_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13.5,
            "axes.labelsize": 15.5,
            "legend.fontsize": 12.5,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    specs = [
        ("vx_m_s", "(a) Horizontal velocity", r"Horizontal velocity [m s$^{-1}$]", (-0.5, 0.5)),
        (
            "vy_m_s",
            "(b) Vertical velocity (downward positive)",
            r"Vertical velocity [m s$^{-1}$]",
            (-0.25, 1.0),
        ),
        ("vz_m_s", "(c) Axial velocity", r"Axial velocity [m s$^{-1}$]", (-0.5, 0.5)),
    ]
    fig = plt.figure(figsize=(11.8, 5.25), dpi=220)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.026), wspace=0.035)
    panels = outer[0, 0].subgridspec(1, 3, wspace=0.43)
    axes = [fig.add_subplot(panels[0, i]) for i in range(3)]
    cax = fig.add_subplot(outer[0, 1])
    mappable = None
    beard_handle = None
    for ax, (component, subcaption, ylabel, ylim) in zip(axes, specs, strict=True):
        x = tracks["diameter_um"].to_numpy(np.float64)
        y = tracks[component].to_numpy(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        hb = ax.hexbin(
            x,
            y,
            gridsize=(72, 62),
            extent=(25.0, 300.0, ylim[0], ylim[1]),
            mincnt=1,
            bins="log",
            cmap="viridis",
            linewidths=0.0,
            rasterized=True,
        )
        mappable = hb
        ax.set_xlim(25.0, 300.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Particle diameter [µm]", labelpad=7)
        ax.set_ylabel(ylabel, labelpad=7)
        ax.grid(color="white", alpha=0.20, lw=0.45)
        ax.text(
            0.5,
            -0.31,
            subcaption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13.5,
        )

        if component == "vy_m_s":
            curve_d = np.linspace(25.0, 300.0, 600)
            terminal = (
                np.asarray(
                    terminal_velocity(curve_d, pressure_mb=900.0, kelvin=307.182512),
                    dtype=np.float64,
                )
                / 1.0e6
            )
            (beard_handle,) = ax.plot(
                curve_d,
                terminal - 0.26,
                color="black",
                lw=2.0,
                ls="--",
                label="Beard (1976) $-$ 0.26 m s$^{-1}$",
                zorder=6,
            )

    if beard_handle is not None:
        fig.legend(
            handles=[beard_handle],
            labels=["Beard (1976) $-$ 0.26 m s$^{-1}$"],
            loc="upper center",
            bbox_to_anchor=(0.49, 0.985),
            frameon=True,
            framealpha=0.92,
        )

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Tracks per bin", labelpad=9)
        cbar.ax.tick_params(labelsize=12.5)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.24, top=0.86)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
