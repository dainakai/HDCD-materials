"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
FIGURE_ROOT=Path('.')

def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.labelsize": 13.5,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
        }
    )

def equal_limits(x: np.ndarray, y: np.ndarray, padding_fraction: float = 0.04) -> tuple[float, float]:
    values = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if not len(values):
        return 0.0, 1.0
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, max(abs(low), abs(high), 1.0) * 0.1)
    return low - padding_fraction * span, high + padding_fraction * span

def scatter_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    expected: str,
    measured: str,
    xlabel: str,
    ylabel: str,
):
    """Draw FP density first and place every confirmed collision above it."""

    valid = frame.loc[
        np.isfinite(pd.to_numeric(frame[expected], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame[measured], errors="coerce"))
    ].copy()
    x = valid[expected].to_numpy(np.float64)
    y = valid[measured].to_numpy(np.float64)
    low, high = equal_limits(x, y)
    fp = valid.loc[valid["classification_role"].eq("FP")]
    density = None
    if not fp.empty:
        density = axis.hexbin(
            fp[expected],
            fp[measured],
            gridsize=44,
            extent=(low, high, low, high),
            mincnt=1,
            bins="log",
            cmap="Blues",
            linewidths=0.0,
            alpha=0.86,
            rasterized=True,
            zorder=1,
        )
    axis.plot(
        [low, high],
        [low, high],
        color="black",
        linewidth=1.3,
        label="1:1",
        zorder=2,
    )
    tp = valid.loc[valid["classification_role"].eq("TP")]
    axis.scatter(
        tp[expected],
        tp[measured],
        s=34,
        alpha=0.90,
        marker="o",
        facecolor="#e66101",
        edgecolor="white",
        linewidth=0.65,
        label=f"TP: confirmed collision (n={len(tp)})",
        rasterized=True,
        zorder=4,
    )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(False)
    axis.tick_params(direction="out", width=0.9)
    return density

def combined_scatter_figure(
    diameter_events: pd.DataFrame,
    velocity_events: pd.DataFrame,
) -> None:
    configure_plotting()
    fig, axes_array = plt.subplots(1, 2, figsize=(10.4, 5.35), facecolor="white")
    axes = list(axes_array)
    fig.subplots_adjust(
        left=0.075,
        right=0.955,
        bottom=0.22,
        top=0.82,
        wspace=0.42,
    )
    diameter_density = scatter_panel(
        axes[0],
        diameter_events,
        expected="expected_post_diameter_um",
        measured="measured_post_diameter_um",
        xlabel="Expected diameter\nafter event [µm]",
        ylabel="Measured diameter after event [µm]",
    )
    velocity_density = scatter_panel(
        axes[1],
        velocity_events,
        expected="expected_post_vy_m_s",
        measured="measured_post_vy_m_s",
        xlabel="Momentum-predicted " + r"$v_y$" + "\nafter event " + r"[m s$^{-1}$]",
        ylabel=r"Measured $v_y$ after event [m s$^{-1}$]",
    )
    for axis in axes:
        axis.legend(
            frameon=False,
            fontsize=10.2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncols=2,
            borderaxespad=0.0,
            columnspacing=1.2,
            handletextpad=0.5,
        )
    for axis, density in zip(
        axes,
        (diameter_density, velocity_density),
        strict=True,
    ):
        if density is not None:
            colorbar = fig.colorbar(
                density,
                ax=axis,
                shrink=0.70,
                pad=0.025,
                aspect=18,
            )
            colorbar.set_label("FP candidates per hexagonal bin")
    for axis, frame in zip(axes, (diameter_events, velocity_events), strict=True):
        fp_count = int(frame.classification_role.eq("FP").sum())
        axis.text(0.98, 0.025, f"FP: n={fp_count}", transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=9.5, color="#215778",
                  bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2})
    fig.canvas.draw()
    for axis, caption in zip(
        axes,
        (
            "(a) Post-event diameter consistency",
            "(b) Post-event vertical-velocity consistency",
        ),
        strict=True,
    ):
        bounds = axis.get_position()
        fig.text(
            0.5 * (bounds.x0 + bounds.x1),
            0.055,
            caption,
            ha="center",
            va="center",
            fontsize=12.0,
        )
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = "physical_consistency_combined_260715"
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
