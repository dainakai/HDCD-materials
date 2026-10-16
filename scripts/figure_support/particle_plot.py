"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, FuncFormatter, MultipleLocator
FIGSIZE=(8.0,5.8)
PERCENTILE_COLOR='#111111'
OUTPUT_DIR=Path('.')

def configure_plotting() -> None:
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
            "font.size": 15.0,
            "axes.labelsize": 18.0,
            "axes.titlesize": 18.0,
            "legend.fontsize": 14.5,
            "legend.frameon": False,
            "legend.handlelength": 3.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "axes.linewidth": 1.35,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
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
            "axes.unicode_minus": False,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )

def style_box(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.35)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

def paper_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=220)
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.17, top=0.975)
    return fig, ax

def save_pdf(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / filename,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "Matplotlib", "Subject": "Seed 260745 particle validation"},
    )
    plt.close(fig)

def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]

def plot_pr(curve: dict[str, np.ndarray | float | int]) -> None:
    recall = np.asarray(curve["recall"], dtype=np.float64)
    precision = np.asarray(curve["precision"], dtype=np.float64)
    keep = np.unique(
        np.linspace(0, len(recall) - 1, min(14000, len(recall))).astype(np.int64)
    )

    fig, ax = paper_figure()
    ax.plot(
        recall[keep],
        precision[keep],
        color="#0072B2",
        linewidth=2.6,
        zorder=2,
    )
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("Recall", labelpad=8)
    ax.set_ylabel("Precision", labelpad=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "" if np.isclose(value, 0.9) else f"{value:.2f}")
    )
    style_box(ax)
    save_pdf(fig, "particle_detection_pr_curve_seed260745.pdf")

def plot_positive_histogram(
    values: np.ndarray,
    percentiles: np.ndarray,
    *,
    xlim: tuple[float, float],
    bins: int,
    xlabel: str,
    unit: str,
    facecolor: str,
    edgecolor: str,
    filename: str,
    decimals: int,
    major_tick: float,
) -> None:
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    fig, ax = paper_figure()
    ax.stairs(
        counts,
        edges,
        fill=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        alpha=0.93,
        zorder=1,
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    styles = ("-", (0, (6.0, 3.0)), (0, (1.4, 2.0)))
    widths = (2.2, 1.9, 1.9)
    labels = ("p50", "p90", "p95")
    for value, label, linestyle, linewidth in zip(
        percentiles, labels, styles, widths, strict=True
    ):
        ax.axvline(
            float(value),
            color=PERCENTILE_COLOR,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{label} = {float(value):.{decimals}f} {unit}",
            zorder=4,
        )
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Number of particle observations per bin", labelpad=9)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.55, handlelength=2.8)
    style_box(ax)
    save_pdf(fig, filename)

def plot_diameter_histogram(values_percent: np.ndarray, quantiles: np.ndarray) -> None:
    xlim = (-50.0, 50.0)
    edges = np.linspace(xlim[0], xlim[1], 161)
    counts, _ = np.histogram(values_percent, bins=edges)
    q025, q05, q50, q95, q975 = [float(value) for value in quantiles]

    fig, ax = paper_figure()
    ax.stairs(
        counts,
        edges,
        fill=True,
        facecolor="#B96F52",
        edgecolor="#754433",
        linewidth=0.9,
        alpha=0.92,
        zorder=1,
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)

    for value in (q025, q975):
        ax.axvline(
            value,
            color=PERCENTILE_COLOR,
            linestyle=(0, (1.4, 2.0)),
            linewidth=1.9,
            zorder=4,
        )
    for value in (q05, q95):
        ax.axvline(
            value,
            color=PERCENTILE_COLOR,
            linestyle=(0, (6.0, 3.0)),
            linewidth=1.9,
            zorder=4,
        )
    ax.axvline(q50, color=PERCENTILE_COLOR, linestyle="-", linewidth=2.2, zorder=4)

    legend_handles = [
        Line2D([0], [0], color=PERCENTILE_COLOR, lw=2.2, linestyle="-", label=f"p50 = {q50:.2f}%"),
        Line2D(
            [0],
            [0],
            color=PERCENTILE_COLOR,
            lw=1.9,
            linestyle=(0, (6.0, 3.0)),
            label=f"p90: [{q05:.2f}, {q95:.2f}]%",
        ),
        Line2D(
            [0],
            [0],
            color=PERCENTILE_COLOR,
            lw=1.9,
            linestyle=(0, (1.4, 2.0)),
            label=f"p95: [{q025:.2f}, {q975:.2f}]%",
        ),
    ]
    ax.set_xlim(*xlim)
    ax.set_xlabel(
        r"Relative diameter error, $\Delta d/d_{\mathrm{true}}$ [%]",
        labelpad=8,
    )
    ax.set_ylabel("Number of particle observations per bin", labelpad=9)
    ax.xaxis.set_major_locator(MultipleLocator(10.0))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.985),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderaxespad=0.0,
        borderpad=0.3,
        handlelength=2.4,
    )
    style_box(ax)
    save_pdf(fig, "diameter_relative_error_hist_seed260745.pdf")
