"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import polars as pl
FPS = 4000.0
INK = MUTED_INK = FRAME = '#000000'
OUTPUT_DIR = Path('.')

def style_publication() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 17.0,
            "axes.titlesize": 20.0,
            "axes.labelsize": 19.0,
            "xtick.labelsize": 17.0,
            "ytick.labelsize": 17.0,
            "legend.fontsize": 11.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": FRAME,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

def scene_summary(frame_metrics: pl.DataFrame, scenes: list[str]) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        subset = frame_metrics.filter(pl.col("scene") == scene)
        count = subset["particles_per_hologram"].to_numpy().astype(float)
        lwc = subset["lwc_g_m3"].to_numpy().astype(float)
        record: dict[str, object] = {
            "scene_index": scene_index,
            "scene": scene,
            "scene_number": int(subset["scene_number"][0]),
            "frames": subset.height,
            "duration_s": subset.height / FPS,
        }
        for prefix, values in (("particles", count), ("lwc_g_m3", lwc)):
            record[f"{prefix}_mean"] = float(np.mean(values))
            record[f"{prefix}_std"] = float(np.std(values, ddof=1))
            record[f"{prefix}_q10"] = float(np.quantile(values, 0.10))
            record[f"{prefix}_q25"] = float(np.quantile(values, 0.25))
            record[f"{prefix}_median"] = float(np.quantile(values, 0.50))
            record[f"{prefix}_q75"] = float(np.quantile(values, 0.75))
            record[f"{prefix}_q90"] = float(np.quantile(values, 0.90))
        records.append(record)
    return pl.DataFrame(records)

def draw_mean_sd(
    ax: plt.Axes,
    summary: pl.DataFrame,
    prefix: str,
    xlabel: str,
    show_scene_labels: bool,
) -> None:
    y = summary["scene_index"].to_numpy().astype(float)
    mean = summary[f"{prefix}_mean"].to_numpy()
    standard_deviation = summary[f"{prefix}_std"].to_numpy()
    ax.errorbar(
        mean,
        y,
        xerr=standard_deviation,
        fmt="o",
        markersize=6.8,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=1.35,
        ecolor=INK,
        elinewidth=1.3,
        capsize=4.5,
        capthick=1.3,
        linestyle="none",
        zorder=3,
    )
    low = mean - standard_deviation
    high = mean + standard_deviation
    span = float(np.max(high) - np.min(low))
    pad = 0.09 * span if span > 0 else 1.0
    ax.set_xlim(float(np.min(low)) - pad, float(np.max(high)) + pad)
    ax.set_ylim(summary.height + 0.65, 0.35)
    ax.set_yticks(y)
    if show_scene_labels:
        ax.set_yticklabels([str(int(value)) for value in y])
        ax.set_ylabel("Scene number", labelpad=10)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(xlabel, labelpad=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
    ax.tick_params(axis="both", direction="out", length=5.0, width=0.9)
    ax.tick_params(axis="y", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(FRAME)
    ax.grid(False)

def render_scene_summary(summary: pl.DataFrame) -> tuple[list[str], dict[str, object]]:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.8),
        gridspec_kw={
            "left": 0.095,
            "right": 0.975,
            "bottom": 0.115,
            "top": 0.975,
            "wspace": 0.105,
        },
    )
    count_ax, lwc_ax = axes
    draw_mean_sd(
        count_ax,
        summary,
        "particles",
        "Droplets per hologram",
        show_scene_labels=True,
    )
    draw_mean_sd(
        lwc_ax,
        summary,
        "lwc_g_m3",
        r"Liquid water content [g m$^{-3}$]",
        show_scene_labels=False,
    )
    files = save_figure_set(figure, "particle_count_lwc_scene_summary")
    plt.close(figure)
    return files, {
        "layout": "two horizontally aligned scientific error-bar panels",
        "scene_axis_labels": "sequential 1–14 in acquisition order",
        "color_mapping": "none; pure-black markers, error bars, text, ticks, and box frames",
        "summary_marks": "open circle=mean; horizontal error bar=one sample standard deviation",
        "standard_deviation": "one sample SD across all raw frames, ddof=1",
        "axis_frame_style": "box; all four spines visible",
        "font_sizes_pt": {"base": 17.0, "axis_labels": 19.0, "tick_labels": 17.0},
        "subplot_wspace": 0.105,
    }

def save_figure_set(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if suffix == "png":
            kwargs["dpi"] = 260
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return paths
