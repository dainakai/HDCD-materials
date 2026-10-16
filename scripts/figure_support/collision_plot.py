"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
OUTPUT=Path('.')
COHORT = "terminal"
VALIDATION_SEED = 260745
FIGSIZE = (6.5, 4.8)
PNG_DPI = 300
PREVIEW_DPI = 120
METHODS = ["ours", "depth_free", "minip_2d"]
METHOD_LABEL = dict(zip(METHODS, ["Ours", "Depth-free variant", "MinIP-based 2D"]))
METHOD_LEGEND_LABEL = METHOD_LABEL.copy()
METHOD_COLOR = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73"]))
METHOD_LINESTYLE = dict(zip(METHODS, ["-", (0, (5, 2.5)), (0, (5, 2, 1, 2))]))
METHOD_MARKER = dict(zip(METHODS, ["o", "s", "^"]))

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 150,
            "savefig.dpi": PNG_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 14.5,
            "axes.labelsize": 16,
            "axes.linewidth": 1.15,
            "axes.facecolor": "white",
            "axes.edgecolor": "#252525",
            "axes.labelcolor": "#111111",
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3.5,
            "ytick.minor.size": 3.5,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "legend.fontsize": 13.2,
            "legend.frameon": False,
            "lines.linewidth": 2.4,
            "lines.markersize": 7.0,
            "grid.color": "#C9C9C9",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.55,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "path.simplify": True,
            "path.simplify_threshold": 0.08,
        }
    )

def new_figure(
    *,
    figsize: tuple[float, float] = FIGSIZE,
    left: float = 0.16,
    right: float = 0.97,
    bottom: float = 0.16,
    top: float = 0.97,
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return fig, ax

def polish(ax, *, grid_axis: str = "both") -> None:
    del grid_axis
    ax.grid(False, which="both", axis="both")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#252525")

def plot_precision_recall(pr: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = new_figure(left=0.18)
    for method in METHODS:
        d = pr[(pr["cohort"] == COHORT) & (pr["method"] == method)].sort_values("recall")
        ap = float(
            summary[(summary["cohort"] == COHORT) & (summary["method"] == method)]["event_ap"].iloc[0]
        )
        ax.plot(
            d["recall"],
            d["precision"],
            color=METHOD_COLOR[method],
            linestyle=METHOD_LINESTYLE[method],
            drawstyle="steps-post",
            label=f"{METHOD_LEGEND_LABEL[method]}: AP = {ap:.3f}",
        )
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xticks(np.linspace(0.5, 1.0, 6))
    ax.set_yticks(np.linspace(0.5, 1.0, 6))
    polish(ax)
    ax.legend(loc="lower left", handlelength=2.7)
    save_figure(fig, "01_validation_precision_recall")

def plot_fp_burden(target: pd.DataFrame) -> None:
    fig, ax = new_figure(left=0.18)
    for method in METHODS:
        d = target[
            (target["cohort"] == COHORT)
            & (target["method"] == method)
            & (target["target_recall"] <= 0.90)
        ].sort_values("target_recall")
        ax.plot(
            d["target_recall"],
            d["false_positive_events"],
            color=METHOD_COLOR[method],
            linestyle=METHOD_LINESTYLE[method],
            marker=METHOD_MARKER[method],
            markerfacecolor="white",
            markeredgewidth=1.7,
            label=METHOD_LEGEND_LABEL[method],
        )
    ax.set_yscale("log")
    ax.set_xlim(0.49, 0.91)
    ax.set_ylim(1.6, 1500)
    ax.set_xlabel("Target recall")
    ax.set_ylabel("Number of false positives")
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90])
    polish(ax)
    ax.legend(loc="upper left", handlelength=2.7)
    save_figure(fig, "02_validation_false_positive_burden")

def plot_fp_composition(composition: pd.DataFrame) -> None:
    d = composition[composition["cohort"] == COHORT].copy()
    categories = ["duplicate_or_competition", "no_frame_xy_truth"]
    category_label = {
        "duplicate_or_competition": "Duplicate or competing",
        "no_frame_xy_truth": "No truth within matching gate",
    }
    category_color = {
        "duplicate_or_competition": "#56B4E9",
        "no_frame_xy_truth": "#E69F00",
    }

    fig, ax = new_figure(left=0.255, right=0.96, top=0.82)
    y = np.arange(len(METHODS))
    left = np.zeros(len(METHODS))
    for category in categories:
        values = np.array(
            [
                int(d[(d["method"] == method) & (d["category"] == category)]["count"].iloc[0])
                for method in METHODS
            ]
        )
        ax.barh(
            y,
            values,
            left=left,
            height=0.48,
            color=category_color[category],
            label=category_label[category],
        )
        left += values

    for yi, total in zip(y, left):
        ax.text(total + 4.4, yi, f"{total:.0f}", va="center", ha="left", fontsize=14.5, color="#111111")

    ax.set_yticks(y, [METHOD_LABEL[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.17)
    ax.set_xlabel("Number of false positives")
    ax.set_ylabel("")
    polish(ax, grid_axis="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.53, 1.015),
        handlelength=1.4,
        borderaxespad=0,
    )
    save_figure(fig, "04_validation_false_positive_composition")

def plot_topn(pr: pd.DataFrame) -> None:
    fig, ax = new_figure(
        figsize=FIGSIZE,
        left=0.17,
        right=0.97,
        bottom=0.17,
        top=0.97,
    )
    ax.axvline(2000, color="#111111", linewidth=1.6, zorder=1)
    for method in METHODS:
        d = pr[(pr["cohort"] == COHORT) & (pr["method"] == method)].sort_values("predicted_events")
        for metric, metric_label, linestyle in [
            ("precision", "Precision", "-"),
            ("recall", "Recall", (0, (5, 2.5))),
        ]:
            ax.plot(
                d["predicted_events"],
                d[metric],
                color=METHOD_COLOR[method],
                linestyle=linestyle,
                linewidth=2.25,
                zorder=2,
            )
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 1.015)
    ax.set_xlabel("Reviewed candidates, N")
    ax.set_ylabel("Top-N metric value")
    ax.set_yticks(np.linspace(0, 1, 6))
    polish(ax)
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR[method],
            linewidth=2.25,
            label=METHOD_LEGEND_LABEL[method],
        )
        for method in METHODS
    ]
    metric_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.25, linestyle="-", label="Precision"),
        Line2D([0], [0], color="#333333", linewidth=2.25, linestyle=(0, (5, 2.5)), label="Recall"),
    ]
    ax.legend(
        handles=method_handles + metric_handles,
        loc="center left",
        bbox_to_anchor=(0.015, 0.52),
        ncol=2,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=0.9,
        borderaxespad=0,
    )
    save_figure(fig, "05_validation_topn_precision_recall")


def save_figure(fig, stem):
    stem = {"05_validation_topn_precision_recall": "02_validation_topn_precision_recall",
            "02_validation_false_positive_burden": "03_validation_false_positive_burden"}.get(stem, stem)
    ax = fig.axes[0]
    if stem == "03_validation_false_positive_burden":
        ax.set_ylim(1.6, 4000)
    if stem == "04_validation_false_positive_composition":
        ax.set_yticklabels(["Ours", "Depth-free\nvariant", "MinIP-based 2D"])
    if stem == "02_validation_topn_precision_recall":
        for text in ax.get_legend().get_texts():
            text.set_fontsize(12.0)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=PNG_DPI)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)
