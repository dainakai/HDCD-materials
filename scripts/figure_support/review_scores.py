"""Top-80 score plot; source hash and paper reference in provenance/paper_figures.json."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCORE_RANGE = (0.98, 1.0)
SCORE_FIGURE_SIZE = (6.6, 4.6)
SCORE_BINS = 40
TP_COLOR = "#e66101"
FP_COLOR = "#2b83ba"
STEMS = ("review_tp_fp_score_distributions_260715", "physical_consistency_combined_260715")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 13.0, "legend.fontsize": 10.5,
        "xtick.labelsize": 11.0, "ytick.labelsize": 11.0,
        "axes.linewidth": 1.0, "axes.spines.top": True,
        "axes.spines.right": True, "axes.grid": False,
    })

def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.pdf", facecolor="white")
    figure.savefig(directory / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)

def plot_scores(binary: pd.DataFrame, directory: Path) -> pd.DataFrame:
    configure_plotting()
    figure, axis = plt.subplots(figsize=SCORE_FIGURE_SIZE, layout="constrained")
    low, high = SCORE_RANGE
    edges = np.linspace(low, high, SCORE_BINS + 1)
    rows = []
    for status, role, color, alpha, zorder in (
        ("non_collision", "FP: non-collision candidate", FP_COLOR, 0.44, 2),
        ("collision", "TP: confirmed collision", TP_COLOR, 0.62, 3),
    ):
        scores = binary.loc[binary.status.eq(status), "collision_score"].to_numpy(float)
        values, _ = np.histogram(scores, bins=edges)
        require(int(values.sum()) == len(scores), "Figure 11 must include every binary review")
        rows.extend({"panel": "single", "status": status, "low": float(a),
                     "high": float(b), "count": int(n)}
                    for a, b, n in zip(edges[:-1], edges[1:], values, strict=True))
        axis.bar(edges[:-1], values, width=np.diff(edges), align="edge", color=color,
                 edgecolor=color, linewidth=0.9, alpha=alpha, zorder=zorder,
                 label=f"{role} (n={len(scores)})")
    axis.set(xlim=(low, high), xlabel="Score", ylabel="Candidate count")
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axis.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(low, high, 5)))
    axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    axis.tick_params(direction="out", width=0.8)
    axis.legend(frameon=False, fontsize=10.5, loc="upper left")
    save_figure(figure, directory, STEMS[0])
    return pd.DataFrame(rows)

def plot(labels: pd.DataFrame, output: Path) -> pd.DataFrame:
    binary = labels.loc[labels.status.isin(["collision", "non_collision"])].copy()
    require(binary.review_stratum.eq("top_score").all(), "Figure 11 requires the top-80 cohort")
    require(binary.review_inclusion_probability.eq(1).all(), "Top-80 review inclusion must be one")
    return plot_scores(binary, output)
