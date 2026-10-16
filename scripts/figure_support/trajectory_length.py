"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
SOURCE=OUTPUT=Path('.')
TRUTH_COLOR='#202020'
MEASURED_COLOR='#0072B2'

def cumulative_count_curve(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(lengths, dtype=np.int64))
    length, first_index = np.unique(ordered, return_index=True)
    return length, len(ordered) - first_index

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
            "axes.labelweight": "normal",
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
            "legend.fontsize": 14.5,
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
    truth = pd.read_csv(SOURCE / "truth_trajectory_summary.csv")
    measured = pd.read_csv(SOURCE / "measured_trajectory_summary.csv")
    truth = truth[
        truth["diameter_um"].ge(25.0) & truth["diameter_um"].lt(300.0)
    ]
    measured = measured[
        measured["diameter_um"].ge(25.0) & measured["diameter_um"].lt(300.0)
    ]

    truth_length, truth_count = cumulative_count_curve(truth["length"].to_numpy())
    measured_length, measured_count = cumulative_count_curve(
        measured["length"].to_numpy()
    )

    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    ax.step(
        truth_length,
        truth_count,
        where="post",
        color=TRUTH_COLOR,
        linewidth=2.5,
        linestyle="-",
        label=f"Ground truth ($n$ = {len(truth):,})",
        zorder=2,
    )
    ax.step(
        measured_length,
        measured_count,
        where="post",
        color=MEASURED_COLOR,
        linewidth=2.5,
        linestyle="-",
        label=f"Measurement ($n$ = {len(measured):,})",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 4000.0)
    ax.set_ylim(0.8, 3.2e4)
    ax.set_xlabel(r"Trajectory length, $L$ [frames]", labelpad=8)
    ax.set_ylabel(r"Number of trajectories with length $\geq L$", labelpad=9)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Explicitly retain all four spines for a boxed scientific axis.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.35)

    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False, which="both")
    ax.legend(loc="lower left", borderaxespad=0.8)

    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.17, top=0.975)
    stem = OUTPUT / "paper_trajectory_length_counts_25_300um_seed260745"
    fig.savefig(f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(f"{stem}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    truth_exact = truth["length"].value_counts().sort_index()
    measured_exact = measured["length"].value_counts().sort_index()
    pd.concat(
        [
            pd.DataFrame(
                {
                    "source": "ground_truth",
                    "trajectory_length_frames": truth_exact.index.astype(int),
                    "trajectory_count": truth_exact.to_numpy(dtype=int),
                }
            ),
            pd.DataFrame(
                {
                    "source": "measurement",
                    "trajectory_length_frames": measured_exact.index.astype(int),
                    "trajectory_count": measured_exact.to_numpy(dtype=int),
                }
            ),
        ],
        ignore_index=True,
    ).to_csv(
        OUTPUT / "paper_trajectory_length_exact_counts_25_300um_seed260745.csv",
        index=False,
    )

