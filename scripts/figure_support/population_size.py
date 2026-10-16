"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .scene_index import SCENE_NUMBERS as SCENES

def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.labelsize": 14.0,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
        }
    )

def set_diameter_ticks(axis: plt.Axes, plot_min: float, plot_max: float) -> None:
    regular = np.arange(50.0, plot_max + 0.1, 50.0)
    ticks = np.unique(np.concatenate([[float(plot_min)], regular]))
    ticks = ticks[(ticks >= plot_min) & (ticks <= plot_max)]
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))

def style_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(False)
    axis.tick_params(direction="out", width=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("Droplet diameter [µm]")
    axis.set_ylabel(r"dn/dD [cm$^{-3}$ µm$^{-1}$]")
def load_scene_curves(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    required = {
        "scene_num",
        "diameter_low_um",
        "diameter_high_um",
        "diameter_center_um",
        "dn_dD_cm3_um",
    }
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"missing required columns in {path}: {missing}")
    if frame.duplicated(["scene_num", "diameter_center_um"]).any():
        raise RuntimeError("duplicate scene/bin rows in particle-size table")

    reference: pd.DataFrame | None = None
    curves: list[np.ndarray] = []
    for scene in SCENES:
        selected = frame.loc[frame["scene_num"].eq(scene)].sort_values(
            "diameter_center_um", kind="mergesort"
        )
        if selected.empty:
            raise RuntimeError(f"particle-size table is missing scene {scene}")
        bins = selected[
            ["diameter_low_um", "diameter_high_um", "diameter_center_um"]
        ].reset_index(drop=True)
        if reference is None:
            reference = bins
        elif not np.allclose(
            bins.to_numpy(float), reference.to_numpy(float), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"particle-size bins differ for scene {scene}")
        values = pd.to_numeric(selected["dn_dD_cm3_um"], errors="raise").to_numpy(
            float
        )
        if not np.all(np.isfinite(values) & (values >= 0.0)):
            raise RuntimeError(f"invalid density values for scene {scene}")
        curves.append(values)

    assert reference is not None
    observed_scenes = set(pd.to_numeric(frame["scene_num"], errors="raise").astype(int))
    expected_scenes = set(int(scene) for scene in SCENES)
    if observed_scenes != expected_scenes:
        raise RuntimeError(
            "particle-size table scene set differs from the expected fourteen scenes"
        )
    return reference, np.vstack(curves)

def bootstrap_mean_interval(
    matrix: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    if chunk_size < 1:
        raise ValueError("bootstrap chunk size must be positive")
    scene_count, bin_count = matrix.shape
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((replicates, bin_count), dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sampled = rng.integers(
            0,
            scene_count,
            size=(stop - start, scene_count),
        )
        bootstrap_means[start:stop] = matrix[sampled].mean(axis=1)
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
        axis=0,
    )
    return ci_low, ci_high

def plot_distribution(
    figure_stem: Path,
    centers: np.ndarray,
    mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    *,
    plot_min: float,
    plot_max: float,
) -> None:
    visible = (centers >= plot_min) & (centers <= plot_max)
    if not np.any(visible):
        raise ValueError("plot range does not contain any histogram-bin centers")
    if not np.all(
        np.isfinite(mean[visible])
        & np.isfinite(ci_low[visible])
        & np.isfinite(ci_high[visible])
    ):
        raise RuntimeError("non-finite mean or confidence limit in plot range")
    if not np.all(
        (mean[visible] > 0.0) & (ci_low[visible] > 0.0) & (ci_high[visible] > 0.0)
    ):
        raise RuntimeError(
            "log-scale confidence-interval plot requires positive values in plot range"
        )
    if not np.all(ci_low <= ci_high):
        raise RuntimeError("bootstrap confidence limits are reversed")

    configure_plotting()
    fig, axis = plt.subplots(figsize=(6.3, 4.65), constrained_layout=True)
    axis.fill_between(
        centers[visible],
        ci_low[visible],
        ci_high[visible],
        color="#8ecae6",
        alpha=0.42,
        linewidth=0,
        label="Pointwise 95% scene-bootstrap CI",
    )
    axis.plot(
        centers[visible],
        mean[visible],
        color="#0072b2",
        linewidth=2.3,
        label="14-scene arithmetic mean",
    )
    style_axis(axis)
    axis.set_xlim(plot_min, plot_max)
    set_diameter_ticks(axis, plot_min, plot_max)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=1,
        fontsize=10.5,
        borderaxespad=0.0,
    )
    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300)
    fig.savefig(figure_stem.with_suffix(".pdf"))
    plt.close(fig)
