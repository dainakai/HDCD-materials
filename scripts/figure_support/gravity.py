"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

import math
from typing import Any

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
FPS = 4_000.0
RECORDED_FRAME_COUNT = 10_917
MAX_FINAL_FRAME_GAP = 12
EFFECTIVE_FRAME_COUNT = (RECORDED_FRAME_COUNT - 1) - MAX_FINAL_FRAME_GAP
FINAL_EXCLUDED_FRAME_COUNT = RECORDED_FRAME_COUNT - EFFECTIVE_FRAME_COUNT
VOLUME_CM3 = 10.71644672
VOLUME_M3 = VOLUME_CM3 * 1.0e-6
EXPOSURE_S = EFFECTIVE_FRAME_COUNT / FPS
EXPOSURE_VOLUME_M3_S = EXPOSURE_S * VOLUME_M3
FINE_MIN_UM = 25.0
FINE_MAX_UM = 300.0
FINE_WIDTH_UM = 5.0
PAIR_WIDTH_UM = 25.0
TIME_BLOCKS = 11
BOOTSTRAP_REPLICATES = 4_000
RANDOM_SEED = 260724
# Default manuscript calculation uses the complete recording and Table 1 volume.
# The constants above remain the historical saved-figure convention.
MANUSCRIPT_VOLUME_CM3 = 10.24 * 10.24 * 102.3 / 1000.0
MANUSCRIPT_EXPOSURE_M3_S = MANUSCRIPT_VOLUME_CM3 * 1e-6 * RECORDED_FRAME_COUNT / FPS

def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.labelsize": 13.5,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
        }
    )

class HallCorrectedLaminarKernel:
    """Davis--Jonas--Hall efficiency with the source-code interpolation rules."""

    def __init__(self, root: Path):
        self.table_path = root / "collision_efficiency_table_long.csv"
        self.metadata_path = root / "extraction_metadata.csv"
        table = pd.read_csv(self.table_path)
        metadata = pd.read_csv(self.metadata_path).set_index("quantity")["value"]
        self.large_nodes_um = np.sort(table["r_large_um"].unique())
        self.ratio_nodes = np.sort(table["radius_ratio"].unique())
        pivot = table.pivot(
            index="r_large_um",
            columns="radius_ratio",
            values="collision_efficiency",
        )
        self.efficiency = pivot.loc[
            self.large_nodes_um, self.ratio_nodes
        ].to_numpy()
        self.surface_pressure_hpa = float(metadata["surface_pressure_hpa"])
        self.surface_temperature_k = float(metadata["derived_surface_temperature_k"])
        self.surface_density_kg_m3 = float(metadata["derived_surface_density_kg_m3"])

    def collision_efficiency(
        self, radius_1_um: np.ndarray, radius_2_um: np.ndarray
    ) -> np.ndarray:
        radius_1_um, radius_2_um = np.broadcast_arrays(radius_1_um, radius_2_um)
        large = np.maximum(radius_1_um, radius_2_um)
        ratio = np.minimum(radius_1_um, radius_2_um) / large
        ratio_hi = np.searchsorted(self.ratio_nodes, ratio, side="left")
        ratio_hi = np.clip(ratio_hi, 1, len(self.ratio_nodes) - 1)
        ratio_lo = ratio_hi - 1
        q = (ratio - self.ratio_nodes[ratio_lo]) / (
            self.ratio_nodes[ratio_hi] - self.ratio_nodes[ratio_lo]
        )
        large_hi = np.searchsorted(self.large_nodes_um, large, side="left")
        below = large <= self.large_nodes_um[0]
        above = large > self.large_nodes_um[-1]
        inside = ~(below | above)
        output = np.empty_like(large, dtype=float)
        output[below] = (
            (1.0 - q[below]) * self.efficiency[0, ratio_lo[below]]
            + q[below] * self.efficiency[0, ratio_hi[below]]
        )
        if np.any(inside):
            hi = large_hi[inside]
            lo = hi - 1
            p = (large[inside] - self.large_nodes_um[lo]) / (
                self.large_nodes_um[hi] - self.large_nodes_um[lo]
            )
            qi = q[inside]
            rlo = ratio_lo[inside]
            rhi = ratio_hi[inside]
            output[inside] = (
                (1.0 - p) * (1.0 - qi) * self.efficiency[lo, rlo]
                + p * (1.0 - qi) * self.efficiency[hi, rlo]
                + (1.0 - p) * qi * self.efficiency[lo, rhi]
                + p * qi * self.efficiency[hi, rhi]
            )
        if np.any(above):
            interpolated = (
                (1.0 - q[above]) * self.efficiency[-1, ratio_lo[above]]
                + q[above] * self.efficiency[-1, ratio_hi[above]]
            )
            output[above] = np.minimum(interpolated, 1.0)
        return output

    def terminal_velocity_m_s(self, radius_um: np.ndarray) -> np.ndarray:
        radius_um = np.asarray(radius_um, dtype=float)
        grav_cm_s2 = 9.80665e2
        rho_water_g_cm3 = 1.0
        lb4l_cm = 6.62e-6
        pb4l_hpa = 1013.25
        tb4l_k = 293.15
        visb4l = 1.818e-4
        rho_air_g_cm3 = self.surface_density_kg_m3 * 1e-3
        diameter_cm = 2.0 * radius_um * 1e-4
        temperature_c = self.surface_temperature_k - 273.15
        if temperature_c >= 0:
            viscosity = (1.7180 + 4.9e-3 * temperature_c) * 1e-4
        else:
            viscosity = (
                1.7180 + 4.9e-3 * temperature_c - 1.2e-5 * temperature_c**2
            ) * 1e-4
        mean_free_path_cm = (
            lb4l_cm
            * (viscosity / visb4l)
            * (pb4l_hpa / self.surface_pressure_hpa)
            * math.sqrt(self.surface_temperature_k / tb4l_k)
        )
        gxdrow = grav_cm_s2 * (rho_water_g_cm3 - rho_air_g_cm3)
        slip = 1.0 + 2.510 * mean_free_path_cm / diameter_cm
        output = np.empty_like(radius_um)
        small = diameter_cm <= 1.9e-3
        output[small] = (
            (gxdrow / (18.0 * viscosity))
            * slip[small]
            * diameter_cm[small] ** 2
            * 1e-2
        )
        if np.any(~small):
            coefficients = np.asarray(
                [
                    -0.318657e1,
                    0.9926960,
                    -0.153193e-2,
                    -0.987059e-3,
                    -0.578878e-3,
                    0.855176e-4,
                    -0.327815e-5,
                ]
            )
            diameter_large = diameter_cm[~small]
            nda = (
                rho_air_g_cm3
                * (4.0 * gxdrow)
                / (3.0 * viscosity**2)
                * diameter_large**3
            )
            x1 = np.log(nda)
            x2 = x1**2
            x3 = x1 * x2
            y = (
                coefficients[0]
                + coefficients[1] * x1
                + coefficients[2] * x2
                + coefficients[3] * x3
                + coefficients[4] * x2**2
                + coefficients[5] * x3 * x2
                + coefficients[6] * x3**2
            )
            reynolds = slip[~small] * np.exp(y)
            output[~small] = (
                viscosity
                * reynolds
                / (rho_air_g_cm3 * diameter_large)
                * 1e-2
            )
        return output

    def kernel_m3_s(
        self, diameter_1_um: np.ndarray, diameter_2_um: np.ndarray
    ) -> np.ndarray:
        diameter_1_um, diameter_2_um = np.broadcast_arrays(
            diameter_1_um, diameter_2_um
        )
        radius_1_um = diameter_1_um / 2.0
        radius_2_um = diameter_2_um / 2.0
        efficiency = self.collision_efficiency(radius_1_um, radius_2_um)
        velocity_1 = self.terminal_velocity_m_s(radius_1_um)
        velocity_2 = self.terminal_velocity_m_s(radius_2_um)
        return (
            efficiency
            * math.pi
            * ((radius_1_um + radius_2_um) * 1e-6) ** 2
            * np.abs(velocity_1 - velocity_2)
        )

def pair_prediction(
    density_m3: np.ndarray,
    kernel_matrix: np.ndarray,
    fine_centers: np.ndarray,
    pair_edges: np.ndarray,
    *,
    exposure_volume_m3_s: float = MANUSCRIPT_EXPOSURE_M3_S,
) -> np.ndarray:
    density_m3 = np.asarray(density_m3, dtype=float)
    if density_m3.ndim == 1:
        density_m3 = density_m3[None, :]
    size = len(pair_edges) - 1
    output = np.full((len(density_m3), size, size), np.nan)
    for small_bin in range(size):
        small = (fine_centers >= pair_edges[small_bin]) & (
            fine_centers < pair_edges[small_bin + 1]
        )
        for large_bin in range(small_bin, size):
            large = (fine_centers >= pair_edges[large_bin]) & (
                fine_centers < pair_edges[large_bin + 1]
            )
            diagonal_factor = 0.5 if small_bin == large_bin else 1.0
            output[:, small_bin, large_bin] = (
                exposure_volume_m3_s
                * diagonal_factor
                * np.einsum(
                    "bi,ij,bj->b",
                    density_m3[:, small],
                    kernel_matrix[np.ix_(small, large)],
                    density_m3[:, large],
                    optimize=True,
                )
            )
    return output

def observed_collision_counts(
    labels: pd.DataFrame,
    pair_edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    collisions = labels.loc[labels["binary_label"].eq(1)].copy()
    small = pd.to_numeric(collisions["pre_diameter_small_um"], errors="coerce")
    large = pd.to_numeric(collisions["pre_diameter_large_um"], errors="coerce")
    size = len(pair_edges) - 1
    matrix = np.zeros((size, size), dtype=int)
    small_index = np.digitize(small, pair_edges, right=False) - 1
    large_index = np.digitize(large, pair_edges, right=False) - 1
    in_range = 0
    for i, j in zip(small_index, large_index, strict=True):
        if 0 <= i < size and 0 <= j < size:
            matrix[min(i, j), max(i, j)] += 1
            in_range += 1
    return matrix, {
        "confirmed_collisions": int(len(collisions)),
        "in_pair_map": int(in_range),
        "outside_pair_map": int(len(collisions) - in_range),
    }

def predictive_standard_deviation(
    expected: np.ndarray,
    bootstrap_expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density_sd = np.nanstd(bootstrap_expected, axis=0, ddof=1)
    poisson_sd = np.sqrt(np.maximum(expected, 0.0))
    total_sd = np.sqrt(np.maximum(expected, 0.0) + density_sd**2)
    return density_sd, poisson_sd, total_sd

def pair_labels(edges: np.ndarray) -> list[str]:
    return [f"{int(low)}–{int(high)}" for low, high in zip(edges[:-1], edges[1:])]

def setup_pair_axis(axis: plt.Axes, labels: list[str]) -> None:
    positions = np.arange(len(labels))
    axis.set_xticks(positions, labels, rotation=52, ha="right")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Larger-droplet diameter [µm]")
    axis.set_ylabel("Smaller-droplet diameter [µm]")
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylim(-0.5, len(labels) - 0.5)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(False)

def annotation_color(image: mpl.image.AxesImage, value: float) -> str:
    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.58 else "white"

def save_pair_map(
    path_stem: Path,
    observed: np.ndarray,
    expected: np.ndarray,
    predictive_sd: np.ndarray,
    pair_edges: np.ndarray,
) -> None:
    configure_plotting()
    size = len(pair_edges) - 1
    invalid = np.tril(np.ones((size, size), dtype=bool), k=-1)
    shared_max = float(
        math.ceil(max(float(observed.max()), float(np.nanmax(expected)), 1.0))
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=shared_max)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    figure = plt.figure(figsize=(10.6, 5.15), layout="constrained")
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        height_ratios=(1.0, 0.075),
        wspace=0.06,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])
    subcaption_axes = [
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    figure.add_subplot(grid[1, 2]).axis("off")
    observed_image = axes[0].imshow(
        np.ma.masked_where(invalid, observed.astype(float)),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    expected_image = axes[1].imshow(
        np.ma.masked_where(invalid, expected),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    labels = pair_labels(pair_edges)
    for axis in axes:
        setup_pair_axis(axis, labels)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    for axis, text in zip(
        subcaption_axes,
        (
            "(a) Confirmed collisions",
            "(b) Gravitational-kernel prediction",
        ),
        strict=True,
    ):
        axis.axis("off")
        axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=12.2)
    for small_bin in range(size):
        for large_bin in range(small_bin, size):
            observed_value = int(observed[small_bin, large_bin])
            observed_color = annotation_color(observed_image, observed_value)
            axes[0].text(
                large_bin,
                small_bin,
                str(observed_value),
                ha="center",
                va="center",
                fontsize=11.2,
                fontweight="bold",
                color=observed_color,
            )
            expected_value = float(expected[small_bin, large_bin])
            sd_value = float(predictive_sd[small_bin, large_bin])
            expected_color = annotation_color(expected_image, expected_value)
            axes[1].text(
                large_bin,
                small_bin,
                f"{expected_value:.1f}\n±{sd_value:.1f}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="semibold",
                linespacing=0.82,
                color=expected_color,
            )
    colorbar = figure.colorbar(
        expected_image,
        cax=colorbar_axis,
    )
    locator = mticker.MaxNLocator(nbins=7, integer=True)
    ticks = locator.tick_values(0.0, shared_max)
    ticks = ticks[(ticks >= 0.0) & (ticks <= shared_max)]
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    colorbar.set_label("Collision count")
    figure.savefig(path_stem.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(path_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

def add_count_axis(
    axis: plt.Axes,
    exposure_volume_m3_s: float,
    *,
    label: str,
    integer_ticks: bool,
) -> plt.Axes:
    if exposure_volume_m3_s <= 0.0:
        raise ValueError("exposure volume-time must be positive")
    count_axis = axis.secondary_yaxis(
        "right",
        functions=(
            lambda rate_scaled: np.asarray(rate_scaled)
            * 1.0e5
            * exposure_volume_m3_s,
            lambda count: np.asarray(count)
            / exposure_volume_m3_s
            / 1.0e5,
        ),
    )
    count_axis.set_ylabel(label)
    count_axis.yaxis.set_major_locator(
        mticker.MaxNLocator(
            nbins=6,
            integer=integer_ticks,
            steps=[1.0, 2.0, 2.5, 5.0, 10.0],
        )
    )
    count_axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    count_axis.tick_params(direction="out", width=0.9)
    return count_axis

def plot_count_columns(
    axis: plt.Axes,
    table: pd.DataFrame,
    x: np.ndarray,
) -> tuple[Any, Any]:
    exposure = table["exposure_volume_m3_s"].to_numpy(float)
    predicted_count = table["predicted_expected_count"].to_numpy(float)
    predicted_sd_count = table["predicted_total_sd"].to_numpy(float)
    confirmed_count = table["confirmed_collision_lower_bound"].to_numpy(float)
    predicted_rate = predicted_count / exposure / 1.0e5
    predicted_sd_rate = predicted_sd_count / exposure / 1.0e5
    confirmed_rate = confirmed_count / exposure / 1.0e5
    prediction_handle = axis.errorbar(
        x,
        predicted_rate,
        yerr=predicted_sd_rate,
        fmt="D",
        markersize=7.8,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color="#0072b2",
        ecolor="#56b4e9",
        elinewidth=2.2,
        capsize=4.5,
        capthick=1.5,
        label="Gravitational-kernel prediction",
        zorder=2,
    )
    confirmed_handle = axis.scatter(
        x,
        confirmed_rate,
        marker="o",
        s=82,
        facecolor="none",
        edgecolor="#d55e00",
        linewidth=2.0,
        label="Confirmed collisions",
        zorder=3,
    )
    return confirmed_handle, prediction_handle

def style_count_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(False)
    axis.tick_params(direction="out", width=0.9)

def save_scene_comparison(
    path_stem: Path,
    scene_table: pd.DataFrame,
    total_row: pd.Series,
) -> None:
    configure_plotting()
    if not scene_table["row_type"].eq("scene").all():
        raise ValueError("the scene comparison must not contain the total row")
    scene_exposure = scene_table["exposure_volume_m3_s"].to_numpy(float)
    if not np.allclose(scene_exposure, scene_exposure[0]):
        raise ValueError("all scene rows must share one exposure volume-time")
    total_table = total_row.to_frame().T
    figure = plt.figure(figsize=(9.4, 5.8), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(14.0, 1.8),
        left=0.085,
        right=0.905,
        top=0.95,
        bottom=0.20,
        wspace=0.42,
    )
    scene_axis = figure.add_subplot(grid[0, 0])
    total_axis = figure.add_subplot(grid[0, 1], sharey=scene_axis)
    scene_x = np.arange(len(scene_table), dtype=float)
    confirmed_handle, prediction_handle = plot_count_columns(
        scene_axis,
        scene_table,
        scene_x,
    )
    plot_count_columns(total_axis, total_table, np.asarray([0.0]))

    all_rates = np.concatenate(
        [
            scene_table["confirmed_collision_rate_lower_bound_m3_s"].to_numpy(float),
            scene_table["predicted_collision_rate_m3_s"].to_numpy(float)
            + scene_table["predicted_collision_rate_total_sd_m3_s"].to_numpy(float),
            np.asarray(
                [
                    float(total_row["confirmed_collision_rate_lower_bound_m3_s"]),
                    float(total_row["predicted_collision_rate_m3_s"])
                    + float(total_row["predicted_collision_rate_total_sd_m3_s"]),
                ]
            ),
        ]
    ) / 1.0e5
    top_limit = max(1.0, 0.5 * math.ceil(2.0 * 1.10 * float(all_rates.max())))
    scene_axis.set_xlim(-0.55, len(scene_table) - 0.45)
    scene_axis.set_ylim(0.0, top_limit)
    scene_axis.set_xticks(
        scene_x,
        scene_table["scene_index"].astype(int).astype(str),
    )
    scene_axis.set_xlabel("Scene")
    scene_axis.set_ylabel(
        r"Collision rate [$10^5$ m$^{-3}$ s$^{-1}$]"
    )
    scene_axis.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=6, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    )
    scene_axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    add_count_axis(
        scene_axis,
        float(scene_exposure[0]),
        label="Collision count",
        integer_ticks=True,
    )

    total_axis.set_xlim(-0.72, 0.72)
    total_axis.set_xticks([0.0], ["All scenes"])
    total_axis.tick_params(
        axis="y",
        left=True,
        labelleft=True,
    )
    add_count_axis(
        total_axis,
        float(total_row["exposure_volume_m3_s"]),
        label="Collision count",
        integer_ticks=True,
    )
    for axis in (scene_axis, total_axis):
        style_count_axis(axis)
    figure.legend(
        [confirmed_handle, prediction_handle],
        [
            "Confirmed collisions",
            "Gravitational-kernel prediction",
        ],
        frameon=False,
        fontsize=11.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=2,
    )
    figure.savefig(
        path_stem.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    figure.savefig(
        path_stem.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)
