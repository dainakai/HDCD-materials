"""Validated HoloD3-Linker and legacy trajectory CSV adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Trajectory

PATH_PATTERN = re.compile(r"^path_([0-9]+)\.csv$")


@dataclass
class LoadedTracks:
    trajectories: list[Trajectory]
    raw: pd.DataFrame
    source: Path
    input_files: list[Path]


def resolve_input(source: str | Path | object) -> Path:
    if hasattr(source, "output_dir") and hasattr(source, "trajectory_dir"):
        source = source.output_dir
    elif hasattr(source, "trajectory_dir"):
        source = source.trajectory_dir
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if path.name == "particles_3d.csv":
            raise ValueError(
                "particles_3d.csv contains per-detection IDs, not temporal tracks. "
                "Run holod3-link first, then provide its output directory."
            )
        return path
    # Prefer the full linker table so row_id and depth/diameter provenance survive.
    if (path / "tracks_with_diam_depth.csv").is_file():
        return path / "tracks_with_diam_depth.csv"
    if (path / "trajectory3d").is_dir():
        return path / "trajectory3d"
    if (path / "best_tiebreak_z010").is_dir():
        return resolve_input(path / "best_tiebreak_z010")
    if any(PATH_PATTERN.fullmatch(p.name) for p in path.iterdir() if p.is_file()):
        return path
    if (path / "particles_3d.csv").exists():
        raise ValueError("HoloD3 detections are not linked yet. Run holod3-link on this run first.")
    raise ValueError(f"No linker tracks or path_*.csv files found in {path}")


def _numbers(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Column {column} must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column} contains non-finite or missing values")
    return values


def _integers(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numbers(frame, column)
    if np.any(values < 0) or np.any(values >= 2**53) or np.any(values != np.floor(values)):
        raise ValueError(f"Column {column} must contain non-negative integer values < 2**53")
    return values.astype(np.int64)


def load_tracks(source: str | Path | object, *, file_limit: int = 0) -> LoadedTracks:
    """Read a linker table, linker/run directory, path directory or single path CSV.

    Input row order is normalized by track and frame; duplicate track/frame pairs
    and empty trajectories fail explicitly. No detections are silently discarded.
    ``file_limit`` deterministically limits tracks after sorting by numeric ID.
    """
    if isinstance(file_limit, bool) or not isinstance(file_limit, int) or file_limit < 0:
        raise ValueError("file_limit must be a non-negative integer")
    path = resolve_input(source)
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.is_file() and PATH_PATTERN.fullmatch(p.name)),
            key=lambda p: int(PATH_PATTERN.fullmatch(p.name).group(1)),
        )
        if file_limit:
            files = files[:file_limit]
    else:
        files = [path]
    tables = []
    for file in files:
        table = pd.read_csv(file)
        if table.empty:
            raise ValueError(f"Empty trajectory table: {file.name}")
        match = PATH_PATTERN.fullmatch(file.name)
        if match:
            expected_id = int(match.group(1))
            if "track_id" in table and np.any(_integers(table, "track_id") != expected_id):
                raise ValueError(f"track_id disagrees with filename {file.name}")
            table["track_id"] = expected_id
        if "track_id" not in table:
            raise ValueError("Input requires temporal track_id values from a linker")
        x_col = "seg_xc" if "seg_xc" in table else ("x_lap" if "x_lap" in table else "xc")
        y_col = "seg_yc" if "seg_yc" in table else ("y_lap" if "y_lap" in table else "yc")
        required = {"track_id", "frame", x_col, y_col, "slice", "diameter_px"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{file.name}: missing required columns: {', '.join(sorted(missing))}")
        table["track_id"] = _integers(table, "track_id")
        table["frame"] = _integers(table, "frame")
        for column in (x_col, y_col, "slice", "diameter_px"):
            _numbers(table, column)
        # Internal columns never escape into exported provenance.
        if any(c.startswith("__h3s_") for c in table):
            raise ValueError("Input uses reserved __h3s_ column prefix")
        table["__h3s_x"] = _numbers(table, x_col)
        table["__h3s_y"] = _numbers(table, y_col)
        tables.append(table)
    if not tables:
        raise ValueError("No trajectory CSV files found")
    raw = pd.concat(tables, ignore_index=True).sort_values(["track_id", "frame"], kind="stable")
    if raw.duplicated(["track_id", "frame"]).any():
        raise ValueError("Duplicate (track_id, frame) pairs in input")
    if file_limit:
        raw = raw.loc[raw.track_id.isin(sorted(raw.track_id.unique())[:file_limit])]
    trajectories = [
        Trajectory(
            int(track_id),
            group.frame.to_numpy(),
            group["__h3s_x"].to_numpy(),
            group["__h3s_y"].to_numpy(),
            group.slice.to_numpy(),
            group.diameter_px.to_numpy(),
        )
        for track_id, group in raw.groupby("track_id", sort=True)
    ]
    return LoadedTracks(
        trajectories, raw.drop(columns=["__h3s_x", "__h3s_y"]).reset_index(drop=True), path, files
    )
