"""Public in-memory and filesystem APIs with explicit unit conversion."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SmootherConfig
from .core import fit_z_gmm, predict_z, smooth_xy
from .io import load_tracks
from .model import GMMState, Trajectory, validate_trajectories
from .tuning import tune_hyperparameters

SOURCE_SHA256 = "7412c0e5653dfce61019c7f448c9ec0ba53ed58f6a6b25d800dab8dfbcabd853"
COMPACT_COLUMNS = ["frame", "x", "vx", "y", "vy", "z", "vz", "diameter_px", "postOutProb"]


@dataclass(frozen=True)
class Units:
    """Explicit calibration. z_um = z_origin_um + (slice - slice_origin) * spacing.

    x/y follow image coordinates (y down). fps converts per-frame velocities
    to per-second velocities. No calibration or frame rate is guessed.
    """

    pixel_pitch_um: float
    slice_spacing_um: float
    fps: float
    z_origin_um: float = 0.0
    slice_origin: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("pixel_pitch_um", "slice_spacing_um", "fps") and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class SmoothedBatch:
    tracks: pd.DataFrame
    trajectories: list[Trajectory]
    state: GMMState
    config: SmootherConfig
    tuning: dict


@dataclass(frozen=True)
class SmoothResult:
    output_dir: Path
    tracks_csv: Path
    trajectory_dir: Path
    summary_json: Path
    tuning_json: Path
    config_json: Path
    report_html: Path

    def tracks(self) -> pd.DataFrame:
        return pd.read_csv(self.tracks_csv)

    def summary(self) -> dict:
        return json.loads(self.summary_json.read_text(encoding="utf-8"))


def split_at_gaps(trajectories: list[Trajectory], max_gap: int | None) -> list[Trajectory]:
    if max_gap is None:
        return trajectories
    if any(t.segment != 0 for t in trajectories):
        raise ValueError("max_gap expects unsplit trajectories (segment=0)")
    result = []
    for t in trajectories:
        cuts = np.r_[0, np.flatnonzero(np.diff(t.frames) > max_gap) + 1, len(t)]
        for segment, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
            idx = slice(start, stop)
            result.append(
                Trajectory(
                    t.track_id, t.frames[idx], t.x[idx], t.y[idx], t.z[idx], t.diameter_px[idx], segment
                )
            )
    return result


def smooth_trajectories(
    trajectories: list[Trajectory],
    config: SmootherConfig | None = None,
    *,
    units: Units | None = None,
    progress=None,
) -> SmoothedBatch:
    """Smooth without file I/O. All tracks share a noise mixture and coefficient prior.

    Optional gap segmentation changes fitting segments, never temporal track IDs.
    The returned table includes raw positions, derivatives and outlier probability.
    """
    config = config or SmootherConfig()
    validate_trajectories(trajectories)
    pairs = [(t.track_id, int(f)) for t in trajectories for f in t.frames]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate (track_id, frame) pairs across segments")
    segments = split_at_gaps(trajectories, config.max_gap)
    if units and config.xy_mode == "legacy" and any(np.any(np.diff(t.frames) != 1) for t in segments):
        raise ValueError("Physical XY velocities with frame gaps require xy_mode='auto' or 'local'")
    if config.selection == "auto":
        config, tuning = tune_hyperparameters(segments, config, progress=progress)
    else:
        tuning = {
            "selection": "fixed",
            "candidates": [],
            "selected": {key: getattr(config, key) for key in ("degree", "alpha", "lambda_other")},
        }
    state = fit_z_gmm(segments, config)
    tables = []
    for t, beta, posterior in zip(segments, state.betas, state.post_out_prob, strict=True):
        x, vx = smooth_xy(t.x, t.frames, config)
        y, vy = smooth_xy(t.y, t.frames, config)
        z, vz = predict_z(t, beta, state.time_scale)
        table = pd.DataFrame(
            {
                "track_id": t.track_id,
                "frame": t.frames,
                "segment": t.segment,
                "x_raw": t.x,
                "y_raw": t.y,
                "z_raw": t.z,
                "x": x,
                "vx": vx,
                "y": y,
                "vy": vy,
                "z": z,
                "vz": vz,
                "diameter_px": t.diameter_px,
                "postOutProb": posterior,
                "velocity_resolved": len(t) >= 3 and config.sg_window >= 3 and config.sg_order >= 1,
            }
        )
        if units:
            table["x_um_smooth"] = x * units.pixel_pitch_um
            table["y_um_smooth"] = y * units.pixel_pitch_um
            table["z_um_smooth"] = units.z_origin_um + (z - units.slice_origin) * units.slice_spacing_um
            table["vx_um_s"] = vx * units.pixel_pitch_um * units.fps
            table["vy_um_s"] = vy * units.pixel_pitch_um * units.fps
            table["vz_um_s"] = vz * units.slice_spacing_um * units.fps
            table["diameter_um_smooth"] = t.diameter_px * units.pixel_pitch_um
        tables.append(table)
    tracks = pd.concat(tables, ignore_index=True).sort_values(["track_id", "frame"], kind="stable")
    tuning["final_fit"] = {
        **state.summary(),
        "trajectories": len(segments),
        "degree": config.degree,
        "alpha": config.alpha,
        "lambda0": config.lambda0,
        "lambda_other": config.lambda_other,
    }
    return SmoothedBatch(tracks.reset_index(drop=True), segments, state, config, tuning)


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_destination(destination: Path, input_files: list[Path], overwrite: bool) -> None:
    if destination == Path(destination.anchor) or any(
        p.resolve().is_relative_to(destination) for p in input_files
    ):
        raise ValueError("Output must not replace or contain the input files")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Output path must be a directory")
        if any(destination.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output exists: {destination}; choose a new directory or use --overwrite"
                )
            marker = destination / "holod3_smooth_summary.json"
            if not marker.is_file() or json.loads(marker.read_text()).get("package") != "holod3-smoother":
                raise ValueError("--overwrite only replaces a previous HoloD3-Smoother output directory")
            known = {
                "smoothedtrajs",
                "smoothed_tracks.csv",
                "holod3_smooth_summary.json",
                "zsmooth_tuning_summary.json",
                "smoothedtrajs_zgmm_params.csv",
                "selected_config.json",
                "report.html",
                "segments.csv",
            }
            if any(p.name not in known for p in destination.iterdir()):
                raise ValueError("Output contains unrecognized files; use a new output directory")


def smooth_linker_output(
    source: str | Path | object,
    *,
    output_dir: str | Path | None = None,
    config: SmootherConfig | None = None,
    units: Units | None = None,
    overwrite: bool = False,
    file_limit: int = 0,
    progress=None,
) -> SmoothResult:
    """Process a LinkResult or supported CSV/run path and atomically publish its outputs.

    The default destination is ``<linker output>/smoothing``. Input files are
    never edited. An existing owned output is replaced only after a successful
    fit and complete staging. Do not run concurrent writers on one destination.
    """
    loaded = load_tracks(source, file_limit=file_limit)
    base = (
        loaded.source.parent
        if loaded.source.is_file() or loaded.source.name == "trajectory3d"
        else loaded.source
    )
    requested = Path(output_dir).expanduser() if output_dir is not None else base / "smoothing"
    if requested.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    destination = requested.resolve()
    _validate_destination(destination, loaded.input_files, overwrite)
    requested_config = config or SmootherConfig()
    batch = smooth_trajectories(loaded.trajectories, requested_config, units=units, progress=progress)
    # Preserve all original detection/provenance columns; fitted columns have explicit suffixes.
    additions = batch.tracks.drop(columns="diameter_px").rename(
        columns={c: f"{c}_smooth" for c in ("x", "y", "z", "vx", "vy", "vz")}
    )
    collision = (set(additions) & set(loaded.raw)) - {"track_id", "frame"}
    if collision:
        raise ValueError(f"Input contains reserved smoother output columns: {', '.join(sorted(collision))}")
    merged = loaded.raw.merge(additions, on=["track_id", "frame"], how="left", validate="one_to_one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    backup = None
    try:
        (stage / "smoothedtrajs").mkdir()
        for track_id, group in batch.tracks.groupby("track_id", sort=True):
            group[COMPACT_COLUMNS].to_csv(
                stage / "smoothedtrajs" / f"path_{int(track_id):08d}.csv", index=False
            )
        merged.to_csv(stage / "smoothed_tracks.csv", index=False)
        segment_rows = [
            {
                "track_id": int(t.track_id),
                "segment": t.segment,
                "points": len(t),
                "first_frame": int(t.frames[0]),
                "last_frame": int(t.frames[-1]),
                **{f"beta_{i}": float(b) for i, b in enumerate(beta)},
            }
            for t, beta in zip(batch.trajectories, batch.state.betas, strict=True)
        ]
        pd.DataFrame(segment_rows).to_csv(stage / "segments.csv", index=False)
        _json(stage / "zsmooth_tuning_summary.json", batch.tuning)
        replace(batch.config, selection="fixed").save(stage / "selected_config.json")
        params = {
            "pi_out": batch.state.pi_out,
            "sigma2_z0": batch.state.sigma2_normal,
            "sigma2_z1": batch.state.sigma2_outlier,
            "sigma_z0": math.sqrt(batch.state.sigma2_normal),
            "sigma_z1": math.sqrt(batch.state.sigma2_outlier),
            "time_scale": batch.state.time_scale,
            "degree": batch.config.degree,
            "alpha": batch.config.alpha,
            "lambda0": batch.config.lambda0,
            "lambda_other": batch.config.lambda_other,
            "selection_auto": float(batch.config.selection == "auto"),
            "sg_window": batch.config.sg_window,
            "sg_order": batch.config.sg_order,
            "em_tol": batch.config.em_tol,
            "em_max_iter": batch.config.em_max_iter,
            "em_iterations": batch.state.iterations,
            "em_final_diff": batch.state.final_diff,
            "em_converged": float(batch.state.converged),
        }
        pd.DataFrame({"parameter": list(params), "value": list(params.values())}).to_csv(
            stage / "smoothedtrajs_zgmm_params.csv", index=False
        )
        summary = {
            "schema_version": 1,
            "package": "holod3-smoother",
            "version": "0.1.0",
            "source_algorithm_sha256": SOURCE_SHA256,
            "input": str(loaded.source),
            "input_files": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in loaded.input_files
            ],
            "requested_config": requested_config.to_dict(),
            "resolved_config": batch.config.to_dict(),
            "units": asdict(units) if units else None,
            "native_units": {
                "x_y": "pixel",
                "z": "slice",
                "vz": "slice/frame",
                "vx_vy": "pixel/observation" if batch.config.xy_mode == "legacy" else "pixel/frame",
            },
            "counts": {
                "tracks": len(loaded.trajectories),
                "segments": len(batch.trajectories),
                "points": len(batch.tracks),
                "outlier_points_p_gt_0_5": int((batch.tracks.postOutProb > 0.5).sum()),
                "short_segments_lt_3": sum(len(t) < 3 for t in batch.trajectories),
            },
            "fit": batch.state.summary(),
            "outputs": {
                "tracks_csv": "smoothed_tracks.csv",
                "trajectory_dir": "smoothedtrajs",
                "report_html": "report.html",
                "tuning_json": "zsmooth_tuning_summary.json",
            },
        }
        _json(stage / "holod3_smooth_summary.json", summary)
        from .report import write_report

        write_report(batch, stage / "report.html")
        _validate_destination(destination, loaded.input_files, overwrite)
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}-backup-", dir=destination.parent))
            backup.rmdir()
            destination.rename(backup)
        try:
            stage.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return SmoothResult(
        destination,
        destination / "smoothed_tracks.csv",
        destination / "smoothedtrajs",
        destination / "holod3_smooth_summary.json",
        destination / "zsmooth_tuning_summary.json",
        destination / "selected_config.json",
        destination / "report.html",
    )
