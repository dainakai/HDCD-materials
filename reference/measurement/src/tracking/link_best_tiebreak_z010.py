#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def repo_path(path: Path | str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Connect particle point outputs to the bundled fixed-lag best_tiebreak_z0.10 linker, "
            "optionally evaluate synthetic tracking and run zsmooth."
        )
    )
    p.add_argument("--pipeline-run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <pipeline-run-dir>/best_tiebreak_z010")
    p.add_argument("--input-source", choices=("final", "depth"), default="final")
    p.add_argument("--truth-csv", type=Path, default=None)
    p.add_argument("--eval-out-dir", type=Path, default=None, help="Default: <out-dir>/eval_xy")
    p.add_argument("--laptrack-dir", type=Path, default=SCRIPT_DIR / "laptrack")
    p.add_argument("--link-input-csv", type=Path, default=None, help="Default: <out-dir>/link_input.csv")
    p.add_argument(
        "--linking-mode",
        choices=("fixed-lag", "legacy"),
        default="fixed-lag",
        help="fixed-lag is the production default; legacy reproduces the linker used before fixed-lag adoption.",
    )
    p.add_argument("--xy-source", choices=("seg", "bbox", "x_lap"), default="seg")
    p.add_argument("--size-source", choices=("bbox_area", "diameter"), default="bbox_area")
    p.add_argument("--z-mode", choices=("tiebreak", "distance"), default="tiebreak")
    p.add_argument("--z-weight", type=float, default=0.10)
    p.add_argument("--z-tiebreak-margin", type=float, default=0.04)
    p.add_argument("--z-tiebreak-penalty", type=float, default=0.10)
    p.add_argument("--z-tiebreak-cap", type=float, default=1.0)
    p.add_argument("--z-velocity-alpha", type=float, default=0.08)
    p.add_argument("--z-start-velocity-alpha", type=float, default=0.20)
    p.add_argument(
        "--zsmooth-selection",
        choices=("auto", "fixed"),
        default="auto",
        help="Search Z-smoothing hyperparameters by robust PRESS (default), or use the fixed values below.",
    )
    p.add_argument("--zsmooth-degree", type=int, default=2, help="Used only with --zsmooth-selection fixed.")
    p.add_argument("--zsmooth-alpha", type=float, default=0.0, help="Used only with --zsmooth-selection fixed.")
    p.add_argument(
        "--zsmooth-lambda-other",
        type=float,
        default=1.0e-4,
        help="Used only with --zsmooth-selection fixed.",
    )
    p.add_argument("--zsmooth-degree-grid", default=None, help="Optional comma-separated auto-tune degree grid.")
    p.add_argument(
        "--zsmooth-lambda-other-grid",
        default=None,
        help="Optional comma-separated auto-tune base-precision grid.",
    )
    p.add_argument("--zsmooth-alpha-grid", default=None, help="Optional comma-separated auto-tune alpha grid.")
    p.add_argument("--zsmooth-tune-max-trajectories", type=int, default=None)
    p.add_argument("--zsmooth-tune-min-points", type=int, default=None)
    p.add_argument("--zsmooth-tune-em-max-iter", type=int, default=None)
    p.add_argument("--zsmooth-tune-relative-tolerance", type=float, default=None)
    p.add_argument("--match-space", choices=("xy", "xyz"), default="xy")
    p.add_argument("--skip-pre-zsmooth-eval", action="store_true")
    p.add_argument("--skip-zsmooth", action="store_true")
    p.add_argument("--skip-post-zsmooth-eval", action="store_true")
    p.add_argument(
        "--linker-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to depth3d_online_link.py. Repeat for multiple arguments.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep-intermediate-csv",
        action="store_true",
        help="Keep link_input.csv and the linker's large consolidated CSV copies for debugging.",
    )
    return p.parse_args()


def finite_positive(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    return np.isfinite(values) & (values > 0.0)


def prepare_link_input(pipeline_run_dir: Path, out_csv: Path, source: str) -> dict[str, Any]:
    final_csv = pipeline_run_dir / "particles_3d.csv"
    depth_csv = pipeline_run_dir / "tracks_with_diam_depth.csv"
    if source == "final" and final_csv.exists():
        input_csv = final_csv
    else:
        input_csv = depth_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"missing pipeline link source: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"frame", "file", "conf", "xc", "yc", "w", "h", "slice", "diameter_px"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{input_csv} missing required linker columns: {missing}")

    diameter_source = "diameter_px"
    updated = 0
    if "final_diameter_um" in df.columns:
        good = finite_positive(df["final_diameter_um"])
        if good.any():
            final_um = pd.to_numeric(df["final_diameter_um"], errors="coerce").to_numpy(np.float64)
            current_px = pd.to_numeric(df["diameter_px"], errors="coerce").to_numpy(np.float64)
            current_um = pd.to_numeric(df.get("diameter_um", df["diameter_px"] * 10.0), errors="coerce").to_numpy(np.float64)
            df["diameter_um"] = np.where(good, final_um, current_um)
            df["diameter_px"] = np.where(good, final_um / 10.0, current_px)
            updated = int(good.sum())
            diameter_source = "final_diameter_um"

    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df = df.sort_values(["frame", "row_id"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return {
        "input_csv": str(input_csv),
        "output_csv": str(out_csv),
        "rows": int(len(df)),
        "frames": int(df["frame"].nunique()) if len(df) else 0,
        "diameter_source": diameter_source,
        "diameter_rows_updated": updated,
    }


def run_command(name: str, cmd: list[str], records: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    print(json.dumps({"event": "step_start", "step": name, "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - started
    records.append({"step": name, "elapsed_sec": elapsed, "cmd": cmd})
    print(json.dumps({"event": "step_done", "step": name, "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    pipeline_run_dir = repo_path(args.pipeline_run_dir)
    out_dir = repo_path(args.out_dir) if args.out_dir else pipeline_run_dir / "best_tiebreak_z010"
    eval_out_dir = repo_path(args.eval_out_dir) if args.eval_out_dir else out_dir / "eval_xy"
    laptrack_dir = repo_path(args.laptrack_dir)
    work_dir = out_dir if args.keep_intermediate_csv else out_dir / "_work"
    link_input_csv = repo_path(args.link_input_csv) if args.link_input_csv else work_dir / "link_input.csv"
    truth_csv = repo_path(args.truth_csv) if args.truth_csv else None
    py = sys.executable

    started = time.perf_counter()
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    prepared = prepare_link_input(pipeline_run_dir, link_input_csv, args.input_source)

    link_cmd = [
        py,
        str(laptrack_dir / "depth3d_online_link.py"),
        "--input",
        str(link_input_csv),
        "--out-dir",
        str(out_dir),
        "--z-mode",
        args.z_mode,
        "--z-weight",
        str(args.z_weight),
        "--z-tiebreak-margin",
        str(args.z_tiebreak_margin),
        "--z-tiebreak-penalty",
        str(args.z_tiebreak_penalty),
        "--z-tiebreak-cap",
        str(args.z_tiebreak_cap),
        "--z-velocity-alpha",
        str(args.z_velocity_alpha),
        "--z-start-velocity-alpha",
        str(args.z_start_velocity_alpha),
        "--xy-source",
        args.xy_source,
        "--size-source",
        args.size_source,
    ]
    link_cmd += list(args.linker_arg)
    link_cmd += ["--linking-mode", args.linking_mode]
    run_command("link_best_tiebreak_z010", link_cmd, steps)

    if truth_csv is not None and not args.skip_pre_zsmooth_eval:
        run_command(
            "evaluate_tracking_reconstructed_pre_zsmooth",
            [
                py,
                str(laptrack_dir / "evaluate_tracking_scores.py"),
                "--truth-csv",
                str(truth_csv),
                "--trajectory-dir",
                str(out_dir / "trajectory3d"),
                "--which",
                "reconstructed",
                "--match-space",
                args.match_space,
                "--out-dir",
                str(out_dir / "eval_xy_reconstructed_pre_zsmooth"),
            ],
            steps,
        )

    if not args.skip_zsmooth:
        zsmooth_cmd = [
            "julia",
            str(laptrack_dir / "zsmooth_new.jl"),
            f"--run-dir={out_dir}",
            f"--selection={args.zsmooth_selection}",
            "--force",
        ]
        if args.zsmooth_selection == "fixed":
            zsmooth_cmd += [
                f"--degree={args.zsmooth_degree}",
                f"--alpha={args.zsmooth_alpha}",
                f"--lambda-other={args.zsmooth_lambda_other}",
            ]
        for value, option in [
            (args.zsmooth_degree_grid, "--degree-grid"),
            (args.zsmooth_lambda_other_grid, "--lambda-other-grid"),
            (args.zsmooth_alpha_grid, "--alpha-grid"),
            (args.zsmooth_tune_max_trajectories, "--tune-max-trajectories"),
            (args.zsmooth_tune_min_points, "--tune-min-points"),
            (args.zsmooth_tune_em_max_iter, "--tune-em-max-iter"),
            (args.zsmooth_tune_relative_tolerance, "--tune-relative-tolerance"),
        ]:
            if value is not None:
                zsmooth_cmd.append(f"{option}={value}")
        run_command(
            "zsmooth",
            zsmooth_cmd,
            steps,
        )

    if truth_csv is not None and not args.skip_post_zsmooth_eval:
        which = "reconstructed" if args.skip_zsmooth else "both"
        eval_cmd = [
            py,
            str(laptrack_dir / "evaluate_tracking_scores.py"),
            "--truth-csv",
            str(truth_csv),
            "--run-dir",
            str(out_dir),
            "--which",
            which,
            "--match-space",
            args.match_space,
            "--out-dir",
            str(eval_out_dir),
        ]
        if not args.skip_zsmooth:
            eval_cmd += ["--smoothed-match-trajectory-dir", str(out_dir / "trajectory3d")]
        run_command("evaluate_tracking_post_zsmooth", eval_cmd, steps)

    if not args.keep_intermediate_csv:
        for path in [
            link_input_csv,
            out_dir / "tracks_with_diam_depth.csv",
            out_dir / "tracks_with_diameter.csv",
        ]:
            path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)

    zsmooth_tuning_path = out_dir / "zsmooth_tuning_summary.json"
    zsmooth_tuning = (
        json.loads(zsmooth_tuning_path.read_text(encoding="utf-8"))
        if zsmooth_tuning_path.is_file()
        else None
    )
    zsmooth_selected = zsmooth_tuning.get("selected", {}) if zsmooth_tuning else {}

    summary = {
        "method": "best_tiebreak_z0.10",
        "pipeline_run_dir": str(pipeline_run_dir),
        "out_dir": str(out_dir),
        "prepared_input": prepared,
        "truth_csv": str(truth_csv) if truth_csv else None,
        "eval_out_dir": str(eval_out_dir) if truth_csv else None,
        "parameters": {
            "linking_mode": args.linking_mode,
            "z_mode": args.z_mode,
            "z_weight": args.z_weight,
            "z_tiebreak_margin": args.z_tiebreak_margin,
            "z_tiebreak_penalty": args.z_tiebreak_penalty,
            "z_tiebreak_cap": args.z_tiebreak_cap,
            "z_velocity_alpha": args.z_velocity_alpha,
            "z_start_velocity_alpha": args.z_start_velocity_alpha,
            "xy_source": args.xy_source,
            "size_source": args.size_source,
            "zsmooth_selection": args.zsmooth_selection,
            "zsmooth_degree": zsmooth_selected.get("degree", args.zsmooth_degree),
            "zsmooth_alpha": zsmooth_selected.get("alpha", args.zsmooth_alpha),
            "zsmooth_lambda_other": zsmooth_selected.get(
                "lambda_other", args.zsmooth_lambda_other
            ),
            "zsmooth_auto_tuning": zsmooth_tuning,
            "match_space": args.match_space,
            "linker_args": list(args.linker_arg),
        },
        "outputs": {
            "trajectory3d": str(out_dir / "trajectory3d"),
            "smoothedtrajs": str(out_dir / "smoothedtrajs"),
            "zsmooth_tuning_summary": str(zsmooth_tuning_path),
            "summary_json": str(out_dir / "best_tiebreak_z010_summary.json"),
        },
        "intermediate_csv_retained": bool(args.keep_intermediate_csv),
        "intermediate_csv_dir": str(work_dir) if args.keep_intermediate_csv else None,
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
    }
    summary_path = out_dir / "best_tiebreak_z010_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
