"""Console entry point for smoothing, reproducible demos, and validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

from . import __version__
from .config import SmootherConfig
from .pipeline import Units, smooth_linker_output
from .synthetic import evaluate_tracks, write_synthetic
from .tuning import TuningError


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Robust HoloD3-Linker trajectory smoothing (CPU only)")
    root.add_argument("--version", action="version", version=f"holod3-smoother {__version__}")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Smooth a linker result, table, or path directory")
    run.add_argument("source")
    run.add_argument("--output-dir")
    run.add_argument(
        "--config", type=Path, help="JSON configuration; explicit CLI options override its values"
    )
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--file-limit", type=int, default=0, help="Limit tracks in numeric ID order; 0 means all"
    )
    run.add_argument("--selection", choices=["auto", "fixed"])
    run.add_argument("--xy-mode", choices=["auto", "legacy", "local"])
    for name in (
        "degree",
        "em_max_iter",
        "sg_window",
        "sg_order",
        "tune_max_trajectories",
        "tune_min_points",
        "tune_em_max_iter",
        "max_gap",
    ):
        run.add_argument("--" + name.replace("_", "-"), type=int)
    for name in (
        "alpha",
        "lambda0",
        "lambda_other",
        "pi_init",
        "sigma2z0_init",
        "sigma2z1_init",
        "outlier_variance_multiplier",
        "em_tol",
        "tune_relative_tolerance",
    ):
        run.add_argument("--" + name.replace("_", "-"), type=float)
    for name in ("degree_grid", "alpha_grid", "lambda_other_grid"):
        run.add_argument("--" + name.replace("_", "-"), help="Comma-separated candidate values")
    for name in ("pixel_pitch_um", "slice_spacing_um", "fps", "z_origin_um", "slice_origin"):
        run.add_argument("--" + name.replace("_", "-"), type=float)
    demo = sub.add_parser("demo", help="Generate seeded analytic tracks, smooth, evaluate, and report")
    demo.add_argument("--output-dir", default="runs/demo")
    demo.add_argument("--seed", type=int, default=260712)
    demo.add_argument("--tracks", type=int, default=8)
    demo.add_argument("--points", type=int, default=80)
    demo.add_argument("--selection", choices=["auto", "fixed"], default="auto")
    evaluate = sub.add_parser(
        "evaluate", help="Compare smoothed CSV with independently supplied aligned truth"
    )
    evaluate.add_argument("prediction")
    evaluate.add_argument("truth")
    evaluate.add_argument("--output", help="Optional new JSON file (never overwrites)")
    config = sub.add_parser("config", help="Print the complete default JSON configuration")
    config.add_argument("--selection", choices=["auto", "fixed"], default="auto")
    ui = sub.add_parser("ui", help="Launch the optional local parameter-tuning app")
    ui.add_argument("--port", type=int, default=8501)
    return root


def _progress(candidate: dict) -> None:
    score = candidate.get("objective")
    score_text = f"{score:.5g}" if score is not None else "failed"
    print(
        f"candidate {candidate['index']}: degree={candidate['degree']} alpha={candidate['alpha']:g} "
        f"lambda_other={candidate['lambda_other']:g} PRESS={score_text} "
        f"converged={candidate.get('em_converged', False)}",
        file=sys.stderr,
        flush=True,
    )


def _run(args) -> int:
    if args.command == "config":
        print(json.dumps(SmootherConfig(selection=args.selection).to_dict(), indent=2))
        return 0
    if args.command == "ui":
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        try:
            import streamlit  # noqa: F401
        except ImportError as error:
            raise ValueError("Install the UI extra: pip install '.[ui]' or uv sync --extra ui") from error
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).with_name("ui.py")),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(args.port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
        )
    if args.command == "evaluate":
        output = Path(args.output) if args.output else None
        if output is not None and output.exists():
            raise FileExistsError(f"Metrics output exists: {output}")
        metrics = evaluate_tracks(pd.read_csv(args.prediction), pd.read_csv(args.truth))
        payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        if output:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    if args.command == "demo":
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Demo destination already contains files: {root}")
        source = write_synthetic(root / "input", seed=args.seed, tracks=args.tracks, points=args.points)
        result = smooth_linker_output(
            source,
            output_dir=root / "smoothing",
            config=SmootherConfig(selection=args.selection),
            progress=_progress,
        )
        metrics = evaluate_tracks(result.tracks(), pd.read_csv(source / "truth.csv"))
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"output_dir": str(result.output_dir), "report": str(result.report_html), **metrics}, indent=2
            )
        )
        return 0
    config_values = SmootherConfig.load(args.config).to_dict() if args.config else {}
    for field in fields(SmootherConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            if field.name.endswith("_grid"):
                converter = int if field.name == "degree_grid" else float
                value = [converter(token.strip()) for token in value.split(",")]
            config_values[field.name] = value
    config = SmootherConfig.from_dict(config_values)
    unit_values = {
        field.name: getattr(args, field.name)
        for field in fields(Units)
        if getattr(args, field.name) is not None
    }
    if unit_values and not {"pixel_pitch_um", "slice_spacing_um", "fps"}.issubset(unit_values):
        raise ValueError("Physical outputs require --pixel-pitch-um, --slice-spacing-um and --fps together")
    units = Units(**unit_values) if unit_values else None
    result = smooth_linker_output(
        args.source,
        output_dir=args.output_dir,
        config=config,
        units=units,
        overwrite=args.overwrite,
        file_limit=args.file_limit,
        progress=_progress,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "tracks_csv": str(result.tracks_csv),
                "report_html": str(result.report_html),
                "fit": result.summary()["fit"],
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "demo", "evaluate", "config", "ui"}
    if arguments and not arguments[0].startswith("-") and arguments[0] not in commands:
        arguments.insert(0, "run")
    args = parser().parse_args(arguments)
    try:
        return _run(args)
    except TuningError as error:
        print(f"Tuning failed: {error}", file=sys.stderr)
        print(json.dumps(error.summary, indent=2, allow_nan=False), file=sys.stderr)
        return 2
    except (ValueError, OSError, FloatingPointError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
