#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PIPELINE_DIR.parent
SCRIPT_DIR = SRC_DIR
DETECTION_DIR = SRC_DIR / "detection"


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "New 260624 particle pipeline: YOLO on MinIP images, ROI diameter for depth conditioning, "
            "depth_model_estimate.py, and SliceDiamModel diameter at estimated depth. No tracking."
        )
    )
    p.add_argument("--data-dir", type=Path, required=True, help="Dataset root containing 01_bkgrem, 02_calibration, variables.yaml.")
    p.add_argument("--image-dir", type=Path, default=None, help="MinIP image directory. Default: <data-dir>/05_03_xyproj/C000H001S0011")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=30, help="Maximum frames to process. Use 0 or a negative value for all selected frames.")
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--yolo-weights", type=Path, default=Path("models/yolo/best.pt"))
    p.add_argument("--depth-checkpoint", type=Path, default=Path("models/depth/depth_compare_best.pt"))
    p.add_argument("--slice-diam-weights", type=Path, default=Path("models/slice_diam/slice_diammodel_best.pt"))
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.15)
    p.add_argument("--max-det", type=int, default=600)
    p.add_argument("--yolo-batch-size", type=int, default=8)
    p.add_argument("--depth-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--depth-batch-size", type=int, default=256)
    p.add_argument("--depth-slice-block", type=int, default=8)
    p.add_argument("--diam-batch-size", type=int, default=512)
    p.add_argument("--scenenum", type=int, default=11)
    p.add_argument("--calibration-scene", default="C000H001S0010")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run_step(name: str, cmd: list[str], steps: list[dict[str, object]]) -> None:
    started = time.perf_counter()
    print(json.dumps({"event": "step_start", "step": name, "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - started
    steps.append({"step": name, "elapsed_sec": elapsed, "cmd": cmd})
    print(json.dumps({"event": "step_done", "step": name, "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    data_dir = repo_path(args.data_dir)
    image_dir = repo_path(args.image_dir) if args.image_dir else data_dir / "05_03_xyproj" / "C000H001S0011"
    run_dir = repo_path(args.run_dir)
    yolo_weights = repo_path(args.yolo_weights)
    depth_checkpoint = repo_path(args.depth_checkpoint)
    slice_diam_weights = repo_path(args.slice_diam_weights)
    py = sys.executable

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = run_dir / "raw_detections.csv"
    frame_stats_csv = run_dir / "frame_stats.csv"
    raw_summary_json = run_dir / "raw_detection_summary.json"
    measured_preids_csv = run_dir / "tracks_with_diameter_preids.csv"
    measured_csv = run_dir / "tracks_with_diameter.csv"
    measured_summary_json = run_dir / "tracks_with_diameter.summary.json"
    depth_csv = run_dir / "tracks_with_diam_depth.csv"
    depth_metrics_json = run_dir / "model_depth_metrics.json"
    slice_csv = run_dir / "tracks_with_estdepth_slice_diameter.csv"
    slice_metrics_json = run_dir / "slice_diameter_metrics.json"
    final_csv = run_dir / "particles_3d.csv"
    hybrid_metrics_json = run_dir / "hybrid_diameter_metrics.json"
    pipeline_summary_json = run_dir / "pipeline_summary.json"

    overwrite = ["--overwrite"] if args.overwrite else []
    yolo_device = args.yolo_device if args.yolo_device is not None else args.device
    if yolo_device.startswith("cuda:"):
        yolo_device = yolo_device.split(":", 1)[1]

    steps: list[dict[str, object]] = []
    started = time.perf_counter()

    yolo_cmd = [
        py,
        str(DETECTION_DIR / "raw_detect_best.py"),
        "--image-dir",
        str(image_dir),
        "--weights",
        str(yolo_weights),
        "--out-csv",
        str(raw_csv),
        "--frame-stats-csv",
        str(frame_stats_csv),
        "--summary-json",
        str(raw_summary_json),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--device",
        str(yolo_device),
        "--batch-size",
        str(args.yolo_batch_size),
    ]
    if args.limit > 0:
        yolo_cmd += ["--limit", str(args.limit)]
    if args.start_index is not None:
        yolo_cmd += ["--start-index", str(args.start_index)]
    if args.end_index is not None:
        yolo_cmd += ["--end-index", str(args.end_index)]
    yolo_cmd += overwrite
    run_step("yolo_raw_minip_detection", yolo_cmd, steps)

    run_step(
        "measure_detection_roi_for_depth_input",
        [
            py,
            str(DETECTION_DIR / "raw_detections_depth_input.py"),
            "--raw-csv",
            str(raw_csv),
            "--image-dir",
            str(image_dir),
            "--out-csv",
            str(measured_preids_csv),
            "--summary-json",
            str(measured_summary_json),
            "--min-conf",
            str(args.conf),
            "--device",
            str(args.device),
            *overwrite,
        ],
        steps,
    )

    run_step(
        "assign_detection_ids_no_tracking",
        [
            py,
            str(DETECTION_DIR / "add_detection_track_ids.py"),
            "--input",
            str(measured_preids_csv),
            "--output",
            str(measured_csv),
            *overwrite,
        ],
        steps,
    )

    run_step(
        "estimate_depth_model",
        [
            py,
            str(DETECTION_DIR / "depth_model_estimate.py"),
            "--run-dir",
            str(run_dir),
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(depth_checkpoint),
            "--input",
            str(measured_csv),
            "--output",
            str(depth_csv),
            "--metrics-output",
            str(depth_metrics_json),
            "--device",
            str(args.device),
            "--model-backend",
            str(args.depth_model_backend),
            "--batch-size",
            str(args.depth_batch_size),
            "--slice-block",
            str(args.depth_slice_block),
            "--scenenum",
            str(args.scenenum),
            "--calibration-scene",
            str(args.calibration_scene),
        ],
        steps,
    )

    run_step(
        "infer_slice_diameter_at_estimated_depth",
        [
            py,
            str(DETECTION_DIR / "infer_slice_diameter_at_estimated_depth.py"),
            "--input",
            str(depth_csv),
            "--data-dir",
            str(data_dir),
            "--weights",
            str(slice_diam_weights),
            "--output",
            str(slice_csv),
            "--metrics-output",
            str(slice_metrics_json),
            "--device",
            str(args.device),
            "--batch-size",
            str(args.diam_batch_size),
            "--scenenum",
            str(args.scenenum),
            "--calibration-scene",
            str(args.calibration_scene),
        ],
        steps,
    )

    run_step(
        "apply_hybrid_diameter_rule",
        [
            py,
            str(DETECTION_DIR / "apply_hybrid_diameter.py"),
            "--input",
            str(slice_csv),
            "--output",
            str(final_csv),
            "--metrics-output",
            str(hybrid_metrics_json),
            *overwrite,
        ],
        steps,
    )

    total_elapsed = time.perf_counter() - started
    summary = {
        "method": "newpipeline_260624_yolo_depth_slicediam_no_tracking",
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "run_dir": str(run_dir),
        "limit": int(args.limit) if args.limit > 0 else None,
        "device": str(args.device),
        "outputs": {
            "raw_detections_csv": str(raw_csv),
            "tracks_with_diameter_csv": str(measured_csv),
            "tracks_with_diam_depth_csv": str(depth_csv),
            "slice_diameter_csv": str(slice_csv),
            "particles_3d_csv": str(final_csv),
        },
        "steps": steps,
        "elapsed_sec": total_elapsed,
    }
    pipeline_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "pipeline_done", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
