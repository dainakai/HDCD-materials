#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT_FOR_IMPORT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from src.pipeline.run_pipeline import DETECTION_DIR, REPO_ROOT, repo_path, run_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "New 260624 particle pipeline with depth_model_estimate and SliceDiamModel fused "
            "into one per-frame hologram pass. No tracking."
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
    p.add_argument("--depth-checkpoint", type=Path, default=Path("models/depth/depth_compare_robust_v1.pt"))
    p.add_argument("--depth-fallback-checkpoint", type=Path, default=Path("models/depth/depth_compare_best.pt"))
    p.add_argument("--slice-diam-weights", type=Path, default=Path("models/slice_diam/slice_diammodel_best.pt"))
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.15)
    p.add_argument("--max-det", type=int, default=600)
    p.add_argument("--yolo-batch-size", type=int, default=8)
    p.add_argument("--yolo-image-load-workers", type=int, default=4)
    p.add_argument("--depth-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--depth-batch-size", type=int, default=256)
    p.add_argument("--depth-slice-block", type=int, default=8)
    p.add_argument("--depth-padlen", type=int, default=1536)
    p.add_argument("--diam-batch-size", type=int, default=512)
    p.add_argument("--diam-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--direct-roi-track-ids", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roi-stream-csv-writes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roi-stream-write-queue-size", type=int, default=8)
    p.add_argument("--roi-image-prefetch-workers", type=int, default=4)
    p.add_argument("--roi-image-prefetch-frames", type=int, default=32)
    p.add_argument("--stream-csv-writes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stream-write-chunk-frames", type=int, default=50)
    p.add_argument("--stream-write-queue-size", type=int, default=4)
    p.add_argument("--prefetch-holo-workers", type=int, default=4)
    p.add_argument("--prefetch-holo-frames", type=int, default=16)
    p.add_argument("--slice-diam-frame-batch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-buffer-reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-inplace-propagation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-final-hybrid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--recenter-on-slice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine particle centers on the estimated reconstructed slice inside the fused step.",
    )
    p.add_argument("--scenenum", type=int, default=11)
    p.add_argument("--calibration-scene", default="C000H001S0010")
    p.add_argument("--glassplate-to-volume-um", type=float, default=10000.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--reuse-preprocessing",
        action="store_true",
        help=(
            "Reuse existing raw detections and measured ROI input in the run directory, "
            "then start at fused depth/diameter inference."
        ),
    )
    p.add_argument(
        "--stop-after-preprocessing",
        action="store_true",
        help="Stop after YOLO detection and ROI measurement, retaining their staging CSV.",
    )
    p.add_argument(
        "--keep-intermediate-csv",
        action="store_true",
        help="Keep large Depth/Diam staging CSVs for debugging. By default they are deleted after particles_3d.csv is complete.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = repo_path(args.data_dir)
    image_dir = repo_path(args.image_dir) if args.image_dir else data_dir / "05_03_xyproj" / "C000H001S0011"
    run_dir = repo_path(args.run_dir)
    yolo_weights = repo_path(args.yolo_weights)
    depth_checkpoint = repo_path(args.depth_checkpoint)
    depth_fallback_checkpoint = repo_path(args.depth_fallback_checkpoint)
    slice_diam_weights = repo_path(args.slice_diam_weights)
    py = sys.executable

    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir if args.keep_intermediate_csv else run_dir / "_work" / "depth_diam"
    if args.reuse_preprocessing and args.stop_after_preprocessing:
        raise ValueError("--reuse-preprocessing and --stop-after-preprocessing are mutually exclusive")
    retain_preprocessing = bool(args.keep_intermediate_csv or args.stop_after_preprocessing)
    if args.overwrite and not retain_preprocessing and not args.reuse_preprocessing and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = run_dir / "raw_detections.csv"
    frame_stats_csv = run_dir / "frame_stats.csv"
    raw_summary_json = run_dir / "raw_detection_summary.json"
    measured_preids_csv = work_dir / "tracks_with_diameter_preids.csv"
    measured_csv = work_dir / "tracks_with_diameter.csv"
    measured_summary_json = work_dir / "tracks_with_diameter.summary.json"
    depth_csv = work_dir / "tracks_with_diam_depth.csv"
    slice_csv = work_dir / "tracks_with_estdepth_slice_diameter.csv"
    fused_metrics_json = run_dir / "fused_depth_slice_metrics.json"
    final_csv = run_dir / "particles_3d.csv"
    hybrid_metrics_json = run_dir / "hybrid_diameter_metrics.json"
    pipeline_summary_json = run_dir / "pipeline_summary.json"

    overwrite = ["--overwrite"] if args.overwrite else []
    yolo_device = args.yolo_device if args.yolo_device is not None else args.device
    if str(yolo_device).startswith("cuda:"):
        yolo_device = str(yolo_device).split(":", 1)[1]

    steps: list[dict[str, object]] = []
    started = time.perf_counter()

    roi_output_csv = measured_csv if args.direct_roi_track_ids else measured_preids_csv
    if args.reuse_preprocessing:
        required = [raw_csv, frame_stats_csv, raw_summary_json, measured_csv, measured_summary_json]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse preprocessing because required files are missing or empty: "
                + ", ".join(missing)
            )
        steps.append(
            {
                "step": "reuse_yolo_and_roi_preprocessing",
                "elapsed_sec": 0.0,
                "outputs": [str(path) for path in required],
            }
        )
    else:
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
            "--image-load-workers",
            str(args.yolo_image_load_workers),
        ]
        if args.limit > 0:
            yolo_cmd += ["--limit", str(args.limit)]
        if args.start_index is not None:
            yolo_cmd += ["--start-index", str(args.start_index)]
        if args.end_index is not None:
            yolo_cmd += ["--end-index", str(args.end_index)]
        yolo_cmd += overwrite
        run_step("yolo_raw_minip_detection", yolo_cmd, steps)

        roi_cmd = [
            py,
            str(DETECTION_DIR / "raw_detections_depth_input.py"),
            "--raw-csv",
            str(raw_csv),
            "--image-dir",
            str(image_dir),
            "--out-csv",
            str(roi_output_csv),
            "--summary-json",
            str(measured_summary_json),
            "--min-conf",
            str(args.conf),
            "--device",
            str(args.device),
            "--stream-write-queue-size",
            str(args.roi_stream_write_queue_size),
            "--image-prefetch-workers",
            str(args.roi_image_prefetch_workers),
            "--image-prefetch-frames",
            str(args.roi_image_prefetch_frames),
            *overwrite,
        ]
        if args.direct_roi_track_ids:
            roi_cmd.append("--assign-track-ids")
        if args.roi_stream_csv_writes:
            roi_cmd.append("--stream-csv-writes")
        run_step(
            "measure_detection_roi_for_depth_input",
            roi_cmd,
            steps,
        )

    if not args.direct_roi_track_ids and not args.reuse_preprocessing:
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

    if args.stop_after_preprocessing:
        summary = {
            "event": "pipeline_preprocessing_done",
            "data_dir": str(data_dir),
            "image_dir": str(image_dir),
            "run_dir": str(run_dir),
            "raw_detections_csv": str(raw_csv),
            "measured_roi_csv": str(measured_csv),
            "steps": steps,
            "elapsed_sec": time.perf_counter() - started,
        }
        (run_dir / "pipeline_preprocessing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    fused_cmd = [
        py,
        str(DETECTION_DIR / "depth_and_slice_fused.py"),
        "--run-dir",
        str(run_dir),
        "--data-dir",
        str(data_dir),
        "--input",
        str(measured_csv),
        "--depth-output",
        str(depth_csv),
        "--slice-output",
        str(slice_csv),
        "--metrics-output",
        str(fused_metrics_json),
        "--depth-checkpoint",
        str(depth_checkpoint),
        "--depth-fallback-checkpoint",
        str(depth_fallback_checkpoint),
        "--slice-diam-weights",
        str(slice_diam_weights),
        "--device",
        str(args.device),
        "--model-backend",
        str(args.depth_model_backend),
        "--depth-batch-size",
        str(args.depth_batch_size),
        "--slice-block",
        str(args.depth_slice_block),
        "--padlen",
        str(args.depth_padlen),
        "--diam-batch-size",
        str(args.diam_batch_size),
        "--diam-model-backend",
        str(args.diam_model_backend),
        "--stream-write-chunk-frames",
        str(args.stream_write_chunk_frames),
        "--stream-write-queue-size",
        str(args.stream_write_queue_size),
        "--prefetch-holo-workers",
        str(args.prefetch_holo_workers),
        "--prefetch-holo-frames",
        str(args.prefetch_holo_frames),
        "--scenenum",
        str(args.scenenum),
        "--calibration-scene",
        str(args.calibration_scene),
        "--glassplate-to-volume-um",
        str(args.glassplate_to_volume_um),
    ]
    if args.fused_final_hybrid:
        fused_cmd += [
            "--final-output",
            str(final_csv),
            "--hybrid-metrics-output",
            str(hybrid_metrics_json),
        ]
    if args.stream_csv_writes:
        fused_cmd.append("--stream-csv-writes")
    fused_cmd.append("--slice-diam-frame-batch" if args.slice_diam_frame_batch else "--no-slice-diam-frame-batch")
    fused_cmd.append("--depth-buffer-reuse" if args.depth_buffer_reuse else "--no-depth-buffer-reuse")
    fused_cmd.append(
        "--depth-inplace-propagation"
        if args.depth_inplace_propagation
        else "--no-depth-inplace-propagation"
    )
    if args.recenter_on_slice:
        fused_cmd.append("--recenter-on-slice")
    run_step("estimate_depth_and_slice_diameter_fused", fused_cmd, steps)

    if not args.fused_final_hybrid:
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
        "method": "newpipeline_260624_yolo_depth_slicediam_fused_no_tracking",
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "run_dir": str(run_dir),
        "limit": int(args.limit) if args.limit > 0 else None,
        "device": str(args.device),
        "glassplate_to_volume_um": float(args.glassplate_to_volume_um),
        "recenter_on_slice": bool(args.recenter_on_slice),
        "performance_options": {
            "yolo_batch_size": int(args.yolo_batch_size),
            "yolo_image_load_workers": int(args.yolo_image_load_workers),
            "roi_image_prefetch_workers": int(args.roi_image_prefetch_workers),
            "roi_image_prefetch_frames": int(args.roi_image_prefetch_frames),
            "prefetch_holo_workers": int(args.prefetch_holo_workers),
            "prefetch_holo_frames": int(args.prefetch_holo_frames),
            "depth_padlen": int(args.depth_padlen),
            "slice_diam_frame_batch": bool(args.slice_diam_frame_batch),
            "depth_buffer_reuse": bool(args.depth_buffer_reuse),
            "depth_inplace_propagation": bool(args.depth_inplace_propagation),
        },
        "outputs": {
            "raw_detections_csv": str(raw_csv),
            "particles_3d_csv": str(final_csv),
            "fused_depth_slice_metrics_json": str(fused_metrics_json),
            "hybrid_metrics_json": str(hybrid_metrics_json),
        },
        "intermediate_csv_retained": bool(args.keep_intermediate_csv),
        "intermediate_csv_dir": str(work_dir) if args.keep_intermediate_csv else None,
        "steps": steps,
        "elapsed_sec": total_elapsed,
    }
    pipeline_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.keep_intermediate_csv and work_dir.exists():
        shutil.rmtree(work_dir)
        if work_dir.parent.is_dir() and not any(work_dir.parent.iterdir()):
            work_dir.parent.rmdir()
    print(json.dumps({"event": "pipeline_done", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
