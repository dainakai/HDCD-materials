#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.cuda_roi_kernels import CudaRoiAbs2Cropper  # noqa: E402
from src.detection.depth_model_estimate import compute_holo, fft2, ifft2, normalized_image_name, transfer_setup  # noqa: E402
from src.detection.train_slice_diammodel import load_model  # noqa: E402


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def crop_start_xy_tensors(x: np.ndarray, y: np.ndarray, crop_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = np.rint(x.astype(np.float64) - crop_size / 2).astype(np.int32)
    y0 = np.rint(y.astype(np.float64) - crop_size / 2).astype(np.int32)
    return (
        torch.from_numpy(x0).to(device=device, dtype=torch.int32).contiguous(),
        torch.from_numpy(y0).to(device=device, dtype=torch.int32).contiguous(),
    )


def crop_with_mean_padding_cpu(field: torch.Tensor, x0: np.ndarray, y0: np.ndarray, crop_size: int, datlen: int, offset: int) -> torch.Tensor:
    intensity = (field.real.square() + field.imag.square()).clamp(0.0, 1.0)
    central = intensity[offset : offset + datlen, offset : offset + datlen]
    crops = []
    for x_start, y_start in zip(x0, y0, strict=True):
        xs = torch.arange(int(x_start), int(x_start) + crop_size, device=field.device)
        ys = torch.arange(int(y_start), int(y_start) + crop_size, device=field.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        valid = (xx >= 0) & (xx < datlen) & (yy >= 0) & (yy < datlen)
        xx = xx.clamp(0, datlen - 1)
        yy = yy.clamp(0, datlen - 1)
        crop = central[yy, xx]
        if not bool(valid.all()):
            mean = crop[valid].mean() if bool(valid.any()) else central.mean()
            crop = torch.where(valid, crop, mean)
        crops.append(crop)
    return torch.stack(crops, dim=0)


@torch.inference_mode()
def predict_diameter_batch(
    model: torch.nn.Module,
    crops: torch.Tensor,
    norm: dict[str, float],
    calibration_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.round(crops.clamp(0.0, 1.0) * 255.0).div(255.0)[:, None, :, :]
    x = x.contiguous()
    mu_norm, sigma_norm = model(x)
    mu_log = mu_norm.detach().float().cpu().numpy() * norm["log_diam_std"] + norm["log_diam_mean"]
    sigma_log = sigma_norm.detach().float().cpu().numpy() * norm["log_diam_std"] * calibration_scale
    pred_um = np.exp(mu_log)
    sigma_um = pred_um * sigma_log
    return pred_um, sigma_um, mu_log, sigma_log


def optimize_slice_model_backend(
    model: torch.nn.Module,
    *,
    backend: str,
    device: torch.device,
    batch_size: int,
    crop_size: int,
) -> tuple[torch.nn.Module, dict[str, object], float]:
    started = time.perf_counter()
    info: dict[str, object] = {
        "model_backend": backend,
        "inference_precision": "fp16" if backend == "tensorrt" else "fp32",
        "trt_dynamic_max_batch": None,
    }
    if backend == "torch":
        return model, info, time.perf_counter() - started
    if backend != "tensorrt":
        raise RuntimeError(f"Unknown diameter model backend: {backend}")
    if device.type != "cuda":
        raise RuntimeError("TensorRT diameter backend requires CUDA. Use --model-backend torch on CPU.")

    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError("TensorRT diameter backend requires torch-tensorrt. Install torch-tensorrt first.") from exc

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 1, crop_size, crop_size),
                opt_shape=(batch_size, 1, crop_size, crop_size),
                max_shape=(batch_size, 1, crop_size, crop_size),
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
        truncate_double=True,
    )
    trt_model.eval()
    info["model_backend"] = "tensorrt"
    info["inference_precision"] = "fp16"
    info["trt_dynamic_max_batch"] = int(batch_size)
    return trt_model, info, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer diameter from a reconstructed slice at depth_model_estimate's estimated depth.")
    p.add_argument("--input", type=Path, required=True, help="tracks_with_diam_depth.csv from depth_model_estimate.py")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics-output", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--priter", type=int, default=3)
    p.add_argument("--scenenum", type=int, default=11)
    p.add_argument("--calibration-scene", default="C000H001S0010")
    p.add_argument("--glassplate-to-volume-um", type=float, default=10000.0)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--row-limit", type=int, default=0)
    p.add_argument("--only-file", default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--fft-backend", choices=["torch"], default="torch")
    return p.parse_args()


def frame_groups(df: pd.DataFrame, frames: int, only_file: str) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    for file_name, group in df.groupby("file", sort=True):
        norm = normalized_image_name(file_name)
        if only_file and norm != normalized_image_name(only_file):
            continue
        groups.append((norm, group.index.to_numpy(np.int64)))
        if frames > 0 and len(groups) >= frames:
            break
    return groups


def main() -> None:
    args = parse_args()
    input_path = repo_path(args.input)
    data_dir = repo_path(args.data_dir)
    weights = repo_path(args.weights)
    output_path = repo_path(args.output)
    metrics_output = repo_path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".metrics.json")

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    setup = transfer_setup(data_dir, device, args.calibration_scene, args.glassplate_to_volume_um, 0)
    model, norm, calibration_scale, ckpt = load_model(weights, device)
    model.eval()
    model, model_backend_info, model_compile_sec = optimize_slice_model_backend(
        model,
        backend=args.model_backend,
        device=device,
        batch_size=int(args.batch_size),
        crop_size=int(args.crop_size),
    )
    roi_cropper = CudaRoiAbs2Cropper.build(device) if device.type == "cuda" else None

    df = pd.read_csv(input_path, nrows=args.row_limit if args.row_limit > 0 else None)
    required = {"frame", "file", "track_id", "seg_xc", "seg_yc", "slice", "depth_um", "diameter_um"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{input_path} is missing required columns: {missing}")
    df = df.reset_index(drop=True)
    df["file"] = [normalized_image_name(v) for v in df["file"]]
    df["slice"] = df["slice"].round().astype(np.int32).clip(1, setup.slices)
    groups = frame_groups(df, args.frames, args.only_file)
    if not groups:
        raise RuntimeError("No frames selected")

    pred = np.full(len(df), np.nan, dtype=np.float32)
    sigma = np.full(len(df), np.nan, dtype=np.float32)
    pred_log = np.full(len(df), np.nan, dtype=np.float32)
    sigma_log = np.full(len(df), np.nan, dtype=np.float32)

    padlen = setup.datlen * 2
    offset = setup.datlen // 2
    holo1dir = data_dir / "01_bkgrem" / f"C002H001S{args.scenenum:04d}"
    holo2dir = data_dir / "01_bkgrem" / f"C001H001S{args.scenenum:04d}"
    holo_times: list[float] = []
    infer_times: list[float] = []
    unique_slice_count = 0
    total_crops = 0
    started = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "start",
                "rows": int(sum(len(idx) for _, idx in groups)),
                "frames": len(groups),
                "device": str(device),
                "weights": str(weights),
                "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "model_backend": str(model_backend_info["model_backend"]),
                "inference_precision": str(model_backend_info["inference_precision"]),
                "model_compile_sec": round(float(model_compile_sec), 4),
                "input": str(input_path),
                "data_dir": str(data_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with torch.inference_mode():
        for frame_i, (file_name, indices) in enumerate(groups, start=1):
            rows = df.loc[indices]
            h0 = time.perf_counter()
            d_holo = compute_holo(file_name, holo1dir, holo2dir, setup, device, args.priter, args.fft_backend)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            holo_sec = time.perf_counter() - h0
            holo_times.append(holo_sec)

            padded = torch.empty((padlen, padlen), device=device, dtype=torch.complex64)
            padded.fill_(d_holo.mean())
            padded[offset : offset + setup.datlen, offset : offset + setup.datlen] = d_holo
            freq = fft2(padded, args.fft_backend) * setup.d_tf_unshifted
            current_slice = 1
            step_cache: dict[int, torch.Tensor] = {}

            t0 = time.perf_counter()
            for slice_value in sorted(rows["slice"].unique().tolist()):
                slice_int = int(slice_value)
                gap = slice_int - current_slice
                if gap > 0:
                    step_transfer = step_cache.get(gap)
                    if step_transfer is None:
                        step_transfer = setup.d_slice_unshifted ** gap
                        step_cache[gap] = step_transfer
                    freq = freq * step_transfer
                    current_slice = slice_int

                local = rows.index[rows["slice"].to_numpy() == slice_int].to_numpy(np.int64)
                sub = df.loc[local]
                field = ifft2(freq, args.fft_backend)
                if not field.is_contiguous():
                    field = field.contiguous()
                if roi_cropper is not None:
                    x0_t, y0_t = crop_start_xy_tensors(
                        sub["seg_xc"].to_numpy(np.float64),
                        sub["seg_yc"].to_numpy(np.float64),
                        args.crop_size,
                        device,
                    )
                    crops = torch.empty((len(local), args.crop_size, args.crop_size), device=device, dtype=torch.float32)
                    roi_cropper.crop_abs2_meanpad(field, x0_t, y0_t, crops, setup.datlen, offset)
                else:
                    x0 = np.rint(sub["seg_xc"].to_numpy(np.float64) - args.crop_size / 2).astype(np.int32)
                    y0 = np.rint(sub["seg_yc"].to_numpy(np.float64) - args.crop_size / 2).astype(np.int32)
                    crops = crop_with_mean_padding_cpu(field, x0, y0, args.crop_size, setup.datlen, offset)

                for start in range(0, len(local), args.batch_size):
                    end = min(start + args.batch_size, len(local))
                    p_um, s_um, p_log, s_log = predict_diameter_batch(model, crops[start:end], norm, calibration_scale)
                    pred[local[start:end]] = p_um.astype(np.float32)
                    sigma[local[start:end]] = s_um.astype(np.float32)
                    pred_log[local[start:end]] = p_log.astype(np.float32)
                    sigma_log[local[start:end]] = s_log.astype(np.float32)
                unique_slice_count += 1
                total_crops += len(local)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_times.append(time.perf_counter() - t0)

    out = df.copy()
    out["slice_diam_pred_um"] = pred
    out["slice_diam_sigma_um"] = sigma
    out["slice_diam_pred_log"] = pred_log
    out["slice_diam_sigma_log"] = sigma_log
    out["slice_diam_model"] = str(weights)
    out["slice_diam_input_depth"] = "depth_model_estimate.slice"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - started
    metrics = {
        "method": "slice_diameter_model_at_depth_model_estimated_slice",
        "input": str(input_path),
        "output": str(output_path),
        "data_dir": str(data_dir),
        "weights": str(weights),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "device": str(device),
        "model_backend": str(model_backend_info["model_backend"]),
        "inference_precision": str(model_backend_info["inference_precision"]),
        "model_compile_sec": float(model_compile_sec),
        "trt_dynamic_max_batch": model_backend_info["trt_dynamic_max_batch"],
        "rows": int(len(out)),
        "processed_rows": int(total_crops),
        "frames": int(len(groups)),
        "crop_size": int(args.crop_size),
        "priter": int(args.priter),
        "z0_um": float(setup.z0_um),
        "dz_um": float(setup.dz_um),
        "unique_frame_slices": int(unique_slice_count),
        "elapsed_sec": float(elapsed),
        "mean_holo_sec_per_frame": float(np.mean(holo_times)) if holo_times else None,
        "mean_slice_infer_sec_per_frame": float(np.mean(infer_times)) if infer_times else None,
        "crops_per_sec_including_holo": float(total_crops / elapsed) if elapsed > 0 else None,
        "model_norm": norm,
        "calibration_scale": float(calibration_scale),
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
