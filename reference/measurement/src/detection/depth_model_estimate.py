#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.cuda_roi_kernels import CudaRoiAbs2Cropper  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "models/depth/depth_compare_best.pt"


INPUT_CHANNELS = {
    "raw": 1,
    "raw_mask": 2,
    "raw_diam": 2,
    "raw_diam_scalar": 1,
    "raw_mask_diam": 3,
    "raw_mask_diam_scalar": 2,
}
FULL_DIAM_CHANNEL_MODES = {"raw_diam", "raw_mask_diam"}
SCALAR_DIAM_MODES = {"raw_diam_scalar", "raw_mask_diam_scalar"}


@dataclass(frozen=True)
class TransferSetup:
    variables: dict[str, float]
    coeffs: np.ndarray
    dz_um: float
    datlen: int
    slices: int
    z0_um: float
    d_pr: torch.Tensor
    d_pr_inv: torch.Tensor
    d_tf_unshifted: torch.Tensor
    d_slice_unshifted: torch.Tensor


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocusScoreNet(nn.Module):
    def __init__(
        self,
        width: int = 24,
        in_channels: int = 1,
        cond_dim: int = 0,
        arch: str = "default",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        if arch == "default":
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
                DSConv(width, width * 2, stride=2),
                DSConv(width * 2, width * 2, stride=1),
                DSConv(width * 2, width * 3, stride=2),
                DSConv(width * 3, width * 4, stride=2),
                DSConv(width * 4, width * 5, stride=2),
                DSConv(width * 5, width * 6, stride=2),
            )
        elif arch == "faststem":
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
                DSConv(width, width * 2, stride=1),
                DSConv(width * 2, width * 3, stride=2),
                DSConv(width * 3, width * 4, stride=2),
                DSConv(width * 4, width * 5, stride=2),
                DSConv(width * 5, width * 6, stride=2),
            )
        else:
            raise ValueError(f"Unknown arch={arch}. Use default or faststem.")
        self.head = nn.Sequential(
            nn.Linear(width * 6 + cond_dim, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.08),
            nn.Linear(width * 4, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x).mean(dim=(2, 3))
        if self.cond_dim:
            if cond is None:
                raise RuntimeError("Condition tensor is required for this model")
            feat = torch.cat([feat, cond.to(feat.dtype)], dim=1)
        return self.head(feat).squeeze(1)


def input_channels(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return INPUT_CHANNELS[input_mode]


def condition_dim(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return 1 if input_mode in SCALAR_DIAM_MODES else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate particle depth with a learned focus-score model using CUDA/PyTorch angular-spectrum propagation."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "260106")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--input", type=Path, default=None, help="Default: <run-dir>/tracks_with_diameter.csv")
    p.add_argument("--output", type=Path, default=None, help="Default: <run-dir>/tracks_with_diam_depth.csv")
    p.add_argument("--metrics-output", type=Path, default=None, help="Default: <run-dir>/model_depth_metrics.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scenenum", type=int, default=11)
    p.add_argument("--glassplate-to-volume-um", type=float, default=10000.0)
    p.add_argument("--calibration-scene", default="C000H001S0010")
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--priter", type=int, default=3)
    p.add_argument("--frames", type=int, default=0, help="0 means all frames.")
    p.add_argument("--row-limit", type=int, default=0, help="0 means all rows.")
    p.add_argument("--only-file", default="")
    p.add_argument("--slices", type=int, default=0, help="0 means variables.yaml slices.")
    p.add_argument("--slice-start", type=int, default=1)
    p.add_argument("--slice-end", type=int, default=0, help="0 means final slice.")
    p.add_argument("--slice-step", type=int, default=1)
    p.add_argument("--slice-block", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--amp", action="store_true", help="Deprecated for TensorRT backend; kept for torch fallback.")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile-model", action="store_true", help="Only used with --model-backend torch.")
    p.add_argument("--model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--fft-backend", choices=["torch"], default="torch")
    p.add_argument("--mask-radius-diam-scale", type=float, default=None)
    p.add_argument("--mask-min-radius-px", type=float, default=None)
    p.add_argument("--mask-softness-px", type=float, default=None)
    p.add_argument("--flush-every-frames", type=int, default=0, help="Write partial CSV periodically if >0.")
    return p.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_simple_yaml(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return out


def transfer_sqrt_arr(datlen: int, wavlen: float, dx: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(datlen, device=device, dtype=torch.float32) - (datlen / 2)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    scale = float(wavlen) / float(datlen) / float(dx)
    return 1.0 - (xx * scale) ** 2 - (yy * scale) ** 2


def transfer(z_um: float, datlen: int, wavlen: float, sqrt_part: torch.Tensor) -> torch.Tensor:
    phase = (2.0 * math.pi * float(z_um) / float(wavlen)) * torch.sqrt(sqrt_part)
    return torch.exp(1j * phase).to(torch.complex64)


def transfer_setup(
    data_dir: Path,
    device: torch.device,
    calibration_scene: str,
    glassplate_to_volume_um: float,
    requested_slices: int,
) -> TransferSetup:
    variables = read_simple_yaml(data_dir / "variables.yaml")
    coeffs_raw = np.loadtxt(data_dir / "02_calibration" / calibration_scene / "coeffs.txt", dtype=np.float64)
    focal_raw = np.loadtxt(data_dir / "02_calibration" / calibration_scene / "focal_adjustment.txt", dtype=np.float64)
    coeffs = coeffs_raw[:, 0] if coeffs_raw.ndim == 2 else coeffs_raw
    focaldists = focal_raw[:, 0] if focal_raw.ndim == 2 else focal_raw
    wavlen = float(variables["lambda"])
    dx = float(variables["dx"])
    dz = float(variables["dz"])
    datlen = int(variables["datlen"])
    yaml_slices = int(variables["slices"])
    slices = min(requested_slices, yaml_slices) if requested_slices > 0 else yaml_slices
    z0 = float(focaldists[0]) + float(glassplate_to_volume_um)
    prz = float(focaldists[1])
    sqrt_1x = transfer_sqrt_arr(datlen, wavlen, dx, device)
    sqrt_2x = transfer_sqrt_arr(datlen * 2, wavlen, dx, device)
    return TransferSetup(
        variables=variables,
        coeffs=coeffs,
        dz_um=dz,
        datlen=datlen,
        slices=slices,
        z0_um=z0,
        d_pr=transfer(prz, datlen, wavlen, sqrt_1x),
        d_pr_inv=transfer(-prz, datlen, wavlen, sqrt_1x),
        d_tf_unshifted=torch.fft.ifftshift(transfer(-z0, datlen * 2, wavlen, sqrt_2x)),
        d_slice_unshifted=torch.fft.ifftshift(transfer(-dz, datlen * 2, wavlen, sqrt_2x)),
    )


def load_gray2float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


_DISTORTION_MAP_CACHE: dict[tuple[int, bytes], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def distortion_maps(n: int, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff_key = np.ascontiguousarray(coeffs, dtype=np.float64).tobytes()
    cache_key = (n, coeff_key)
    cached = _DISTORTION_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    i, j = np.indices((n, n), dtype=np.float64)
    i += 1.0
    j += 1.0
    ref_x = np.rint(
        coeffs[0]
        + coeffs[1] * j
        + coeffs[2] * i
        + coeffs[3] * j * j
        + coeffs[4] * i * j
        + coeffs[5] * i * i
    ).astype(np.int64)
    ref_y = np.rint(
        coeffs[6]
        + coeffs[7] * j
        + coeffs[8] * i
        + coeffs[9] * j * j
        + coeffs[10] * i * j
        + coeffs[11] * i * i
    ).astype(np.int64)
    valid = (ref_x >= 1) & (ref_x <= n) & (ref_y >= 1) & (ref_y <= n)
    maps = (ref_y - 1, ref_x - 1, valid)
    _DISTORTION_MAP_CACHE[cache_key] = maps
    return maps


def quadratic_distortion_correction(img: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    n = img.shape[0]
    ref_y, ref_x, valid = distortion_maps(n, coeffs)
    out = np.full((n, n), float(img.mean()), dtype=np.float32)
    out[valid] = img[ref_y[valid], ref_x[valid]]
    return out


def normalized_image_name(file_value: Any, imgext: str = "png") -> str:
    stem = Path(str(file_value)).stem
    return f"{stem}.{imgext}"


def fft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.fft2(x)


def ifft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.ifft2(x)


def phase_retrieval_holo(
    holo1: torch.Tensor,
    holo2: torch.Tensor,
    d_pr: torch.Tensor,
    d_pr_inv: torch.Tensor,
    priter: int,
    fft_backend: str,
) -> torch.Tensor:
    sqrt_i1 = torch.sqrt(holo1)
    sqrt_i2 = torch.sqrt(holo2)
    light1 = sqrt_i1.to(torch.complex64)
    d_pr_unshifted = torch.fft.ifftshift(d_pr)
    d_pr_inv_unshifted = torch.fft.ifftshift(d_pr_inv)
    for _ in range(priter):
        light2 = ifft2(fft2(light1, fft_backend) * d_pr_unshifted, fft_backend)
        light2 = sqrt_i2 * torch.exp(1j * torch.angle(light2))
        light1 = ifft2(fft2(light2, fft_backend) * d_pr_inv_unshifted, fft_backend)
        light1 = sqrt_i1 * torch.exp(1j * torch.angle(light1))
    return light1


def compute_holo(
    file_name: str,
    holo1dir: Path,
    holo2dir: Path,
    setup: TransferSetup,
    device: torch.device,
    priter: int,
    fft_backend: str,
) -> torch.Tensor:
    img1_np = load_gray2float(holo1dir / file_name)
    img2_np = quadratic_distortion_correction(load_gray2float(holo2dir / file_name), setup.coeffs)
    img1 = torch.from_numpy(img1_np).to(device=device, dtype=torch.float32)
    img2 = torch.from_numpy(img2_np).to(device=device, dtype=torch.float32)
    return phase_retrieval_holo(img1, img2, setup.d_pr, setup.d_pr_inv, priter, fft_backend)


def diameter_condition_values(diameter_um: torch.Tensor) -> torch.Tensor:
    lo = math.log(25.0)
    hi = math.log(500.0)
    values = (torch.log(torch.clamp(diameter_um, min=1e-6)) - lo) / max(hi - lo, 1e-6)
    return torch.clamp(values * 2.0 - 1.0, -1.5, 1.5)


def add_condition_channels(
    raw: torch.Tensor,
    diameter_um: torch.Tensor,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
) -> torch.Tensor:
    if input_mode == "raw":
        return raw
    batch, _, crop_size, _ = raw.shape
    channels = [raw]
    if "mask" in input_mode:
        coords = torch.arange(crop_size, device=raw.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (crop_size - 1) / 2.0
        rr = torch.sqrt((xx - center).square() + (yy - center).square())[None, None, :, :]
        diameter_px = diameter_um.view(batch, 1, 1, 1) / 10.0
        radius = torch.maximum(
            torch.full_like(diameter_px, mask_min_radius_px),
            diameter_px * mask_radius_diam_scale,
        )
        mask = torch.sigmoid((radius - rr) / max(mask_softness_px, 1e-3))
        channels.append(mask)
    if input_mode in FULL_DIAM_CHANNEL_MODES:
        values = diameter_condition_values(diameter_um)
        channels.append(values.view(batch, 1, 1, 1).expand(batch, 1, crop_size, crop_size))
    return torch.cat(channels, dim=1)


def make_condition_tensor(diameter_um: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode not in SCALAR_DIAM_MODES:
        return torch.empty((diameter_um.numel(), 0), device=diameter_um.device, dtype=torch.float32)
    return diameter_condition_values(diameter_um).view(-1, 1)


def load_model(
    checkpoint: Path,
    device: torch.device,
    compile_model: bool,
    channels_last: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model_info = ckpt.get("model", {})
    input_mode = model_info.get("input_mode") or ckpt_args.get("input_mode", "raw")
    arch = model_info.get("arch", ckpt_args.get("arch", "default"))
    width = int(model_info.get("width", ckpt_args.get("width", 24)))
    in_ch = int(model_info.get("in_channels", input_channels(input_mode)))
    cond_dim = int(model_info.get("cond_dim", condition_dim(input_mode)))
    model = FocusScoreNet(width=width, in_channels=in_ch, cond_dim=cond_dim, arch=arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    if compile_model:
        model = torch.compile(model)
    mask_radius_diam_scale = float(ckpt_args.get("mask_radius_diam_scale", 2.0))
    mask_min_radius_px = float(ckpt_args.get("mask_min_radius_px", 8.0))
    mask_softness_px = float(ckpt_args.get("mask_softness_px", 3.0))
    return model, {
        "input_mode": input_mode,
        "arch": arch,
        "width": width,
        "in_channels": in_ch,
        "cond_dim": cond_dim,
        "mask_radius_diam_scale": mask_radius_diam_scale,
        "mask_min_radius_px": mask_min_radius_px,
        "mask_softness_px": mask_softness_px,
    }


class NoCondWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, None)


def optimize_model_backend(
    model: nn.Module,
    model_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float]:
    compile_start = time.perf_counter()
    backend = str(args.model_backend)
    cond_dim = int(model_info.get("cond_dim", 0))
    model_info["model_backend"] = backend
    model_info["inference_precision"] = "fp16" if backend == "tensorrt" else "fp32"
    model_info["trt_dynamic_max_batch"] = None
    model_info["trt_num_inputs"] = 2 if cond_dim > 0 else 1

    if backend == "torch":
        if args.compile_model:
            model = torch.compile(model)
            model_info["model_backend"] = "torch_compile"
        return model, time.perf_counter() - compile_start

    if backend != "tensorrt":
        raise RuntimeError(f"Unknown model backend: {backend}")
    if device.type != "cuda":
        raise RuntimeError("TensorRT backend requires CUDA. Use --model-backend torch on CPU.")

    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError("TensorRT backend requires torch-tensorrt. Install torch-tensorrt first.") from exc

    batch_size = int(args.batch_size)
    crop_size = int(args.crop_size)
    in_channels = int(model_info["in_channels"])
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, in_channels, crop_size, crop_size),
            opt_shape=(batch_size, in_channels, crop_size, crop_size),
            max_shape=(batch_size, in_channels, crop_size, crop_size),
            dtype=torch.float32,
        )
    ]
    if cond_dim > 0:
        trt_target: nn.Module = model
        inputs.append(
            torch_tensorrt.Input(
                min_shape=(1, cond_dim),
                opt_shape=(batch_size, cond_dim),
                max_shape=(batch_size, cond_dim),
                dtype=torch.float32,
            )
        )
        model_info["trt_num_inputs"] = 2
    else:
        trt_target = NoCondWrapper(model)
        model_info["trt_num_inputs"] = 1

    trt_model = torch_tensorrt.compile(
        trt_target,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
        truncate_double=True,
    )
    trt_model.eval()
    model_info["model_backend"] = "tensorrt"
    model_info["inference_precision"] = "fp16"
    model_info["trt_dynamic_max_batch"] = batch_size
    return trt_model, time.perf_counter() - compile_start


def crop_start_tensors(rows: pd.DataFrame, crop_size: int, datlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rows["seg_xc"].to_numpy(np.float64)
    y = rows["seg_yc"].to_numpy(np.float64)
    x0 = np.rint(x - crop_size / 2).astype(np.int64)
    y0 = np.rint(y - crop_size / 2).astype(np.int64)
    offsets = torch.arange(crop_size, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    x_idx = torch.from_numpy(x0).to(device=device, dtype=torch.long)[:, None, None] + xx[None, :, :]
    y_idx = torch.from_numpy(y0).to(device=device, dtype=torch.long)[:, None, None] + yy[None, :, :]
    valid = (x_idx >= 0) & (x_idx < datlen) & (y_idx >= 0) & (y_idx < datlen)
    x_idx = x_idx.clamp_(0, datlen - 1)
    y_idx = y_idx.clamp_(0, datlen - 1)
    return x_idx, y_idx, valid


def crop_start_xy_tensors(rows: pd.DataFrame, crop_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = rows["seg_xc"].to_numpy(np.float64)
    y = rows["seg_yc"].to_numpy(np.float64)
    x0 = np.rint(x - crop_size / 2).astype(np.int32)
    y0 = np.rint(y - crop_size / 2).astype(np.int32)
    return (
        torch.from_numpy(x0).to(device=device, dtype=torch.int32).contiguous(),
        torch.from_numpy(y0).to(device=device, dtype=torch.int32).contiguous(),
    )


def centered_crops_with_mean_padding(
    central_intensity: torch.Tensor,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    crops = central_intensity[y_idx, x_idx]
    masked = torch.where(valid, crops, torch.zeros((), device=crops.device, dtype=crops.dtype))
    counts = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    means = masked.sum(dim=(1, 2), keepdim=True) / counts
    return torch.where(valid, crops, means).clamp_(0.0, 1.0)


def run_model_batches(
    model: nn.Module,
    crops: torch.Tensor,
    diameters_um: torch.Tensor,
    model_info: dict[str, Any],
    batch_size: int,
    amp: bool,
    channels_last: bool,
) -> torch.Tensor:
    scores = torch.empty((crops.shape[0],), device=crops.device, dtype=torch.float32)
    input_mode = str(model_info["input_mode"])
    model_backend = str(model_info.get("model_backend", "torch"))
    trt_num_inputs = int(model_info.get("trt_num_inputs", 2))
    for start in range(0, crops.shape[0], batch_size):
        end = min(start + batch_size, crops.shape[0])
        raw = crops[start:end, None, :, :].to(dtype=torch.float32)
        raw = raw.sub(0.5).div(0.25)
        diam = diameters_um[start:end]
        tensor = add_condition_channels(
            raw,
            diam,
            input_mode,
            float(model_info["mask_radius_diam_scale"]),
            float(model_info["mask_min_radius_px"]),
            float(model_info["mask_softness_px"]),
        )
        cond = make_condition_tensor(diam, input_mode)
        if model_backend == "tensorrt":
            tensor = tensor.contiguous()
            if trt_num_inputs == 1:
                out = model(tensor)
            else:
                out = model(tensor, cond)
        else:
            if channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            if amp and crops.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(tensor, cond)
            else:
                out = model(tensor, cond)
        scores[start:end] = out.detach().float()
    return scores


def slice_values(setup: TransferSetup, args: argparse.Namespace) -> list[int]:
    if args.slice_step <= 0:
        raise ValueError("--slice-step must be positive")
    lo = max(1, min(setup.slices, int(args.slice_start)))
    hi_req = int(args.slice_end) if int(args.slice_end) > 0 else setup.slices
    hi = max(lo, min(setup.slices, hi_req))
    values = list(range(lo, hi + 1, int(args.slice_step)))
    if values[-1] != hi:
        values.append(hi)
    return values


def contiguous_frame_groups(df: pd.DataFrame, max_frames: int, only_file: str) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    for file_name, group in df.groupby("file", sort=True):
        norm = normalized_image_name(file_name)
        if only_file and norm != normalized_image_name(only_file):
            continue
        groups.append((norm, group.index.to_numpy(np.int64)))
        if max_frames > 0 and len(groups) >= max_frames:
            break
    return groups


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def estimate_depth(args: argparse.Namespace) -> None:
    run_dir = resolve_path(args.run_dir)
    data_dir = resolve_path(args.data_dir)
    checkpoint = resolve_path(args.checkpoint)
    input_path = resolve_path(args.input) if args.input else run_dir / "tracks_with_diameter.csv"
    output_path = resolve_path(args.output) if args.output else run_dir / "tracks_with_diam_depth.csv"
    metrics_output = resolve_path(args.metrics_output) if args.metrics_output else run_dir / "model_depth_metrics.json"

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    setup = transfer_setup(
        data_dir,
        device,
        args.calibration_scene,
        args.glassplate_to_volume_um,
        args.slices,
    )
    model, model_info = load_model(checkpoint, device, False, args.channels_last)
    if args.mask_radius_diam_scale is not None:
        model_info["mask_radius_diam_scale"] = float(args.mask_radius_diam_scale)
    if args.mask_min_radius_px is not None:
        model_info["mask_min_radius_px"] = float(args.mask_min_radius_px)
    if args.mask_softness_px is not None:
        model_info["mask_softness_px"] = float(args.mask_softness_px)
    model, model_compile_sec = optimize_model_backend(model, model_info, args, device)

    df = pd.read_csv(input_path, nrows=args.row_limit if args.row_limit > 0 else None)
    required = {"frame", "file", "seg_xc", "seg_yc", "diameter_um"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{input_path} is missing required columns: {missing}")
    df = df.reset_index(drop=True)
    df["file"] = [normalized_image_name(v) for v in df["file"]]

    groups = contiguous_frame_groups(df, args.frames, args.only_file)
    if not groups:
        raise RuntimeError("No frames to process.")

    slices = slice_values(setup, args)
    padlen = setup.datlen * 2
    offset = setup.datlen // 2
    holo1dir = data_dir / "01_bkgrem" / f"C002H001S{args.scenenum:04d}"
    holo2dir = data_dir / "01_bkgrem" / f"C001H001S{args.scenenum:04d}"
    use_cuda_roi_crop = device.type == "cuda"
    roi_kernel_compile_sec = 0.0
    roi_cropper: CudaRoiAbs2Cropper | None = None
    if use_cuda_roi_crop:
        k0 = time.perf_counter()
        roi_cropper = CudaRoiAbs2Cropper.build(device)
        roi_kernel_compile_sec = time.perf_counter() - k0

    best_slice = np.zeros((len(df),), dtype=np.int32)
    best_score = np.full((len(df),), -np.inf, dtype=np.float32)
    holo_times: list[float] = []
    depth_times: list[float] = []
    crops_scored = 0
    t0 = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "start",
                "frames": len(groups),
                "rows": int(sum(len(indices) for _, indices in groups)),
                "slices": len(slices),
                "device": str(device),
                "checkpoint": str(checkpoint),
                "input_mode": model_info["input_mode"],
                "slice_block": int(args.slice_block),
                "batch_size": int(args.batch_size),
                "amp": bool(args.amp),
                "model_backend": str(model_info.get("model_backend", args.model_backend)),
                "inference_precision": str(model_info.get("inference_precision", "fp32")),
                "roi_crop_backend": "cuda" if use_cuda_roi_crop else "torch",
                "model_compile_sec": round(float(model_compile_sec), 4),
                "roi_kernel_compile_sec": round(float(roi_kernel_compile_sec), 4),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with torch.inference_mode():
        for frame_i, (file_name, indices) in enumerate(groups, start=1):
            rows = df.loc[indices].reset_index(drop=True)
            nrows = len(rows)
            diam_frame = torch.from_numpy(rows["diameter_um"].to_numpy(np.float32)).to(device=device)
            if use_cuda_roi_crop:
                x0, y0 = crop_start_xy_tensors(rows, args.crop_size, device)
                x_idx = y_idx = valid = None
            else:
                x_idx, y_idx, valid = crop_start_tensors(rows, args.crop_size, setup.datlen, device)
                x0 = y0 = None

            h0 = time.perf_counter()
            d_holo = compute_holo(file_name, holo1dir, holo2dir, setup, device, args.priter, args.fft_backend)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            holo_times.append(time.perf_counter() - h0)

            padded = torch.empty((padlen, padlen), device=device, dtype=torch.complex64)
            padded.fill_(d_holo.mean())
            padded[offset : offset + setup.datlen, offset : offset + setup.datlen] = d_holo
            freq = fft2(padded, args.fft_backend) * setup.d_tf_unshifted
            if slices[0] > 1:
                freq = freq * (setup.d_slice_unshifted ** int(slices[0] - 1))
            current_slice = int(slices[0])

            frame_best_slice = torch.full((nrows,), int(slices[0]), device=device, dtype=torch.int32)
            frame_best_score = torch.full((nrows,), -torch.inf, device=device, dtype=torch.float32)
            step_cache: dict[int, torch.Tensor] = {}

            d0 = time.perf_counter()
            block_crops: list[torch.Tensor] = []
            block_slices: list[int] = []
            for zi in slices:
                gap = int(zi - current_slice)
                if gap > 0:
                    step_transfer = step_cache.get(gap)
                    if step_transfer is None:
                        step_transfer = setup.d_slice_unshifted ** gap
                        step_cache[gap] = step_transfer
                    freq = freq * step_transfer
                    current_slice = int(zi)

                field = ifft2(freq, args.fft_backend)
                if use_cuda_roi_crop:
                    if roi_cropper is None or x0 is None or y0 is None:
                        raise RuntimeError("CUDA ROI cropper was not initialized")
                    if not field.is_contiguous():
                        field = field.contiguous()
                    crop_out = torch.empty(
                        (nrows, args.crop_size, args.crop_size),
                        device=device,
                        dtype=torch.float32,
                    )
                    roi_cropper.crop_abs2_meanpad(field, x0, y0, crop_out, setup.datlen, offset)
                    block_crops.append(crop_out)
                else:
                    if x_idx is None or y_idx is None or valid is None:
                        raise RuntimeError("Torch ROI crop tensors were not initialized")
                    intensity = (field.real.square() + field.imag.square()).clamp_(0.0, 1.0)
                    central = intensity[offset : offset + setup.datlen, offset : offset + setup.datlen]
                    block_crops.append(centered_crops_with_mean_padding(central, x_idx, y_idx, valid))
                block_slices.append(int(zi))

                if len(block_crops) >= args.slice_block:
                    crops = torch.cat(block_crops, dim=0)
                    diam = diam_frame.repeat(len(block_crops))
                    scores = run_model_batches(
                        model, crops, diam, model_info, args.batch_size, args.amp, args.channels_last
                    ).view(len(block_crops), nrows)
                    values, arg = scores.max(dim=0)
                    improve = values > frame_best_score
                    if improve.any():
                        slice_tensor = torch.tensor(block_slices, device=device, dtype=torch.int32)
                        frame_best_score[improve] = values[improve]
                        frame_best_slice[improve] = slice_tensor[arg[improve]]
                    crops_scored += int(crops.shape[0])
                    block_crops.clear()
                    block_slices.clear()

            if block_crops:
                crops = torch.cat(block_crops, dim=0)
                diam = diam_frame.repeat(len(block_crops))
                scores = run_model_batches(
                    model, crops, diam, model_info, args.batch_size, args.amp, args.channels_last
                ).view(len(block_crops), nrows)
                values, arg = scores.max(dim=0)
                improve = values > frame_best_score
                if improve.any():
                    slice_tensor = torch.tensor(block_slices, device=device, dtype=torch.int32)
                    frame_best_score[improve] = values[improve]
                    frame_best_slice[improve] = slice_tensor[arg[improve]]
                crops_scored += int(crops.shape[0])

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            depth_times.append(time.perf_counter() - d0)

            best_slice[indices] = frame_best_slice.cpu().numpy().astype(np.int32)
            best_score[indices] = frame_best_score.cpu().numpy().astype(np.float32)
            if args.flush_every_frames > 0 and frame_i % args.flush_every_frames == 0:
                partial = df.copy()
                partial["slice"] = best_slice
                partial["depth_um"] = setup.z0_um + (best_slice.astype(np.float64) - 1.0) * setup.dz_um
                partial["model_score"] = best_score
                partial["depth_method"] = "focus_score_model_argmax_partial"
                partial.to_csv(output_path, index=False)

    total_elapsed = time.perf_counter() - t0
    out = df.copy()
    out["slice"] = best_slice
    out["depth_um"] = setup.z0_um + (best_slice.astype(np.float64) - 1.0) * setup.dz_um
    out["model_score"] = best_score
    out["depth_method"] = "focus_score_model_argmax"
    if use_cuda_roi_crop:
        out["model_crop_method"] = "cuda_field_roi_abs2_centered_64px_valid_region_mean_padding"
    else:
        out["model_crop_method"] = "torch_full_intensity_centered_64px_valid_region_mean_padding"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    metrics = {
        "method": "focus_score_model_argmax_cuda_roi_abs2_tensorrt_fp16",
        "input": str(input_path),
        "output": str(output_path),
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "rows": int(len(out)),
        "processed_rows": int(sum(len(indices) for _, indices in groups)),
        "frames": int(len(groups)),
        "slices": int(len(slices)),
        "slice_start": int(slices[0]),
        "slice_end": int(slices[-1]),
        "slice_step": int(args.slice_step),
        "crop_size": int(args.crop_size),
        "crop_center": "seg_xc_seg_yc",
        "crop_padding": "valid_region_mean",
        "priter": int(args.priter),
        "z0_um": float(setup.z0_um),
        "dz_um": float(setup.dz_um),
        "batch_size": int(args.batch_size),
        "slice_block": int(args.slice_block),
        "amp": bool(args.amp),
        "channels_last": bool(args.channels_last),
        "compile_model": bool(args.compile_model),
        "roi_crop_backend": "cuda" if use_cuda_roi_crop else "torch",
        "model_backend": str(model_info.get("model_backend", args.model_backend)),
        "inference_precision": str(model_info.get("inference_precision", "fp32")),
        "model_compile_sec": float(model_compile_sec),
        "roi_kernel_compile_sec": float(roi_kernel_compile_sec),
        "fft_backend": str(args.fft_backend),
        "model": model_info,
        "crops_scored": int(crops_scored),
        "elapsed_sec": float(total_elapsed),
        "mean_holo_sec_per_frame": float(np.mean(holo_times)) if holo_times else None,
        "mean_depth_sec_per_frame": float(np.mean(depth_times)) if depth_times else None,
        "crops_per_sec": float(crops_scored / sum(depth_times)) if sum(depth_times) > 0 else None,
    }
    write_json(metrics_output, metrics)
    print(json.dumps({"event": "done", **metrics}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    estimate_depth(parse_args())


if __name__ == "__main__":
    main()
