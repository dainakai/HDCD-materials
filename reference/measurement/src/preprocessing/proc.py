#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.depth_model_estimate import (  # noqa: E402
    fft2,
    ifft2,
    load_gray2float,
    phase_retrieval_holo,
    quadratic_distortion_correction,
    read_simple_yaml,
    transfer,
    transfer_sqrt_arr,
)
from src.preprocessing.calibration import resolve_calibration  # noqa: E402


@dataclass(frozen=True)
class ProjectionSetup:
    variables: dict[str, float]
    coeffs: np.ndarray
    datlen: int
    padlen: int
    slices: int
    dz_um: float
    slice_step_mult: int
    effective_dz_um: float
    z0_um: float
    d_pr: torch.Tensor
    d_pr_inv: torch.Tensor
    d_tf_unshifted: torch.Tensor
    d_slice_unshifted: torch.Tensor


@dataclass(frozen=True)
class ImagePair:
    name: str
    camera1_path: Path
    camera2_path: Path


@dataclass(frozen=True)
class LoadedPair:
    name: str
    img1: np.ndarray
    img2: np.ndarray


def env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return int(value)
    return default


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    return int(value) if value else None


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def data_dir_from_env() -> Path:
    date = os.environ.get("DATE", "260106")
    date_dir = Path(os.environ.get("DATE_DIR", date)).expanduser()
    return date_dir.resolve() if date_dir.is_absolute() else (PROJECT_ROOT / date_dir).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate XY minimum intensity projection PNGs from two-camera background-removed holograms. "
            "This uses threaded image loading/saving and torch FFTs."
        )
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Default follows DATE_DIR/DATE.")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <data-dir>/<xyproj-dir>/C000H001Sxxxx")
    p.add_argument("--xyproj-dir", default=os.environ.get("XYPROJ_DIR", "05_xyproj"))
    p.add_argument("--scene-num", type=int, default=env_int(("SCENE_NUM", "SCENENUM"), 11))
    p.add_argument(
        "--bundle-num",
        type=int,
        default=None,
        help=(
            "Calibration bundle scene number. Default: BUNDLE_NUM/BUNDLENUM, "
            "or the largest valid 02_calibration scene lower than --scene-num. "
            "If none exists, processing stops before reconstruction."
        ),
    )
    p.add_argument("--start-index", type=int, default=env_optional_int("START_INDEX"))
    p.add_argument("--end-index", type=int, default=env_optional_int("END_INDEX"))
    p.add_argument("--limit", type=int, default=0, help="0 means all matched frames.")
    p.add_argument("--camera1-id", default=os.environ.get("CAMERA1_ID", "C002"))
    p.add_argument("--camera2-id", default=os.environ.get("CAMERA2_ID", "C001"))
    p.add_argument("--backrem", action=argparse.BooleanOptionalAction, default=env_bool("BACKREM", True))
    p.add_argument("--device", default=os.environ.get("TORCH_DEVICE", "cuda:0"))
    p.add_argument("--priter", type=int, default=env_int(("PRITER",), 12))
    p.add_argument("--slices", type=int, default=0, help="0 means variables.yaml slices.")
    p.add_argument("--padlen", type=int, default=env_int(("PADLEN",), 1536))
    p.add_argument("--slice-step-mult", type=int, default=env_int(("SLICE_STEP_MULT",), 2))
    p.add_argument("--glassplate-to-volume-um", type=float, default=10000.0)
    p.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) // 2)))
    p.add_argument("--prefetch", type=int, default=8)
    p.add_argument("--save-workers", type=int, default=2)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--summary", type=Path, default=None)
    return p.parse_args()


def scene_name(scene_num: int) -> str:
    return f"H001S{scene_num:04d}"


def infer_bundle_num(data_dir: Path, scene_num: int) -> int:
    explicit = os.environ.get("BUNDLE_NUM") or os.environ.get("BUNDLENUM")
    if explicit:
        return int(explicit)
    data_date = next((part for part in reversed(data_dir.parts) if len(part) == 6 and part.isdigit()), "")
    return resolve_calibration(
        data_dir / "02_calibration",
        scene_num,
        data_date=data_date,
        data_root=data_dir,
    ).scene_num


def image_index(path: Path) -> int:
    return int(path.stem)


def filter_index_range(paths: list[Path], start_index: int | None, end_index: int | None) -> list[Path]:
    out = []
    for path in paths:
        idx = image_index(path)
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx > end_index:
            continue
        out.append(path)
    return sorted(out, key=image_index)


def matched_pairs(
    data_dir: Path,
    camera1_id: str,
    camera2_id: str,
    scene: str,
    backrem: bool,
    start_index: int | None,
    end_index: int | None,
    limit: int,
) -> list[ImagePair]:
    imgdir = "01_bkgrem" if backrem else "00_img"
    dir1 = data_dir / imgdir / f"{camera1_id}{scene}"
    dir2 = data_dir / imgdir / f"{camera2_id}{scene}"
    if not dir1.is_dir():
        raise FileNotFoundError(dir1)
    if not dir2.is_dir():
        raise FileNotFoundError(dir2)

    files1 = filter_index_range(list(dir1.glob("*.png")), start_index, end_index)
    files2 = filter_index_range(list(dir2.glob("*.png")), start_index, end_index)
    by1 = {path.name: path for path in files1}
    by2 = {path.name: path for path in files2}
    matched_names = sorted(set(by1).intersection(by2), key=lambda name: int(Path(name).stem))
    if not matched_names:
        raise RuntimeError(f"No matched PNG names between {dir1} and {dir2}")
    if len(matched_names) != len(files1) or len(matched_names) != len(files2):
        print(
            json.dumps(
                {
                    "warning": "some camera images do not have matching names",
                    "matched": len(matched_names),
                    "camera1": len(files1),
                    "camera2": len(files2),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    names = matched_names[:limit] if limit > 0 else matched_names
    return [ImagePair(name=name, camera1_path=by1[name], camera2_path=by2[name]) for name in names]


def load_pair(pair: ImagePair, coeffs: np.ndarray) -> LoadedPair:
    img1 = load_gray2float(pair.camera1_path)
    img2 = quadratic_distortion_correction(load_gray2float(pair.camera2_path), coeffs)
    return LoadedPair(pair.name, img1, img2)


def iter_prefetched_pairs(
    pairs: list[ImagePair],
    coeffs: np.ndarray,
    workers: int,
    prefetch: int,
) -> Any:
    if workers <= 0:
        for pair in pairs:
            yield load_pair(pair, coeffs)
        return

    max_pending = max(1, prefetch)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[int, Future[LoadedPair]] = {}
        submit_idx = 0
        yield_idx = 0

        while yield_idx < len(pairs):
            while submit_idx < len(pairs) and len(pending) < max_pending:
                pending[submit_idx] = executor.submit(load_pair, pairs[submit_idx], coeffs)
                submit_idx += 1
            future = pending.pop(yield_idx)
            yield future.result()
            yield_idx += 1


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def start_save_worker(
    save_queue: queue.Queue,
    stop_token: object,
    errors: queue.Queue[BaseException],
    abort_event: threading.Event,
) -> threading.Thread:
    def run() -> None:
        while True:
            item = save_queue.get()
            try:
                if item is stop_token:
                    return
                if abort_event.is_set():
                    continue
                path, arr = item
                try:
                    save_png(path, arr)
                except BaseException as exc:
                    errors.put(exc)
                    abort_event.set()
            finally:
                save_queue.task_done()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def build_setup(
    data_dir: Path,
    bundle_num: int,
    device: torch.device,
    requested_slices: int,
    requested_padlen: int,
    slice_step_mult: int,
    glassplate_to_volume_um: float = 10000.0,
) -> ProjectionSetup:
    variables = read_simple_yaml(data_dir / "variables.yaml")
    calib_name = f"C000H001S{bundle_num:04d}"
    calib_dir = data_dir / "02_calibration" / calib_name
    coeffs_raw = np.loadtxt(calib_dir / "coeffs.txt", dtype=np.float64)
    focal_raw = np.loadtxt(calib_dir / "focal_adjustment.txt", dtype=np.float64)
    coeffs = coeffs_raw[:, 0] if coeffs_raw.ndim == 2 else coeffs_raw
    focaldists = focal_raw[:, 0] if focal_raw.ndim == 2 else focal_raw

    wavlen = float(variables["lambda"])
    dx = float(variables["dx"])
    dz = float(variables["dz"])
    datlen = int(variables["datlen"])
    padlen = int(requested_padlen) if int(requested_padlen) > 0 else datlen * 2
    if padlen < datlen:
        raise ValueError(f"padlen must be >= datlen: {padlen} < {datlen}")
    if (padlen - datlen) % 2 != 0:
        raise ValueError(f"padlen - datlen must be even for centered crop: {padlen} - {datlen}")
    yaml_slices = int(variables["slices"])
    base_slices = min(requested_slices, yaml_slices) if requested_slices > 0 else yaml_slices
    step_mult = max(1, int(slice_step_mult))
    slices = max(1, (base_slices + step_mult - 1) // step_mult)
    effective_dz = dz * step_mult
    z0 = float(focaldists[0]) + float(glassplate_to_volume_um)
    prz = float(focaldists[1])

    sqrt_1x = transfer_sqrt_arr(datlen, wavlen, dx, device)
    sqrt_pad = transfer_sqrt_arr(padlen, wavlen, dx, device)
    return ProjectionSetup(
        variables=variables,
        coeffs=coeffs,
        datlen=datlen,
        padlen=padlen,
        slices=slices,
        dz_um=dz,
        slice_step_mult=step_mult,
        effective_dz_um=effective_dz,
        z0_um=z0,
        d_pr=transfer(prz, datlen, wavlen, sqrt_1x),
        d_pr_inv=transfer(-prz, datlen, wavlen, sqrt_1x),
        d_tf_unshifted=torch.fft.ifftshift(transfer(-z0, padlen, wavlen, sqrt_pad)),
        d_slice_unshifted=torch.fft.ifftshift(transfer(-effective_dz, padlen, wavlen, sqrt_pad)),
    )


def mean_pad_complex(field: torch.Tensor, outlen: int) -> torch.Tensor:
    inlen = int(field.shape[0])
    out = torch.empty((outlen, outlen), device=field.device, dtype=torch.complex64)
    out.fill_(field.mean())
    offset = (outlen - inlen) // 2
    out[offset : offset + inlen, offset : offset + inlen] = field
    return out


def xy_min_projection(
    holo: torch.Tensor,
    setup: ProjectionSetup,
    priter: int,
    fft_backend: str = "torch",
) -> torch.Tensor:
    padlen = setup.padlen
    offset = (setup.padlen - setup.datlen) // 2
    padded = mean_pad_complex(holo, padlen)
    freq = fft2(padded, fft_backend) * setup.d_tf_unshifted
    xy_min = None
    for zi in range(setup.slices):
        if zi > 0:
            freq = freq * setup.d_slice_unshifted
        field = ifft2(freq, fft_backend)
        intensity = (field.real.square() + field.imag.square()).clamp_(0.0, 1.0)
        central = intensity[offset : offset + setup.datlen, offset : offset + setup.datlen]
        xy_min = central if xy_min is None else torch.minimum(xy_min, central)
    if xy_min is None:
        raise RuntimeError("No slices were reconstructed")
    return xy_min


def process(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = (args.data_dir.expanduser().resolve() if args.data_dir is not None else data_dir_from_env())
    bundle_num = int(args.bundle_num) if args.bundle_num is not None else infer_bundle_num(data_dir, args.scene_num)
    scene = scene_name(args.scene_num)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / args.xyproj_dir / f"C000{scene}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    setup = build_setup(
        data_dir,
        bundle_num,
        device,
        args.slices,
        args.padlen,
        args.slice_step_mult,
        args.glassplate_to_volume_um,
    )
    pairs = matched_pairs(
        data_dir,
        args.camera1_id,
        args.camera2_id,
        scene,
        args.backrem,
        args.start_index,
        args.end_index,
        args.limit,
    )

    save_queue: queue.Queue = queue.Queue(maxsize=max(1, args.save_workers * 2))
    stop_token = object()
    save_errors: queue.Queue[BaseException] = queue.Queue()
    save_abort = threading.Event()
    save_threads = [
        start_save_worker(save_queue, stop_token, save_errors, save_abort)
        for _ in range(max(1, args.save_workers))
    ]

    def raise_save_error() -> None:
        try:
            error = save_errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("PNG save worker failed") from error

    timings: list[float] = []
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            iterator = iter_prefetched_pairs(pairs, setup.coeffs, args.workers, args.prefetch)
            for loaded in tqdm(iterator, total=len(pairs), desc=f"C000{scene}", unit="img"):
                raise_save_error()
                out_path = output_dir / loaded.name
                if out_path.exists() and not args.overwrite:
                    continue
                t0 = time.perf_counter()
                img1 = torch.from_numpy(loaded.img1).to(device=device, dtype=torch.float32)
                img2 = torch.from_numpy(loaded.img2).to(device=device, dtype=torch.float32)
                holo = phase_retrieval_holo(img1, img2, setup.d_pr, setup.d_pr_inv, args.priter, "torch")
                xyproj = xy_min_projection(holo, setup, args.priter).detach().cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - t0)
                save_queue.put((out_path, xyproj))
                raise_save_error()
    finally:
        save_queue.join()
        for _ in save_threads:
            save_queue.put(stop_token)
        for thread in save_threads:
            thread.join()
    raise_save_error()

    wall_sec = time.perf_counter() - start
    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "scene": f"C000{scene}",
        "bundle": f"C000H001S{bundle_num:04d}",
        "images": len(pairs),
        "datlen": setup.datlen,
        "padlen": setup.padlen,
        "slices": setup.slices,
        "base_dz_um": setup.dz_um,
        "slice_step_mult": setup.slice_step_mult,
        "effective_dz_um": setup.effective_dz_um,
        "glassplate_to_volume_um": float(args.glassplate_to_volume_um),
        "z0_um": setup.z0_um,
        "approx_z_span_um": setup.effective_dz_um * max(0, setup.slices - 1),
        "priter": args.priter,
        "device": str(device),
        "workers": args.workers,
        "save_workers": args.save_workers,
        "wall_sec": wall_sec,
        "mean_compute_sec_per_image": float(np.mean(timings)) if timings else 0.0,
    }
    summary_path = args.summary.expanduser().resolve() if args.summary is not None else output_dir / "proc_py_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = process(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
