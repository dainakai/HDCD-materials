#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


@dataclass(frozen=True)
class CropRow:
    row_id: str
    split: str
    image_path: str
    file: str
    frame: int
    particle_id: int
    diameter_um: float
    diameter_px: float
    z_um: float
    z_slice: int


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_rows(path: Path) -> list[CropRow]:
    path = repo_path(path)
    rows: list[CropRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(
                CropRow(
                    row_id=raw["row_id"],
                    split=raw["split"],
                    image_path=raw["image_path"],
                    file=raw["file"],
                    frame=int(raw["frame"]),
                    particle_id=int(raw["particle_id"]),
                    diameter_um=float(raw["diameter_um"]),
                    diameter_px=float(raw["diameter_px"]),
                    z_um=float(raw["z_um"]),
                    z_slice=int(raw["z_slice"]),
                )
            )
    return rows


class SliceCropDataset(Dataset):
    def __init__(
        self,
        rows: list[CropRow],
        *,
        training: bool,
        log_diam_mean: float,
        log_diam_std: float,
    ) -> None:
        self.rows = rows
        self.training = training
        self.log_diam_mean = log_diam_mean
        self.log_diam_std = log_diam_std
        self.cache = {row.image_path: read_gray(REPO_ROOT / row.image_path) for row in rows}

    def __len__(self) -> int:
        return len(self.rows)

    def image_tensor(self, row: CropRow) -> torch.Tensor:
        crop = self.cache[row.image_path].astype(np.float32) / 255.0
        if self.training:
            pad = 4
            padded = np.pad(crop, ((pad, pad), (pad, pad)), mode="edge")
            ox = random.randint(0, 2 * pad)
            oy = random.randint(0, 2 * pad)
            crop = padded[oy : oy + crop.shape[0], ox : ox + crop.shape[1]]
            if random.random() < 0.5:
                crop = np.flip(crop, axis=1)
            if random.random() < 0.5:
                crop = np.flip(crop, axis=0)
            if random.random() < 0.5:
                crop = np.rot90(crop, k=random.randrange(4))
            scale = random.uniform(0.90, 1.12)
            offset = random.uniform(-0.04, 0.04)
            gamma = random.uniform(0.88, 1.14)
            crop = np.clip(crop * scale + offset, 0.0, 1.0)
            crop = np.clip(crop, 1e-4, 1.0) ** gamma
            if random.random() < 0.20:
                crop = cv2.GaussianBlur(crop, (3, 3), sigmaX=random.uniform(0.2, 0.65))
            noise = np.random.normal(0.0, random.uniform(0.002, 0.012), crop.shape).astype(np.float32)
            crop = np.clip(crop + noise, 0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(crop[None, :, :]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        log_diam = math.log(row.diameter_um)
        target = (log_diam - self.log_diam_mean) / self.log_diam_std
        return {
            "image": self.image_tensor(row),
            "target": torch.tensor(target, dtype=torch.float32),
            "diameter_um": torch.tensor(row.diameter_um, dtype=torch.float32),
        }


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.silu(y + self.skip(x), inplace=True)


class DiameterNet(nn.Module):
    def __init__(self, min_sigma_norm: float = 0.025, dropout: float = 0.10) -> None:
        super().__init__()
        self.min_sigma_norm = min_sigma_norm
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(32, 32, stride=1, dropout=0.02),
            ResBlock(32, 48, stride=2, dropout=0.02),
            ResBlock(48, 64, stride=1, dropout=0.03),
            ResBlock(64, 96, stride=2, dropout=0.04),
            ResBlock(96, 128, stride=2, dropout=0.05),
            ResBlock(128, 160, stride=2, dropout=0.06),
            ResBlock(160, 192, stride=2, dropout=0.06),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.SiLU(inplace=True),
        )
        self.mu_head = nn.Linear(96, 1)
        self.sigma_head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.blocks(self.stem(x)).mean(dim=(2, 3))
        h = self.head(feat)
        mu = self.mu_head(h).squeeze(1)
        sigma = F.softplus(self.sigma_head(h).squeeze(1)) + self.min_sigma_norm
        return mu, sigma


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z = (target - mu) / sigma
    return (torch.log(sigma) + 0.5 * z.pow(2)).mean()


def make_loader(
    rows: list[CropRow],
    *,
    batch_size: int,
    training: bool,
    log_diam_mean: float,
    log_diam_std: float,
    num_workers: int,
) -> DataLoader:
    ds = SliceCropDataset(rows, training=training, log_diam_mean=log_diam_mean, log_diam_std=log_diam_std)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(rows) >= batch_size,
    )


def train_epoch(
    model: DiameterNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "nll": 0.0, "huber": 0.0, "mae_norm": 0.0, "items": 0.0}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            mu, sigma = model(x)
            nll = gaussian_nll(mu, sigma, target)
            huber = F.smooth_l1_loss(mu, target, beta=0.10)
            loss = nll + 0.04 * huber
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        n = x.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["nll"] += float(nll.detach().cpu()) * n
        totals["huber"] += float(huber.detach().cpu()) * n
        totals["mae_norm"] += float(torch.mean(torch.abs(mu.detach() - target)).cpu()) * n
        totals["items"] += n
    denom = max(1.0, totals["items"])
    return {k: v / denom for k, v in totals.items() if k != "items"}


def save_checkpoint(
    path: Path,
    model: DiameterNet,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    norm: dict[str, float],
    calibration_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "epoch": epoch,
            "metrics": metrics,
            "norm": norm,
            "calibration_scale": calibration_scale,
        },
        path,
    )


def load_model(weights: Path, device: torch.device) -> tuple[DiameterNet, dict[str, float], float, dict[str, Any]]:
    ckpt = torch.load(repo_path(weights), map_location=device, weights_only=False)
    model = DiameterNet()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt["norm"], float(ckpt.get("calibration_scale", 1.0)), ckpt


@torch.no_grad()
def predict_rows(
    model: DiameterNet,
    rows: list[CropRow],
    *,
    device: torch.device,
    batch_size: int,
    norm: dict[str, float],
    calibration_scale: float,
) -> list[dict[str, Any]]:
    ds = SliceCropDataset(
        rows,
        training=False,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    for batch in loader:
        x = batch["image"].to(device)
        mu_norm, sigma_norm = model(x)
        mu_log = mu_norm.cpu().numpy() * norm["log_diam_std"] + norm["log_diam_mean"]
        sigma_log = sigma_norm.cpu().numpy() * norm["log_diam_std"] * calibration_scale
        for ml, sl in zip(mu_log.tolist(), sigma_log.tolist(), strict=True):
            row = rows[offset]
            pred_um = math.exp(ml)
            out.append(
                {
                    "row_id": row.row_id,
                    "split": row.split,
                    "image_path": row.image_path,
                    "file": row.file,
                    "frame": row.frame,
                    "particle_id": row.particle_id,
                    "z_um": row.z_um,
                    "z_slice": row.z_slice,
                    "true_diameter_um": row.diameter_um,
                    "true_diameter_px": row.diameter_px,
                    "pred_log_diameter": ml,
                    "pred_sigma_log": sl,
                    "pred_diameter_um": pred_um,
                    "pred_sigma_um": pred_um * sl,
                    "pred_low68_um": math.exp(ml - sl),
                    "pred_high68_um": math.exp(ml + sl),
                    "bbox_64crop_upper_bound_um": 640.0,
                }
            )
            offset += 1
    return out


def _float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def compute_metrics(preds: list[dict[str, Any]]) -> dict[str, float]:
    truth = _float_array(preds, "true_diameter_um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma = _float_array(preds, "pred_sigma_um")
    abs_err = np.abs(pred - truth)
    return {
        "count": int(len(preds)),
        "mae_um": float(np.mean(abs_err)),
        "median_ae_um": float(np.median(abs_err)),
        "rmse_um": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(np.mean(abs_err / np.maximum(truth, 1e-9))),
        "within_5pct": float(np.mean(abs_err <= 0.05 * truth)),
        "within_10pct": float(np.mean(abs_err <= 0.10 * truth)),
        "coverage_1sigma": float(np.mean(abs_err <= np.maximum(sigma, 1e-9))),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * np.maximum(sigma, 1e-9))),
        "mean_sigma_um": float(np.mean(sigma)),
    }


def bin_metrics(preds: list[dict[str, Any]], bins: int = 12) -> list[dict[str, Any]]:
    truth = _float_array(preds, "true_diameter_um")
    edges = np.linspace(float(truth.min()), float(truth.max()), bins + 1)
    out: list[dict[str, Any]] = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sub = [p for p in preds if (lo <= float(p["true_diameter_um"]) <= hi if i == bins - 1 else lo <= float(p["true_diameter_um"]) < hi)]
        if not sub:
            continue
        row = {
            "diameter_bin_lo_um": lo,
            "diameter_bin_hi_um": hi,
            "diameter_bin_mid_um": 0.5 * (lo + hi),
            "true_diameter_mean_um": float(np.mean(_float_array(sub, "true_diameter_um"))),
        }
        row.update(compute_metrics(sub))
        out.append(row)
    return out


def calibrate_sigma(preds: list[dict[str, Any]]) -> float:
    truth = _float_array(preds, "true_diameter_um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma_log = _float_array(preds, "pred_sigma_log")
    err_log = np.abs(np.log(pred) - np.log(truth))
    scale = float(np.sqrt(np.mean((err_log / np.maximum(sigma_log, 1e-6)) ** 2)))
    return float(np.clip(scale, 0.20, 10.0))


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    plt.figure(figsize=(9.0, 5.0))
    plt.plot(epochs, [float(r["train_loss"]) for r in history], label="train loss")
    plt.plot(epochs, [float(r["valid_loss"]) for r in history], label="valid loss")
    plt.plot(epochs, [float(r["valid_mae_norm"]) for r in history], label="valid MAE norm")
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_pred_vs_true(preds: list[dict[str, Any]], path: Path) -> None:
    truth = _float_array(preds, "true_diameter_um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma = _float_array(preds, "pred_sigma_um")
    lo = min(float(truth.min()), float(pred.min()))
    hi = max(float(truth.max()), float(pred.max()))
    plt.figure(figsize=(6.2, 6.0))
    sc = plt.scatter(truth, pred, c=sigma, s=8, cmap="viridis", alpha=0.65, linewidths=0)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.xlabel("true diameter (um)")
    plt.ylabel("predicted diameter (um)")
    plt.colorbar(sc, label="predicted sigma (um)")
    plt.grid(True, alpha=0.20)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_error_bins(bins: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([float(r["true_diameter_mean_um"]) for r in bins])
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(x, [float(r["mae_um"]) for r in bins], marker="o", label="MAE")
    plt.plot(x, [float(r["median_ae_um"]) for r in bins], marker="s", label="median AE")
    plt.xlabel("true diameter (um)")
    plt.ylabel("absolute error (um)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_montage(preds: list[dict[str, Any]], path: Path, *, mode: str) -> None:
    if not preds:
        return
    selected = sorted(
        preds,
        key=lambda r: abs(float(r["pred_diameter_um"]) - float(r["true_diameter_um"])),
        reverse=(mode == "worst"),
    )
    if mode != "worst":
        idx = np.linspace(0, len(selected) - 1, min(48, len(selected)), dtype=int)
        selected = [selected[int(i)] for i in idx]
    else:
        selected = selected[:48]
    tile = 128
    label_h = 44
    cols = 8
    rows_n = math.ceil(len(selected) / cols)
    canvas = Image.new("RGB", (cols * tile, rows_n * (tile + label_h)), "white")
    font = ImageFont.load_default()
    for i, pred in enumerate(selected):
        crop = read_gray(REPO_ROOT / str(pred["image_path"]))
        im = Image.fromarray(crop, mode="L")
        im = ImageOps.autocontrast(im, cutoff=0.5).resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle([1, 1, tile - 2, tile - 2], outline=(30, 120, 80), width=2)
        x = (i % cols) * tile
        y = (i // cols) * (tile + label_h)
        canvas.paste(im, (x, y))
        label_draw = ImageDraw.Draw(canvas)
        truth = float(pred["true_diameter_um"])
        pred_um = float(pred["pred_diameter_um"])
        label_draw.text((x + 3, y + tile + 1), f"p={pred_um:.0f} t={truth:.0f}", fill=(0, 0, 0), font=font)
        label_draw.text((x + 3, y + tile + 15), f"e={pred_um - truth:+.0f} z={float(pred['z_um'])/1000:.1f}mm", fill=(0, 0, 0), font=font)
        label_draw.text((x + 3, y + tile + 29), Path(str(pred["file"])).name[:18], fill=(80, 80, 80), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def write_outputs(preds: list[dict[str, Any]], out_dir: Path, prefix: str) -> dict[str, float]:
    metrics = compute_metrics(preds)
    bins = bin_metrics(preds)
    write_csv(out_dir / f"{prefix}_predictions.csv", preds)
    write_json(out_dir / f"{prefix}_metrics.json", metrics)
    write_csv(out_dir / f"{prefix}_metrics_by_diameter.csv", bins)
    plot_pred_vs_true(preds, out_dir / f"{prefix}_pred_vs_true.png")
    plot_error_bins(bins, out_dir / f"{prefix}_error_by_diameter.png")
    make_montage(preds, out_dir / f"{prefix}_worst_examples.png", mode="worst")
    make_montage(preds, out_dir / f"{prefix}_size_examples.png", mode="sizes")
    return metrics


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.crops_csv)
    train_rows = [r for r in rows if r.split == "train"]
    valid_rows = [r for r in rows if r.split == "valid"]
    test_rows = [r for r in rows if r.split == "test"]
    if not train_rows or not valid_rows:
        raise RuntimeError(f"Need train and valid rows, got train={len(train_rows)} valid={len(valid_rows)}")
    log_train = np.log(np.asarray([r.diameter_um for r in train_rows], dtype=np.float64))
    norm = {
        "log_diam_mean": float(log_train.mean()),
        "log_diam_std": float(max(log_train.std(), 1e-6)),
        "target": "log(diameter_um)",
    }
    write_json(
        out_dir / "run_meta.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "crops_csv": str(repo_path(args.crops_csv)),
            "out_dir": str(out_dir),
            "device": args.device,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "input": "64x64 focused phase-retrieval reconstruction slice crop at true particle z",
            "norm": norm,
        },
    )
    train_loader = make_loader(
        train_rows,
        batch_size=args.batch_size,
        training=True,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_rows,
        batch_size=args.batch_size,
        training=False,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    model = DiameterNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    history: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, device, optimizer)
        valid_m = train_epoch(model, valid_loader, device, None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"valid_{k}": v for k, v in valid_m.items()},
        }
        history.append(row)
        write_csv(out_dir / "history.csv", history)
        if valid_m["mae_norm"] < best_mae:
            best_mae = valid_m["mae_norm"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(out_dir / "slice_diammodel_best.pt", model, args, epoch, valid_m, norm, calibration_scale=1.0)
        else:
            stale += 1
        save_checkpoint(out_dir / "slice_diammodel_last.pt", model, args, epoch, valid_m, norm, calibration_scale=1.0)
        print(
            f"epoch {epoch:03d} train_loss={train_m['loss']:.4f} valid_loss={valid_m['loss']:.4f} "
            f"valid_mae_norm={valid_m['mae_norm']:.4f}"
        )
        if stale >= args.patience:
            print(f"early stopping after epoch {epoch}; best_epoch={best_epoch}")
            break

    model, norm, _, _ = load_model(out_dir / "slice_diammodel_best.pt", device)
    raw_valid = predict_rows(model, valid_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=1.0)
    calibration_scale = calibrate_sigma(raw_valid)
    save_checkpoint(
        out_dir / "slice_diammodel_best.pt",
        model,
        args,
        best_epoch,
        {"best_valid_mae_norm": best_mae},
        norm,
        calibration_scale,
    )
    model, norm, calibration_scale, _ = load_model(out_dir / "slice_diammodel_best.pt", device)
    valid_preds = predict_rows(model, valid_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=calibration_scale)
    valid_metrics = write_outputs(valid_preds, out_dir, "valid")
    valid_metrics["calibration_scale"] = calibration_scale
    metrics = {"valid": valid_metrics}
    if test_rows:
        test_preds = predict_rows(model, test_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=calibration_scale)
        metrics["test"] = write_outputs(test_preds, out_dir, "test")
    write_json(out_dir / "metrics.json", metrics)
    plot_history(history, out_dir / "results.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a diameter model from focused reconstruction slice crops.")
    parser.add_argument("--crops-csv", type=Path, default=Path("data/slice_crops/slice_crops.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/slice_resnet64"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=260624)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
