"""Trajectory and fit results independent of on-disk input/output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    frames: ArrayLike
    x: ArrayLike
    y: ArrayLike
    z: ArrayLike
    diameter_px: ArrayLike
    segment: int = 0

    def __post_init__(self) -> None:
        for name in ("track_id", "segment"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        n = None
        for name in ("frames", "x", "y", "z", "diameter_px"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
            if n is not None and len(arr) != n:
                raise ValueError("Trajectory arrays must have equal lengths")
            n = len(arr)
            if name == "frames":
                if np.any(arr < 0) or np.any(arr >= 2**53) or np.any(arr != np.floor(arr)):
                    raise ValueError("frames must be non-negative exactly representable integers < 2**53")
                if np.any(np.diff(arr) <= 0):
                    raise ValueError("frames must be strictly increasing; duplicate frames are not allowed")
                arr = arr.astype(np.int64)
            if name == "diameter_px" and np.any(arr < 0):
                raise ValueError("diameter_px must be non-negative")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def filename(self) -> str:
        return f"path_{self.track_id:08d}.csv"


@dataclass
class GMMState:
    pi_out: float
    sigma2_normal: float
    sigma2_outlier: float
    betas: NDArray[np.float64]
    post_out_prob: list[NDArray[np.float64]]
    time_scale: float
    prior_mean: NDArray[np.float64]
    iterations: int
    final_diff: float
    converged: bool

    def summary(self) -> dict:
        return {
            "pi_out": self.pi_out,
            "sigma2_normal": self.sigma2_normal,
            "sigma2_outlier": self.sigma2_outlier,
            "sigma_z_normal": float(np.sqrt(self.sigma2_normal)),
            "sigma_z_outlier": float(np.sqrt(self.sigma2_outlier)),
            "time_scale": self.time_scale,
            "prior_mean": self.prior_mean.tolist(),
            "em_iterations": self.iterations,
            "em_final_diff": self.final_diff,
            "em_converged": self.converged,
        }


def validate_trajectories(trajectories: list[Trajectory]) -> None:
    if not trajectories:
        raise ValueError("No non-empty trajectories were supplied")
    keys = [(t.track_id, t.segment) for t in trajectories]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (track_id, segment) keys")
