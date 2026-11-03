"""Validated, serializable smoothing options; defaults follow the 260712 Julia implementation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from pathlib import Path


@dataclass(frozen=True)
class SmootherConfig:
    selection: str = "auto"
    degree: int = 2
    alpha: float = 0.0
    lambda0: float = 1e-6
    lambda_other: float = 1e-4
    degree_grid: tuple[int, ...] = (1, 2)
    lambda_other_grid: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
    alpha_grid: tuple[float, ...] = (0.0, 1e-4, 1e-2)
    tune_max_trajectories: int = 512
    tune_min_points: int = 8
    tune_em_max_iter: int = 100
    tune_relative_tolerance: float = 0.01
    pi_init: float = 0.05
    sigma2z0_init: float | None = None
    sigma2z1_init: float | None = None
    outlier_variance_multiplier: float = 50.0
    em_tol: float = 5e-5
    em_max_iter: int = 100
    sg_window: int = 7
    sg_order: int = 2
    xy_mode: str = "auto"
    max_gap: int | None = None

    def __post_init__(self) -> None:
        if self.selection not in ("auto", "fixed"):
            raise ValueError("selection must be 'auto' or 'fixed'")
        if self.xy_mode not in ("auto", "legacy", "local"):
            raise ValueError("xy_mode must be 'auto', 'legacy', or 'local'")
        for name in (
            "degree",
            "sg_order",
            "sg_window",
            "em_max_iter",
            "tune_max_trajectories",
            "tune_min_points",
            "tune_em_max_iter",
        ):
            value = getattr(self, name)
            minimum = 0 if name in ("degree", "sg_order") else (2 if name == "tune_min_points" else 1)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "alpha",
            "lambda0",
            "lambda_other",
            "pi_init",
            "em_tol",
            "outlier_variance_multiplier",
            "tune_relative_tolerance",
            "sigma2z0_init",
            "sigma2z1_init",
        ):
            value = getattr(self, name)
            if value is None and name in ("sigma2z0_init", "sigma2z1_init"):
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            positive = name in ("lambda0", "em_tol", "sigma2z0_init", "sigma2z1_init")
            if value < 0 or (positive and value == 0):
                raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
        if not 0 < self.pi_init < 1:
            raise ValueError("pi_init must be between 0 and 1")
        if self.outlier_variance_multiplier <= 1:
            raise ValueError("outlier_variance_multiplier must be > 1")
        if self.max_gap is not None and (
            isinstance(self.max_gap, bool) or not isinstance(self.max_gap, Integral) or self.max_gap < 1
        ):
            raise ValueError("max_gap must be a positive integer or null")
        for name in ("degree_grid", "alpha_grid", "lambda_other_grid"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a non-empty list")
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value < 0
                    or (name == "degree_grid" and not isinstance(value, Integral))
                ):
                    raise ValueError(f"Invalid {name} value: {value}")
            object.__setattr__(self, name, tuple(values))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> SmootherConfig:
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> SmootherConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
