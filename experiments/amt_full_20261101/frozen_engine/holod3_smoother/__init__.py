"""Robust holographic particle trajectory smoothing in Python."""

from .config import SmootherConfig
from .core import fit_z_gmm, predict_z
from .io import load_tracks
from .model import GMMState, Trajectory
from .pipeline import SmoothedBatch, SmoothResult, Units, smooth_linker_output, smooth_trajectories
from .tuning import TuningError, tune_hyperparameters

__version__ = "0.1.0"
__all__ = [
    "GMMState",
    "SmoothedBatch",
    "SmoothResult",
    "SmootherConfig",
    "Trajectory",
    "TuningError",
    "Units",
    "fit_z_gmm",
    "load_tracks",
    "predict_z",
    "smooth_linker_output",
    "smooth_trajectories",
    "tune_hyperparameters",
]
