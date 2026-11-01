"""Deterministic approximate robust PRESS selection, preserving the Julia rule."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .config import SmootherConfig
from .core import design_matrix, fit_z_gmm, regularization
from .model import GMMState, Trajectory, validate_trajectories


class TuningError(ValueError):
    """No valid tuning result. ``summary`` retains every evaluated candidate."""

    def __init__(self, message: str, summary: dict):
        super().__init__(message)
        self.summary = summary


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), np.clip(quantile, 0, 1) * weights.sum())
    return float(values[order[min(index, len(order) - 1)]])


def press_metrics(state: GMMState, trajectories: list[Trajectory], config: SmootherConfig) -> dict:
    """Approximate leave-one-out errors with the fitted mixture/prior held fixed.

    This is the source's robust tuning surrogate, not independently refitted
    cross-validation or a measurement of ground-truth reconstruction accuracy.
    """
    errors, score_weights, track_maes, leverages = [], [], [], []
    precision = regularization(config)
    for trajectory, beta, resp in zip(trajectories, state.betas, state.post_out_prob, strict=True):
        phi = design_matrix((trajectory.frames - trajectory.frames[0]) / state.time_scale, config.degree)
        weights = (1 - resp) / state.sigma2_normal + resp / state.sigma2_outlier
        a = phi.T @ (phi * weights[:, None]) + precision
        try:
            solved = np.linalg.solve(a, phi.T)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(a, phi.T, rcond=None)[0]
        leverage = weights * np.sum(phi * solved.T, axis=1)
        error = np.abs((trajectory.z - phi @ beta) / np.maximum(1 - leverage, 1e-6))
        inlier = np.clip(1 - resp, 1e-6, 1)
        if not np.isfinite(error).all():
            raise FloatingPointError("Non-finite PRESS residuals")
        errors.append(error)
        score_weights.append(inlier)
        track_maes.append(float(np.average(error, weights=inlier)))
        leverages.append(leverage)
    error, weights = np.concatenate(errors), np.concatenate(score_weights)
    p99 = weighted_quantile(error, weights, 0.99)
    return {
        "objective": float(np.sqrt(np.average(np.minimum(error, p99) ** 2, weights=weights))),
        "objective_name": "inlier_weighted_press_trimmed_rmse_99",
        "press_weighted_mae": float(np.average(error, weights=weights)),
        "press_weighted_rmse": float(np.sqrt(np.average(error**2, weights=weights))),
        "press_weighted_p95": weighted_quantile(error, weights, 0.95),
        "press_weighted_p99": p99,
        "press_track_macro_mae": float(np.mean(track_maes)),
        "mean_leverage": float(np.mean(np.concatenate(leverages))),
        "scored_tracks": len(track_maes),
        "scored_points": len(error),
    }


def tuning_candidates(config: SmootherConfig) -> list[dict]:
    return [
        {"degree": degree, "lambda_other": penalty, "alpha": alpha}
        for degree in sorted(set(config.degree_grid))
        for penalty in sorted(set(config.lambda_other_grid))
        for alpha in (sorted(set(config.alpha_grid)) if degree >= 2 else [0.0])
    ]


def select_candidate(candidates: list[dict], tolerance: float) -> tuple[dict, float]:
    converged = [
        c for c in candidates if c["status"] == "passed" and c["em_converged"] and np.isfinite(c["objective"])
    ]
    if not converged:
        raise ValueError(
            "No candidate converged; increase tune_em_max_iter and em_max_iter or use fixed selection"
        )
    minimum = min(c["objective"] for c in converged)
    near = [c for c in converged if c["objective"] <= minimum * (1 + tolerance)]
    return min(near, key=lambda c: (c["degree"], c["objective"], -c["lambda_other"], -c["alpha"])), minimum


def tune_hyperparameters(
    trajectories: list[Trajectory],
    config: SmootherConfig | None = None,
    *,
    progress: Callable[[dict], None] | None = None,
) -> tuple[SmootherConfig, dict]:
    validate_trajectories(trajectories)
    config = config or SmootherConfig()
    eligible = sorted(
        (t for t in trajectories if len(t) >= config.tune_min_points),
        key=lambda t: (len(t), t.filename, t.segment),
    )
    if not eligible:
        raise TuningError(
            f"No trajectories have at least {config.tune_min_points} points. "
            "Lower tune_min_points or use selection='fixed'.",
            {"candidates": []},
        )
    sample = eligible
    if len(sample) > config.tune_max_trajectories:
        # Match Julia's round-to-even 1-based index selection, including count=1.
        indices = np.unique(
            np.rint(np.linspace(1, len(sample), config.tune_max_trajectories)).astype(int) - 1
        )
        sample = [sample[i] for i in indices]
    candidates = []
    for index, spec in enumerate(tuning_candidates(config), 1):
        started = time.perf_counter()
        candidate = {"index": index, **spec}
        candidate_config = replace(
            config, selection="fixed", **spec, em_max_iter=min(config.em_max_iter, config.tune_em_max_iter)
        )
        try:
            state = fit_z_gmm(sample, candidate_config, warn=False)
            candidate.update(press_metrics(state, sample, candidate_config))
            candidate.update(
                status="passed",
                em_iterations=state.iterations,
                em_converged=state.converged,
                em_final_diff=state.final_diff,
                sigma_z_normal=float(np.sqrt(state.sigma2_normal)),
                pi_out=state.pi_out,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            candidate.update(status="failed", error=str(error), objective=None)
        candidate["elapsed_sec"] = time.perf_counter() - started
        candidates.append(candidate)
        if progress:
            progress(candidate.copy())
    lengths = [len(t) for t in sample]
    summary = {
        "selection": "auto",
        "objective_name": "inlier_weighted_press_trimmed_rmse_99",
        "selection_rule": "lowest degree within relative tolerance, then minimum objective",
        "relative_tolerance": config.tune_relative_tolerance,
        "sample": {
            "selection_method": "length_stratified_deterministic",
            "eligible_trajectories": len(eligible),
            "sampled_trajectories": len(sample),
            "sampled_points": sum(lengths),
            "minimum_track_points": min(lengths),
            "median_track_points": float(np.median(lengths)),
            "maximum_track_points": max(lengths),
            "keys": [[int(t.track_id), int(t.segment)] for t in sample],
        },
        "grid": {
            "degree": list(config.degree_grid),
            "lambda_other": list(config.lambda_other_grid),
            "alpha_for_degree_ge_2": list(config.alpha_grid),
        },
        "candidates": candidates,
    }
    try:
        selected, minimum = select_candidate(candidates, config.tune_relative_tolerance)
    except ValueError as error:
        raise TuningError(str(error), summary) from error
    summary.update(
        selected=selected,
        minimum_objective=minimum,
        near_optimal_threshold=minimum * (1 + config.tune_relative_tolerance),
        converged_candidates=sum(c.get("em_converged", False) for c in candidates),
    )
    resolved = replace(
        config, degree=selected["degree"], alpha=selected["alpha"], lambda_other=selected["lambda_other"]
    )
    return resolved, summary
