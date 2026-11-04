"""Pooled two-Gaussian EM polynomial smoother ported from 260712/zsmooth_new.jl.

Coordinates are in input units and time is in frames. Batched normal equations
reuse the polynomial basis across EM iterations. No random initialization is used.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_coeffs

from .config import SmootherConfig
from .model import GMMState, Trajectory, validate_trajectories


def design_matrix(times: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(times, degree + 1, increasing=True)


def regularization(config: SmootherConfig) -> np.ndarray:
    diagonal = config.lambda_other + config.alpha * np.arange(config.degree + 1, dtype=float) ** 2
    diagonal[0] = config.lambda0
    return np.diag(diagonal)


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched SPD solve, with minimum-norm fallback for unregularized short tracks."""
    try:
        factor = np.linalg.cholesky(a)
        return np.linalg.solve(factor.swapaxes(-1, -2), np.linalg.solve(factor, b))
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(ai, bi, rcond=None)[0] for ai, bi in zip(a, b, strict=True)])


class _System:
    def __init__(self, trajectories: list[Trajectory], degree: int, time_scale: float):
        self.lengths = np.array([len(t) for t in trajectories])
        self.starts = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.tau = np.concatenate([(t.frames - t.frames[0]) / time_scale for t in trajectories])
        self.phi = design_matrix(self.tau, degree)
        self.z = np.concatenate([t.z for t in trajectories])

    def predict(self, betas: np.ndarray) -> np.ndarray:
        return np.einsum("np,np->n", self.phi, np.repeat(betas, self.lengths, axis=0))

    def normal_matrix(self, weights: np.ndarray, precision: np.ndarray) -> np.ndarray:
        # Build one coefficient pair at a time to avoid an N x degree² allocation.
        p = self.phi.shape[1]
        a = np.empty((len(self.lengths), p, p))
        for j in range(p):
            for k in range(j + 1):
                a[:, j, k] = a[:, k, j] = np.add.reduceat(
                    weights * self.phi[:, j] * self.phi[:, k], self.starts
                )
        return a + precision

    def ridge(self, weights: np.ndarray, precision: np.ndarray, mean: np.ndarray) -> np.ndarray:
        a = self.normal_matrix(weights, precision)
        b = np.add.reduceat(self.phi * (weights * self.z)[:, None], self.starts, axis=0)
        b += precision @ mean
        return _solve(a, b[..., None])[..., 0]

    def split(self, values: np.ndarray) -> list[np.ndarray]:
        return [a.copy() for a in np.split(values, self.starts[1:])]


def _posterior(residual: np.ndarray, pi: float, s0: float, s1: float) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        r2 = residual**2
        lp0 = np.log(1 - pi + 1e-12) - 0.5 * (np.log(2 * np.pi * s0) + r2 / s0)
        lp1 = np.log(pi + 1e-12) - 0.5 * (np.log(2 * np.pi * s1) + r2 / s1)
        return np.clip(np.exp(lp1 - np.logaddexp(lp0, lp1)), 0, 1)


def fit_z_gmm(
    trajectories: list[Trajectory], config: SmootherConfig | None = None, *, warn: bool = True
) -> GMMState:
    """Fit the supplied degree/regularization jointly across trajectories.

    This low-level function does not run parameter selection. Use
    ``smooth_trajectories`` or ``tune_hyperparameters`` for automatic selection.
    """
    validate_trajectories(trajectories)
    config = config or SmootherConfig(selection="fixed")
    scale = float(max(max(int(t.frames[-1] - t.frames[0]), 1) for t in trajectories))
    system = _System(trajectories, config.degree, scale)
    precision = regularization(config)
    weights = np.ones(len(system.z))
    initial = system.ridge(weights, precision, np.zeros(config.degree + 1))
    mean = initial.mean(axis=0)
    betas = system.ridge(weights, precision, mean)
    residual = system.z - system.predict(betas)
    s0 = max(1e-9, float(np.mean(residual**2)) if config.sigma2z0_init is None else config.sigma2z0_init)
    s1 = max(
        s0 * 1.000001,
        config.sigma2z1_init if config.sigma2z1_init is not None else s0 * config.outlier_variance_multiplier,
    )
    pi = config.pi_init
    converged = False
    for iteration in range(1, config.em_max_iter + 1):  # noqa: B007 - retained for convergence diagnostics
        old = np.array([pi, s0, s1, np.linalg.norm(betas, axis=1).mean()])
        resp = _posterior(residual, pi, s0, s1)
        pi = float(np.clip(resp.mean(), 1e-9, 1 - 1e-9))
        betas = system.ridge((1 - resp) / s0 + resp / s1, precision, mean)
        residual = system.z - system.predict(betas)
        squared = residual**2
        s0 = max(1e-9, float(np.sum((1 - resp) * squared) / max(np.sum(1 - resp), 1e-9)))
        s1 = max(s0 * 1.000001, float(np.sum(resp * squared) / max(np.sum(resp), 1e-9)))
        current = np.array([pi, s0, s1, np.linalg.norm(betas, axis=1).mean()])
        diff = float(np.max(np.abs(current - old) / (np.abs(old) + 1e-8)))
        if not np.isfinite(current).all() or not np.isfinite(betas).all():
            raise FloatingPointError("Non-finite EM state; check coordinate magnitudes and regularization")
        if diff <= config.em_tol:
            converged = True
            break
    if not converged and warn:
        warnings.warn(
            f"Z GMM did not converge after {iteration} iterations (relative change {diff:.3g})",
            RuntimeWarning,
            stacklevel=2,
        )
    post = system.split(_posterior(residual, pi, s0, s1))
    return GMMState(pi, s0, s1, betas, post, scale, mean, iteration, diff, converged)


def predict_z(trajectory: Trajectory, beta: np.ndarray, time_scale: float) -> tuple[np.ndarray, np.ndarray]:
    tau = (trajectory.frames - trajectory.frames[0]) / time_scale
    position = design_matrix(tau, len(beta) - 1) @ beta
    velocity = np.zeros(len(trajectory))
    if len(beta) > 1:
        velocity = design_matrix(tau, len(beta) - 2) @ (np.arange(1, len(beta)) * beta[1:]) / time_scale
    return position, velocity


def smooth_xy(
    values: np.ndarray, frames: np.ndarray, config: SmootherConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy reflected SG at unit spacing; local polynomial on actual times otherwise.

    ``legacy`` deliberately retains the source's sample-index derivative even
    with gaps. ``auto`` uses it only for consecutive frames. For 1–2 points we
    retain source behavior: unchanged positions and zero (unresolved) velocity.
    """
    n = len(values)
    window = min(config.sg_window, n)
    window -= 1 - window % 2
    if window < 3:
        return values.copy(), np.zeros(n)
    order = min(config.sg_order, window - 2)
    use_legacy = config.xy_mode == "legacy" or (config.xy_mode == "auto" and np.all(np.diff(frames) == 1))
    if use_legacy:
        hw = window // 2
        padded = np.r_[
            2 * values[0] - values[1 : hw + 1][::-1], values, 2 * values[-1] - values[-hw - 1 : -1][::-1]
        ]
        smooth = np.convolve(padded, savgol_coeffs(window, order), mode="valid")
        velocity = (
            np.convolve(padded, savgol_coeffs(window, order, deriv=1), mode="valid")
            if order >= 1
            else np.zeros(n)
        )
        return smooth, velocity
    smooth, velocity = np.empty(n), np.zeros(n)
    for i in range(n):
        start = min(max(0, i - window // 2), n - window)
        idx = slice(start, start + window)
        dt = (frames[idx] - frames[i]).astype(float)
        scale = max(float(np.max(np.abs(dt))), 1)
        coeff = np.linalg.lstsq(design_matrix(dt / scale, order), values[idx], rcond=None)[0]
        smooth[i] = coeff[0]
        if order >= 1:
            velocity[i] = coeff[1] / scale
    return smooth, velocity
