"""Paper plotting/scientific functions; source hashes in provenance/paper_figures.json."""
from __future__ import annotations

import numpy as np

def terminal_velocity_um_s(
    diameter_um: float | np.ndarray,
    *,
    visco_g_cm_s: float = 0.0001818,
    pressure_mb: float = 1013.25,
    kelvin: float = 293.15,
) -> float | np.ndarray:
    """Return Beard (1976) terminal fall velocity in um/s.

    This implementation covers the same range as the original script:
    0.5 <= diameter_um < 1070.0.
    """

    scalar_input = np.isscalar(diameter_um)
    dp = np.asarray(diameter_um, dtype=np.float64)
    if np.any((dp < 0.5) | (dp >= 1070.0) | ~np.isfinite(dp)):
        raise ValueError("diameter_um must be finite and in the range [0.5, 1070.0)")

    eta = visco_g_cm_s * 0.1
    gravity = 9.817
    d0 = dp * 1e-6
    l0 = 6.62e-6 * 0.01
    p0 = 1013.25
    eta0 = 0.0001818 * 0.1
    t0 = 293.15
    rho_w = 1000.0
    rho_f = 1.225
    delta_rho = rho_w - rho_f

    l_val = l0 * (eta / eta0) * (p0 / pressure_mb) * (kelvin / t0) ** 0.5
    c_ac = 1.0 + 2.51 * l_val / d0

    vel_m_s = np.empty_like(dp, dtype=np.float64)
    small = dp < 19.0
    if np.any(small):
        c_l = delta_rho * gravity / (18.0 * eta)
        vel_m_s[small] = c_l * c_ac[small] * d0[small] ** 2

    mid = ~small
    if np.any(mid):
        c2 = 4.0 * rho_f * delta_rho * gravity / (3.0 * eta**2)
        n_da = c2 * d0[mid] ** 3
        x_val = np.log(n_da)
        y_val = (
            -0.318657e1
            + 0.992696 * x_val
            - 0.153193e-2 * x_val**2
            - 0.987059e-3 * x_val**3
            - 0.578878e-3 * x_val**4
            + 0.855176e-4 * x_val**5
            - 0.327815e-5 * x_val**6
        )
        n_re = c_ac[mid] * np.exp(y_val)
        vel_m_s[mid] = eta * n_re / (rho_f * d0[mid])

    out = vel_m_s * 1e6
    return float(out) if scalar_input else out

def terminal_velocity_beard_1976_um_s(
    diameter_um: float | np.ndarray,
    *,
    visco_g_cm_s: float = 0.0001818,
    pressure_mb: float = 1013.25,
    kelvin: float = 293.15,
) -> float | np.ndarray:
    """Compatibility alias for the historical function name."""

    return terminal_velocity_um_s(
        diameter_um,
        visco_g_cm_s=visco_g_cm_s,
        pressure_mb=pressure_mb,
        kelvin=kelvin,
    )
