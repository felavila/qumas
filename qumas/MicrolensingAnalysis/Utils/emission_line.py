from dataclasses import dataclass
import numpy as np 
from typing import Any

from qumas.MicrolensingAnalysis.Utils.extras import _linear_params,_area_under_line,_trapz_in_range 

@dataclass
class _BootRes:
    slope_samples: np.ndarray
    intercept_samples: np.ndarray
    area_cont_samples: np.ndarray
    core_flux_samples: np.ndarray
    slope_med: float
    slope_std: float
    intercept_med: float
    intercept_std: float
    area_cont_med: float
    area_cont_std: float
    core_line_med: float
    core_line_std: float


    
    
def _bootstrap_linear_and_areas(
    x: np.ndarray,
    y: np.ndarray,
    left_window: tuple[float, float],
    right_window: tuple[float, float],
    core_window: tuple[float, float],
    n_bootstrap: int = 5000,
    random_state: int | None = None,
    y_err: np.ndarray | None = None,
) -> _BootRes:
    """
    Bootstrap continuum + areas, optionally including flux errors y_err.

    If y_err is provided, each bootstrap realization uses a perturbed
    spectrum y_boot ~ N(y, y_err) consistently for continuum and core.
    """
    rng = np.random.default_rng(random_state)

    lmin, lmax = float(left_window[0]), float(left_window[1])
    rmin, rmax = float(right_window[0]), float(right_window[1])
    cmin, cmax = float(core_window[0]), float(core_window[1])

    mL = (x >= lmin) & (x <= lmax)
    mR = (x >= rmin) & (x <= rmax)

    xL = x[mL]
    xR = x[mR]
    nL, nR = xL.size, xR.size

    if (nL + nR) < 2:
        zeros = np.zeros(n_bootstrap, dtype=float)
        return _BootRes(
            zeros, zeros, zeros, zeros,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    slope_samples = np.empty(n_bootstrap, dtype=float)
    intercept_samples = np.empty(n_bootstrap, dtype=float)
    area_cont_samples = np.empty(n_bootstrap, dtype=float)
    core_flux_samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        # Draw a noisy realization of the whole spectrum, if errors are given
        if y_err is not None:
            y_boot = rng.normal(loc=y, scale=y_err)
        else:
            y_boot = y

        yL = y_boot[mL]
        yR = y_boot[mR]

        idxL = rng.integers(0, nL, size=nL) if nL > 0 else np.array([], dtype=int)
        idxR = rng.integers(0, nR, size=nR) if nR > 0 else np.array([], dtype=int)

        if nL and nR:
            x_res = np.concatenate([xL[idxL], xR[idxR]])
            y_res = np.concatenate([yL[idxL], yR[idxR]])
        elif nL:
            x_res, y_res = xL[idxL], yL[idxL]
        else:
            x_res, y_res = xR[idxR], yR[idxR]

        # Fit continuum on the noisy resampled points
        m, b = _linear_params(x_res, y_res)
        slope_samples[i] = m
        intercept_samples[i] = b

        # Area under continuum between lmin and rmax
        area_cont_samples[i] = _area_under_line(m, b, lmin, rmax)

        # Continuum-subtracted core flux using the same noisy realization
        y_curve = y_boot - (m * x + b)
        core_flux_samples[i] = _trapz_in_range(x, y_curve, cmin, cmax)

    def _med_std(a: np.ndarray) -> tuple[float, float]:
        return float(np.median(a)), float(np.std(a, ddof=1))

    sm, ss = _med_std(slope_samples)
    bm, bs = _med_std(intercept_samples)
    am, as_ = _med_std(area_cont_samples)
    cm, cs = _med_std(core_flux_samples)

    return _BootRes(
        slope_samples, intercept_samples, area_cont_samples, core_flux_samples,
        sm, ss, bm, bs, am, as_, cm, cs
    )
    
def _compute_metrics_for_image(
        X: np.ndarray,
        Y: np.ndarray,
        left_window: tuple[float, float],
        right_window: tuple[float, float],
        core_window: tuple[float, float],
        n_bootstrap: int = 5000,
        random_state: int | None = None,
        y_err: np.ndarray | None = None,
    ) -> dict:
        boot = _bootstrap_linear_and_areas(
            X, Y, left_window, right_window, core_window,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            y_err=y_err,
        )

        # Use the *median* continuum parameters on the original spectrum Y
        m_med, b_med = boot.slope_med, boot.intercept_med
        cont_med = m_med * X + b_med
        y_curve_med = (Y - cont_med).astype(float)

        return dict(
            slope_fit=boot.slope_med,
            slope_fit_err=boot.slope_std,
            intercept_fit=boot.intercept_med,
            intercept_fit_err=boot.intercept_std,
            area_continuo=boot.area_cont_med,
            area_continuo_error=boot.area_cont_std,
            core_line=boot.core_line_med,
            core_line_error=boot.core_line_std,
            y_curve=y_curve_med.tolist(),
            bootstrap_slope=boot.slope_samples.tolist(),
            bootstrap_intercept=boot.intercept_samples.tolist(),
            bootstrap_area_cont=boot.area_cont_samples.tolist(),
            bootstrap_core_flux=boot.core_flux_samples.tolist(),
        )