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
    
# def _compute_metrics_for_image(
#         X: np.ndarray,
#         Y: np.ndarray,
#         left_window: tuple[float, float],
#         right_window: tuple[float, float],
#         core_window: tuple[float, float],
#         n_bootstrap: int = 5000,
#         random_state: int | None = None,
#         y_err: np.ndarray | None = None,
#     ) -> dict:
#         # boot = _bootstrap_linear_and_areas(
#         #     X, Y, left_window, right_window, core_window,
#         #     n_bootstrap=n_bootstrap,
#         #     random_state=random_state,
#         #     y_err=y_err,
#         # )

#         for x,y in zip(X,Y):
#             boot = _bootstrap_linear_and_areas(
#             x,y, left_window, right_window, core_window,
#             n_bootstrap=n_bootstrap,
#             random_state=random_state,
#             y_err=y_err,)

#             # Use the *median* continuum parameters on the original spectrum Y
#             m_med, b_med = boot.slope_med, boot.intercept_med
#             #print(X.shape,Y.shape)
#             cont_med = m_med * X + b_med
#             y_curve_med = (Y - cont_med).astype(float)

#         return dict(
#             slope_fit=boot.slope_med,
#             slope_fit_err=boot.slope_std,
#             intercept_fit=boot.intercept_med,
#             intercept_fit_err=boot.intercept_std,
#             area_continuo=boot.area_cont_med,
#             area_continuo_error=boot.area_cont_std,
#             core_line=boot.core_line_med,
#             core_line_error=boot.core_line_std,
#             y_curve=y_curve_med.tolist(),
#             bootstrap_slope=boot.slope_samples.tolist(),
#             bootstrap_intercept=boot.intercept_samples.tolist(),
#             bootstrap_area_cont=boot.area_cont_samples.tolist(),
#             bootstrap_core_flux=boot.core_flux_samples.tolist(),
#         )


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
    """
    Compute continuum + line metrics for one or many spectra.

    Parameters
    ----------
    X : array-like
        1D array (n_pix,) or 2D array (N, n_pix) with the x-axis (e.g. wavelength).
    Y : array-like
        1D array (n_pix,) or 2D array (N, n_pix) with the flux values.
    left_window, right_window, core_window : tuple[float, float]
        Windows in X used by _bootstrap_linear_and_areas.
    n_bootstrap : int
        Number of bootstrap realizations.
    random_state : int or None
        RNG seed passed to the bootstrap routine.
    y_err : array-like or None
        Optional uncertainties. Accepted shapes:
          - None
          - (n_pix,)
          - (N, n_pix)

    Returns
    -------
    dict
        All quantities are per object (length N) or (N, n_bootstrap) for samples.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Normalize shapes: allow 1D or 2D, but end up with (N, n_pix)
    if X.ndim == 1 and Y.ndim == 1:
        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have the same shape, got {X.shape} vs {Y.shape}")
        X_2d = X[None, :]
        Y_2d = Y[None, :]
    elif X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape:
        X_2d, Y_2d = X, Y
    else:
        raise ValueError(
            f"X and Y must both be 1D (n_pix,) or 2D (N, n_pix) with the same shape; "
            f"got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    n_obj, n_pix = Y_2d.shape

    # Handle y_err broadcasting
    if y_err is None:
        y_err_2d = None
    else:
        y_err = np.asarray(y_err)
        if y_err.ndim == 1:
            if y_err.shape[0] != n_pix:
                raise ValueError(
                    f"y_err 1D must have length n_pix={n_pix}, got {y_err.shape[0]}"
                )
            # Same error for all objects
            y_err_2d = np.broadcast_to(y_err, (n_obj, n_pix))
        elif y_err.ndim == 2:
            if y_err.shape != (n_obj, n_pix):
                raise ValueError(
                    f"y_err 2D must have shape (N, n_pix)=({n_obj}, {n_pix}), "
                    f"got {y_err.shape}"
                )
            y_err_2d = y_err
        else:
            raise ValueError("y_err must be None, 1D, or 2D")

    # Allocate outputs (per object)
    slope_fit = np.empty(n_obj, dtype=float)
    slope_fit_err = np.empty(n_obj, dtype=float)
    intercept_fit = np.empty(n_obj, dtype=float)
    intercept_fit_err = np.empty(n_obj, dtype=float)
    area_continuo = np.empty(n_obj, dtype=float)
    area_continuo_error = np.empty(n_obj, dtype=float)
    core_line = np.empty(n_obj, dtype=float)
    core_line_error = np.empty(n_obj, dtype=float)
    y_curve = np.empty_like(Y_2d, dtype=float)

    # We’ll allocate bootstrap arrays after first iteration when we know len(samples)
    bootstrap_slope = None
    bootstrap_intercept = None
    bootstrap_area_cont = None
    bootstrap_core_flux = None

    for i, (x_row, y_row) in enumerate(zip(X_2d, Y_2d)):
        if y_err_2d is not None:
            y_err_row = y_err_2d[i]
        else:
            y_err_row = None

        boot = _bootstrap_linear_and_areas(
            x_row,
            y_row,
            left_window,
            right_window,
            core_window,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            y_err=y_err_row,
        )

        # Continuum with median parameters for this object
        m_med, b_med = boot.slope_med, boot.intercept_med
        cont_med = m_med * x_row + b_med
        y_curve[i] = (y_row - cont_med).astype(float)

        # Store per-object summary stats
        slope_fit[i] = boot.slope_med
        slope_fit_err[i] = boot.slope_std
        intercept_fit[i] = boot.intercept_med
        intercept_fit_err[i] = boot.intercept_std
        area_continuo[i] = boot.area_cont_med
        area_continuo_error[i] = boot.area_cont_std
        core_line[i] = boot.core_line_med
        core_line_error[i] = boot.core_line_std

        # Allocate bootstrap samples arrays once, using first object's sample length
        if bootstrap_slope is None:
            n_samp = len(boot.slope_samples)
            bootstrap_slope = np.empty((n_obj, n_samp), dtype=float)
            bootstrap_intercept = np.empty((n_obj, n_samp), dtype=float)
            bootstrap_area_cont = np.empty((n_obj, n_samp), dtype=float)
            bootstrap_core_flux = np.empty((n_obj, n_samp), dtype=float)

        bootstrap_slope[i] = np.asarray(boot.slope_samples, dtype=float)
        bootstrap_intercept[i] = np.asarray(boot.intercept_samples, dtype=float)
        bootstrap_area_cont[i] = np.asarray(boot.area_cont_samples, dtype=float)
        bootstrap_core_flux[i] = np.asarray(boot.core_flux_samples, dtype=float)

    return dict(
        slope_fit=slope_fit.tolist(),
        slope_fit_err=slope_fit_err.tolist(),
        intercept_fit=intercept_fit.tolist(),
        intercept_fit_err=intercept_fit_err.tolist(),
        area_continuo=area_continuo.tolist(),
        area_continuo_error=area_continuo_error.tolist(),
        core_line=core_line.tolist(),
        core_line_error=core_line_error.tolist(),
        y_curve=y_curve.tolist(),  # shape (N, n_pix)
        bootstrap_slope=bootstrap_slope.tolist(),          # (N, n_bootstrap)
        bootstrap_intercept=bootstrap_intercept.tolist(),  # (N, n_bootstrap)
        bootstrap_area_cont=bootstrap_area_cont.tolist(),  # (N, n_bootstrap)
        bootstrap_core_flux=bootstrap_core_flux.tolist(),  # (N, n_bootstrap)
    )
