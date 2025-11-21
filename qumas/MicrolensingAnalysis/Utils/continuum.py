from dataclasses import dataclass
import numpy as np 
from typing import Any



@dataclass
class _BootSumRes:
    sum_samples: np.ndarray
    sum_med: float
    sum_std: float




def _bootstrap_sum_y(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    y_err: np.ndarray | None = None,
    n_bootstrap: int = 5000,
    random_state: int | None = None,
) -> _BootSumRes:
    """
    Bootstrap the sum of y values within `window`, with optional y-errors.

    Parameters
    ----------
    x : array-like
        Wavelength (or x-axis) array.
    y : array-like
        Flux (or y-axis) array.
    window : (float, float)
        [xmin, xmax] region over which to consider points.
    y_err : array-like or None, optional
        1σ uncertainties on y. If provided, each bootstrap realization
        draws y' ~ N(y, y_err) for the resampled points.
    n_bootstrap : int, optional
        Number of bootstrap realizations.
    random_state : int or None, optional
        Seed for the RNG.

    Returns
    -------
    _BootSumRes
        sum_samples : array
            Bootstrap realizations of the sum.
        sum_med : float
            Median of the bootstrap sums.
        sum_std : float
            Standard deviation (ddof=1) of the bootstrap sums.
    """
    rng = np.random.default_rng(random_state)

    xmin, xmax = float(window[0]), float(window[1])
    mask = (x >= xmin) & (x <= xmax)

    xw = np.asarray(x)[mask]
    yw = np.asarray(y)[mask]
    if y_err is not None:
        y_errw = np.asarray(y_err)[mask]
    else:
        y_errw = None

    n = yw.size
    if n == 0:
        zeros = np.zeros(n_bootstrap, dtype=float)
        return _BootSumRes(zeros, 0.0, 0.0)

    sum_samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        # resample indices with replacement
        idx = rng.integers(0, n, size=n)

        y_res = yw[idx]
        if y_errw is not None:
            # draw a realization using the errors
            y_res = rng.normal(loc=y_res, scale=y_errw[idx])

        # here is the statistic: simple sum of y
        sum_samples[i] = float(np.sum(y_res))
        # if you prefer an integrated flux over x:
        # sum_samples[i] = float(np.trapz(y_res, xw[idx]))

    sum_med = float(np.median(sum_samples))
    sum_std = float(np.std(sum_samples, ddof=1)) if n_bootstrap > 1 else 0.0

    return _BootSumRes(sum_samples, sum_med, sum_std)




# def _compute_metrics_for_image_continuum(
#     x: np.ndarray,
#     y: np.ndarray,
#     core_window: tuple[float, float],
#     y_err: np.ndarray | None = None,
#     n_bootstrap: int = 5000,
#     random_state: int | None = None,
# ) -> dict[str, Any]:
#     """
#     Compute bootstrap metrics (sum over y) for a single image/spectrum.

#     Parameters
#     ----------
#     x : np.ndarray
#         X axis (e.g. wavelength).
#     y : np.ndarray
#         Y axis (e.g. flux).
#     core_window : (float, float)
#         [xmin, xmax] over which the sum is computed.
#     y_err : np.ndarray or None, optional
#         1σ uncertainties on y. If provided, bootstrap draws
#         y' ~ N(y, y_err) for the resampled points.
#     n_bootstrap : int, optional
#         Number of bootstrap realizations.
#     random_state : int or None, optional
#         Seed for RNG.

#     Returns
#     -------
#     dict
#         {
#           "core_sum_samples": np.ndarray,  # all bootstrap sums
#           "core_sum": float,               # median of bootstrap sums
#           "core_sum_err": float,           # std (ddof=1) of bootstrap sums
#         }
#     """
#     core_res = _bootstrap_sum_y(
#         x=x,
#         y=y,
#         window=core_window,
#         y_err=y_err,
#         n_bootstrap=n_bootstrap,
#         random_state=random_state,
#     )

#     return {
#         "core_sum_samples": core_res.sum_samples,
#         "core_sum": core_res.sum_med,
#         "core_sum_err": core_res.sum_std,
#     }
    
    
    

def _compute_metrics_for_image_continuum(
    x: np.ndarray,
    y: np.ndarray,
    core_window: tuple[float, float],
    y_err: np.ndarray | None = None,
    n_bootstrap: int = 5000,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Compute bootstrap metrics (sum over y) for one or many spectra.

    Parameters
    ----------
    x : np.ndarray
        1D array (n_pix,) or 2D array (N, n_pix) with the x-axis (e.g. wavelength).
    y : np.ndarray
        1D array (n_pix,) or 2D array (N, n_pix) with the y-axis (e.g. flux).
    core_window : (float, float)
        [xmin, xmax] over which the sum is computed.
    y_err : np.ndarray or None, optional
        1σ uncertainties on y. Accepted shapes:
          - None
          - (n_pix,)
          - (N, n_pix)
        If provided, bootstrap draws y' ~ N(y, y_err) for the resampled points.
    n_bootstrap : int, optional
        Number of bootstrap realizations.
    random_state : int or None, optional
        Seed for RNG.

    Returns
    -------
    dict
        {
          "core_sum_samples": np.ndarray,  # shape (N, n_bootstrap)
          "core_sum": np.ndarray,          # shape (N,)
          "core_sum_err": np.ndarray,      # shape (N,)
        }

        For 1D inputs, N = 1 (i.e. length-1 arrays).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Normalize shapes: allow 1D or 2D, but end up with (N, n_pix)
    if x.ndim == 1 and y.ndim == 1:
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        x_2d = x[None, :]
        y_2d = y[None, :]
    elif x.ndim == 2 and y.ndim == 2 and x.shape == y.shape:
        x_2d, y_2d = x, y
    else:
        raise ValueError(
            f"x and y must both be 1D (n_pix,) or 2D (N, n_pix) with the same shape; "
            f"got x.shape={x.shape}, y.shape={y.shape}"
        )

    n_obj, n_pix = y_2d.shape

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

    # Allocate outputs
    core_sum = np.empty(n_obj, dtype=float)
    core_sum_err = np.empty(n_obj, dtype=float)

    core_sum_samples = None  # we’ll allocate once we know n_bootstrap from the first call

    for i, (x_row, y_row) in enumerate(zip(x_2d, y_2d)):
        if y_err_2d is not None:
            y_err_row = y_err_2d[i]
        else:
            y_err_row = None

        core_res = _bootstrap_sum_y(
            x=x_row,
            y=y_row,
            window=core_window,
            y_err=y_err_row,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        # Allocate samples array on first iteration
        if core_sum_samples is None:
            n_samp = len(core_res.sum_samples)
            core_sum_samples = np.empty((n_obj, n_samp), dtype=float)

        core_sum_samples[i] = np.asarray(core_res.sum_samples, dtype=float)
        core_sum[i] = core_res.sum_med
        core_sum_err[i] = core_res.sum_std

    return {
        "core_sum_samples": core_sum_samples,
        "core_sum": core_sum,
        "core_sum_err": core_sum_err,
    }