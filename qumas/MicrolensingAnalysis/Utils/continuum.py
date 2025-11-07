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




def _compute_metrics_for_image_continuum(
    x: np.ndarray,
    y: np.ndarray,
    core_window: tuple[float, float],
    y_err: np.ndarray | None = None,
    n_bootstrap: int = 5000,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Compute bootstrap metrics (sum over y) for a single image/spectrum.

    Parameters
    ----------
    x : np.ndarray
        X axis (e.g. wavelength).
    y : np.ndarray
        Y axis (e.g. flux).
    core_window : (float, float)
        [xmin, xmax] over which the sum is computed.
    y_err : np.ndarray or None, optional
        1σ uncertainties on y. If provided, bootstrap draws
        y' ~ N(y, y_err) for the resampled points.
    n_bootstrap : int, optional
        Number of bootstrap realizations.
    random_state : int or None, optional
        Seed for RNG.

    Returns
    -------
    dict
        {
          "core_sum_samples": np.ndarray,  # all bootstrap sums
          "core_sum": float,               # median of bootstrap sums
          "core_sum_err": float,           # std (ddof=1) of bootstrap sums
        }
    """
    core_res = _bootstrap_sum_y(
        x=x,
        y=y,
        window=core_window,
        y_err=y_err,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    return {
        "core_sum_samples": core_res.sum_samples,
        "core_sum": core_res.sum_med,
        "core_sum_err": core_res.sum_std,
    }