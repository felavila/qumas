import numpy as np 
from typing import Any
from qumas.MicrolensingAnalysis.functions import linear_model,linear_func


def calculate_line(X, Y, window_range_left, window_range_right):
    """
    Parameters
    ----------
    X : (N, M) array or list of arrays
    Y : (N, M) array or list of arrays
    window_range_left, window_range_right : (lo, hi)

    Returns
    -------
    Y_wo : (N, M) ndarray
        Continuum-subtracted spectra per object.
    CONT : (N, M) ndarray
        Fitted linear continuum evaluated over full X per object.
    """
    Y_wo = []
    CONT = []

    for x, y in zip(X, Y):
        mask_lc = (x >= window_range_left[0]) & (x <= window_range_left[1])
        mask_rc = (x >= window_range_right[0]) & (x <= window_range_right[1])

        x_combined = np.concatenate((x[mask_lc], x[mask_rc]))
        y_combined = np.concatenate((y[mask_lc], y[mask_rc]))

        slope_fit, intercept_fit = linear_model(x_combined, y_combined)

        cont = linear_func(x, slope_fit, intercept_fit)       # continuum on full grid
        y_wo = y - cont                                       # continuum-subtracted

        CONT.append(cont)
        Y_wo.append(y_wo)

    return np.asarray(Y_wo), np.asarray(CONT)


def _linear_params(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """NaN-safe linear fit y ~ m x + b. Falls back to flat line if too few points."""
        msk = np.isfinite(x) & np.isfinite(y)
        if msk.sum() < 2:
            return 0.0, (float(np.nanmedian(y[msk])) if msk.any() else 0.0)
        m, b = np.polyfit(x[msk], y[msk], 1)
        return float(m), float(b)
    
    
def _area_under_line(m: float, b: float, a: float, c: float) -> float:
    """Analytic ∫(m x + b) dx from a to c."""
    return m * 0.5 * (c**2 - a**2) + b * (c - a)

def _trapz_in_range(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Trapz integral of y over x in [lo, hi]."""
    lo, hi = sorted((lo, hi))
    m = (x >= lo) & (x <= hi)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(y[m], x[m]))

def convert_none_to_nan(item):
    if item is None:
        return np.nan
    elif isinstance(item, list):
        return [convert_none_to_nan(x) for x in item]
    elif isinstance(item, dict):
        return {key: convert_none_to_nan(value) if isinstance(value[0],str) else convert_none_to_nan(value) for key,value in item.items()}
    else:
        return item
    