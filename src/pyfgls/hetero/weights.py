"""From a fitted log-variance to FGLS weights, with diagnostics."""

from __future__ import annotations

import warnings

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls.exceptions import InputError, WeightWarning

#: Warn when the largest fitted variance exceeds the smallest by this factor.
MAX_VARIANCE_RATIO = 1e6

#: Fitted log-variances are clipped to this range before exponentiation (overflow guard).
_LOG_CLIP = 700.0


def log_squared_residuals(u: FloatArray, delta: float) -> FloatArray:
    """Response of the skedastic regression, ``log(max(u_t^2, delta^2))`` ([GP24] Section 2).

    The floor ``delta`` (default 0.1, in the units of ``y``) avoids ``log 0``.
    """
    if not delta > 0:
        raise InputError(f"delta must be positive, got {delta}.")
    return np.asarray(np.log(np.maximum(u**2, delta**2)), dtype=np.float64)


def variances_from_log(
    h: FloatArray,
    *,
    clip: tuple[float, float] | None = None,
) -> FloatArray:
    """Turn fitted log-variances into variances normalised to mean one.

    Parameters
    ----------
    h : ndarray
        Fitted log-variance.
    clip : (lo, hi), optional
        Quantile winsorisation of ``h`` before exponentiation, e.g. ``(0.01, 0.99)``.
        Default ``None`` (as in the papers).

    Notes
    -----
    Normalising the variances (equivalently the weights) by a constant changes neither
    the FGLS coefficients nor the sandwich variance.
    """
    hh = np.asarray(h, dtype=np.float64)
    if clip is not None:
        lo, hi = clip
        if not 0.0 <= lo < hi <= 1.0:
            raise InputError(f"weight_clip must satisfy 0 <= lo < hi <= 1, got {clip}.")
        ql, qh = np.quantile(hh, [lo, hi])
        hh = np.clip(hh, ql, qh)
    hh = np.clip(hh - np.max(hh), -_LOG_CLIP, 0.0)  # max at 0: no overflow
    v = np.exp(hh)
    v = v / v.mean()
    ratio = float(v.max() / v.min())
    if ratio > MAX_VARIANCE_RATIO:
        warnings.warn(
            f"The fitted variances span a factor of {ratio:.3g}; a few observations get "
            "almost all the weight. Consider weight_clip=(0.01, 0.99), a smaller candidate "
            "set z, or delta_scale=True.",
            WeightWarning,
            stacklevel=3,
        )
    return np.asarray(v, dtype=np.float64)


def known_variances(v: FloatArray) -> FloatArray:
    """Validate user-supplied variances (positive, finite) and normalise to mean one."""
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise InputError("Known variances must be positive and finite.")
    return np.asarray(arr / arr.mean(), dtype=np.float64)
