"""White-type heteroskedasticity-consistent covariances (HC0--HC3) for OLS."""

from __future__ import annotations

import numpy as np

from pyfgls._linalg import FloatArray, hat_diag, inv_psd, make_psd
from pyfgls.exceptions import InputError


def hc_covariance(X: FloatArray, resid: FloatArray, cov_type: str = "HC3") -> FloatArray:
    """``(X'X)^{-1} X' diag(w_t u_t^2) X (X'X)^{-1}`` with HC0--HC3 weights.

    ``"nonrobust"`` gives ``s^2 (X'X)^{-1}`` with ``s^2 = u'u / (T - q)``.
    """
    T, q = X.shape
    A = inv_psd(X.T @ X, name="X'X")
    u2 = resid**2
    if cov_type == "nonrobust":
        return np.asarray(float(u2.sum()) / (T - q) * A, dtype=np.float64)
    if cov_type == "HC0":
        w = u2
    elif cov_type == "HC1":
        w = u2 * T / (T - q)
    elif cov_type in ("HC2", "HC3"):
        h = np.clip(hat_diag(X), 0.0, 1.0 - 1e-12)
        w = u2 / (1.0 - h) if cov_type == "HC2" else u2 / (1.0 - h) ** 2
    else:
        raise InputError(f"Unknown cov_type {cov_type!r} (HC0-HC3 or nonrobust).")
    return make_psd(A @ ((X * w[:, None]).T @ X) @ A, name="HC covariance")
