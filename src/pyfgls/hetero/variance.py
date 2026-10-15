"""Sandwich variances for FGLS with an estimated skedastic function ([GP24] Section 4).

Because the skedastic model is only an approximation, inference uses the sandwich
([GP24] eq. (6))

    Var(b) = (T^{-1} X'W^{-1}X)^{-1} Omega (T^{-1} X'W^{-1}X)^{-1} / T,
    Omega  = T^{-1} sum_t Sigma_tt x_t x_t',

with ``W = diag(v_t)`` and, for the Lasso and SVR skedastic fits, the Miller--Startz
type bias correction of eq. (8):

    Sigma_tt = (u_t^2 / v_t^2) * ( 1/(1 - h_t^FGLS)^2 + 4 (h_t^OLS / k) df ),

``u_t = y_t - x_t' b_FGLS``, ``h_t^FGLS = [X (X'W^{-1}X)^{-1} X'W^{-1}]_tt``,
``h_t^OLS = [X(X'X)^{-1}X']_tt``, ``k`` the number of regressors and ``df`` the
degrees of freedom of the skedastic fit (non-zero Lasso coefficients including the
intercept, or the SVR count).

For the Romano--Wolf WLS fits (eq. (7)), ``Sigma_tt = e_t^2 / v_t^2`` with the HC3-type
OLS residual ``e_t^2 = u_t^{OLS 2} / (1 - h_t^OLS)^2`` (open item B: the printed
definition is read as HC3 on the OLS residuals).

The same code serves the combined estimator ([PG26] Supplement eqs. (A.4)--(A.5)),
where ``X`` holds the *filtered* regressors and "OLS" is the unweighted filtered (F-C)
regression.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pyfgls._linalg import FloatArray, hat_diag, inv_psd, make_psd

HeteroCovType = Literal["robust", "nonrobust", "HC0", "HC1", "HC2", "HC3"]
HETERO_COV_TYPES: tuple[str, ...] = ("robust", "nonrobust", "HC0", "HC1", "HC2", "HC3")


def weighted_leverage(X: FloatArray, v: FloatArray) -> FloatArray:
    """``h_t = [X (X'W^{-1}X)^{-1} X'W^{-1}]_tt`` = leverages of the rows ``x_t / sqrt(v_t)``."""
    return hat_diag(X / np.sqrt(v)[:, None])


def sandwich(X: FloatArray, v: FloatArray, sigma_tt: FloatArray) -> FloatArray:
    """``(T^{-1}X'W^{-1}X)^{-1} Omega (T^{-1}X'W^{-1}X)^{-1} / T`` with ``Omega = T^{-1} X'diag(sigma)X``."""
    T = X.shape[0]
    A = inv_psd((X / v[:, None]).T @ X / T, name="weighted moment matrix X'W^-1 X")
    Omega = (X * sigma_tt[:, None]).T @ X / T
    return make_psd(A @ Omega @ A / T, name="FGLS sandwich covariance")


def hetero_sigma(
    cov_type: str,
    *,
    X: FloatArray,
    v: FloatArray,
    u_fgls: FloatArray,
    u_ols: FloatArray,
    h_ols: FloatArray,
    df: float,
    method: str,
) -> FloatArray:
    """Diagonal ``Sigma_tt`` of the middle matrix for each covariance type.

    Parameters
    ----------
    cov_type : {"robust", "HC0", "HC1", "HC2", "HC3"}
        ``"robust"`` is eq. (8) for Lasso/SVR fits, eq. (7) for WLS-S1/S2 fits and
        HC3 otherwise (known variances, OLS).
    X : regressors (T, k); v : variances (T,)
    u_fgls : FGLS residuals ``y - X b_FGLS``; u_ols : unweighted (OLS) residuals
    h_ols : unweighted leverages; df : skedastic degrees of freedom
    method : skedastic method name
    """
    T, k = X.shape
    h_w = np.clip(weighted_leverage(X, v), 0.0, 1.0 - 1e-12)
    base = u_fgls**2 / v**2
    if cov_type == "robust":
        if method in ("lasso", "svr"):
            # [GP24] eq. (8): Miller-Startz / Rothenberg bias correction
            return np.asarray(base * (1.0 / (1.0 - h_w) ** 2 + 4.0 * (h_ols / k) * df))
        if method in ("wls_s1", "wls_s2"):
            # [GP24] eq. (7): HC3-type OLS residuals
            h_o = np.clip(h_ols, 0.0, 1.0 - 1e-12)
            return np.asarray(u_ols**2 / (1.0 - h_o) ** 2 / v**2)
        cov_type = "HC3"
    if cov_type == "HC0":
        return np.asarray(base)
    if cov_type == "HC1":
        return np.asarray(base * (T / max(T - k, 1)))
    if cov_type == "HC2":
        return np.asarray(base / (1.0 - h_w))
    if cov_type == "HC3":
        return np.asarray(base / (1.0 - h_w) ** 2)
    raise ValueError(f"unknown cov_type {cov_type!r}")


def hetero_covariance(
    cov_type: str,
    *,
    X: FloatArray,
    v: FloatArray,
    u_fgls: FloatArray,
    u_ols: FloatArray,
    h_ols: FloatArray,
    df: float,
    method: str,
) -> FloatArray:
    """Covariance matrix of the weighted (FGLS) coefficients.

    ``"nonrobust"`` assumes the variance model is correct:
    ``s^2 (X'W^{-1}X)^{-1}`` with ``s^2`` the mean squared weighted residual.
    """
    T, k = X.shape
    if cov_type == "nonrobust":
        s2 = float(np.sum(u_fgls**2 / v) / max(T - k, 1))
        return s2 * inv_psd((X / v[:, None]).T @ X, name="weighted moment matrix")
    sig = hetero_sigma(
        cov_type, X=X, v=v, u_fgls=u_fgls, u_ols=u_ols, h_ols=h_ols, df=df, method=method
    )
    return sandwich(X, v, sig)
