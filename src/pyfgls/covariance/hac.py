"""Heteroskedasticity and autocorrelation consistent (HAC) covariance for OLS.

* Andrews (1991) quadratic-spectral kernel
  ``k(x) = (3/z^2) (sin z / z - cos z)``, ``z = 6 pi x / 5`` with the automatic
  bandwidth ``S_T = 1.3221 (alpha(2) T)^{1/5}`` from AR(1) approximations of the
  score components (the benchmark of [PG26]);
* Newey and West (1987) Bartlett kernel, with bandwidth
  ``floor(4 (T/100)^{2/9})`` by default or Andrews' automatic choice
  ``1.1447 (alpha(1) T)^{1/3}``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pyfgls._linalg import FloatArray, inv_psd, lstsq, make_psd
from pyfgls.exceptions import InputError

Kernel = Literal["qs", "bartlett"]


def qs_kernel(x: FloatArray) -> FloatArray:
    """Quadratic-spectral kernel ``(3/z^2)(sin z/z - cos z)``, ``z = 6 pi x / 5``."""
    xa = np.asarray(x, dtype=np.float64)
    out = np.ones_like(xa)
    nz = xa != 0
    z = 6.0 * np.pi * xa[nz] / 5.0
    out[nz] = 3.0 / z**2 * (np.sin(z) / z - np.cos(z))
    return out


def bartlett_kernel(x: FloatArray) -> FloatArray:
    """Bartlett kernel ``max(1 - |x|, 0)``."""
    return np.asarray(np.clip(1.0 - np.abs(x), 0.0, None), dtype=np.float64)


def andrews_bandwidth(
    scores: FloatArray, kernel: Kernel = "qs", *, skip: int | None = None
) -> float:
    """Andrews (1991) automatic bandwidth from AR(1) fits to each score column.

    ``skip`` excludes one column (conventionally the intercept score) from the
    weighting, as Andrews suggests.
    """
    T, q = scores.shape
    num = den = 0.0
    for a in range(q):
        if skip is not None and a == skip and q > 1:
            continue
        s = scores[:, a]
        s0, s1 = s[:-1], s[1:]
        denom = float(s0 @ s0)
        rho = float(s0 @ s1) / denom if denom > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        sig2 = float(np.mean((s1 - rho * s0) ** 2))
        if kernel == "qs":
            num += 4.0 * rho**2 * sig2**2 / (1.0 - rho) ** 8
        else:
            num += 4.0 * rho**2 * sig2**2 / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2)
        den += sig2**2 / (1.0 - rho) ** 4
    if den <= 0:
        return 1.0
    alpha = num / den
    if kernel == "qs":
        return float(1.3221 * (alpha * T) ** 0.2)
    return float(1.1447 * (alpha * T) ** (1.0 / 3.0))


def newey_west_lags(T: int) -> int:
    """Rule-of-thumb Bartlett truncation ``floor(4 (T/100)^{2/9})``."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def long_run_covariance(
    scores: FloatArray,
    *,
    kernel: Kernel = "qs",
    bandwidth: float | Literal["andrews", "nw"] = "andrews",
    skip: int | None = None,
) -> tuple[FloatArray, float]:
    """Kernel estimate of ``lim Var(T^{-1/2} sum_t s_t)`` and the bandwidth used."""
    T, _ = scores.shape
    if kernel not in ("qs", "bartlett"):
        raise InputError(f"kernel must be 'qs' or 'bartlett', got {kernel!r}.")
    if bandwidth == "andrews":
        bw = andrews_bandwidth(scores, kernel, skip=skip)
    elif bandwidth == "nw":
        bw = (
            float(newey_west_lags(T) + 1)
            if kernel == "bartlett"
            else andrews_bandwidth(scores, kernel, skip=skip)
        )
    else:
        bw = float(bandwidth)
        if bw <= 0:
            raise InputError("bandwidth must be positive.")
    S = scores.T @ scores / T
    lags = np.arange(1, T)
    w = qs_kernel(lags / bw) if kernel == "qs" else bartlett_kernel(lags / bw)
    for j, wj in zip(lags, w, strict=True):
        if kernel == "bartlett" and wj == 0.0:
            break
        if abs(wj) < 1e-12 and kernel == "qs" and j > 50 * bw:
            break
        G = scores[j:].T @ scores[:-j] / T
        S += wj * (G + G.T)
    return make_psd(S, name="long-run covariance"), bw


def hac_covariance(
    X: FloatArray,
    resid: FloatArray,
    *,
    kernel: Kernel = "qs",
    bandwidth: float | Literal["andrews", "nw"] = "andrews",
    small_sample: bool = False,
    const_col: int | None = None,
) -> tuple[FloatArray, float]:
    """HAC covariance ``(X'X)^{-1} T S (X'X)^{-1}`` of OLS coefficients.

    Returns the covariance and the bandwidth. ``small_sample`` multiplies by
    ``T / (T - q)``.
    """
    T, q = X.shape
    S, bw = long_run_covariance(
        X * resid[:, None], kernel=kernel, bandwidth=bandwidth, skip=const_col
    )
    A = inv_psd(X.T @ X / T, name="X'X")
    V = A @ S @ A / T
    if small_sample:
        V = V * T / (T - q)
    return make_psd(V, name="HAC covariance"), bw


def ols_with_hac(
    y: FloatArray, X: FloatArray, **kwargs: object
) -> tuple[FloatArray, FloatArray, float]:
    """OLS coefficients with their HAC covariance (convenience)."""
    b = lstsq(X, y)
    V, bw = hac_covariance(X, y - X @ b, **kwargs)  # type: ignore[arg-type]
    return b, V, bw
