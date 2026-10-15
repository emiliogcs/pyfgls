"""Variance estimators for the serial-correlation FGLS estimator ([PG26] Theorem 2).

Notation (effective sample ``t = k+1..T``, ``n = T - k``):

* ``xf_t``: filtered regressors ``x_t - sum_j rho_j x_{t-j}``;
* ``u_t``: *unfiltered* FGLS residuals ``y_t - x_t' beta``;
* ``e_t``: residuals of the filtered regression, ``sigma2 = n^{-1} sum e_t^2``;
* ``Q = n^{-1} sum xf_t xf_t'``.

Naive variance, eq. (27) — valid only for exogenous regressors::

    AVar_N = sigma2 * Q^{-1}

Exogeneity-robust variance, eqs. (28)--(29) — the default::

    g_j   = n^{-1} sum_t xf_t u_{t-j},   G = [g_1, ..., g_k]                (q x k)
    psi_t = residual of (y_{t-1}, ..., y_{t-k})' regressed on the other
            Durbin regressors W_t = (x_t', x_{t-1}', ..., x_{t-k}')'           (k x 1)
    Sigma = n^{-1} sum_t psi_t psi_t'
    AVar_R = sigma2 * (Q^{-1} + Q^{-1} G Sigma^{-1} G' Q^{-1})

and ``Var(beta) = AVar / n``. The second term accounts for the first-stage estimation
of the AR coefficients, which matters exactly when the regressors are not exogenous
(``G != 0``). It is positive semidefinite by construction.

The same correction can be written through the influence function of the estimator,

    a_t = xf_t - G C_t,    C = J (Z'Z/n)^{-1} Z',

with ``Z`` the Durbin design and ``J`` the Jacobian of ``rho`` with respect to the
Durbin coefficients. When ``rho`` are the coefficients on the lagged ``y`` (``J`` a
selector), ``C_t = Sigma^{-1} psi_t`` and ``n^{-1} sum a_t a_t' = Q + G Sigma^{-1} G'``
exactly, so ``AVar_R = sigma2 Q^{-1} (n^{-1} sum a_t a_t') Q^{-1}``. This general form
is used (i) when ``rho`` is recovered by the Wallis ratio (lagged dependent variables,
[PG26] Section 4.2) and (ii) for the heteroskedasticity-robust ``HC0``--``HC3``
variants, which replace ``sigma2`` by ``e_t^2`` inside the sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg

from pyfgls._linalg import FloatArray, hat_diag, inv_psd, lstsq, make_psd
from pyfgls.serial.durbin import DurbinDesign

SerialCovType = Literal["robust", "naive", "HC0", "HC1", "HC2", "HC3"]
SERIAL_COV_TYPES: tuple[str, ...] = ("robust", "naive", "HC0", "HC1", "HC2", "HC3")


@dataclass(frozen=True)
class SerialVariance:
    """Variance of the serial-correlation FGLS estimator.

    Attributes
    ----------
    cov : ndarray (q, q)
        Estimated ``Var(beta_hat) = AVar / n``.
    avar : ndarray (q, q)
        Asymptotic variance of ``sqrt(n) (beta_hat - beta)``.
    naive_avar : ndarray (q, q)
        ``sigma2 Q^{-1}`` ([PG26] eq. (27)).
    correction : ndarray (q, q)
        ``AVar_R - AVar_N``: the first-stage (non-exogeneity) correction.
    G : ndarray (q, k)
        ``[g_1, ..., g_k]``.
    Sigma : ndarray (k, k) or None
        Variance of ``psi_t`` (standard case only).
    sigma2 : float
    n : int
    cov_type : str
    """

    cov: FloatArray
    avar: FloatArray
    naive_avar: FloatArray
    correction: FloatArray
    G: FloatArray
    Sigma: FloatArray | None
    sigma2: float
    n: int
    cov_type: str

    @property
    def nonexogeneity_by_coef(self) -> FloatArray:
        """Correction relative to the naive variance, coefficient by coefficient.

        ``(AVar_R - AVar_N)_ii / (AVar_N)_ii``; zero when the regressors behave as
        exogenous.
        """
        d = np.diag(self.naive_avar)
        out = np.divide(np.diag(self.correction), d, out=np.zeros_like(d), where=d > 0).astype(
            np.float64
        )
        return out

    @property
    def nonexogeneity_share(self) -> float:
        """Average of :attr:`nonexogeneity_by_coef` (scale-free summary)."""
        v = self.nonexogeneity_by_coef
        return float(v.mean()) if v.size else 0.0


def g_matrix(Xf: FloatArray, uhat: FloatArray, start: int, k: int) -> FloatArray:
    """``G = [g_1..g_k]`` with ``g_j = n^{-1} sum_t xf_t u_{t-j}`` ([PG26] eq. (28)).

    ``Xf`` holds rows ``t = start..T-1`` and ``uhat`` the unfiltered residuals on the
    full sample; the *raw* lagged residual ``u_{t-j}`` is used.
    """
    n, q = Xf.shape
    T = uhat.shape[0]
    G = np.empty((q, k))
    for j in range(1, k + 1):
        G[:, j - 1] = Xf.T @ uhat[start - j : T - j] / n
    return G


def psi_residuals(design: DurbinDesign, k: int) -> FloatArray:
    """Residuals of the lagged ``y`` regressed on the other Durbin regressors.

    ``psi_t`` of [PG26] eq. (29): the part of ``(y_{t-1}, ..., y_{t-k})`` not explained
    by ``W_t = (x_t', x_{t-1}', ..., x_{t-k}')'`` (collinear lagged columns already
    removed from the design).
    """
    Z = design.prefix(k)
    ycols = design.ylag_columns(k)
    other = [c for c in range(Z.shape[1]) if c not in set(ycols)]
    Yl = Z[:, ycols]
    W = Z[:, other]
    return np.asarray(Yl - W @ lstsq(W, Yl), dtype=np.float64)


def influence_weights(Z: FloatArray, J: FloatArray) -> FloatArray:
    """``C = J (Z'Z/n)^{-1} Z'`` (k x n): first-stage influence of the Durbin regression."""
    n = Z.shape[0]
    Q, R = np.linalg.qr(Z)
    JRinv = scipy.linalg.solve_triangular(R, J.T, trans="T", lower=False).T  # J R^{-1}
    return np.asarray(n * (JRinv @ Q.T), dtype=np.float64)


def _hc_weights(cov_type: str, ehat: FloatArray, Xf: FloatArray) -> FloatArray:
    n, q = int(Xf.shape[0]), int(Xf.shape[1])
    e2: FloatArray = ehat**2
    if cov_type == "HC0":
        return e2
    if cov_type == "HC1":
        return e2 * (n / max(n - q, 1))
    h: FloatArray = np.clip(hat_diag(Xf), 0.0, 1.0 - 1e-12)
    if cov_type == "HC2":
        return e2 / (1.0 - h)
    return e2 / (1.0 - h) ** 2


def serial_variance(
    Xf: FloatArray,
    ehat: FloatArray,
    uhat: FloatArray,
    design: DurbinDesign,
    *,
    J: FloatArray | None = None,
    cov_type: str = "robust",
) -> SerialVariance:
    """Variance of the FGLS estimator with an estimated AR(k) filter.

    Parameters
    ----------
    Xf : ndarray (n, q)
        Filtered regressors on ``t = k..T-1``.
    ehat : ndarray (n,)
        Residuals of the filtered regression.
    uhat : ndarray (T,)
        Unfiltered FGLS residuals.
    design : DurbinDesign
        Durbin design of order ``k`` on the same sample (``design.start == k``).
    J : ndarray (k, p), optional
        Jacobian of ``rho`` with respect to the Durbin coefficients. ``None`` means
        ``rho`` are the coefficients on the lagged ``y`` (the standard case, eq. (29)).
    cov_type : {"robust", "naive", "HC0", "HC1", "HC2", "HC3"}

    Returns
    -------
    SerialVariance
    """
    n, q = Xf.shape
    k = design.order
    start = design.start
    sigma2 = float(ehat @ ehat / n)
    Qinv = inv_psd(Xf.T @ Xf / n, name="filtered regressor moment matrix")
    naive = sigma2 * Qinv
    G = g_matrix(Xf, uhat, start, k)

    if k == 0:
        C: FloatArray = np.zeros((0, n))
        Sigma: FloatArray | None = np.zeros((0, 0))
    elif J is None:
        Psi = psi_residuals(design, k)
        Sigma = Psi.T @ Psi / n
        C = inv_psd(Sigma, name="variance of psi_t (eq. 29)") @ Psi.T
    else:
        Sigma = None
        C = influence_weights(design.prefix(k), J)

    if cov_type == "naive":
        avar = naive
    elif cov_type == "robust" and J is None:
        # [PG26] eq. (29), literally
        if k == 0:
            avar = naive
        else:
            assert Sigma is not None
            corr = Qinv @ G @ inv_psd(Sigma, name="variance of psi_t (eq. 29)") @ G.T @ Qinv
            avar = sigma2 * (Qinv + make_psd(corr, name="eq. (29) correction"))
    else:
        A = Xf - C.T @ G.T  # influence a_t, rows t
        if cov_type == "robust":
            meat = sigma2 * (A.T @ A) / n
        elif cov_type in ("HC0", "HC1", "HC2", "HC3"):
            w = _hc_weights(cov_type, ehat, Xf)
            meat = (A * w[:, None]).T @ A / n
        else:
            raise ValueError(f"unknown cov_type {cov_type!r}")
        avar = Qinv @ meat @ Qinv
    avar = make_psd(avar, name="FGLS covariance matrix")
    return SerialVariance(
        cov=avar / n,
        avar=avar,
        naive_avar=naive,
        correction=avar - naive if cov_type == "robust" else np.zeros((q, q)),
        G=G,
        Sigma=Sigma,
        sigma2=sigma2,
        n=n,
        cov_type=cov_type,
    )
