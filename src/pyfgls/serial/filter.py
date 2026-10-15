"""Quasi-differencing and the second-step FGLS regression ([PG26] eq. (15))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyfgls._linalg import FloatArray, lstsq
from pyfgls._utils import ar_filter


@dataclass(frozen=True)
class FilteredRegression:
    """OLS on quasi-differenced data.

    Attributes
    ----------
    beta : ndarray, shape (q,)
        FGLS coefficients. Because every column (including a constant) is filtered,
        the intercept is reported on the original scale.
    yf, Xf : ndarray
        Filtered data ``y_t - sum_j rho_j y_{t-j}`` and ``x_t - sum_j rho_j x_{t-j}``,
        ``t = k, ..., T-1`` (0-based).
    ehat : ndarray
        Residuals of the filtered regression (estimated innovations).
    uhat : ndarray, shape (T,)
        Unfiltered residuals ``y_t - x_t' beta`` on the full sample.
    start : int
        Number of dropped initial observations (``k``).
    """

    beta: FloatArray
    yf: FloatArray
    Xf: FloatArray
    ehat: FloatArray
    uhat: FloatArray
    start: int

    @property
    def n_eff(self) -> int:
        """Effective sample size ``T - k``."""
        return int(self.yf.shape[0])

    @property
    def sigma2(self) -> float:
        """``n_eff^{-1} sum ehat^2`` ([PG26] notation ``sigma^2_{e,k}``)."""
        return float(self.ehat @ self.ehat / self.n_eff)


def quasi_difference(
    y: FloatArray, X: FloatArray, rho: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Apply ``A(L) = 1 - sum_j rho_j L^j`` to ``y`` and every column of ``X``.

    The first ``k = len(rho)`` observations are dropped (no Prais--Winsten
    transformation of the initial observations, as in [PG26]).
    """
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    return ar_filter(y, r), ar_filter(X, r)


def filtered_regression(y: FloatArray, X: FloatArray, rho: npt.ArrayLike) -> FilteredRegression:
    """Second FGLS step: OLS of the quasi-differenced ``y`` on the quasi-differenced ``X``.

    With known ``rho`` this is exactly GLS for AR(k) errors conditional on the first
    ``k`` observations.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([np.ones(200), rng.normal(size=200)])
    >>> u = np.zeros(200)
    >>> e = rng.normal(size=200)
    >>> for t in range(1, 200):
    ...     u[t] = 0.5 * u[t - 1] + e[t]
    >>> fr = filtered_regression(X @ [1.0, 2.0] + u, X, [0.5])
    >>> fr.n_eff
    199
    >>> round(float(fr.beta[1]), 1)
    2.0
    """
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    yf, Xf = quasi_difference(y, X, r)
    beta = lstsq(Xf, yf)
    return FilteredRegression(
        beta=beta,
        yf=yf,
        Xf=Xf,
        ehat=yf - Xf @ beta,
        uhat=y - X @ beta,
        start=r.size,
    )
