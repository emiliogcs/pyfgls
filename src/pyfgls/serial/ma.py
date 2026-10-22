"""GMA: feasible GLS for MA(1) errors ([PG26] Section 5).

For ``u_t = e_t + theta e_{t-1}`` the GLS transformation inverts the MA filter:

    y*_t = sum_{j=0}^{t-1} (-theta)^j y_{t-j}     (i.e. y*_t = y_t - theta y*_{t-1}),

for ``t = 1, ..., T`` and likewise for every column of ``X``; then
``y*_t = x*_t' beta + e_t`` is estimated by OLS with ``Var = sigma_e^2 (X*'X*)^{-1}``.

``theta`` is the exact Gaussian maximum-likelihood estimate (innovations algorithm,
``sigma^2`` profiled out) computed from ``u_t = y_t - x_t' beta_tilde``, where
``beta_tilde`` is the Durbin-regression estimate ([PG26] eq. (14)) with
``k_T = int(4 (T/100)^{2/9})`` lags — consistent whether or not the regressors are
exogenous, unlike OLS.

GMA is useful when prior knowledge says the errors are MA(1) with a large negative
coefficient (e.g. two-step-ahead forecast errors), where the BIC-selected AR
approximation may need many lags ([PG26] Section 6).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal

from pyfgls._linalg import FloatArray, inv_psd, lstsq
from pyfgls._validation import RegressionData
from pyfgls.exceptions import ConvergenceWarning, NonStationaryWarning
from pyfgls.results import FGLSResults, SerialInfo
from pyfgls.serial.durbin import durbin_design, fit_durbin

#: theta is searched in [-THETA_BOUND, THETA_BOUND].
THETA_BOUND = 0.995


def gma_durbin_order(T: int) -> int:
    """Order of the preliminary Durbin regression, ``k_T = int(4 (T/100)^{2/9})``."""
    return int(4.0 * (T / 100.0) ** (2.0 / 9.0))


def ma1_negloglik(theta: float, u: FloatArray) -> float:
    """Profile negative Gaussian log-likelihood of an MA(1) (innovations algorithm)."""
    n = u.size
    g0, g1 = 1.0 + theta**2, theta
    r = np.empty(n)
    innov = np.empty(n)
    r[0] = g0
    innov[0] = u[0]
    for t in range(1, n):
        th = g1 / r[t - 1]
        r[t] = g0 - th**2 * r[t - 1]
        innov[t] = u[t] - th * innov[t - 1]
    s2 = float(np.mean(innov**2 / r))
    return float(0.5 * n * np.log(s2) + 0.5 * np.sum(np.log(r)))


def ma1_mle(u: FloatArray) -> float:
    """Exact Gaussian MLE of ``theta`` in ``u_t = e_t + theta e_{t-1}``."""
    uc = u - u.mean()
    res = scipy.optimize.minimize_scalar(
        ma1_negloglik,
        bounds=(-THETA_BOUND, THETA_BOUND),
        args=(uc,),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not res.success:  # pragma: no cover
        warnings.warn(
            "MA(1) likelihood maximisation did not converge.", ConvergenceWarning, stacklevel=2
        )
    theta = float(res.x)
    if abs(theta) > 0.98:
        warnings.warn(
            f"Estimated MA coefficient {theta:.3f} is close to the non-invertibility boundary.",
            NonStationaryWarning,
            stacklevel=2,
        )
    return theta


def inverse_ma_filter(a: FloatArray, theta: float) -> FloatArray:
    """``a*_t = a_t - theta a*_{t-1}`` with ``a*_{-1} = 0`` (column-wise)."""
    return np.asarray(scipy.signal.lfilter([1.0], [1.0, theta], a, axis=0), dtype=np.float64)


@dataclass(frozen=True)
class GMAEstimate:
    """GMA estimates."""

    beta: FloatArray
    cov: FloatArray
    theta: float
    durbin_order: int
    ystar: FloatArray
    Xstar: FloatArray
    ehat: FloatArray
    uhat: FloatArray

    @property
    def n_eff(self) -> int:
        """Observations in the transformed regression (all ``T``)."""
        return int(self.ystar.shape[0])


def estimate_gma(
    y: FloatArray,
    X: FloatArray,
    x_names: Sequence[str] | None = None,
    *,
    durbin_order: int | None = None,
) -> GMAEstimate:
    """GMA estimator of [PG26] Section 5 for MA(1) errors.

    Parameters
    ----------
    y, X : ndarray
    x_names : sequence of str, optional
    durbin_order : int, optional
        Lags in the preliminary Durbin regression (default ``int(4 (T/100)^{2/9})``).

    Examples
    --------
    >>> from pyfgls.simulate import serial_dgp
    >>> sim = serial_dgp(400, ar=(), ma=(0.6,), gamma=0.0, burn=50, random_state=1)
    >>> est = estimate_gma(sim.y, sim.X)
    >>> round(est.theta, 1)
    0.6
    """
    T, q = X.shape
    names = list(x_names) if x_names is not None else [f"x{j}" for j in range(q)]
    kT = gma_durbin_order(T) if durbin_order is None else int(durbin_order)
    beta_tilde = fit_durbin(durbin_design(y, X, names, kT)).beta
    theta = ma1_mle(y - X @ beta_tilde)
    ys = inverse_ma_filter(y, theta)
    Xs = inverse_ma_filter(X, theta)
    beta = lstsq(Xs, ys)
    e = ys - Xs @ beta
    n = ys.size
    s2 = float(e @ e) / (n - q)
    cov = s2 * inv_psd(Xs.T @ Xs, name="GMA transformed X'X")
    return GMAEstimate(
        beta=beta,
        cov=cov,
        theta=theta,
        durbin_order=kT,
        ystar=ys,
        Xstar=Xs,
        ehat=e,
        uhat=y - X @ beta,
    )


def build_gma_results(
    data: RegressionData, est: GMAEstimate, *, use_t: bool, options: dict[str, Any]
) -> FGLSResults:
    """Assemble :class:`FGLSResults` for the GMA estimator."""
    names = list(data.x_names)
    idx = data.index if data.index is not None else pd.RangeIndex(data.nobs)
    info = SerialInfo(
        ar_order=0,
        rho=pd.Series(np.zeros(0), name="rho", dtype=np.float64),
        method="ma1",
        kmax=None,
        ic_path=None,
        durbin_params=None,
        dropped_lags=(),
        nonexogeneity_by_coef=pd.Series(np.zeros(len(names)), index=names),
        extra={"MA coef theta": est.theta, "Durbin order k_T": est.durbin_order},
    )
    return FGLSResults(
        params=pd.Series(est.beta, index=names, name="params"),
        cov=pd.DataFrame(est.cov, index=names, columns=names),
        method="GMA (MA(1))",
        cov_type="nonrobust",
        nobs=data.nobs,
        n_eff=est.n_eff,
        use_t=use_t,
        resid=pd.Series(est.uhat, index=idx, name="resid"),
        resid_final=pd.Series(est.ehat, index=idx, name="innovations"),
        data=data,
        serial=info,
        options=options,
        notes=("GMA for MA(1) errors ([PG26] Section 5); covariance sigma^2 (X*'X*)^{-1}.",),
        estimate=est,
    )
