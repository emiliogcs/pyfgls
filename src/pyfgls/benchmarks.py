"""Benchmark estimators used for comparisons, tests and the papers' simulations.

* :func:`ols` — OLS with HAC (Andrews quadratic-spectral or Newey--West) or HC
  standard errors;
* :func:`cochrane_orcutt` — the classic (iterated) Cochrane--Orcutt FGLS. **It is
  not valid with non-exogenous regressors**: it estimates the AR coefficient from OLS
  residuals, and OLS is inconsistent in that case ([PG26] Section 3);
* :func:`durbin` — the Durbin-regression estimator ``beta^D`` of [PG26] eq. (14);
* :func:`infeasible_gls` — GLS with known AR coefficients or a known covariance
  matrix (simulation benchmark only);
* :func:`hybrid_gls` — the "hybrid" estimator of [PG26] Supplement S-1 (exogenous
  regressors only).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
import scipy.stats

from pyfgls._builders import row_index
from pyfgls._linalg import FloatArray, inv_psd, lstsq
from pyfgls._utils import RandomState, ar_filter, check_random_state
from pyfgls._validation import RegressionData, check_choice, prepare_data
from pyfgls.covariance.hac import hac_covariance
from pyfgls.covariance.hc import hc_covariance
from pyfgls.exceptions import ConvergenceWarning, InputError
from pyfgls.results import FGLSResults, SerialInfo
from pyfgls.serial.durbin import durbin_design, fit_durbin
from pyfgls.serial.lagselect import KmaxRule, resolve_kmax, select_ar_order

__all__ = ["cochrane_orcutt", "durbin", "hybrid_gls", "infeasible_gls", "ols"]

OLSCovType = Literal["HAC", "NW", "HC0", "HC1", "HC2", "HC3", "nonrobust"]


def _data(y_or_data: Any, X: Any, time_series: bool) -> RegressionData:
    if isinstance(y_or_data, RegressionData):
        return y_or_data
    if X is None:
        raise InputError("X is required.")
    return prepare_data(y_or_data, X, time_series=time_series)


def _result(
    data: RegressionData,
    beta: FloatArray,
    cov: FloatArray,
    *,
    method: str,
    cov_type: str,
    use_t: bool,
    n_eff: int | None = None,
    resid_final: FloatArray | None = None,
    start: int = 0,
    serial: SerialInfo | None = None,
    notes: tuple[str, ...] = (),
    options: dict[str, Any] | None = None,
) -> FGLSResults:
    names = list(data.x_names)
    idx = row_index(data)
    u = data.y - data.X @ beta
    return FGLSResults(
        params=pd.Series(beta, index=names, name="params"),
        cov=pd.DataFrame(cov, index=names, columns=names),
        method=method,
        cov_type=cov_type,
        nobs=data.nobs,
        n_eff=data.nobs if n_eff is None else n_eff,
        use_t=use_t,
        resid=pd.Series(u, index=idx, name="resid"),
        resid_final=pd.Series(u if resid_final is None else resid_final, index=idx[start:]),
        data=data,
        serial=serial,
        options=options or {},
        notes=notes,
    )


def _simple_serial_info(rho: FloatArray, method: str, names: list[str], **extra: Any) -> SerialInfo:
    k = rho.size
    return SerialInfo(
        ar_order=k,
        rho=pd.Series(rho, index=[f"rho{j}" for j in range(1, k + 1)], name="rho"),
        method=method,
        kmax=None,
        ic_path=None,
        durbin_params=None,
        dropped_lags=(),
        nonexogeneity_by_coef=pd.Series(np.zeros(len(names)), index=names),
        extra=dict(extra),
    )


# ------------------------------------------------------------------------------ OLS
def ols(
    y: Any,
    X: Any = None,
    *,
    cov_type: OLSCovType = "HAC",
    kernel: Literal["qs", "bartlett"] | None = None,
    bandwidth: float | Literal["andrews", "nw"] | None = None,
    small_sample: bool = False,
    use_t: bool = False,
) -> FGLSResults:
    """OLS with HAC or heteroskedasticity-robust standard errors.

    Parameters
    ----------
    y, X : data (or a validated :class:`~pyfgls._validation.RegressionData` as ``y``)
    cov_type : {"HAC", "NW", "HC0", "HC1", "HC2", "HC3", "nonrobust"}
        ``"HAC"``: Andrews (1991) quadratic-spectral kernel with automatic bandwidth
        (AR(1) approximation); ``"NW"``: Newey--West Bartlett kernel with
        ``floor(4 (T/100)^{2/9})`` lags; ``"HC*"``: White-type.
    kernel, bandwidth : optional overrides for the HAC estimator.
    small_sample : bool
        Multiply HAC covariances by ``T/(T-q)``.
    use_t : bool

    Examples
    --------
    >>> from pyfgls.simulate import serial_dgp
    >>> sim = serial_dgp(300, random_state=0)
    >>> ols(sim.y, sim.X).method
    'OLS'
    """
    check_choice("cov_type", cov_type, ("HAC", "NW", "HC0", "HC1", "HC2", "HC3", "nonrobust"))
    d = _data(y, X, time_series=cov_type in ("HAC", "NW"))
    b = lstsq(d.X, d.y)
    u = d.y - d.X @ b
    notes: tuple[str, ...] = ()
    if cov_type in ("HAC", "NW"):
        kern = kernel or ("qs" if cov_type == "HAC" else "bartlett")
        bw = bandwidth or ("andrews" if cov_type == "HAC" else "nw")
        V, used = hac_covariance(
            d.X, u, kernel=kern, bandwidth=bw, small_sample=small_sample, const_col=d.const_col
        )
        label = f"HAC ({'QS' if kern == 'qs' else 'Bartlett'}, bw={used:.2f})"
        notes = (f"Covariance: {label}.",)
    else:
        V = hc_covariance(d.X, u, cov_type)
        label = cov_type
    return _result(d, b, V, method="OLS", cov_type=label, use_t=use_t, notes=notes)


# ------------------------------------------------------------------ Cochrane-Orcutt
def cochrane_orcutt(
    y: Any,
    X: Any = None,
    *,
    ar_order: int = 1,
    iterate: bool = False,
    max_iter: int = 100,
    tol: float = 1e-8,
    use_t: bool = False,
) -> FGLSResults:
    """Cochrane--Orcutt FGLS for AR(p) errors (two-step by default, or iterated).

    The two-step version (the "CO-FGLS" of [PG26] Section 3 and Table 1) estimates the
    AR coefficients by regressing the OLS residuals on their own lags and then applies
    OLS to the quasi-differenced data. **It is invalid with non-exogenous regressors**:
    the OLS residuals inherit the inconsistency of OLS and so does the estimated filter
    (in the Table 1 design, rho = 0.8 is estimated with a bias of about -0.17) — use
    :class:`~pyfgls.SerialFGLS` instead.

    With ``iterate=True`` the two steps are repeated until the AR coefficients
    converge; the fixed point is the nonlinear least-squares estimate of the
    common-factor AR(p) model, which (unlike the two-step version) does not rely on OLS
    residuals, but it is only valid when the AR(p) error model is correctly specified.

    The covariance is the conventional ``s^2 (Xf'Xf)^{-1}`` of the last filtered
    regression.
    """
    d = _data(y, X, time_series=True)
    if ar_order < 1:
        raise InputError("ar_order must be at least 1.")
    b = lstsq(d.X, d.y)
    rho = np.zeros(ar_order)
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        u = d.y - d.X @ b
        L = np.column_stack([u[ar_order - j : d.nobs - j] for j in range(1, ar_order + 1)])
        rho_new = lstsq(L, u[ar_order:])
        b = lstsq(ar_filter(d.X, rho_new), ar_filter(d.y, rho_new))
        done = np.max(np.abs(rho_new - rho)) < tol
        rho = rho_new
        if not iterate or done:
            break
    else:  # pragma: no cover - exceedingly rare
        import warnings

        warnings.warn("Cochrane-Orcutt did not converge.", ConvergenceWarning, stacklevel=2)
    Xf, yf = ar_filter(d.X, rho), ar_filter(d.y, rho)
    e = yf - Xf @ b
    n, q = Xf.shape
    V = float(e @ e) / (n - q) * inv_psd(Xf.T @ Xf, name="filtered X'X")
    info = _simple_serial_info(rho, "cochrane-orcutt", list(d.x_names), iterations=n_iter)
    return _result(
        d,
        b,
        V,
        method="Cochrane-Orcutt" + (" (iterated)" if iterate else " (two-step)"),
        cov_type="nonrobust",
        use_t=use_t,
        n_eff=n,
        resid_final=e,
        start=ar_order,
        serial=info,
        notes=(
            "Two-step Cochrane-Orcutt is not valid with non-exogenous regressors "
            "([PG26] Section 3)."
            if not iterate
            else "Iterated Cochrane-Orcutt: NLS of the AR(p) common-factor model.",
        ),
    )


# ---------------------------------------------------------------------------- Durbin
def durbin(
    y: Any,
    X: Any = None,
    *,
    ar_order: int | None = None,
    kmax: KmaxRule = None,
    use_t: bool = False,
) -> FGLSResults:
    """Estimate ``beta^D`` by the Durbin regression ([PG26] eq. (14)).

    The coefficients on the contemporaneous regressors of the Durbin regression (AR
    order by BIC unless ``ar_order`` is given), with their conventional OLS covariance.
    """
    d = _data(y, X, time_series=True)
    if ar_order is None:
        km = resolve_kmax(kmax, d.nobs, d.k_vars, has_const=d.has_const)
        k = select_ar_order(d.y, d.X, d.x_names, km).order
    else:
        k = int(ar_order)
    fit = fit_durbin(durbin_design(d.y, d.X, d.x_names, k, y_name=d.y_name))
    q = d.k_vars
    Z = fit.design.Z
    n, p = Z.shape
    s2 = float(fit.resid @ fit.resid) / (n - p)
    V = s2 * inv_psd(Z.T @ Z, name="Durbin design")[:q, :q]
    rho = fit.coef[fit.design.ylag_columns(k)]
    info = _simple_serial_info(rho, "durbin", list(d.x_names))
    return _result(
        d, fit.beta, V, method="Durbin", cov_type="nonrobust", use_t=use_t, n_eff=n, serial=info
    )


# ------------------------------------------------------------------- infeasible GLS
def infeasible_gls(
    y: Any,
    X: Any = None,
    *,
    rho: npt.ArrayLike | None = None,
    omega: npt.ArrayLike | None = None,
    use_t: bool = False,
) -> FGLSResults:
    """GLS with a known error structure (simulation benchmark).

    Parameters
    ----------
    rho : array_like, optional
        Known AR coefficients: the data are quasi-differenced (first ``k`` observations
        dropped) and OLS is applied — exactly GLS conditional on the initial values.
    omega : array_like (T, T), optional
        Known error covariance matrix (up to scale): GLS by Cholesky whitening.
    """
    d = _data(y, X, time_series=rho is not None)
    if (rho is None) == (omega is None):
        raise InputError("Give exactly one of rho or omega.")
    if rho is not None:
        r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        Xw, yw = ar_filter(d.X, r), ar_filter(d.y, r)
        start = r.size
        serial = _simple_serial_info(r, "known", list(d.x_names))
    else:
        Om = np.asarray(omega, dtype=np.float64)
        if Om.shape != (d.nobs, d.nobs):
            raise InputError(f"omega must be {d.nobs} x {d.nobs}.")
        Lc = scipy.linalg.cholesky(Om, lower=True)
        Xw = scipy.linalg.solve_triangular(Lc, d.X, lower=True)
        yw = scipy.linalg.solve_triangular(Lc, d.y, lower=True)
        start = 0
        serial = None
    b = lstsq(Xw, yw)
    e = yw - Xw @ b
    n, q = Xw.shape
    V = float(e @ e) / (n - q) * inv_psd(Xw.T @ Xw, name="whitened X'X")
    return _result(
        d,
        b,
        V,
        method="GLS (infeasible)",
        cov_type="nonrobust",
        use_t=use_t,
        n_eff=n,
        resid_final=e,
        start=start,
        serial=serial,
    )


# --------------------------------------------------------------------- hybrid GLS
def hybrid_gls(
    y: Any,
    X: Any = None,
    *,
    alpha: float = 0.05,
    lags: int = 1,
    random_state: RandomState = None,
    use_t: bool = False,
) -> FGLSResults:
    """Hybrid OLS/GLS of [PG26] Supplement S-1 (**exogenous regressors only**).

    Run the Breusch--Godfrey LM test for serial correlation of the OLS residuals
    (``lags`` lags). If it does not reject at level ``alpha`` return OLS (HAC
    standard errors); otherwise quasi-difference with an AR(1) coefficient drawn from
    ``U(0.1, 0.9)`` (or ``U(-0.9, -0.1)``) with the sign of the first-order sample
    autocorrelation of the OLS residuals. With exogenous regressors and AR(1)-type
    errors, quasi-differencing with any coefficient of the right sign reduces the
    MSE relative to OLS (Supplement Corollary A.1).
    """
    d = _data(y, X, time_series=True)
    b = lstsq(d.X, d.y)
    u = d.y - d.X @ b
    T = d.nobs
    L = np.column_stack([np.r_[np.zeros(j), u[:-j]] for j in range(1, lags + 1)])
    aux = np.column_stack([d.X, L])
    e = u - aux @ lstsq(aux, u)
    r2 = 1.0 - float(e @ e) / float((u - u.mean()) @ (u - u.mean()))
    lm = T * r2
    pval = float(scipy.stats.chi2.sf(lm, lags))
    if pval >= alpha:
        res = ols(d, cov_type="HAC", use_t=use_t)
        return dataclasses.replace(
            res, method="Hybrid (OLS)", notes=(f"BG LM p-value {pval:.3g}.",)
        )
    rng = check_random_state(random_state)
    r1 = float(u[1:] @ u[:-1] / (u @ u))
    rho = rng.uniform(0.1, 0.9) * (1.0 if r1 >= 0 else -1.0)
    Xf, yf = ar_filter(d.X, [rho]), ar_filter(d.y, [rho])
    bh = lstsq(Xf, yf)
    eh = yf - Xf @ bh
    Vh, _ = hac_covariance(Xf, eh, const_col=d.const_col)
    return _result(
        d,
        bh,
        Vh,
        method="Hybrid (GLS)",
        cov_type="HAC",
        use_t=use_t,
        n_eff=T - 1,
        resid_final=eh,
        start=1,
        serial=_simple_serial_info(np.array([rho]), "hybrid", list(d.x_names), lm_pvalue=pval),
        notes=(f"BG LM p-value {pval:.3g}; quasi-differenced with rho = {rho:.3f}.",),
    )
