"""Heteroskedasticity FGLS of [GP24] Section 2: orchestration.

1. Unweighted regression (OLS) and its residuals ``u_t``;
2. skedastic fit of ``log(max(u_t^2, delta^2))`` on the candidate covariates ``z_t``;
3. weighted least squares with weights ``1 / exp(fitted)`` (rows scaled by
   ``exp(-fitted/2)``, then OLS);
4. sandwich variance with the bias correction of eq. (8) (or eq. (7) for WLS fits).

The function works on whatever ``(y, X)`` it is given; the combined estimator of
[PG26] Supplement S-4 passes the quasi-differenced data, so that step 1 is the F-C
regression and step 3 gives F-CH.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pyfgls._linalg import FloatArray, hat_diag, lstsq
from pyfgls._utils import RandomState
from pyfgls.exceptions import InputError
from pyfgls.hetero.adaptive_lasso import (
    PSI_GRID,
    CVWeights,
    FoldScheme,
    LambdaRule,
    SkedasticFit,
    adaptive_lasso,
)
from pyfgls.hetero.svr import svr_skedastic
from pyfgls.hetero.variance import hetero_covariance
from pyfgls.hetero.weights import known_variances, log_squared_residuals, variances_from_log
from pyfgls.hetero.wls import wls_skedastic

SkedasticMethod = Literal["lasso", "svr", "wls_s1", "wls_s2", "known", "none"]
SKEDASTIC_METHODS: tuple[str, ...] = ("lasso", "svr", "wls_s1", "wls_s2", "known", "none")


@dataclass(frozen=True)
class HeteroEstimate:
    """All intermediate results of the heteroskedasticity FGLS estimator.

    Attributes
    ----------
    beta : FGLS (weighted) coefficients
    cov : covariance matrix of ``beta``
    v : fitted variances (mean one)
    fit : the skedastic fit (``None`` for known variances / no correction)
    beta_unweighted, u_unweighted, h_unweighted : first-step regression
    u_weighted : ``y - X beta`` (unweighted scale)
    delta : floor used in ``log(max(u^2, delta^2))``
    method, cov_type : str
    """

    beta: FloatArray
    cov: FloatArray
    v: FloatArray
    fit: SkedasticFit | None
    beta_unweighted: FloatArray
    u_unweighted: FloatArray
    h_unweighted: FloatArray
    u_weighted: FloatArray
    delta: float
    method: str
    cov_type: str

    @property
    def df(self) -> float:
        """Degrees of freedom of the skedastic fit."""
        return 1.0 if self.fit is None else self.fit.df


def fit_skedastic(
    r: FloatArray,
    Z: FloatArray,
    names: Sequence[str],
    method: str,
    *,
    psi_grid: Sequence[float] = PSI_GRID,
    cv_folds: int = 10,
    lambda_rule: LambdaRule = "min",
    fold_scheme: FoldScheme = "random",
    cv_weights: CVWeights = "fold",
    random_state: RandomState = None,
    n_jobs: int | None = None,
) -> SkedasticFit:
    """Dispatch to the skedastic-function estimator."""
    if method == "lasso":
        return adaptive_lasso(
            r,
            Z,
            names,
            psi_grid=psi_grid,
            cv_folds=cv_folds,
            lambda_rule=lambda_rule,
            fold_scheme=fold_scheme,
            cv_weights=cv_weights,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if method == "svr":
        return svr_skedastic(
            r,
            Z,
            names,
            cv_folds=cv_folds,
            fold_scheme=fold_scheme,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if method in ("wls_s1", "wls_s2"):
        return wls_skedastic(r, Z, names, kind=method)  # type: ignore[arg-type]
    raise InputError(f"Unknown skedastic method {method!r}; choose from {SKEDASTIC_METHODS}.")


def estimate_hetero(
    y: FloatArray,
    X: FloatArray,
    Z: FloatArray,
    z_names: Sequence[str],
    *,
    method: str = "lasso",
    delta: float = 0.1,
    delta_scale: bool = False,
    weight_clip: tuple[float, float] | None = None,
    known: FloatArray | Callable[[FloatArray], Any] | None = None,
    cov_type: str = "robust",
    psi_grid: Sequence[float] = PSI_GRID,
    cv_folds: int = 10,
    lambda_rule: LambdaRule = "min",
    fold_scheme: FoldScheme = "random",
    cv_weights: CVWeights = "fold",
    random_state: RandomState = None,
    n_jobs: int | None = None,
) -> HeteroEstimate:
    """Run the [GP24] FGLS estimator on ``(y, X)`` with candidate covariates ``Z``.

    Parameters
    ----------
    y, X : ndarray
        Data (for the combined estimator: quasi-differenced data).
    Z : ndarray (T, d)
        Candidate covariates for the skedastic function.
    z_names : sequence of str
    method : {"lasso", "svr", "wls_s1", "wls_s2", "known", "none"}
    delta : float
        Floor in ``log(max(u^2, delta^2))``; in the units of ``y`` (default 0.1).
    delta_scale : bool
        Use ``delta * sigma_OLS`` instead (scale-aware floor).
    weight_clip : (lo, hi), optional
        Quantile winsorisation of the fitted log-variance.
    known : array or callable, optional
        Known variances (``method="known"``): an array of length ``T`` or a function
        of ``X`` returning one.
    cov_type : {"robust", "nonrobust", "HC0", "HC1", "HC2", "HC3"}
    psi_grid, cv_folds, lambda_rule, fold_scheme, cv_weights, random_state, n_jobs
        Adaptive-Lasso / SVR tuning (see :func:`pyfgls.hetero.adaptive_lasso.adaptive_lasso`).

    Returns
    -------
    HeteroEstimate
    """
    T, k = X.shape
    b0 = lstsq(X, y)
    u0 = y - X @ b0
    h0 = hat_diag(X)
    floor = delta
    if delta_scale:
        floor = delta * float(np.sqrt(u0 @ u0 / max(T - k, 1)))
        floor = floor if floor > 0 else delta

    fit: SkedasticFit | None = None
    if method == "none":
        v = np.ones(T)
    elif method == "known":
        if known is None:
            raise InputError("skedastic='known' needs the variances (variance=...).")
        vals = known(X) if callable(known) else known
        v = known_variances(np.asarray(vals, dtype=np.float64))
        if v.shape[0] != T:
            raise InputError(f"Known variances have length {v.shape[0]}; expected {T}.")
    else:
        r = log_squared_residuals(u0, floor)
        fit = fit_skedastic(
            r,
            Z,
            z_names,
            method,
            psi_grid=psi_grid,
            cv_folds=cv_folds,
            lambda_rule=lambda_rule,
            fold_scheme=fold_scheme,
            cv_weights=cv_weights,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        v = variances_from_log(fit.fitted, clip=weight_clip)

    s = 1.0 / np.sqrt(v)
    beta = lstsq(X * s[:, None], y * s)
    u = y - X @ beta
    cov = hetero_covariance(
        cov_type,
        X=X,
        v=v,
        u_fgls=u,
        u_ols=u0,
        h_ols=h0,
        df=1.0 if fit is None else fit.df,
        method=method if fit is not None else ("ols" if method == "none" else "known"),
    )
    return HeteroEstimate(
        beta=beta,
        cov=cov,
        v=v,
        fit=fit,
        beta_unweighted=b0,
        u_unweighted=u0,
        h_unweighted=h0,
        u_weighted=u,
        delta=floor,
        method=method,
        cov_type=cov_type,
    )
