"""FGLS for serial correlation **and** heteroskedasticity ([PG26] Supplement S-4).

Steps (Supplement eqs. (A.2)--(A.7)):

1. **F-C** — the serial-correlation FGLS of [PG26] Section 4: AR order by BIC from the
   Durbin regression, quasi-differencing, OLS; residuals ``e_t``, ``t = k*+1..T``.
2. Skedastic fit — adaptive Lasso of ``log(max(e_t^2, delta^2))`` on candidate
   covariates ``z_t`` built from the **unfiltered** regressors (default dictionary),
   aligned to ``t = k*+1..T``; fitted variances ``v_t``.
3. **F-CH** — weighted least squares on the quasi-differenced regression with weights
   ``1/v_t`` (eqs. (A.6)--(A.7)).
4. Variance (eqs. (A.4)--(A.5)) — the [GP24] eq. (6)/(8) sandwich applied to the
   filtered regressors ``xf_t``:

       Var = A^{-1} [n^{-1} sum_t e_t^2 adj_t a_t a_t'] A^{-1} / n,
       A = n^{-1} sum_t xf_t xf_t' / v_t,
       adj_t = 1/(1 - h_t^{F-CH})^2 + 4 (h_t^{F-C} / q) df,

   with ``a_t = xf_t / v_t``. ``h^{F-CH}`` is the leverage of the weighted filtered
   regression, ``h^{F-C}`` that of the unweighted one, ``df`` the Lasso degrees of
   freedom (intercept included). F-C intervals are built the same way with unit
   weights.

**First-stage correction (open item A).** The Supplement does not write out the
[PG26] eq. (29) term for the combined estimator. With ``first_stage_correction=True``
the influence of the estimated AR coefficients is added to ``a_t``:

    a_t = xf_t / v_t - G_w C_t,    G_w = [n^{-1} sum_t xf_t u_{t-j} / v_t]_j,

where ``u`` are the unfiltered residuals and ``C_t`` the Durbin-regression influence
(``Sigma^{-1} psi_t``, see :mod:`pyfgls.serial.variance`). This is the
heteroskedasticity-robust analogue of eq. (29): with ``v_t = 1`` and homoskedastic
innovations it reduces to it, it is positive semidefinite by construction, and it is
zero when the regressors are exogenous (``G_w -> 0``). It is on by default when the
regressors are declared pre-determined.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyfgls._linalg import FloatArray, hat_diag, inv_psd, make_psd
from pyfgls._utils import RandomState
from pyfgls.hetero.adaptive_lasso import PSI_GRID, CVWeights, FoldScheme, LambdaRule
from pyfgls.hetero.fgls import HeteroEstimate, estimate_hetero
from pyfgls.hetero.variance import weighted_leverage
from pyfgls.serial.fgls import SerialEstimate, estimate_serial
from pyfgls.serial.lagselect import Criterion, ICPenalty, KmaxRule
from pyfgls.serial.variance import first_stage_influence, g_matrix

CombinedCovType = Literal["robust", "HC0", "HC3"]
COMBINED_COV_TYPES: tuple[str, ...] = ("robust", "HC0", "HC3")


@dataclass(frozen=True)
class CombinedEstimate:
    """F-C and F-CH estimators with their covariance matrices."""

    serial: SerialEstimate
    hetero: HeteroEstimate
    cov_fc: FloatArray
    cov_fch: FloatArray
    first_stage_correction: bool
    cov_type: str

    @property
    def beta_fc(self) -> FloatArray:
        """F-C coefficients (serial correction only)."""
        return self.serial.beta

    @property
    def beta_fch(self) -> FloatArray:
        """F-CH coefficients (serial and heteroskedasticity correction)."""
        return self.hetero.beta

    @property
    def order(self) -> int:
        """Selected AR order ``k*``."""
        return self.serial.order


def combined_covariance(
    Xf: FloatArray,
    e: FloatArray,
    uhat: FloatArray,
    v: FloatArray,
    *,
    start: int,
    C: FloatArray | None,
    h_fc: FloatArray,
    df: float,
    cov_type: str = "robust",
) -> FloatArray:
    """Sandwich variance of (weighted) FGLS on filtered data, [PG26] Suppl. (A.4)--(A.5).

    Parameters
    ----------
    Xf : filtered regressors (n, q), rows ``t = start..T-1``
    e : residuals of the filtered regression at the estimate (unweighted scale)
    uhat : unfiltered residuals ``y - X beta`` on the full sample (T,)
    v : variances (n,) — ones for F-C
    start : number of initial observations dropped (``k*``)
    C : first-stage influence weights (k x n), or ``None`` for no correction
    h_fc : leverages of the unweighted filtered regression
    df : skedastic degrees of freedom
    cov_type : ``"robust"`` (Miller--Startz adjustment of [GP24] eq. (8)), ``"HC3"`` or
        ``"HC0"``
    """
    n, q = Xf.shape
    a = Xf / v[:, None]
    if C is not None and C.shape[0] > 0:
        k = C.shape[0]
        Gw = g_matrix(Xf / v[:, None], uhat, start, k)
        a = a - C.T @ Gw.T
    h_w = np.clip(weighted_leverage(Xf, v), 0.0, 1.0 - 1e-12)
    if cov_type == "robust":
        adj = 1.0 / (1.0 - h_w) ** 2 + 4.0 * (h_fc / q) * df
    elif cov_type == "HC3":
        adj = 1.0 / (1.0 - h_w) ** 2
    elif cov_type == "HC0":
        adj = np.ones(n)
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")
    meat = (a * (e**2 * adj)[:, None]).T @ a / n
    Ainv = inv_psd((Xf / v[:, None]).T @ Xf / n, name="weighted filtered moment matrix")
    return make_psd(Ainv @ meat @ Ainv / n, name="combined FGLS covariance")


def estimate_combined(
    y: FloatArray,
    X: FloatArray,
    x_names: Sequence[str],
    Z: FloatArray,
    z_names: Sequence[str],
    *,
    y_name: str = "y",
    has_const: bool = True,
    ar_order: int | None = None,
    kmax: KmaxRule = None,
    ic: Criterion = "bic",
    min_lags: int | Literal["adjusted"] | None = None,
    lagged_dep_idx: Sequence[int] = (),
    wallis_regressor: str = "auto",
    method: str = "lasso",
    known: FloatArray | Callable[[FloatArray], Any] | None = None,
    delta: float = 0.1,
    delta_scale: bool = False,
    weight_clip: tuple[float, float] | None = None,
    psi_grid: Sequence[float] = PSI_GRID,
    cv_folds: int = 10,
    lambda_rule: LambdaRule = "min",
    fold_scheme: FoldScheme = "random",
    cv_weights: CVWeights = "fold",
    random_state: RandomState = None,
    n_jobs: int | None = None,
    first_stage_correction: bool = True,
    cov_type: str = "robust",
    ic_penalty: ICPenalty = "order",
    foldid: npt.ArrayLike | None = None,
    lambda_grid: Sequence[float] | None = None,
) -> CombinedEstimate:
    """Run F-C and F-CH ([PG26] Supplement S-4); see the module docstring.

    ``Z`` holds the candidate covariates for all ``T`` observations (unfiltered); the
    rows ``k*..T-1`` are used. ``known`` variances (``method="known"``) are likewise
    given for all ``T`` observations.
    """
    ser = estimate_serial(
        y,
        X,
        x_names,
        y_name=y_name,
        has_const=has_const,
        ar_order=ar_order,
        kmax=kmax,
        ic=ic,
        min_lags=min_lags,
        lagged_dep_idx=lagged_dep_idx,
        wallis_regressor=wallis_regressor,
        cov_type="robust",
        ic_penalty=ic_penalty,
    )
    k = ser.order
    fr = ser.filtered
    known_k: FloatArray | Callable[[FloatArray], Any] | None = known
    if known is not None and not callable(known):
        known_k = np.asarray(known, dtype=np.float64)[k:]
    het = estimate_hetero(
        fr.yf,
        fr.Xf,
        Z[k:],
        z_names,
        method=method,
        delta=delta,
        delta_scale=delta_scale,
        weight_clip=weight_clip,
        known=known_k,
        cov_type="robust" if cov_type == "robust" else cov_type,
        psi_grid=psi_grid,
        cv_folds=cv_folds,
        lambda_rule=lambda_rule,
        fold_scheme=fold_scheme,
        cv_weights=cv_weights,
        random_state=random_state,
        n_jobs=n_jobs,
        foldid=foldid,
        lambda_grid=lambda_grid,
    )
    C = None
    if first_stage_correction and k > 0:
        C = first_stage_influence(ser.design, None if ser.wallis is None else ser.wallis.J)
    h_fc = hat_diag(fr.Xf)
    n = fr.n_eff
    e_fch = fr.yf - fr.Xf @ het.beta
    u_fch = y - X @ het.beta
    cov_fch = combined_covariance(
        fr.Xf, e_fch, u_fch, het.v, start=k, C=C, h_fc=h_fc, df=het.df, cov_type=cov_type
    )
    cov_fc = combined_covariance(
        fr.Xf, fr.ehat, fr.uhat, np.ones(n), start=k, C=C, h_fc=h_fc, df=het.df, cov_type=cov_type
    )
    return CombinedEstimate(
        serial=ser,
        hetero=het,
        cov_fc=cov_fc,
        cov_fch=cov_fch,
        first_stage_correction=first_stage_correction,
        cov_type=cov_type,
    )
