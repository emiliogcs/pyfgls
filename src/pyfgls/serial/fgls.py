"""Serial-correlation FGLS of [PG26] Section 4: orchestration of the three steps.

1. Durbin regression with the AR order chosen by BIC on a common sample (eq. (14));
2. OLS on quasi-differenced data (eq. (15));
3. naive (eq. (27)) or exogeneity-robust (eqs. (28)--(29)) variance.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls._utils import ar_roots
from pyfgls.exceptions import InputError, NonStationaryWarning
from pyfgls.serial.durbin import DurbinDesign, DurbinFit, durbin_design, fit_durbin
from pyfgls.serial.filter import FilteredRegression, filtered_regression
from pyfgls.serial.lagselect import (
    Criterion,
    KmaxRule,
    LagSelection,
    n_durbin_regressors,
    resolve_kmax,
    resolve_min_lags,
    select_ar_order,
)
from pyfgls.serial.variance import SerialVariance, serial_variance
from pyfgls.serial.wallis import WallisResult, wallis_rho

#: Warn when the smallest root of the estimated AR polynomial has modulus below this.
NEAR_UNIT_ROOT = 1.05


@dataclass(frozen=True)
class SerialEstimate:
    """All intermediate results of the serial-correlation FGLS estimator."""

    rho: FloatArray
    order: int
    selection: LagSelection | None
    durbin: DurbinFit
    filtered: FilteredRegression
    variance: SerialVariance
    wallis: WallisResult | None

    @property
    def beta(self) -> FloatArray:
        """FGLS coefficients."""
        return self.filtered.beta

    @property
    def cov(self) -> FloatArray:
        """Estimated covariance matrix of the coefficients."""
        return self.variance.cov

    @property
    def design(self) -> DurbinDesign:
        """Durbin design at the selected order."""
        return self.durbin.design

    @property
    def roots(self) -> FloatArray:
        """Moduli of the roots of ``1 - sum_j rho_j z^j``."""
        return np.asarray(np.abs(ar_roots(self.rho)), dtype=np.float64)


def check_ar_roots(rho: FloatArray, *, threshold: float = NEAR_UNIT_ROOT) -> None:
    """Warn if the estimated AR filter is (nearly) non-invertible ([PG26] Assumption 2).

    Never modifies ``rho``.
    """
    if rho.size == 0:
        return
    mod = np.abs(ar_roots(rho))
    if mod.size == 0:
        return
    m = float(mod.min())
    if m <= 1.0:
        warnings.warn(
            f"The estimated AR filter has a root with modulus {m:.3f} <= 1: the errors look "
            "non-stationary (unit root or explosive). FGLS results are unreliable; consider "
            "differencing the data or testing for a unit root.",
            NonStationaryWarning,
            stacklevel=3,
        )
    elif m < threshold:
        warnings.warn(
            f"The estimated AR filter has a root with modulus {m:.3f} close to the unit "
            "circle (near unit root). Inference may be unreliable; consider differencing "
            "or a unit-root test.",
            NonStationaryWarning,
            stacklevel=3,
        )


def estimate_serial(
    y: FloatArray,
    X: FloatArray,
    x_names: Sequence[str],
    *,
    y_name: str = "y",
    has_const: bool = True,
    ar_order: int | None = None,
    kmax: KmaxRule = None,
    ic: Criterion = "bic",
    min_lags: int | Literal["adjusted"] | None = None,
    lagged_dep_idx: Sequence[int] = (),
    wallis_regressor: str = "auto",
    cov_type: str = "robust",
) -> SerialEstimate:
    """Run the serial-correlation FGLS estimator of [PG26].

    Parameters
    ----------
    y : ndarray (T,)
    X : ndarray (T, q)
        Validated regressors (full column rank).
    x_names : sequence of str
    y_name : str
    has_const : bool
        Whether ``X`` contains an intercept (used for the regressor-count cap).
    ar_order : int, optional
        Fixed AR order; bypasses the information-criterion selection.
    kmax : int, "default", "theory" or None
        Largest order considered by the selection.
    ic : {"bic", "aic"}
    min_lags : int, "adjusted" or None
        Lower bound on the selected order.
    lagged_dep_idx : sequence of int
        Positions of regressors that are lags of ``y``; switches to the Wallis
        recovery of ``rho``.
    wallis_regressor : str
        See :func:`pyfgls.serial.wallis.wallis_rho`.
    cov_type : str
        See :func:`pyfgls.serial.variance.serial_variance`.

    Returns
    -------
    SerialEstimate
    """
    T, q = X.shape
    lagged_dep = tuple(int(i) for i in lagged_dep_idx)

    selection: LagSelection | None = None
    if ar_order is None:
        km = resolve_kmax(kmax, T, q, has_const=has_const)
        selection = select_ar_order(
            y, X, x_names, km, ic=ic, min_lags=resolve_min_lags(min_lags, T), y_name=y_name
        )
        k = selection.order
        if k > selection.kmax:  # min_lags above kmax: cap by what the sample supports
            k = resolve_kmax(k, T, q, has_const=has_const)
    else:
        if not isinstance(ar_order, int | np.integer) or isinstance(ar_order, bool) or ar_order < 0:
            raise InputError(f"ar_order must be a non-negative int, got {ar_order!r}.")
        k = int(ar_order)
        if n_durbin_regressors(q, k, has_const) >= T - k:
            raise InputError(
                f"ar_order={k} is too large for T={T} observations and {q} regressors."
            )

    design = durbin_design(y, X, x_names, k, start=k, y_name=y_name)
    durbin = fit_durbin(design)

    wallis: WallisResult | None = None
    J: FloatArray | None = None
    if lagged_dep:
        wallis = wallis_rho(durbin, k, x_names, lagged_dep, wallis_regressor)
        rho = wallis.rho
        J = wallis.J
    else:
        missing = [j for j in range(1, k + 1) if j not in design.ylag_pos]
        if missing:
            raise InputError(
                f"The lag(s) {missing} of {y_name} are collinear with the regressors, so X "
                "seems to contain lags of the dependent variable. Declare them with "
                "lagged_dep_vars=[...] ([PG26] Section 4.2)."
            )
        rho = durbin.coef[design.ylag_columns(k)]

    check_ar_roots(rho)
    filtered = filtered_regression(y, X, rho)
    variance = serial_variance(
        filtered.Xf, filtered.ehat, filtered.uhat, design, J=J, cov_type=cov_type
    )
    return SerialEstimate(
        rho=np.asarray(rho, dtype=np.float64),
        order=k,
        selection=selection,
        durbin=durbin,
        filtered=filtered,
        variance=variance,
        wallis=wallis,
    )
