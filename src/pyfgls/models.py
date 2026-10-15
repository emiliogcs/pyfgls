"""User-facing estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfgls._validation import (
    RegressionData,
    check_choice,
    detect_lagged_dependent,
    prepare_data,
    warn_lagged_dependent,
)
from pyfgls.exceptions import InputError
from pyfgls.results import FGLSResults, SerialInfo
from pyfgls.serial.fgls import SerialEstimate, estimate_serial
from pyfgls.serial.lagselect import Criterion, KmaxRule
from pyfgls.serial.variance import SERIAL_COV_TYPES

__all__ = ["SerialFGLS"]

ExogAssumption = Literal["predetermined", "exogenous"]


def _row_index(data: RegressionData) -> pd.Index:
    return data.index if data.index is not None else pd.RangeIndex(data.nobs)


def _resolve_columns(
    spec: Sequence[str | int] | str | int | None, names: Sequence[str], what: str
) -> tuple[int, ...]:
    if spec is None:
        return ()
    items = [spec] if isinstance(spec, str | int) else list(spec)
    out: list[int] = []
    for it in items:
        if isinstance(it, str):
            if it not in names:
                raise InputError(f"{what}: {it!r} is not a regressor name ({list(names)}).")
            out.append(list(names).index(it))
        elif isinstance(it, int | np.integer) and 0 <= int(it) < len(names):
            out.append(int(it))
        else:
            raise InputError(f"{what}: invalid column {it!r}.")
    return tuple(dict.fromkeys(out))


def serial_info(est: SerialEstimate, data: RegressionData, lagged_dep: Sequence[int]) -> SerialInfo:
    """Package a :class:`SerialEstimate` into the results' serial block."""
    names = list(data.x_names)
    k = est.order
    sel = est.selection
    return SerialInfo(
        ar_order=k,
        rho=pd.Series(est.rho, index=[f"rho{j}" for j in range(1, k + 1)], name="rho"),
        method="fixed" if sel is None else sel.criterion,
        kmax=None if sel is None else sel.kmax,
        ic_path=None
        if sel is None
        else pd.Series(sel.values, index=pd.RangeIndex(sel.kmax + 1, name="k"), name=sel.criterion),
        durbin_params=pd.Series(est.durbin.coef, index=list(est.design.names), name="durbin"),
        dropped_lags=est.design.dropped,
        nonexogeneity_by_coef=pd.Series(est.variance.nonexogeneity_by_coef, index=names),
        lagged_dep_vars=tuple(names[i] for i in lagged_dep),
        wallis_regressors=() if est.wallis is None else est.wallis.used,
    )


class SerialFGLS:
    """Feasible GLS for regressions with serially correlated errors ([PG26]).

    The errors are approximated by an AR(k) process whose order is selected by BIC and
    whose coefficients come from the Durbin regression ([PG26] eq. (14)); the data are
    quasi-differenced and OLS is applied ([PG26] eq. (15)). The default variance
    (``cov_type="robust"``) is the exogeneity-robust estimator of [PG26] eqs.
    (28)--(29), valid when the regressors are only pre-determined.

    Parameters
    ----------
    y : array_like or pandas.Series
        Dependent variable (time ordered).
    X : array_like or pandas.DataFrame
        Regressors, including a constant column for an intercept.
    ar_order : int, optional
        Fix the AR order instead of selecting it.
    ic : {"bic", "aic"}
        Information criterion for the order selection (default BIC, as in [PG26]).
    kmax : int, "default" or "theory", optional
        Largest order considered. Default ``int(12 * (T/200)**0.25)`` (12 at
        ``T = 200``, as in the [PG26] simulations); ``"theory"`` gives
        ``floor((T/ln T)**(1/3))``. Capped so the Durbin regression uses at most half
        the sample as regressors.
    min_lags : int or "adjusted", optional
        Lower bound on the order; ``"adjusted"`` uses ``ceil((ln T)**1.5)``
        ([PG26] Suppl. Lemma A.2). Default ``None``.
    lagged_dep_vars : list of str or int, optional
        Regressors that are lags of ``y``. The AR coefficients are then recovered by the
        Wallis ratio ([PG26] Section 4.2).
    wallis_regressor : {"auto", "median"} or str
        Regressor used for the Wallis ratio (see :mod:`pyfgls.serial.wallis`).
    exog : {"predetermined", "exogenous"}
        Your assumption about the regressors. It does not change the estimates (the
        robust variance is valid under both); with ``"exogenous"`` a note is added when
        the data suggest otherwise.
    missing : {"raise", "drop"}
        Missing-value policy (only leading/trailing rows can be dropped).
    allow_gaps : bool
        Accept a date index with gaps (lags are then positional).

    Examples
    --------
    >>> from pyfgls.simulate import serial_dgp
    >>> sim = serial_dgp(500, random_state=0)
    >>> res = SerialFGLS(sim.y, sim.X).fit()
    >>> res.ar_order
    1
    >>> round(float(res.params["x1"]), 2)
    1.0
    """

    def __init__(
        self,
        y: Any,
        X: Any,
        *,
        ar_order: int | None = None,
        ic: Criterion = "bic",
        kmax: KmaxRule = None,
        min_lags: int | Literal["adjusted"] | None = None,
        lagged_dep_vars: Sequence[str | int] | str | None = None,
        wallis_regressor: str = "auto",
        exog: ExogAssumption = "predetermined",
        missing: Literal["raise", "drop"] = "raise",
        allow_gaps: bool = False,
    ) -> None:
        check_choice("ic", ic, ("bic", "aic"))
        check_choice("exog", exog, ("predetermined", "exogenous"))
        self.data = prepare_data(y, X, missing=missing, time_series=True, allow_gaps=allow_gaps)
        self.ar_order = ar_order
        self.ic: Criterion = ic
        self.kmax = kmax
        self.min_lags = min_lags
        self.lagged_dep_idx = _resolve_columns(
            lagged_dep_vars, self.data.x_names, "lagged_dep_vars"
        )
        self.wallis_regressor = wallis_regressor
        self.exog: ExogAssumption = exog
        if not self.lagged_dep_idx:
            warn_lagged_dependent(
                detect_lagged_dependent(self.data.y, self.data.X, self.data.x_names)
            )

    def fit(self, cov_type: str = "robust", *, use_t: bool = False) -> FGLSResults:
        """Estimate the model.

        Parameters
        ----------
        cov_type : {"robust", "naive", "HC0", "HC1", "HC2", "HC3"}
            ``"robust"``: [PG26] eq. (29), valid for pre-determined regressors
            (default). ``"naive"``: eq. (27), valid only for exogenous regressors.
            ``"HC*"``: eq. (29) made robust to conditional heteroskedasticity of the
            innovations (White-type weights on the influence function).
        use_t : bool
            Student-t inference with ``n_eff - q`` degrees of freedom.

        Returns
        -------
        FGLSResults
        """
        check_choice("cov_type", cov_type, SERIAL_COV_TYPES)
        d = self.data
        est = estimate_serial(
            d.y,
            d.X,
            d.x_names,
            y_name=d.y_name,
            has_const=d.has_const,
            ar_order=self.ar_order,
            kmax=self.kmax,
            ic=self.ic,
            min_lags=self.min_lags,
            lagged_dep_idx=self.lagged_dep_idx,
            wallis_regressor=self.wallis_regressor,
            cov_type=cov_type,
        )
        return build_serial_results(
            d,
            est,
            cov_type=cov_type,
            use_t=use_t,
            lagged_dep=self.lagged_dep_idx,
            exog=self.exog,
            options={
                "ar_order": self.ar_order,
                "ic": self.ic,
                "kmax": self.kmax,
                "min_lags": self.min_lags,
                "lagged_dep_vars": [d.x_names[i] for i in self.lagged_dep_idx],
                "wallis_regressor": self.wallis_regressor,
                "exog": self.exog,
            },
        )


def build_serial_results(
    data: RegressionData,
    est: SerialEstimate,
    *,
    cov_type: str,
    use_t: bool,
    lagged_dep: Sequence[int],
    exog: str,
    options: dict[str, Any],
    method: str | None = None,
) -> FGLSResults:
    """Assemble :class:`FGLSResults` from a :class:`SerialEstimate`."""
    names = list(data.x_names)
    idx = _row_index(data)
    k = est.order
    info = serial_info(est, data, lagged_dep)
    notes: list[str] = []
    if cov_type == "robust":
        notes.append(
            "Covariance: [PG26] eq. (29), robust to non-exogenous (pre-determined) regressors."
        )
    elif cov_type == "naive":
        notes.append("Covariance: [PG26] eq. (27), valid only for exogenous regressors.")
    else:
        notes.append(f"Covariance: [PG26] eq. (29) with {cov_type} heteroskedasticity weights.")
    if k == 0:
        notes.append("BIC selected k* = 0: no serial correlation detected; FGLS equals OLS.")
    if info.dropped_lags:
        notes.append(
            "Collinear lagged columns dropped from the Durbin regression: "
            + ", ".join(info.dropped_lags)
        )
    if exog == "exogenous" and cov_type == "robust" and info.nonexogeneity_share > 0.1:
        notes.append(
            f"You declared exog='exogenous', but the first-stage correction adds "
            f"{info.nonexogeneity_share:.0%} to the variance on average: the regressors "
            "may not be exogenous."
        )
    if lagged_dep:
        notes.append(
            "AR coefficients recovered by the Wallis ratio from regressor(s) "
            + ", ".join(info.wallis_regressors)
            + " ([PG26] Section 4.2)."
        )
    fr = est.filtered
    return FGLSResults(
        params=pd.Series(fr.beta, index=names, name="params"),
        cov=pd.DataFrame(est.cov, index=names, columns=names),
        method=method or "FGLS (serial)",
        cov_type=cov_type,
        nobs=data.nobs,
        n_eff=fr.n_eff,
        use_t=use_t,
        resid=pd.Series(fr.uhat, index=idx, name="resid"),
        resid_final=pd.Series(fr.ehat, index=idx[k:], name="innovations"),
        data=data,
        serial=info,
        options=options,
        notes=tuple(notes),
    )
