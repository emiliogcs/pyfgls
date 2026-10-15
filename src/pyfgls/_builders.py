"""Assembly of :class:`~pyfgls.results.FGLSResults` from estimator internals (private)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfgls._linalg import FloatArray
from pyfgls._validation import RegressionData
from pyfgls.exceptions import InputError
from pyfgls.hetero.features import make_dictionary
from pyfgls.hetero.fgls import HeteroEstimate
from pyfgls.results import FGLSResults, HeteroInfo, SerialInfo
from pyfgls.serial.fgls import SerialEstimate

ExogAssumption = Literal["predetermined", "exogenous"]


def row_index(data: RegressionData) -> pd.Index:
    return data.index if data.index is not None else pd.RangeIndex(data.nobs)


def resolve_columns(
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
    idx = row_index(data)
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


# ===================================================================== heteroskedasticity
ZSpec = Literal["auto"] | Sequence[str] | Any


def resolve_candidates(
    z: ZSpec,
    data: RegressionData,
    *,
    transforms: Sequence[str] = ("linear", "poly2", "log", "trig"),
    skip: Sequence[int] = (),
) -> tuple[FloatArray, tuple[str, ...]]:
    """Candidate covariates from the ``z`` option.

    ``"auto"``: dictionary built from the regressors (:func:`make_dictionary`);
    a list of regressor names: those raw columns; otherwise the user matrix carried in
    ``data.extra``.
    """
    if isinstance(z, str):
        if z != "auto":
            raise InputError(f"z must be 'auto', a list of regressor names or a matrix; got {z!r}.")
        return make_dictionary(data.X, data.x_names, transforms=transforms, skip=skip)
    if data.extra is not None:
        return data.extra, data.extra_names
    if isinstance(z, Sequence) and all(isinstance(c, str) for c in z):
        idx = resolve_columns(list(z), data.x_names, "z")
        return data.X[:, list(idx)], tuple(data.x_names[i] for i in idx)
    raise InputError("Could not interpret z.")


def z_is_matrix(z: ZSpec) -> bool:
    if isinstance(z, str):
        return False
    if isinstance(z, pd.DataFrame | pd.Series | np.ndarray):
        return True
    return not (isinstance(z, Sequence) and all(isinstance(c, str) for c in z))


def hetero_info(est: HeteroEstimate, z_names: Sequence[str]) -> HeteroInfo:
    """Package a :class:`HeteroEstimate` into the results' heteroskedasticity block."""
    fit = est.fit
    if fit is None:
        coef = None
        selected: tuple[str, ...] = ()
        tuning: dict[str, Any] = {}
    else:
        coef = pd.Series(
            np.r_[fit.intercept, fit.coef], index=["intercept", *z_names], name="skedastic"
        )
        selected = fit.selected if fit.method in ("lasso", "wls_s1", "wls_s2") else tuple(z_names)
        tuning = {k: v for k, v in fit.tuning.items() if k != "cv_path"}
        if fit.method == "svr":
            coef = None
    tuning["delta"] = est.delta
    return HeteroInfo(
        skedastic=est.method,
        skedastic_coef=coef,
        selected_z=selected,
        variance=est.v,
        df_hat=est.df,
        tuning=tuning,
    )


def build_hetero_results(
    data: RegressionData,
    est: HeteroEstimate,
    z_names: Sequence[str],
    *,
    cov_type: str,
    use_t: bool,
    options: dict[str, Any],
    method: str = "FGLS (hetero)",
) -> FGLSResults:
    """Assemble :class:`FGLSResults` from a :class:`HeteroEstimate` (cross section)."""
    names = list(data.x_names)
    idx = row_index(data)
    info = hetero_info(est, z_names)
    notes: list[str] = []
    if cov_type == "robust":
        if est.method in ("lasso", "svr"):
            notes.append("Covariance: [GP24] eqs. (6)+(8), sandwich with bias correction.")
        elif est.method in ("wls_s1", "wls_s2"):
            notes.append("Covariance: [GP24] eq. (7), HC3-type sandwich (Romano-Wolf).")
        else:
            notes.append("Covariance: HC3 sandwich.")
    if est.method == "lasso" and info.skedastic_coef is not None and not info.selected_z:
        notes.append("The Lasso selected no covariate: constant weights, FGLS equals OLS.")
    return FGLSResults(
        params=pd.Series(est.beta, index=names, name="params"),
        cov=pd.DataFrame(est.cov, index=names, columns=names),
        method=method,
        cov_type=cov_type,
        nobs=data.nobs,
        n_eff=data.nobs,
        use_t=use_t,
        resid=pd.Series(est.u_weighted, index=idx, name="resid"),
        resid_final=pd.Series(est.u_weighted / np.sqrt(est.v), index=idx, name="weighted_resid"),
        data=data,
        hetero=info,
        options=options,
        notes=tuple(notes),
    )


# ================================================================== combined (S-4)
def build_combined_results(
    data: RegressionData,
    est: Any,
    z_names: Sequence[str],
    *,
    cov_type: str,
    use_t: bool,
    lagged_dep: Sequence[int],
    exog: str,
    options: dict[str, Any],
) -> FGLSResults:
    """Assemble F-CH results (with the F-C stage) from a ``CombinedEstimate``."""
    names = list(data.x_names)
    idx = row_index(data)
    ser = est.serial
    het = est.hetero
    k = ser.order
    s_info = serial_info(ser, data, lagged_dep)
    h_info = hetero_info(het, z_names)
    fr = ser.filtered
    corr = "with" if est.first_stage_correction else "without"
    notes = [
        f"Covariance: [PG26] Suppl. (A.4)-(A.5) sandwich ({cov_type}), {corr} the "
        "first-stage (Durbin) correction.",
    ]
    if k == 0:
        notes.append("BIC selected k* = 0: no serial correlation detected.")
    if het.method == "lasso" and h_info.skedastic_coef is not None and not h_info.selected_z:
        notes.append("The Lasso selected no covariate: F-CH equals F-C.")
    if exog == "exogenous" and est.first_stage_correction is False:
        notes.append("exog='exogenous': first-stage correction omitted.")

    fc = FGLSResults(
        params=pd.Series(est.beta_fc, index=names, name="params"),
        cov=pd.DataFrame(est.cov_fc, index=names, columns=names),
        method="F-C (serial)",
        cov_type=cov_type,
        nobs=data.nobs,
        n_eff=fr.n_eff,
        use_t=use_t,
        resid=pd.Series(fr.uhat, index=idx, name="resid"),
        resid_final=pd.Series(fr.ehat, index=idx[k:], name="innovations"),
        data=data,
        serial=s_info,
        options=options,
        notes=(notes[0].replace("sandwich", "sandwich with unit weights"),),
    )
    beta = est.beta_fch
    e = fr.yf - fr.Xf @ beta
    return FGLSResults(
        params=pd.Series(beta, index=names, name="params"),
        cov=pd.DataFrame(est.cov_fch, index=names, columns=names),
        method="F-CH (serial+hetero)",
        cov_type=cov_type,
        nobs=data.nobs,
        n_eff=fr.n_eff,
        use_t=use_t,
        resid=pd.Series(data.y - data.X @ beta, index=idx, name="resid"),
        resid_final=pd.Series(e / np.sqrt(het.v), index=idx[k:], name="weighted_innovations"),
        data=data,
        serial=s_info,
        hetero=h_info,
        stages={"serial": fc},
        options=options,
        notes=tuple(notes),
    )
