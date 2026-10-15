"""Diagnostics: pre-determinedness test, AR roots, stationarity screening, weight checks.

* :func:`test_predetermined` — the variable-addition LM test of [PG26] Remark 9
  (**experimental**): are lags of the stochastic regressors missing from the model?
* :func:`adf_test` / :func:`check_stationarity` — augmented Dickey--Fuller screening
  (warnings only);
* :func:`weight_diagnostics` — how concentrated the FGLS weights are;
* :func:`ar_roots`, :func:`nonexogeneity_share` — convenience accessors.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats

from pyfgls._linalg import lstsq
from pyfgls._utils import ar_roots as _ar_roots
from pyfgls._validation import RegressionData, prepare_data
from pyfgls.exceptions import ExperimentalWarning, NonStationaryWarning
from pyfgls.results import FGLSResults
from pyfgls.serial.durbin import durbin_design, fit_durbin
from pyfgls.serial.lagselect import KmaxRule, resolve_kmax, select_ar_order

__all__ = [
    "ADFResult",
    "PredeterminedTest",
    "adf_test",
    "ar_roots",
    "check_stationarity",
    "nonexogeneity_share",
    "test_predetermined",
    "weight_diagnostics",
]


# ------------------------------------------------------------- pre-determinedness
@dataclass(frozen=True)
class PredeterminedTest:
    """Result of :func:`test_predetermined`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per number of added lags ``l``: LM statistic ``n R^2``, degrees of
        freedom, p-value.
    ar_order : int
        AR order of the Durbin regression used for ``beta^D``.
    regressors : tuple of str
        Stochastic regressors whose lags were added.
    alpha : float
    advice : str
    """

    table: pd.DataFrame
    ar_order: int
    regressors: tuple[str, ...]
    alpha: float
    advice: str

    @property
    def rejected(self) -> bool:
        """Whether any of the sequential tests rejects at level ``alpha``."""
        return bool((self.table["p-value"] < self.alpha).any())

    def __repr__(self) -> str:
        return (
            "Pre-determinedness LM test ([PG26] Remark 9, experimental)\n"
            f"Durbin AR order: {self.ar_order}; regressors tested: {', '.join(self.regressors)}\n"
            f"{self.table.to_string(float_format=lambda v: f'{v:.4f}')}\n{self.advice}"
        )


def test_predetermined(
    y: Any,
    X: Any,
    max_lag: int = 4,
    *,
    ar_order: int | None = None,
    kmax: KmaxRule = None,
    regressors: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> PredeterminedTest:
    """Variable-addition LM test for missing lags of the regressors ([PG26] Remark 9).

    **Experimental.** Steps: (a) Durbin regression (AR order by BIC) giving
    ``beta^D``; (b) ``u^D = y - X beta^D``; (c) demean; (d) for ``l = 1, ..., max_lag``
    regress ``u^D`` on ``X`` and lags ``1..l`` of the stochastic regressors and
    compute ``LM_l = n R^2 ~ chi^2(l * m)`` (``m`` regressors). A rejection suggests
    adding lags of the regressors to the model (so that they are pre-determined) and
    refitting.

    Parameters
    ----------
    y, X : data
    max_lag : int
        Largest number of lags added.
    ar_order : int, optional
        AR order of the Durbin regression (default: BIC).
    kmax : int, optional
        Largest order for the BIC selection.
    regressors : list of str, optional
        Regressors to test (default: all non-constant regressors).
    alpha : float
        Level used for the advice.
    """
    warnings.warn(
        "test_predetermined is experimental; its interface and details may change.",
        ExperimentalWarning,
        stacklevel=2,
    )
    d = prepare_data(y, X, time_series=True)
    T = d.nobs
    if ar_order is None:
        km = resolve_kmax(kmax, T, d.k_vars, has_const=d.has_const)
        k = select_ar_order(d.y, d.X, d.x_names, km).order
    else:
        k = int(ar_order)
    fit = fit_durbin(durbin_design(d.y, d.X, d.x_names, k, y_name=d.y_name))
    uD = d.y - d.X @ fit.beta
    uD = uD - uD.mean()
    if regressors is None:
        idx = [j for j in range(d.k_vars) if np.ptp(d.X[:, j]) > 0]
    else:
        missing = [r for r in regressors if r not in d.x_names]
        if missing:
            raise ValueError(f"Unknown regressors {missing}.")
        idx = [d.x_names.index(r) for r in regressors]
    rows = []
    start = max_lag
    u = uD[start:]
    tss = float(u @ u)
    for lag in range(1, max_lag + 1):
        lags = [d.X[start - j : T - j, idx] for j in range(1, lag + 1)]
        A = np.column_stack([d.X[start:], *lags])
        e = u - A @ lstsq(A, u)
        r2 = 1.0 - float(e @ e) / tss
        n = u.size
        dof = lag * len(idx)
        lm = n * r2
        rows.append((lag, lm, dof, float(scipy.stats.chi2.sf(lm, dof))))
    table = pd.DataFrame(rows, columns=["lags", "LM", "df", "p-value"]).set_index("lags")
    names = tuple(d.x_names[j] for j in idx)
    rej = table.index[table["p-value"] < alpha]
    if len(rej):
        advice = (
            f"Rejected at {alpha:.0%} with {int(rej[0])} lag(s): the regressors may not be "
            f"pre-determined. Add lag(s) 1..{int(rej[0])} of {', '.join(names)} as regressors, "
            "then refit."
        )
    else:
        advice = f"No rejection at {alpha:.0%}: no evidence of missing regressor lags."
    return PredeterminedTest(table=table, ar_order=k, regressors=names, alpha=alpha, advice=advice)


# The public name is fixed by the project brief; keep pytest from collecting it.
test_predetermined.__test__ = False  # type: ignore[attr-defined]


# ------------------------------------------------------------------------- ADF
#: MacKinnon (2010) asymptotic critical values, constant only.
ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey--Fuller test with a constant."""

    statistic: float
    lags: int
    nobs: int
    critical_values: dict[float, float]

    @property
    def unit_root_rejected(self) -> bool:
        """Rejection at the 5% level (series looks stationary)."""
        return self.statistic < self.critical_values[0.05]


def adf_test(x: npt.ArrayLike, *, max_lags: int | None = None) -> ADFResult:
    """ADF regression ``dx_t = a + g x_{t-1} + sum_j c_j dx_{t-j} + e_t`` (BIC lags).

    Returns the t statistic on ``g`` with asymptotic MacKinnon critical values.
    """
    v = np.asarray(x, dtype=np.float64)
    T = v.size
    pmax = int(12 * (T / 100) ** 0.25) if max_lags is None else int(max_lags)
    pmax = max(0, min(pmax, T // 4))
    dx = np.diff(v)
    best: tuple[float, int] | None = None
    start = pmax + 1
    for p in range(pmax + 1):
        cols = [np.ones(T - start), v[start - 1 : T - 1]]
        cols += [dx[start - 1 - j : T - 1 - j] for j in range(1, p + 1)]
        A = np.column_stack(cols)
        yy = dx[start - 1 :]
        e = yy - A @ lstsq(A, yy)
        n = yy.size
        bic = math.log(float(e @ e) / n) + A.shape[1] * math.log(n) / n
        if best is None or bic < best[0]:
            best = (bic, p)
    assert best is not None
    p = best[1]
    start = p + 1
    cols = [np.ones(T - start), v[start - 1 : T - 1]]
    cols += [dx[start - 1 - j : T - 1 - j] for j in range(1, p + 1)]
    A = np.column_stack(cols)
    yy = dx[start - 1 :]
    b = lstsq(A, yy)
    e = yy - A @ b
    n, kk = A.shape
    s2 = float(e @ e) / (n - kk)
    se = math.sqrt(s2 * float(np.linalg.inv(A.T @ A)[1, 1]))
    return ADFResult(statistic=float(b[1] / se), lags=p, nobs=n, critical_values=dict(ADF_CRITICAL))


def check_stationarity(data: RegressionData) -> dict[str, ADFResult]:
    """ADF screening of ``y`` and every non-constant regressor (warnings only)."""
    out: dict[str, ADFResult] = {}
    series = [(data.y_name, data.y)] + [
        (n, data.X[:, j]) for j, n in enumerate(data.x_names) if np.ptp(data.X[:, j]) > 0
    ]
    flagged = []
    for name, s in series:
        if np.unique(s).size <= 2:
            continue
        r = adf_test(s)
        out[name] = r
        if not r.unit_root_rejected:
            flagged.append(f"{name} (ADF {r.statistic:.2f})")
    if flagged:
        warnings.warn(
            "ADF tests do not reject a unit root at 5% for: " + ", ".join(flagged) + ". FGLS "
            "assumes stationary data; consider differencing (or cointegration methods).",
            NonStationaryWarning,
            stacklevel=2,
        )
    return out


# ---------------------------------------------------------------------- weights
def weight_diagnostics(res: FGLSResults) -> dict[str, float]:
    """Concentration of the FGLS weights ``1/v_t``.

    Returns the max/min variance ratio, the Kish effective sample size
    ``(sum w)^2 / sum w^2`` (as a share of ``n``) and the share of the total weight
    carried by the top 1% of observations.
    """
    if res.hetero is None:
        raise ValueError("This fit has no heteroskedasticity weights.")
    v = res.hetero.variance
    w = 1.0 / v
    ws = np.sort(w)[::-1]
    top = max(1, math.ceil(0.01 * w.size))
    return {
        "variance_ratio": float(np.max(v) / np.min(v)),
        "effective_n_share": float(w.sum() ** 2 / (w**2).sum() / w.size),
        "top1pct_weight_share": float(ws[:top].sum() / w.sum()),
    }


def ar_roots(res: FGLSResults) -> npt.NDArray[np.complex128]:
    """Roots of the estimated AR polynomial of a serial-correlation fit."""
    if res.serial is None:
        raise ValueError("This fit has no serial-correlation correction.")
    return _ar_roots(res.serial.rho.to_numpy())


def nonexogeneity_share(res: FGLSResults) -> float:
    """[PG26] first-stage correction relative to the naive variance (see FGLSResults)."""
    if res.serial is None:
        raise ValueError("This fit has no serial-correlation correction.")
    return res.serial.nonexogeneity_share
