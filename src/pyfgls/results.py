"""Results objects: coefficient tables, summaries, tests and comparisons."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyfgls._linalg import FloatArray
from pyfgls._utils import critical_value, two_sided_pvalues
from pyfgls._validation import RegressionData
from pyfgls.exceptions import InputError

__all__ = ["FGLSResults", "HeteroInfo", "SerialInfo", "WaldTestResult", "compare"]


# ------------------------------------------------------------------------ info blocks
@dataclass(frozen=True)
class SerialInfo:
    """Serial-correlation block of the results ([PG26]).

    Attributes
    ----------
    ar_order : int
        Selected (or fixed) AR order ``k*``.
    rho : pandas.Series
        Estimated AR coefficients ``rho_1..rho_k``.
    method : str
        ``"bic"``, ``"aic"``, ``"fixed"`` or ``"ma1"``.
    kmax : int or None
        Largest order considered by the selection.
    ic_path : pandas.Series or None
        Information criterion for ``k = 0..kmax`` (common sample).
    durbin_params : pandas.Series
        Coefficients of the Durbin regression at ``k*`` (by-products).
    dropped_lags : tuple of str
        Lagged columns dropped from the Durbin regression as collinear.
    nonexogeneity_by_coef : pandas.Series
        Relative size of the first-stage correction per coefficient.
    lagged_dep_vars : tuple of str
    wallis_regressors : tuple of str
        Regressor(s) used for the Wallis ratio (lagged dependent variables only).
    extra : dict
        Additional method-specific information (e.g. the MA coefficient for GMA).
    """

    ar_order: int
    rho: pd.Series
    method: str
    kmax: int | None
    ic_path: pd.Series | None
    durbin_params: pd.Series | None
    dropped_lags: tuple[str, ...]
    nonexogeneity_by_coef: pd.Series
    lagged_dep_vars: tuple[str, ...] = ()
    wallis_regressors: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def roots(self) -> npt.NDArray[np.complex128]:
        """Roots of ``1 - sum_j rho_j z^j`` (outside the unit circle if invertible)."""
        from pyfgls._utils import ar_roots

        return ar_roots(self.rho.to_numpy())

    @property
    def nonexogeneity_share(self) -> float:
        """Average relative size of the first-stage correction (0 = exogenous-like)."""
        v = self.nonexogeneity_by_coef.to_numpy()
        return float(v.mean()) if v.size else 0.0


@dataclass(frozen=True)
class HeteroInfo:
    """Heteroskedasticity block of the results ([GP24]).

    Attributes
    ----------
    skedastic : str
        Skedastic-function estimator (``"lasso"``, ``"svr"``, ``"wls_s1"``, ...).
    skedastic_coef : pandas.Series or None
        Coefficients of the fitted log-variance (on the original scale of ``z``).
    selected_z : tuple of str
        Candidate covariates with non-zero coefficients.
    variance : ndarray
        Fitted variances ``v_t`` (normalised to mean one).
    df_hat : float
        Degrees of freedom of the skedastic fit used in the bias correction.
    tuning : dict
        Tuning parameters (``lambda``, ``psi``, CV error, ...).
    """

    skedastic: str
    skedastic_coef: pd.Series | None
    selected_z: tuple[str, ...]
    variance: FloatArray
    df_hat: float
    tuning: dict[str, Any] = field(default_factory=dict)

    @property
    def weights(self) -> FloatArray:
        """FGLS weights ``1 / v_t`` (mean of ``v_t`` normalised to one)."""
        out: FloatArray = 1.0 / self.variance
        return out


# ------------------------------------------------------------------------- Wald test
@dataclass(frozen=True)
class WaldTestResult:
    """Wald test of linear restrictions ``R beta = r``.

    Attributes
    ----------
    statistic : float
        Chi-square statistic (or F statistic when ``use_t``).
    pvalue : float
    df : int
        Number of restrictions.
    distribution : str
        ``"chi2"`` or ``"F"``.
    restrictions : tuple of str
    """

    statistic: float
    pvalue: float
    df: int
    distribution: str
    restrictions: tuple[str, ...]
    df_denom: float | None = None

    def __repr__(self) -> str:
        dist = (
            f"F({self.df}, {self.df_denom:g})" if self.distribution == "F" else f"chi2({self.df})"
        )
        return (
            f"<Wald test: {', '.join(self.restrictions)}; {dist} = {self.statistic:.4f}, "
            f"p-value = {self.pvalue:.4g}>"
        )


# ---------------------------------------------------------------------------- results
@dataclass(frozen=True, repr=False)
class FGLSResults:
    """Results of an FGLS (or OLS benchmark) fit.

    Most users access the statsmodels-like attributes ``params``, ``bse``,
    ``tvalues``, ``pvalues``, ``conf_int()``, ``cov_params()`` and ``summary()``.
    Arrays are returned as pandas objects labelled by the regressor names.

    Attributes
    ----------
    params : pandas.Series
        Estimated coefficients.
    cov : pandas.DataFrame
        Estimated covariance matrix of ``params``.
    method : str
        Human-readable estimator name.
    cov_type : str
        Variance estimator.
    nobs : int
        Number of observations ``T``.
    n_eff : int
        Effective number of observations in the final regression (``T - k*``).
    use_t : bool
        Whether inference uses Student-t (``n_eff - q`` dof) instead of the normal.
    resid : pandas.Series
        Residuals ``y_t - x_t' beta`` on the full sample.
    resid_final : pandas.Series
        Residuals of the final (filtered and/or weighted) regression on the effective
        sample — the estimated innovations.
    serial : SerialInfo or None
    hetero : HeteroInfo or None
    stages : dict
        Intermediate estimators (e.g. ``"serial"`` = F-C and ``"hetero"`` = F-CH).
    options : dict
        Options used for the fit.
    """

    params: pd.Series
    cov: pd.DataFrame
    method: str
    cov_type: str
    nobs: int
    n_eff: int
    use_t: bool
    resid: pd.Series
    resid_final: pd.Series
    data: RegressionData = field(repr=False)
    serial: SerialInfo | None = None
    hetero: HeteroInfo | None = None
    stages: Mapping[str, FGLSResults] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------ basic stats
    @property
    def k_vars(self) -> int:
        """Number of coefficients ``q``."""
        return int(self.params.size)

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom ``n_eff - q`` (used when ``use_t``)."""
        return int(self.n_eff - self.k_vars)

    @property
    def _dof(self) -> float | None:
        return float(self.df_resid) if self.use_t else None

    @property
    def bse(self) -> pd.Series:
        """Standard errors."""
        return pd.Series(
            np.sqrt(np.clip(np.diag(self.cov.to_numpy()), 0.0, None)),
            index=self.params.index,
            name="bse",
        )

    @property
    def tvalues(self) -> pd.Series:
        """T (or z) statistics ``params / bse``."""
        b = self.bse.to_numpy()
        t = np.divide(self.params.to_numpy(), b, out=np.full_like(b, np.nan), where=b > 0)
        return pd.Series(t, index=self.params.index, name="tvalues")

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided p-values (normal by default, Student-t if ``use_t``)."""
        return pd.Series(
            two_sided_pvalues(self.tvalues.to_numpy(), df=self._dof),
            index=self.params.index,
            name="pvalues",
        )

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Wald confidence intervals ``params -/+ c_{1-alpha/2} * bse``."""
        c = critical_value(alpha, df=self._dof)
        b, s = self.params.to_numpy(), self.bse.to_numpy()
        lo, hi = f"{alpha / 2:.3g}", f"{1 - alpha / 2:.3g}"
        return pd.DataFrame({lo: b - c * s, hi: b + c * s}, index=self.params.index)

    def cov_params(self) -> pd.DataFrame:
        """Estimated covariance matrix of the coefficients."""
        return self.cov.copy()

    @property
    def fittedvalues(self) -> pd.Series:
        """Fitted values ``x_t' beta`` on the full sample."""
        return pd.Series(
            self.data.X @ self.params.to_numpy(), index=self.resid.index, name="fitted"
        )

    def predict(self, X_new: Any = None) -> pd.Series | FloatArray:
        """Predict ``x' beta`` for new regressors (in-sample fitted values if ``None``).

        ``X_new`` must have the same columns as the estimation ``X`` (same order, or
        the same names for a DataFrame). Returns a Series for pandas input.
        """
        if X_new is None:
            return self.fittedvalues
        names = list(self.params.index)
        if isinstance(X_new, pd.DataFrame):
            missing = [n for n in names if n not in X_new.columns]
            if missing:
                raise InputError(f"X_new lacks columns {missing}.")
            Xn = X_new[names].to_numpy(dtype=np.float64)
            return pd.Series(Xn @ self.params.to_numpy(), index=X_new.index, name="predicted")
        Xa = np.asarray(X_new, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa[None, :] if Xa.size == len(names) else Xa[:, None]
        if Xa.shape[1] != len(names):
            raise InputError(f"X_new has {Xa.shape[1]} columns; expected {len(names)}.")
        out: FloatArray = Xa @ self.params.to_numpy()
        return out

    # ------------------------------------------------------------- serial accessors
    @property
    def ar_order(self) -> int | None:
        """Selected AR order ``k*`` (``None`` without serial correction)."""
        return None if self.serial is None else self.serial.ar_order

    @property
    def rho(self) -> pd.Series | None:
        """Estimated AR coefficients (``None`` without serial correction)."""
        return None if self.serial is None else self.serial.rho

    @property
    def ar_roots(self) -> npt.NDArray[np.complex128] | None:
        """Roots of the estimated AR polynomial."""
        return None if self.serial is None else self.serial.roots

    @property
    def bic_path(self) -> pd.Series | None:
        """Information criterion by AR order (``None`` if the order was fixed)."""
        return None if self.serial is None else self.serial.ic_path

    @property
    def nonexogeneity_share(self) -> float | None:
        """Size of the [PG26] eq. (29) correction relative to the naive variance.

        Average over coefficients of ``(AVar_R - AVar_N)_ii / (AVar_N)_ii``. Values
        near zero mean the regressors behave as exogenous; large values mean the naive
        variance (eq. (27)) would understate the uncertainty.
        """
        return None if self.serial is None else self.serial.nonexogeneity_share

    # ------------------------------------------------------------- hetero accessors
    @property
    def skedastic_coef(self) -> pd.Series | None:
        """Coefficients of the fitted log-variance function."""
        return None if self.hetero is None else self.hetero.skedastic_coef

    @property
    def selected_z(self) -> tuple[str, ...] | None:
        """Candidate covariates selected by the skedastic fit."""
        return None if self.hetero is None else self.hetero.selected_z

    @property
    def weights(self) -> FloatArray | None:
        """FGLS weights ``1/v_t`` (``None`` without heteroskedasticity correction)."""
        return None if self.hetero is None else self.hetero.weights

    @property
    def df_hat(self) -> float | None:
        """Degrees of freedom of the skedastic fit (bias correction, [GP24] eq. (8))."""
        return None if self.hetero is None else self.hetero.df_hat

    # ------------------------------------------------------------------- inference
    def wald_test(
        self,
        restrictions: str | Sequence[str] | tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike,
        value: npt.ArrayLike | None = None,
    ) -> WaldTestResult:
        """Wald test of linear restrictions ``R beta = r``.

        Parameters
        ----------
        restrictions : str, list of str, (R, r) or R
            Restrictions as strings such as ``"x1 = x2"``, ``"x1 + 2*x2 = 1"`` or
            ``"x1 = 0, x2 = 0"``; or a matrix ``R`` (with ``value`` = ``r``, default 0).
        value : array_like, optional
            Right-hand side ``r`` when ``restrictions`` is a matrix.

        Returns
        -------
        WaldTestResult
            Chi-square test (``F = chi2 / m`` with ``(m, n_eff - q)`` dof if ``use_t``).
        """
        from pyfgls._restrictions import parse_restrictions

        names = list(self.params.index)
        if isinstance(restrictions, str) or (
            isinstance(restrictions, Sequence)
            and not isinstance(restrictions, tuple)
            and restrictions
            and all(isinstance(s, str) for s in restrictions)
        ):
            R, r, labels = parse_restrictions(restrictions, names)  # type: ignore[arg-type]
        else:
            if isinstance(restrictions, tuple) and len(restrictions) == 2 and value is None:
                Rm, rv = restrictions
            else:
                Rm, rv = restrictions, (0.0 if value is None else value)
            R = np.atleast_2d(np.asarray(Rm, dtype=np.float64))
            r = np.broadcast_to(np.asarray(rv, dtype=np.float64), (R.shape[0],)).copy()
            if R.shape[1] != len(names):
                raise InputError(f"R must have {len(names)} columns, got {R.shape[1]}.")
            labels = tuple(f"R[{i}] beta = {r[i]:g}" for i in range(R.shape[0]))
        b = self.params.to_numpy()
        V = self.cov.to_numpy()
        diff = R @ b - r
        RVR = R @ V @ R.T
        stat = float(diff @ np.linalg.lstsq(RVR, diff, rcond=None)[0])
        m = R.shape[0]
        import scipy.stats

        if self.use_t:
            F = stat / m
            return WaldTestResult(
                statistic=F,
                pvalue=float(scipy.stats.f.sf(F, m, self.df_resid)),
                df=m,
                distribution="F",
                restrictions=labels,
                df_denom=float(self.df_resid),
            )
        return WaldTestResult(
            statistic=stat,
            pvalue=float(scipy.stats.chi2.sf(stat, m)),
            df=m,
            distribution="chi2",
            restrictions=labels,
        )

    # ----------------------------------------------------------------- comparisons
    def compare_ols(self, *, cov_type: str | None = None, alpha: float = 0.05) -> pd.DataFrame:
        """Side-by-side table of this fit and OLS on the same data.

        The OLS standard errors are HAC (Andrews quadratic-spectral kernel, automatic
        bandwidth) for fits that correct serial correlation and HC3 otherwise; pass
        ``cov_type`` (``"HAC"``, ``"NW"``, ``"HC0"``..``"HC3"``) to override.
        """
        from pyfgls.benchmarks import ols

        if cov_type is None:
            cov_type = "HAC" if self.serial is not None else "HC3"
        base = ols(self.data, cov_type=cov_type, use_t=self.use_t)  # type: ignore[arg-type]
        return compare([base, self], alpha=alpha)

    @property
    def stage_serial(self) -> FGLSResults | None:
        """The serial-correlation-only stage (F-C) of a combined fit."""
        return self.stages.get("serial")

    @property
    def stage_hetero(self) -> FGLSResults | None:
        """The fully corrected stage (F-CH) of a combined fit: this result."""
        return self if self.stages else None

    def compare_stages(self, alpha: float = 0.05) -> pd.DataFrame:
        """Compare the intermediate estimators (e.g. F-C vs F-CH) with this fit."""
        if not self.stages:
            raise InputError("This result has no intermediate stages to compare.")
        return compare([*self.stages.values(), self], alpha=alpha)

    # ----------------------------------------------------------------- formatting
    def to_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """Coefficient table as a DataFrame (coef, std err, t/z, p-value, CI)."""
        stat = "t" if self.use_t else "z"
        ci = self.conf_int(alpha)
        return pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                stat: self.tvalues,
                f"P>|{stat}|": self.pvalues,
                ci.columns[0]: ci.iloc[:, 0],
                ci.columns[1]: ci.iloc[:, 1],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary of the main results."""
        out: dict[str, Any] = {
            "method": self.method,
            "cov_type": self.cov_type,
            "nobs": self.nobs,
            "n_eff": self.n_eff,
            "use_t": self.use_t,
            "params": _series_dict(self.params),
            "bse": _series_dict(self.bse),
            "pvalues": _series_dict(self.pvalues),
            "cov": self.cov.to_numpy().tolist(),
            "names": list(self.params.index),
        }
        if self.serial is not None:
            out["serial"] = {
                "ar_order": self.serial.ar_order,
                "rho": self.serial.rho.tolist(),
                "method": self.serial.method,
                "kmax": self.serial.kmax,
                "nonexogeneity_share": self.serial.nonexogeneity_share,
            }
        if self.hetero is not None:
            out["hetero"] = {
                "skedastic": self.hetero.skedastic,
                "selected_z": list(self.hetero.selected_z),
                "df_hat": self.hetero.df_hat,
                "skedastic_coef": None
                if self.hetero.skedastic_coef is None
                else _series_dict(self.hetero.skedastic_coef),
            }
        return out

    def to_json(self, **kwargs: Any) -> str:
        """JSON string of :meth:`to_dict` (keyword arguments go to ``json.dumps``)."""
        return json.dumps(self.to_dict(), **kwargs)

    def summary(self, alpha: float = 0.05) -> Summary:
        """Return the formatted summary (print it, or use ``str()``)."""
        return Summary(self, alpha)

    def plot_diagnostics(self, **kwargs: Any) -> Any:
        """Diagnostic plots; see :func:`pyfgls.plotting.plot_diagnostics`.

        Requires the ``plot`` extra (``pip install pyfgls[plot]``).
        """
        from pyfgls.plotting import plot_diagnostics

        return plot_diagnostics(self, **kwargs)

    def __repr__(self) -> str:
        coefs = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        extra = ""
        if self.serial is not None:
            extra += f", k*={self.serial.ar_order}"
        if self.hetero is not None:
            extra += f", skedastic={self.hetero.skedastic!r}"
        return f"<FGLSResults {self.method}: {coefs}{extra}, cov_type={self.cov_type!r}>"


def _series_dict(s: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in s.items()}


# ---------------------------------------------------------------------------- summary
class Summary:
    """Printable summary of an :class:`FGLSResults`."""

    width = 78

    def __init__(self, res: FGLSResults, alpha: float = 0.05) -> None:
        self.res = res
        self.alpha = alpha

    def _header(self) -> list[tuple[str, str]]:
        r = self.res
        items: list[tuple[str, str]] = [
            ("Dep. variable", r.data.y_name),
            ("Method", r.method),
            ("Observations", str(r.nobs)),
            ("Effective obs.", str(r.n_eff)),
            ("Covariance", r.cov_type),
            ("Inference", "Student-t" if r.use_t else "normal"),
        ]
        if r.serial is not None:
            s = r.serial
            rho = ", ".join(f"{v:.4f}" for v in s.rho.to_numpy()) or "-"
            how = s.method.upper() if s.method in ("bic", "aic") else s.method
            if s.kmax is not None and s.method in ("bic", "aic"):
                how += f" (kmax={s.kmax})"
            items += [("AR order k*", f"{s.ar_order}  [{how}]"), ("AR coefs rho", rho)]
            if s.method != "ma1":
                items.append(("Non-exog. share", f"{s.nonexogeneity_share:.3f}"))
            if s.lagged_dep_vars:
                items.append(("Lagged dep. vars", ", ".join(s.lagged_dep_vars)))
            for key, val in s.extra.items():
                items.append((key, f"{val:.4f}" if isinstance(val, float) else str(val)))
        if r.hetero is not None:
            h = r.hetero
            items += [
                ("Skedastic fit", h.skedastic),
                ("Selected z", ", ".join(h.selected_z) if h.selected_z else "(none)"),
                ("df_hat", f"{h.df_hat:g}"),
            ]
            v = h.variance
            if v.size:
                items.append(("Var. ratio max/min", f"{float(np.max(v) / np.min(v)):.3g}"))
        return items

    def __str__(self) -> str:
        r = self.res
        W = self.width
        lines = ["=" * W, f"{'FGLS Regression Results':^{W}}", "=" * W]
        hdr = self._header()
        half = (len(hdr) + 1) // 2
        left, right = hdr[:half], hdr[half:]
        for i in range(half):
            lk, lv = left[i]
            lv = lv if len(lv) <= 22 else lv[:19] + "..."
            cell_l = f"{lk + ':':<16}{lv:>22}"
            if i < len(right):
                rk, rv = right[i]
                rv = rv if len(rv) <= 20 else rv[:17] + "..."
                cell_r = f"{rk + ':':<18}{rv:>20}"
            else:
                cell_r = ""
            lines.append(f"{cell_l}  {cell_r}".rstrip())
        lines.append("-" * W)
        tab = r.to_frame(self.alpha)
        stat = "t" if r.use_t else "z"
        cols = ["coef", "std err", stat, f"P>|{stat}|", *tab.columns[4:]]
        namew = max(10, *(len(str(n)) for n in tab.index))
        namew = min(namew, 24)
        head = f"{'':<{namew}}" + "".join(f"{c:>11}" for c in cols)
        lines.append(head)
        lines.append("-" * W)
        for name, row in tab.iterrows():
            nm = str(name)
            nm = nm if len(nm) <= namew else nm[: namew - 3] + "..."
            vals = [
                _fmt(row["coef"]),
                _fmt(row["std err"]),
                f"{row[stat]:.3f}",
                f"{row[f'P>|{stat}|']:.3f}",
                _fmt(row.iloc[4]),
                _fmt(row.iloc[5]),
            ]
            lines.append(f"{nm:<{namew}}" + "".join(f"{v:>11}" for v in vals))
        lines.append("=" * W)
        notes = list(r.notes)
        if notes:
            lines.append("Notes:")
            lines.extend(f"[{i + 1}] {n}" for i, n in enumerate(notes))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        return f"<pre>{self}</pre>"


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-4):
        return f"{x:.3e}"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------- compare
def compare(
    results: Sequence[FGLSResults],
    *,
    alpha: float = 0.05,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side table of several fits (coefficients, standard errors, CI widths).

    Parameters
    ----------
    results : sequence of FGLSResults
    alpha : float
        Level for the confidence-interval width.
    names : sequence of str, optional
        Column labels (default: each result's ``method``, de-duplicated).

    Returns
    -------
    pandas.DataFrame
        Rows ``(coefficient, statistic)`` with statistics ``coef``, ``std err``,
        ``CI width``; columns one per result.
    """
    if not results:
        raise InputError("compare() needs at least one result.")
    if names is None:
        names = []
        for r in results:
            label = r.method
            if label in names or sum(o.method == r.method for o in results) > 1:
                label = f"{r.method} [{r.cov_type}]"
            n = 2
            base = label
            while label in names:
                label = f"{base} ({n})"
                n += 1
            names.append(label)
    if len(names) != len(results):
        raise InputError("names must have one entry per result.")
    blocks = {}
    for nm, r in zip(names, results, strict=True):
        ci = r.conf_int(alpha)
        blocks[nm] = pd.concat(
            {
                "coef": r.params,
                "std err": r.bse,
                "CI width": ci.iloc[:, 1] - ci.iloc[:, 0],
            }
        ).swaplevel()
    table = pd.DataFrame(blocks)
    order = list(dict.fromkeys(i for r in results for i in r.params.index))
    stats = ["coef", "std err", "CI width"]
    return table.reindex(pd.MultiIndex.from_product([order, stats]))
