"""User-facing estimators: :class:`FGLS`, :class:`SerialFGLS`, :class:`LassoFGLS`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfgls._builders import (
    ExogAssumption,
    ZSpec,
    build_combined_results,
    build_hetero_results,
    build_serial_results,
    resolve_candidates,
    resolve_columns,
    z_is_matrix,
)
from pyfgls._linalg import FloatArray
from pyfgls._utils import RandomState
from pyfgls._validation import (
    check_choice,
    detect_lagged_dependent,
    prepare_data,
    warn_lagged_dependent,
)
from pyfgls.combined import COMBINED_COV_TYPES, estimate_combined
from pyfgls.covariance.bootstrap import (
    Scheme,
    WildDist,
    bootstrap_covariance,
    sieve_sample,
    wild_sample,
)
from pyfgls.exceptions import InputError
from pyfgls.hetero.adaptive_lasso import PSI_GRID, CVWeights, FoldScheme, LambdaRule
from pyfgls.hetero.fgls import SKEDASTIC_METHODS, estimate_hetero
from pyfgls.hetero.variance import HETERO_COV_TYPES
from pyfgls.results import FGLSResults
from pyfgls.serial.fgls import estimate_serial
from pyfgls.serial.lagselect import Criterion, ICPenalty, KmaxRule
from pyfgls.serial.variance import SERIAL_COV_TYPES

__all__ = ["FGLS", "LassoFGLS", "SerialFGLS"]

SerialOption = Literal["auto", "fixed", "ma1"] | None
HeteroOption = Literal["lasso", "svr", "wls_s1", "wls_s2", "known", "none"] | None
SERIAL_OPTIONS: tuple[object, ...] = ("auto", "fixed", "ma1", None)
HETERO_OPTIONS: tuple[object, ...] = (*SKEDASTIC_METHODS, None)


class FGLS:
    """Feasible GLS for serial correlation and/or heteroskedasticity.

    ``FGLS(y, X)`` corrects for both problems: the AR filter of [PG26] (Durbin
    regression, BIC lag selection, quasi-differencing) followed by the adaptive-Lasso
    skedastic weights of [GP24] estimated on the filtered residuals ([PG26] Supplement
    S-4, the "F-CH" estimator). ``serial=None`` gives the pure [GP24] estimator and
    ``hetero=None`` the pure [PG26] estimator.

    Parameters
    ----------
    y : array_like or pandas.Series
        Dependent variable (time ordered unless ``serial=None``).
    X : array_like or pandas.DataFrame
        Regressors, including a constant column for an intercept.
    serial : {"auto", "fixed", "ma1", None}
        ``"auto"`` (default): AR order by BIC; ``"fixed"``: use ``ar_order``;
        ``"ma1"``: GMA estimator for MA(1) errors ([PG26] Section 5); ``None``: no
        serial-correlation correction.
    hetero : {"lasso", "svr", "wls_s1", "wls_s2", "known", None}
        Skedastic-function estimator ([GP24]); ``None``: no heteroskedasticity
        correction.
    z : "auto", list of regressor names, or array/DataFrame
        Candidate covariates of the skedastic function (unfiltered). ``"auto"`` builds
        ``{x, x^2, log x, (log x)^2, cos x, cos 2x}`` from each non-constant regressor.
    ar_order, ic, kmax, min_lags, ic_penalty, lagged_dep_vars, wallis_regressor
        Serial-correlation options; see :class:`SerialFGLS`.
    exog : {"predetermined", "exogenous"}
        Assumption on the regressors. Controls the default of
        ``first_stage_correction`` and the diagnostic notes; estimates are unaffected.
    first_stage_correction : bool, optional
        Add the [PG26] eq. (29)-type first-stage term to the combined variance
        (default: ``True`` if ``exog="predetermined"``). See :mod:`pyfgls.combined`.
    variance : array or callable, optional
        Known variances for ``hetero="known"``.
    delta, delta_scale, cv_folds, psi_grid, lambda_rule, fold_scheme, cv_weights, weight_clip, transforms
        Heteroskedasticity options; see :class:`LassoFGLS`.
    random_state : int or Generator, optional
        Seeds the cross-validation folds (and, unless overridden in ``fit``, the
        bootstrap).
    n_jobs : int, optional
        Parallel jobs (cross-validation, bootstrap).
    missing : {"raise", "drop"}
    allow_gaps : bool
        Accept a time index with gaps (lags are then positional).
    check_stationarity : bool
        Screen ``y`` and the regressors with ADF tests and warn
        (:class:`~pyfgls.exceptions.NonStationaryWarning`) when a unit root is not
        rejected. Off by default.

    Examples
    --------
    >>> from pyfgls.simulate import combined_dgp
    >>> sim = combined_dgp(300, ar=(0.5,), skedastic="x2", random_state=1)
    >>> res = FGLS(sim.y, sim.X, random_state=0).fit()
    >>> res.method
    'F-CH (serial+hetero)'
    >>> res.ar_order
    1
    >>> list(res.compare_stages().columns)
    ['F-C (serial)', 'F-CH (serial+hetero)']
    """

    def __init__(
        self,
        y: Any,
        X: Any,
        *,
        serial: SerialOption = "auto",
        hetero: HeteroOption = "lasso",
        z: ZSpec = "auto",
        ar_order: int | None = None,
        ic: Criterion = "bic",
        kmax: KmaxRule = None,
        min_lags: int | Literal["adjusted"] | None = None,
        ic_penalty: ICPenalty = "order",
        lagged_dep_vars: Sequence[str | int] | str | None = None,
        wallis_regressor: str = "auto",
        exog: ExogAssumption = "predetermined",
        first_stage_correction: bool | None = None,
        variance: FloatArray | Callable[[FloatArray], Any] | None = None,
        delta: float = 0.1,
        delta_scale: bool = False,
        cv_folds: int = 10,
        psi_grid: Sequence[float] = PSI_GRID,
        lambda_rule: LambdaRule = "min",
        fold_scheme: FoldScheme = "random",
        cv_weights: CVWeights = "fold",
        weight_clip: tuple[float, float] | None = None,
        transforms: Sequence[str] = ("linear", "poly2", "log", "trig"),
        random_state: RandomState = None,
        n_jobs: int | None = None,
        missing: Literal["raise", "drop"] = "raise",
        allow_gaps: bool = False,
        check_stationarity: bool = False,
    ) -> None:
        check_choice("serial", serial, SERIAL_OPTIONS)
        check_choice("hetero", hetero, HETERO_OPTIONS)
        check_choice("ic", ic, ("bic", "aic"))
        check_choice("ic_penalty", ic_penalty, ("order", "params"))
        check_choice("exog", exog, ("predetermined", "exogenous"))
        check_choice("lambda_rule", lambda_rule, ("min", "1se"))
        check_choice("fold_scheme", fold_scheme, ("random", "blocked"))
        check_choice("cv_weights", cv_weights, ("fold", "full"))
        if serial == "fixed" and ar_order is None:
            raise InputError("serial='fixed' needs ar_order=<int>.")
        if hetero == "none":
            hetero = None
        self.serial: SerialOption = serial
        self.hetero: HeteroOption = hetero
        time_series = serial is not None
        use_z = hetero is not None and hetero != "known"
        self.data = prepare_data(
            y,
            X,
            missing=missing,
            time_series=time_series,
            allow_gaps=allow_gaps,
            extra=z if use_z and z_is_matrix(z) else None,
        )
        d = self.data
        self.lagged_dep_idx = resolve_columns(lagged_dep_vars, d.x_names, "lagged_dep_vars")
        if use_z:
            self.Z, self.z_names = resolve_candidates(z, d, transforms=transforms)
        else:
            self.Z, self.z_names = np.zeros((d.nobs, 0)), ()
        if time_series and not self.lagged_dep_idx:
            warn_lagged_dependent(detect_lagged_dependent(d.y, d.X, d.x_names))
        if check_stationarity and time_series:
            from pyfgls.diagnostics import check_stationarity as _adf_screen

            _adf_screen(d)
        self.z_spec = z
        self.ar_order = ar_order
        self.ic: Criterion = ic
        self.kmax = kmax
        self.min_lags = min_lags
        self.ic_penalty: ICPenalty = ic_penalty
        self.wallis_regressor = wallis_regressor
        self.exog: ExogAssumption = exog
        self.first_stage_correction = (
            exog == "predetermined" if first_stage_correction is None else first_stage_correction
        )
        self.variance = variance
        self.delta = delta
        self.delta_scale = delta_scale
        self.cv_folds = cv_folds
        self.psi_grid = tuple(psi_grid)
        self.lambda_rule: LambdaRule = lambda_rule
        self.fold_scheme: FoldScheme = fold_scheme
        self.cv_weights: CVWeights = cv_weights
        self.weight_clip = weight_clip
        self.random_state = random_state
        self.n_jobs = n_jobs

    @classmethod
    def from_formula(cls, formula: str, data: Any, **kwargs: Any) -> Any:
        """Build the model from a formula such as ``"y ~ x1 + np.log(x2) + lag(x1)"``.

        Requires the ``formula`` extra (formulaic). Missing values created by
        ``lag()`` are left in place, so pass ``missing="drop"`` to drop the leading
        rows. The intercept column is named ``const``. Other keyword arguments go to
        the class constructor.

        Examples
        --------
        >>> import pandas as pd
        >>> from pyfgls.simulate import serial_dgp
        >>> sim = serial_dgp(300, random_state=0)
        >>> df = pd.DataFrame({"y": sim.y, "x": sim.X[:, 1]})
        >>> SerialFGLS.from_formula("y ~ x", data=df).fit().params.index.tolist()
        ['const', 'x']
        """
        from pyfgls._formula import design_from_formula

        y, X = design_from_formula(formula, data)
        return cls(y, X, **kwargs)

    # ------------------------------------------------------------------ helpers
    @property
    def _mode(self) -> str:
        if self.serial is None:
            return "hetero"
        if self.hetero is None:
            return "serial"
        return "combined"

    def _serial_kwargs(self) -> dict[str, Any]:
        d = self.data
        return {
            "y_name": d.y_name,
            "has_const": d.has_const,
            "ar_order": self.ar_order,
            "kmax": self.kmax,
            "ic": self.ic,
            "min_lags": self.min_lags,
            "lagged_dep_idx": self.lagged_dep_idx,
            "wallis_regressor": self.wallis_regressor,
            "ic_penalty": self.ic_penalty,
        }

    def _hetero_kwargs(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "delta_scale": self.delta_scale,
            "weight_clip": self.weight_clip,
            "psi_grid": self.psi_grid,
            "cv_folds": self.cv_folds,
            "lambda_rule": self.lambda_rule,
            "fold_scheme": self.fold_scheme,
            "cv_weights": self.cv_weights,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def _options(self) -> dict[str, Any]:
        d = self.data
        return {
            "serial": self.serial,
            "hetero": self.hetero,
            "ar_order": self.ar_order,
            "ic": self.ic,
            "kmax": self.kmax,
            "min_lags": self.min_lags,
            "ic_penalty": self.ic_penalty,
            "lagged_dep_vars": [d.x_names[i] for i in self.lagged_dep_idx],
            "wallis_regressor": self.wallis_regressor,
            "exog": self.exog,
            "first_stage_correction": self.first_stage_correction,
            "z": "auto" if isinstance(self.z_spec, str) else list(self.z_names),
            "delta": self.delta,
            "delta_scale": self.delta_scale,
            "cv_folds": self.cv_folds,
            "psi_grid": list(self.psi_grid),
            "lambda_rule": self.lambda_rule,
            "fold_scheme": self.fold_scheme,
            "cv_weights": self.cv_weights,
            "weight_clip": self.weight_clip,
        }

    def _allowed_cov_types(self) -> tuple[str, ...]:
        if self._mode == "serial" and self.serial == "ma1":
            return ("robust", "bootstrap")
        base = {
            "serial": SERIAL_COV_TYPES,
            "hetero": HETERO_COV_TYPES,
            "combined": COMBINED_COV_TYPES,
        }[self._mode]
        return (*base, "bootstrap")

    def _estimate(self, y: FloatArray, cov_type: str) -> Any:
        d = self.data
        if self._mode == "serial":
            if self.serial == "ma1":
                from pyfgls.serial.ma import estimate_gma

                return estimate_gma(y, d.X, d.x_names)
            return estimate_serial(y, d.X, d.x_names, **self._serial_kwargs(), cov_type=cov_type)
        if self._mode == "hetero":
            return estimate_hetero(
                y,
                d.X,
                self.Z,
                self.z_names,
                method=self.hetero or "none",
                known=self.variance,
                cov_type=cov_type,
                **self._hetero_kwargs(),
            )
        if self.serial == "ma1":
            raise InputError(
                "serial='ma1' cannot be combined with a heteroskedasticity correction."
            )
        return estimate_combined(
            y,
            d.X,
            d.x_names,
            self.Z,
            self.z_names,
            **self._serial_kwargs(),
            method=self.hetero or "none",
            known=self.variance,
            **self._hetero_kwargs(),
            first_stage_correction=self.first_stage_correction,
            cov_type=cov_type,
        )

    @staticmethod
    def _beta(est: Any) -> FloatArray:
        b = est.beta_fch if hasattr(est, "beta_fch") else est.beta
        return np.asarray(b, dtype=np.float64)

    def _build(self, est: Any, cov_type: str, use_t: bool) -> FGLSResults:
        d = self.data
        opts = self._options()
        if self._mode == "serial":
            if self.serial == "ma1":
                from pyfgls.serial.ma import build_gma_results

                return build_gma_results(d, est, use_t=use_t, options=opts)
            return build_serial_results(
                d,
                est,
                cov_type=cov_type,
                use_t=use_t,
                lagged_dep=self.lagged_dep_idx,
                exog=self.exog,
                options=opts,
            )
        if self._mode == "hetero":
            method = {None: "OLS", "known": "GLS (known variances)"}.get(
                self.hetero, "FGLS (hetero)"
            )
            return build_hetero_results(
                d, est, self.z_names, cov_type=cov_type, use_t=use_t, options=opts, method=method
            )
        return build_combined_results(
            d,
            est,
            self.z_names,
            cov_type=cov_type,
            use_t=use_t,
            lagged_dep=self.lagged_dep_idx,
            exog=self.exog,
            options=opts,
        )

    # ---------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str = "robust",
        *,
        use_t: bool = False,
        n_boot: int = 999,
        bootstrap: Scheme = "wild",
        wild_dist: WildDist = "rademacher",
        boot_random_state: RandomState = None,
    ) -> FGLSResults:
        """Estimate the model.

        Parameters
        ----------
        cov_type : str
            ``"robust"`` (default): [PG26] eq. (29) for serial correlation, [GP24]
            eq. (8)/(7) for heteroskedasticity, [PG26] Supplement (A.4)--(A.5) for
            both. Also ``"naive"`` and ``"HC0"``--``"HC3"`` (serial), ``"nonrobust"``
            and ``"HC0"``--``"HC3"`` (heteroskedasticity), ``"HC0"``/``"HC3"``
            (combined), and ``"bootstrap"`` (all).
        use_t : bool
            Student-t inference with ``n_eff - q`` degrees of freedom.
        n_boot : int
            Bootstrap replications (``cov_type="bootstrap"``).
        bootstrap : {"wild", "iid"}
            Innovation scheme of the sieve bootstrap (serial models).
        wild_dist : {"rademacher", "mammen"}
            Wild-bootstrap multipliers.
        boot_random_state : int or Generator, optional
            Seed of the bootstrap (default: ``random_state`` of the model).

        Returns
        -------
        FGLSResults
        """
        check_choice("cov_type", cov_type, self._allowed_cov_types())
        base_cov = "robust" if cov_type == "bootstrap" else cov_type
        est = self._estimate(self.data.y, base_cov)
        res = self._build(est, base_cov, use_t)
        if cov_type != "bootstrap":
            return res
        return self._bootstrap(est, res, use_t, n_boot, bootstrap, wild_dist, boot_random_state)

    def _bootstrap(
        self,
        est: Any,
        res: FGLSResults,
        use_t: bool,
        n_boot: int,
        scheme: Scheme,
        wild_dist: WildDist,
        random_state: RandomState,
    ) -> FGLSResults:
        d = self.data
        beta = self._beta(est)
        fitted = d.X @ beta
        draw: Callable[[np.random.Generator], FloatArray]
        if self._mode == "hetero":
            resid = d.y - fitted

            def draw(rng: np.random.Generator) -> FloatArray:
                return wild_sample(fitted, resid, wild_dist, rng)

            label = f"bootstrap (wild, {wild_dist})"
        else:
            rho = (
                np.asarray(res.serial.rho.to_numpy(), dtype=np.float64)
                if res.serial is not None and self.serial != "ma1"
                else np.zeros(0)
            )
            k = rho.size
            uhat = d.y - fitted
            innov = uhat[k:].copy()
            for j in range(1, k + 1):
                innov -= rho[j - 1] * uhat[k - j : d.nobs - j]

            def draw(rng: np.random.Generator) -> FloatArray:
                return sieve_sample(fitted, uhat, innov, rho, scheme, wild_dist, rng)

            label = f"bootstrap (sieve AR({k}), {scheme})"

        def refit(ystar: FloatArray) -> FloatArray:
            return self._beta(self._estimate(ystar, "robust"))

        rs = self.random_state if random_state is None else random_state
        boot = bootstrap_covariance(
            refit, draw, n_boot=n_boot, random_state=rs, n_jobs=self.n_jobs, scheme=label
        )
        names = list(d.x_names)
        return FGLSResults(
            params=res.params,
            cov=pd.DataFrame(boot.cov, index=names, columns=names),
            method=res.method,
            cov_type=label,
            nobs=res.nobs,
            n_eff=res.n_eff,
            use_t=use_t,
            resid=res.resid,
            resid_final=res.resid_final,
            data=res.data,
            serial=res.serial,
            hetero=res.hetero,
            stages=res.stages,
            options={**res.options, "n_boot": n_boot},
            notes=(f"Covariance: {label}, {n_boot} replications.",),
        )


class SerialFGLS(FGLS):
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
    ic_penalty : {"order", "params"}
        ``"order"`` (default, as in [PG26] eq. (14)) penalises the AR order;
        ``"params"`` penalises every coefficient a lag adds to the Durbin regression
        (``1 + m`` for ``m`` stochastic regressors). Use ``"params"`` with several
        regressors: the order-based criterion then tends to select ``kmax``.
    lagged_dep_vars : list of str or int, optional
        Regressors that are lags of ``y``. The AR coefficients are then recovered by the
        Wallis ratio ([PG26] Section 4.2).
    wallis_regressor : {"auto", "median"} or str
        Regressor used for the Wallis ratio (see :mod:`pyfgls.serial.wallis`).
    exog : {"predetermined", "exogenous"}
        Your assumption about the regressors. It does not change the estimates (the
        robust variance is valid under both); with ``"exogenous"`` a note is added when
        the data suggest otherwise.
    ma1 : bool
        Use the GMA estimator for MA(1) errors ([PG26] Section 5) instead of the AR
        filter — for errors known to be MA(1), e.g. two-step-ahead forecast errors.
    missing : {"raise", "drop"}
        Missing-value policy (only leading/trailing rows can be dropped).
    allow_gaps : bool
        Accept a date index with gaps (lags are then positional).
    check_stationarity : bool
        ADF screening of ``y`` and the regressors (warnings only).

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
        ic_penalty: ICPenalty = "order",
        lagged_dep_vars: Sequence[str | int] | str | None = None,
        wallis_regressor: str = "auto",
        exog: ExogAssumption = "predetermined",
        ma1: bool = False,
        missing: Literal["raise", "drop"] = "raise",
        allow_gaps: bool = False,
        check_stationarity: bool = False,
    ) -> None:
        serial: SerialOption = "ma1" if ma1 else ("auto" if ar_order is None else "fixed")
        super().__init__(
            y,
            X,
            serial=serial,
            hetero=None,
            ar_order=ar_order,
            ic=ic,
            kmax=kmax,
            min_lags=min_lags,
            ic_penalty=ic_penalty,
            lagged_dep_vars=lagged_dep_vars,
            wallis_regressor=wallis_regressor,
            exog=exog,
            missing=missing,
            allow_gaps=allow_gaps,
            check_stationarity=check_stationarity,
        )


class LassoFGLS(FGLS):
    """Feasible GLS under heteroskedasticity of unknown form ([GP24]).

    The log of the squared OLS residuals is regressed on candidate covariates ``z``
    by an adaptive Lasso tuned by cross-validation; the exponentiated fit gives the
    weights of a weighted least-squares regression. Inference uses the sandwich of
    [GP24] eq. (6) with the bias correction of eq. (8), so it remains valid when the
    skedastic model is misspecified.

    Parameters
    ----------
    y : array_like or pandas.Series
    X : array_like or pandas.DataFrame
        Regressors, including a constant column for an intercept.
    z : "auto", list of regressor names, or array/DataFrame
        Candidate covariates. ``"auto"`` (default) builds
        ``{x, x^2, log x, (log x)^2, cos x, cos 2x}`` from every non-constant regressor
        (logs only for positive regressors; dummies untransformed).
    skedastic : {"lasso", "svr", "wls_s1", "wls_s2", "none", "known"}, array or callable
        Skedastic-function estimator. ``"lasso"`` (default) is the adaptive Lasso of
        [GP24]; ``"svr"`` support-vector regression; ``"wls_s1"``/``"wls_s2"`` the
        Romano--Wolf (2017) parametric fits; ``"none"`` OLS; ``"known"`` (or an array /
        callable of ``X``) uses the given variances (infeasible GLS benchmark).
    variance : array or callable, optional
        Known variances for ``skedastic="known"``.
    delta : float
        Floor in ``log(max(u^2, delta^2))`` (default 0.1, in the units of ``y``).
    delta_scale : bool
        Use the scale-aware floor ``delta * sigma_OLS``.
    cv_folds, psi_grid, lambda_rule, fold_scheme, cv_weights
        Adaptive-Lasso tuning; see :func:`pyfgls.hetero.adaptive_lasso.adaptive_lasso`.
    weight_clip : (lo, hi), optional
        Quantile winsorisation of the fitted log-variance (default none).
    transforms : sequence of str
        Transform families for ``z="auto"``.
    random_state : int or Generator, optional
        Seeds the cross-validation folds.
    n_jobs : int, optional
        Parallel jobs for cross-validation.
    missing : {"raise", "drop"}
        Missing-value policy (rows with missing values are dropped jointly in ``y``,
        ``X`` and a user-supplied ``z``).

    Examples
    --------
    >>> from pyfgls.simulate import hetero_dgp
    >>> sim = hetero_dgp(400, skedastic="x2", random_state=0)
    >>> res = LassoFGLS(sim.y, sim.X, random_state=0).fit()
    >>> res.hetero.skedastic
    'lasso'
    >>> ols = LassoFGLS(sim.y, sim.X, skedastic="none").fit()
    >>> bool(res.bse["x1"] < ols.bse["x1"])
    True
    """

    def __init__(
        self,
        y: Any,
        X: Any,
        *,
        z: ZSpec = "auto",
        skedastic: str | FloatArray | Callable[[FloatArray], Any] = "lasso",
        variance: FloatArray | Callable[[FloatArray], Any] | None = None,
        delta: float = 0.1,
        delta_scale: bool = False,
        cv_folds: int = 10,
        psi_grid: Sequence[float] = PSI_GRID,
        lambda_rule: LambdaRule = "min",
        fold_scheme: FoldScheme = "random",
        cv_weights: CVWeights = "fold",
        weight_clip: tuple[float, float] | None = None,
        transforms: Sequence[str] = ("linear", "poly2", "log", "trig"),
        random_state: RandomState = None,
        n_jobs: int | None = None,
        missing: Literal["raise", "drop"] = "raise",
    ) -> None:
        if callable(skedastic) or not isinstance(skedastic, str):
            variance = skedastic
            skedastic = "known"
        check_choice("skedastic", skedastic, SKEDASTIC_METHODS)
        super().__init__(
            y,
            X,
            serial=None,
            hetero=skedastic,  # type: ignore[arg-type]
            z=z,
            variance=variance,
            delta=delta,
            delta_scale=delta_scale,
            cv_folds=cv_folds,
            psi_grid=psi_grid,
            lambda_rule=lambda_rule,
            fold_scheme=fold_scheme,
            cv_weights=cv_weights,
            weight_clip=weight_clip,
            transforms=transforms,
            random_state=random_state,
            n_jobs=n_jobs,
            missing=missing,
        )
        self.skedastic = skedastic
