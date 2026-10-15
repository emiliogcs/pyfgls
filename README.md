# pyfgls — Feasible GLS for Python

[![CI](https://github.com/emiliogcs/pyfgls/actions/workflows/ci.yml/badge.svg)](https://github.com/emiliogcs/pyfgls/actions/workflows/ci.yml)
[![Docs](https://github.com/emiliogcs/pyfgls/actions/workflows/docs.yml/badge.svg)](https://github.com/emiliogcs/pyfgls/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/pyfgls.svg)](https://pypi.org/project/pyfgls/)
[![Python](https://img.shields.io/pypi/pyversions/pyfgls.svg)](https://pypi.org/project/pyfgls/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/emiliogcs/pyfgls/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational)](https://mypy-lang.org/)

`pyfgls` implements the feasible generalized least squares (FGLS) estimators of

* **[PG26]** Perron, P. and González-Coya, E. (2026). *Feasible GLS for time series
  regression.* Journal of Econometrics 258, 106346 — **serially correlated errors**;
* **[GP24]** González-Coya, E. and Perron, P. (2024). *Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach.* Journal of Econometric
  Methods 13(1), 29–48 — **heteroskedasticity of unknown form**;

and their combination for time series with both problems ([PG26] Supplement S-4).

The message of both papers is the same: GLS is not a fragile luxury. Done the way they
propose, it is consistent in situations where OLS is not, it is (essentially) never less
precise than OLS, it is often far more precise, and its confidence intervals have
accurate coverage and are shorter. `pyfgls` packages those procedures with
statsmodels-style results, pandas support and the papers' simulation designs.

```python
import pyfgls as pf

res = pf.FGLS(y, X).fit()  # time series: serial correlation + heteroskedasticity
print(res.summary())
```

---

## Contents

1. [Install](#install)
2. [Which estimator should I use?](#which-estimator-should-i-use)
3. [Guidance for practitioners](#guidance-for-practitioners)
4. [Quick examples](#quick-examples)
5. [Full functionality and usage](#full-functionality-and-usage)
6. [What the estimators do](#what-the-estimators-do)
7. [Replication of the papers](#replication-of-the-papers)
8. [Implementation choices and open items](#implementation-choices-and-open-items)
9. [Citing](#citing)

---

## Install

`pyfgls` is on [PyPI](https://pypi.org/project/pyfgls/):

```bash
pip install pyfgls                    # core
pip install "pyfgls[formula,plot]"    # + formula interface and diagnostic plots
pip install --upgrade pyfgls          # update to the latest release
```

Requires **Python 3.10 or newer**; tested on 3.10–3.13 on Linux, macOS and Windows. Core
dependencies: NumPy, SciPy, pandas, scikit-learn, joblib. Optional: `formulaic`
(`formula` extra), `matplotlib` (`plot` extra), `statsmodels` (only used in the test
suite).

> **`pip` says "No matching distribution found for pyfgls"?** Your `pip` runs on
> Python 3.9 or older (on macOS, the built-in `python3` is 3.9), and older versions of
> `pip` do not say that the Python version is the reason. Check with
> `python3 --version`, install Python 3.10+ (from python.org, or
> `brew install python@3.12`) and use a virtual environment:
>
> ```bash
> python3.12 -m venv .venv && source .venv/bin/activate
> python -m pip install pyfgls
> ```

Development version from GitHub:

```bash
pip install "git+https://github.com/emiliogcs/pyfgls.git"   # latest main
git clone https://github.com/emiliogcs/pyfgls.git && cd pyfgls
pip install -e ".[dev]"                                      # editable, with test tools
```

See [`CONTRIBUTING.md`](https://github.com/emiliogcs/pyfgls/blob/main/CONTRIBUTING.md) for the development workflow.

---

## Which estimator should I use?

<!-- --8<-- [start:choose] -->
| Your data | Problem | Use | Paper |
|---|---|---|---|
| Time series, stationary | Serially correlated errors | `SerialFGLS(y, X)` | [PG26] |
| Time series, stationary | Serial correlation **and** heteroskedastic innovations | `FGLS(y, X)` (F-CH) | [PG26] Suppl. S-4 |
| Cross section (random sample) | Heteroskedasticity of unknown form | `LassoFGLS(y, X)` | [GP24] |
| Time series with lags of `y` as regressors | Serial correlation | `SerialFGLS(y, X, lagged_dep_vars=[...])` | [PG26] §4.2 |
| Errors known to be MA(1), e.g. 2-step-ahead forecast errors | MA(1) with large negative θ | `SerialFGLS(y, X, ma1=True)` (GMA) | [PG26] §5 |
| Many stochastic regressors in a time series | Lag selection | add `ic_penalty="params"` | see *Implementation choices* below |
| You only want robust standard errors for OLS | — | `pf.benchmarks.ols(y, X, cov_type="HAC")` / `"HC3"` | — |

`FGLS` is the general class: `serial=None` turns off the serial-correlation correction
and `hetero=None` the heteroskedasticity correction, so `FGLS(y, X, hetero=None)` is the
same estimator as `SerialFGLS(y, X)` and `FGLS(y, X, serial=None)` the same as
`LassoFGLS(y, X)`.
<!-- --8<-- [end:choose] -->

---

## Guidance for practitioners

<!-- --8<-- [start:guidance] -->
### Time-series regressions with serially correlated errors ([PG26], Section 6)

The concluding remarks of [PG26] translate into the following recommendations.

1. **Use FGLS as the default.** In a linear time-series regression with stationary
   errors, estimate by FGLS with the AR order chosen by BIC and build confidence
   intervals with the exogeneity-robust variance of eq. (29) — `SerialFGLS(y, X).fit()`
   does exactly this. If the regressors happen to be exogenous, the robust correction
   vanishes asymptotically, so nothing is lost; if they are not, it is what makes the
   intervals reliable. OLS, by contrast, is generally *inconsistent* when the regressors
   react to past shocks, and Cochrane–Orcutt inherits that inconsistency.
2. **Check that the regressors are pre-determined.** FGLS needs the regressors to be
   uncorrelated with current and future innovations (not with past ones). Use the
   variable-addition LM test of Remark 9,
   `pf.diagnostics.test_predetermined(y, X)`. If it rejects, add lags of the relevant
   regressors to the model and refit; if the problem comes from omitted variables that
   are not observed, an instrumental-variables procedure is needed instead (outside the
   scope of this package).
3. **Large negative moving-average components need care.** When the errors may contain an
   MA term with a large negative coefficient (typical of multi-step forecast errors), a
   finite autoregression approximates them poorly and BIC may stop too early, which shows
   up as intervals that under-cover. If you know the MA structure, use a tailored
   transformation such as the GMA estimator (`SerialFGLS(..., ma1=True)`).
4. **Lagged dependent variables.** With lags of `y` among the regressors, OLS is
   inconsistent under serial correlation; declare them (`lagged_dep_vars=["ylag"]`) so the
   AR filter is recovered from the lagged regressors (Section 4.2).
5. **Multi-step predictive regressions.** Under rational expectations the h-step forecast
   errors are MA(h−1) and both OLS and FGLS are consistent; FGLS is more efficient as long
   as the implied MA process is invertible. If it may be non-invertible, FGLS is only
   consistent with exogenous regressors — be cautious.
6. **Know your maintained assumptions.** The advantages of GLS over OLS with HAC standard
   errors rest on stationarity, invertibility of the error process and pre-determined
   regressors. When these are plausible — as they usually are — the gains in precision
   and in the reliability of inference are substantial. Use `check_stationarity=True` to
   screen for unit roots, and watch for `NonStationaryWarning` (an estimated AR filter
   with a root near the unit circle).
7. **Heteroskedastic innovations** ([PG26] Remark 12). Either keep FGLS for the serial
   correlation and make the variance heteroskedasticity-robust (`cov_type="HC3"`), or —
   recommended — add the Lasso-based weighting of [GP24]: `FGLS(y, X)` (F-CH). With
   homoskedastic innovations F-CH is equivalent to correcting for serial correlation only,
   so there is no cost.

How much does `nonexogeneity_share` matter? It reports how large the eq. (29) correction
is relative to the textbook GLS variance: near 0 the regressors behave as exogenous; large
values mean the naive interval would under-cover.

### Cross sections with heteroskedasticity of unknown form ([GP24], Section 6)

1. **Lasso-FGLS is a safe replacement for OLS + robust standard errors.** In all the
   designs of [GP24] it was never noticeably worse than OLS (at most a few percent of MSE
   when there is no heteroskedasticity, because the Lasso then selects nothing and the
   weights are constant), and it is markedly better when heteroskedasticity is present.
2. **You do not need to know the variance function.** It is enough that the candidate
   covariates `z` include at least one variable correlated with what drives the variance,
   plus some non-linear transforms. The Lasso picks what fits the variance pattern in the
   sample; it need not pick the "true" form, and it does not need to.
3. **Prefer many variables and few transforms.** Adding many candidate variables works
   well; adding many functional forms of the same variable creates near-collinearity and
   helps little. Trigonometric terms (`cos w`, `cos 2w`, i.e. Chebyshev-type terms) are a
   good flexible choice: `LassoFGLS(y, X, transforms=("linear", "trig"))` or build your own
   `z` with `pyfgls.hetero.make_dictionary`. The default dictionary
   `{x, x², log x, (log x)², cos x, cos 2x}` per regressor follows the paper's simulations.
4. **Always use the bias-corrected robust variance** (the default, eq. (8)). The fitted
   variance function is an approximation, so model-based standard errors are not
   reliable; the corrected sandwich gives close-to-nominal coverage with intervals that
   are much shorter than OLS-HC intervals under heteroskedasticity.
5. **Avoid unrestricted parametric WLS when the variance driver is uncertain.** The
   Romano–Wolf WLS estimators (`skedastic="wls_s1"`, `"wls_s2"`) can be much *worse* than
   OLS when the true driver is not among their covariates; SVR (`"svr"`) is competitive in
   MSE but tends to give less accurate coverage. The adaptive Lasso dominates both.
<!-- --8<-- [end:guidance] -->

---

## Quick examples

### 1. Serial correlation with a non-exogenous regressor

The [PG26] Table 1 design: AR(1) errors with ρ = 0.8 and a regressor `w_t = v_t + e_{t−1}`
that responds to last period's shock.

```python
import pyfgls as pf
from pyfgls.simulate import serial_dgp

sim = serial_dgp(500, random_state=0)
res = pf.SerialFGLS(sim.y, sim.X).fit()
print(res.summary())
print(res.compare_ols())
```

```text
==============================================================================
                           FGLS Regression Results
==============================================================================
Dep. variable:                       y  Inference:                      normal
Method:                  FGLS (serial)  AR order k*:        1  [BIC (kmax=15)]
Observations:                      500  AR coefs rho:                   0.7668
Effective obs.:                    499  Non-exog. share:                 0.015
Covariance:                     robust
------------------------------------------------------------------------------
                 coef    std err          z      P>|z|      0.025      0.975
------------------------------------------------------------------------------
const          0.8768     0.1951      4.495      0.000     0.4945     1.2591
x1             0.9927     0.0258     38.449      0.000     0.9421     1.0433
==============================================================================
Notes:
[1] Covariance: [PG26] eq. (29), robust to non-exogenous (pre-determined) regressors.
[2] Collinear lagged columns dropped from the Durbin regression: L1.const

                   OLS  FGLS (serial)
const coef      0.9132         0.8768
      std err   0.1315         0.1951
      CI width  0.5154         0.7646
x1    coef      1.4125         0.9927
      std err   0.0676         0.0258
      CI width  0.2650         0.1012
```

The true slope is 1: OLS is off by 0.41 (it is inconsistent here), FGLS is on target with
an interval less than half as wide.

### 2. Heteroskedasticity of unknown form

```python
from pyfgls.simulate import hetero_dgp

sim = hetero_dgp(400, skedastic="x2", random_state=1)  # Var(u | x) = x²
res = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit()
print(res.summary())
```

```text
==============================================================================
                           FGLS Regression Results
==============================================================================
Dep. variable:                       y  Inference:                      normal
Method:                  FGLS (hetero)  Skedastic fit:                   lasso
Observations:                      400  Selected z:              x1^2, log(x1)
Effective obs.:                    400  df_hat:                              3
Covariance:                     robust  Var. ratio max/min:                20.1
------------------------------------------------------------------------------
                 coef    std err          z      P>|z|      0.025      0.975
------------------------------------------------------------------------------
const          1.3531     0.2825      4.790      0.000     0.7995     1.9068
x1             0.7736     0.1455      5.317      0.000     0.4884     1.0588
==============================================================================
Notes:
[1] Covariance: [GP24] eqs. (6)+(8), sandwich with bias correction.
```

### 3. Both problems

```python
from pyfgls.simulate import combined_dgp

sim = combined_dgp(300, ar=(0.5,), skedastic="x2", random_state=2)
res = pf.FGLS(sim.y, sim.X, random_state=0).fit()
print(res.compare_stages())  # F-C: serial correction only; F-CH: both
```

```text
                F-C (serial)  F-CH (serial+hetero)
const coef            0.8496                0.7168
      std err         0.3644                0.2998
      CI width        1.4285                1.1750
x1    coef            1.0020                1.0441
      std err         0.1683                0.1335
      CI width        0.6596                0.5234
```

![OLS-HAC vs FGLS intervals](https://raw.githubusercontent.com/emiliogcs/pyfgls/main/docs/assets/intervals.png)

*Persistent AR(2) errors, T = 200: the OLS-HAC interval is centred far from the truth and
is about three times as wide as the FGLS interval.*

---

## Full functionality and usage

### Input data

* `y`, `X` may be NumPy arrays, lists, pandas `Series`/`DataFrame`; names and the index
  are carried into the results. Include a column of ones for an intercept (a warning is
  issued if there is none; implicit intercepts such as a full set of dummies are
  recognised).
* **Time order matters** for serial-correlation corrections. A pandas index must be sorted
  and unique; a `DatetimeIndex`/`PeriodIndex` with gaps raises `TimeIndexError` unless
  `allow_gaps=True` (lags are then positional, e.g. trading days). Panels (`MultiIndex`)
  are not supported.
* Missing values raise `MissingDataError` by default; `missing="drop"` drops
  leading/trailing rows for time series (interior gaps would corrupt the lags) and any
  incomplete row for cross sections. Infinite values always raise.
* Rank-deficient `X` raises `CollinearityError` naming the offending columns. Lagged
  constants, trends and seasonal dummies in the Durbin regression are dropped
  automatically.
* Formulas (needs `formulaic`): `pf.FGLS.from_formula("y ~ x1 + np.log(x2) + lag(x1)",
  data=df, missing="drop")`; the intercept is named `const`.

### `SerialFGLS` — serial correlation ([PG26])

```python
pf.SerialFGLS(
    y,
    X,
    ar_order=None,
    ic="bic",
    kmax=None,
    min_lags=None,
    ic_penalty="order",
    lagged_dep_vars=None,
    wallis_regressor="auto",
    exog="predetermined",
    ma1=False,
    missing="raise",
    allow_gaps=False,
    check_stationarity=False,
).fit(cov_type="robust", use_t=False)
```

| Option | Meaning |
|---|---|
| `ar_order` | Fix the AR order instead of selecting it. |
| `ic` | `"bic"` (default, as in [PG26]) or `"aic"`. |
| `kmax` | Largest order considered: default `int(12·(T/200)^¼)` (12 at T = 200); `"theory"` = ⌊(T/ln T)^⅓⌋; or an int. Capped so the Durbin regression uses ≤ 50% of the sample as regressors. |
| `min_lags` | Lower bound on the order; `"adjusted"` = ⌈(ln T)^1.5⌉ (Suppl. Lemma A.2). |
| `ic_penalty` | `"order"` (paper: penalise the AR order) or `"params"` (penalise every coefficient a lag adds — use it with several stochastic regressors). |
| `lagged_dep_vars` | Names/positions of regressors that are lags of `y` (Wallis recovery, §4.2). |
| `wallis_regressor` | `"auto"` (largest \|t\|), `"median"`, or a regressor name. |
| `exog` | Your assumption; only affects notes and the default first-stage correction of `FGLS`. |
| `ma1` | GMA estimator for MA(1) errors (§5). |
| `cov_type` | `"robust"` (eq. 29, default), `"naive"` (eq. 27, exogenous regressors only), `"HC0"`–`"HC3"` (eq. 29 with heteroskedasticity weights), `"bootstrap"` (sieve). |

### `LassoFGLS` — heteroskedasticity ([GP24])

```python
pf.LassoFGLS(
    y,
    X,
    z="auto",
    skedastic="lasso",
    variance=None,
    delta=0.1,
    delta_scale=False,
    cv_folds=10,
    psi_grid=(0, 0.25, 0.5, 0.75, 1, 2),
    lambda_rule="min",
    fold_scheme="random",
    cv_weights="fold",
    weight_clip=None,
    transforms=("linear", "poly2", "log", "trig"),
    random_state=None,
    n_jobs=None,
    missing="raise",
).fit(cov_type="robust")
```

| Option | Meaning |
|---|---|
| `z` | Candidate covariates: `"auto"` (dictionary from the regressors), a list of regressor names, or your own matrix/DataFrame (any number of columns, even more than observations). |
| `skedastic` | `"lasso"` (adaptive Lasso, default), `"svr"`, `"wls_s1"`, `"wls_s2"` (Romano–Wolf), `"none"` (OLS + HC3), `"known"` or an array/callable of known variances (infeasible GLS). |
| `delta`, `delta_scale` | Floor in log(max(û², δ²)); δ = 0.1 in the units of `y` (paper), or `delta_scale=True` for δ·σ̂_OLS. |
| `cv_folds`, `psi_grid`, `lambda_rule` | 10-fold CV over (λ, ψ) with common folds; penalty at the CV minimum (`"min"`, paper) or `"1se"`. |
| `fold_scheme` | `"random"` (paper) or `"blocked"` (dependent data). |
| `cv_weights` | Re-estimate the ridge weights within each fold (`"fold"`, default) or once (`"full"`). |
| `weight_clip` | Optional quantile winsorisation of the fitted log-variance. |
| `cov_type` | `"robust"` (eq. 8 for Lasso/SVR, eq. 7 for WLS, HC3 otherwise), `"nonrobust"`, `"HC0"`–`"HC3"`, `"bootstrap"` (wild). |

### `FGLS` — both, and the general interface

```python
pf.FGLS(y, X, serial="auto", hetero="lasso", z="auto", first_stage_correction=None, ...)
```

`serial ∈ {"auto", "fixed", "ma1", None}`, `hetero ∈ {"lasso", "svr", "wls_s1", "wls_s2",
"known", None}`; all `SerialFGLS` and `LassoFGLS` options are accepted. The combined fit
(F-CH) applies the [GP24] weights to the quasi-differenced regression, with candidate
covariates built from the **unfiltered** regressors. `first_stage_correction` (default on
for pre-determined regressors) adds the eq. (29)-type term for the estimated AR filter to
the sandwich variance; in our simulations with non-exogenous regressors it raises the
coverage of 95% intervals from 0.78 to 0.94. `cov_type`: `"robust"`, `"HC0"`, `"HC3"`,
`"bootstrap"`.

### Results (`FGLSResults`)

| Attribute / method | Content |
|---|---|
| `params`, `bse`, `tvalues`, `pvalues` | pandas Series labelled by regressor |
| `conf_int(alpha)`, `cov_params()` | intervals and covariance (normal, or Student-t with `use_t=True`) |
| `summary()`, `to_frame()`, `to_dict()`, `to_json()` | formatted output |
| `resid`, `fittedvalues`, `resid_final`, `predict(X_new)` | full-sample residuals, final-stage (filtered/weighted) residuals, predictions |
| `wald_test("x1 = x2, x3 = 0")` | Wald test of linear restrictions (strings or `(R, r)`) |
| `compare_ols()`, `compare_stages()`, `pf.compare([...])` | side-by-side tables |
| `ar_order`, `rho`, `ar_roots`, `bic_path`, `nonexogeneity_share` | serial block (`res.serial`) |
| `selected_z`, `skedastic_coef`, `weights`, `df_hat` | heteroskedasticity block (`res.hetero`) |
| `stage_serial`, `stage_hetero` | F-C and F-CH fits of a combined estimate |
| `plot_diagnostics()` | residuals, their ACF, BIC path / fitted variance, coefficients vs OLS |

### Diagnostics

```python
from pyfgls import diagnostics

diagnostics.test_predetermined(y, X, max_lag=4)  # [PG26] Remark 9 (experimental)
diagnostics.check_stationarity(res.data)  # ADF screening (warnings)
diagnostics.weight_diagnostics(res)  # concentration of the FGLS weights
```

### Benchmarks, bootstrap, simulation designs

```python
pf.benchmarks.ols(y, X, cov_type="HAC")  # Andrews QS kernel, automatic bandwidth
pf.benchmarks.ols(y, X, cov_type="NW")  # Newey–West
pf.benchmarks.cochrane_orcutt(y, X)  # two-step CO (invalid if not exogenous)
pf.benchmarks.durbin(y, X)  # beta^D from the Durbin regression
pf.benchmarks.infeasible_gls(y, X, rho=[0.8])  # known AR coefficients (or omega=)
pf.benchmarks.hybrid_gls(y, X)  # [PG26] Suppl. S-1, exogenous only

res = pf.SerialFGLS(y, X).fit(cov_type="bootstrap", n_boot=999, boot_random_state=1)

from pyfgls.simulate import serial_dgp, hetero_dgp, combined_dgp, predictive_dgp
```

### scikit-learn

```python
from pyfgls.sklearn import FGLSRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cross_val_score(FGLSRegressor(hetero=None), X_features, y, cv=TimeSeriesSplit(5))
```

### Warnings and errors

All warnings derive from `pyfgls.exceptions.PyFGLSWarning` (`NonStationaryWarning`,
`CollinearityWarning`, `SampleSizeWarning`, `SpecificationWarning`,
`LaggedDependentWarning`, `WeightWarning`, `ConvergenceWarning`, `ExperimentalWarning`)
and all errors from `PyFGLSError` (`InputError`, `MissingDataError`, `TimeIndexError`,
`CollinearityError`, `InsufficientDataError`).

### Performance

`python benchmarks/performance.py` (T = 10,000, 10 regressors, 100 candidate covariates,
one core): serial FGLS ≈ 2 s, Lasso-FGLS ≈ 3 s, combined ≈ 5 s. `n_jobs` parallelises the
cross-validation folds and the bootstrap.

---

## What the estimators do

**Serial correlation ([PG26]).** The errors are approximated by an AR(k) process. Its
coefficients come from the *Durbin regression* of `y_t` on its own lags, `x_t` and the lags
of `x_t` (eq. 14) — every regressor there is uncorrelated with the current innovation, so
the estimates are consistent whether or not the regressors are exogenous (unlike
Cochrane–Orcutt, which uses OLS residuals). The order k* minimises BIC with all orders
estimated on a common sample. The data are then quasi-differenced and OLS is applied
(eq. 15). The estimator is √T-consistent and asymptotically normal under
pre-determinedness; with non-exogenous regressors the first stage inflates its variance
by a term that eq. (29) estimates from by-products of the two steps, with no bandwidth or
other tuning parameter.

**Heteroskedasticity ([GP24]).** `log(max(û², δ²))` from OLS is regressed on the candidate
covariates by an adaptive Lasso: ridge first step, penalty weights `|γ̂_ridge|^(−ψ)`, and
the pair (λ, ψ) chosen jointly by 10-fold cross-validation with common folds. The
exponentiated fit gives the GLS weights. Because the variance function is only
approximated, the variance of the estimator is a heteroskedasticity-robust sandwich with
the Miller–Startz/Rothenberg finite-sample correction (eqs. 6 and 8), which accounts for
the leverage of the weighted regression and the degrees of freedom of the Lasso fit.

**Both ([PG26] Supplement S-4).** F-C (the serial-correlation FGLS), then the Lasso
skedastic fit on the F-C residuals, then weighted least squares on the quasi-differenced
data (F-CH), with the bias-corrected sandwich on the filtered regressors.

More detail: [`docs/theory`](https://github.com/emiliogcs/pyfgls/blob/main/docs/theory/serial.md).

---

## Replication of the papers

The `slow` tests (`pytest --runslow tests/statistical`, nightly in CI) re-run the papers'
simulations with fewer replications. Highlights (full tables in
[`docs/replication.md`](https://github.com/emiliogcs/pyfgls/blob/main/docs/replication.md)):

| Design | Quantity | Paper | pyfgls |
|---|---|---|---|
| [PG26] Table 1 (T = 500) | slope RMSE: OLS / Durbin / GLS / FGLS / CO | 0.399 / 0.036 / 0.025 / 0.025 / 0.041 | 0.398 / 0.037 / 0.024 / 0.026 / 0.041 |
| [PG26] Table 1 | bias of ρ̂: FGLS / Cochrane–Orcutt | −0.008 / −0.171 | −0.008 / −0.170 |
| [PG26] Table 3, AR(2) (1.34, −0.42), γ = 0, T = 200 | MSE×100: OLS / FGLS | 11.50 / 0.41 | 10.7 / 0.41 |
| [PG26] Tables 2–3 | coverage of eq. (29) intervals, AR(1)/AR(2), γ ∈ {0, .25, .5} | 0.91–0.95 | 0.93–0.95 |
| [PG26] Table 4, MA(1) θ = −0.7, γ = 0.5 | coverage FGLS / GMA | 0.87 / 0.91 | 0.85 / 0.91 |
| [GP24] Table 1, T = 400 | Lasso eMSE / OLS: x, x², (log x)², exp, step | 0.92, 0.70, 0.46, 0.45, 0.73 | 0.92, 0.70, 0.46, 0.45, 0.74 |
| [GP24] Table 2, T = 100 | Lasso interval length: x², (log x)², exp | 1.10, 0.32, 1.34 | 1.12, 0.32, 1.35 |

```bash
python benchmarks/replicate_pg26_tables.py --table 1 --reps 10000
python benchmarks/replicate_pg26_tables.py --table general --reps 2000
python benchmarks/replicate_gp24_tables.py --reps 2000 --T 100 200 400
```

---

## Implementation choices and open items

Where the papers leave room, `pyfgls` makes the following choices (all overridable):

* `kmax = int(12·(T/200)^¼)`; the [PG26] simulations use 12 for T = 200 and 500.
* Variances are scaled by the effective sample `T − k*`; normal critical values unless
  `use_t=True`.
* The Lasso degrees of freedom count the intercept; the Romano–Wolf variance uses HC3
  OLS residuals (eq. 7).
* The ridge weights of the adaptive Lasso are re-estimated within each CV fold so the CV
  error is not flattered by weights that have seen the test fold (`cv_weights="full"`
  computes them once).
* The Lasso path is computed exactly (LARS); coordinate descent did not converge on the
  highly correlated default dictionary.
* The first `k*` observations are dropped (no Prais–Winsten).

Open items for the authors:

* **Combined variance.** The Supplement does not write the first-stage term for F-CH;
  `pyfgls` adds a heteroskedasticity-robust analogue of eq. (29) by default (coverage
  0.94 with vs 0.78 without it for non-exogenous regressors).
* **Lag penalty with many regressors.** The eq. (14) BIC penalises only the AR order; with
  several stochastic regressors it tends to select `kmax`. `ic_penalty="params"` is
  available; the default follows the paper.
* **Small-sample Lasso efficiency.** At T = 100–200 our Lasso-FGLS eMSE ratios are
  0.03–0.06 above [GP24] Table 1 (e.g. x²: 0.75–0.77 vs 0.70), although WLS-S1/S2, the
  infeasible GLS and the interval lengths agree with the paper and the T = 400 ratios
  match; the difference must come from details of the glmnet-based tuning.
* **Multi-regressor Wallis ratio** (`wallis_regressor`): `"auto"` uses the regressor with the
  largest |t|; [PG26] reports insensitivity to the choice.
* **Remark 9 test.** Read literally (n·R² of the Durbin residuals on X and lagged x), the
  test rejects correctly specified models whose regressors are pre-determined but not
  exogenous, because the Durbin residuals are not orthogonal to X and still contain past
  innovations. `test_predetermined` therefore uses the variable-addition form
  n(SSR₀ − SSR₁)/SSR₀ on the innovation scale (residuals and regressors filtered with the
  Durbin AR filter); simulated size 0.04, power 1.00 against an omitted x_{t−1}.
  `scale="errors"` keeps the literal reading.
* **GMA.** θ̂ is the exact Gaussian MLE from the residuals of the Durbin regression with
  k_T = int(4(T/100)^{2/9}) lags (§5), and all T observations are transformed.

---

## Citing

If you use `pyfgls`, please cite the papers that the methods come from:

* Perron, P. and González-Coya, E. (2026). Feasible GLS for time series regression.
  *Journal of Econometrics* 258, 106346. https://doi.org/10.1016/j.jeconom.2026.106346
* González-Coya, E. and Perron, P. (2024). Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach. *Journal of Econometric
  Methods* 13(1), 29–48. https://doi.org/10.1515/jem-2023-0007

```bibtex
@article{PerronGonzalezCoya2026,
  author  = {Perron, Pierre and Gonz{\'a}lez-Coya, Emilio},
  title   = {Feasible {GLS} for time series regression},
  journal = {Journal of Econometrics},
  volume  = {258},
  pages   = {106346},
  year    = {2026},
  doi     = {10.1016/j.jeconom.2026.106346}
}

@article{GonzalezCoyaPerron2024,
  author  = {Gonz{\'a}lez-Coya, Emilio and Perron, Pierre},
  title   = {Estimation in the presence of heteroskedasticity of unknown form:
             a {L}asso-based approach},
  journal = {Journal of Econometric Methods},
  volume  = {13},
  number  = {1},
  pages   = {29--48},
  year    = {2024},
  doi     = {10.1515/jem-2023-0007}
}
```

See also [`CITATION.cff`](https://github.com/emiliogcs/pyfgls/blob/main/CITATION.cff). License: MIT © Emilio González-Coya.
