# FAQ

**Why not Cochrane–Orcutt?**
It estimates the AR filter from OLS residuals. When the regressors are only
pre-determined, OLS is inconsistent, and so is the filter: in the [PG26] Table 1 design
the Cochrane–Orcutt estimate of $\rho = 0.8$ is biased by about −0.17 and the slope by
+0.03, while FGLS based on the Durbin regression is essentially unbiased.
`pyfgls.benchmarks.cochrane_orcutt` is available for comparison.

**Why a robust variance for FGLS?**
Estimating the filter adds variability that the textbook GLS variance ignores whenever
the regressors are correlated with past errors ([PG26] eq. (29)). For the heteroskedastic
estimator, the fitted variance function is only an approximation, so a sandwich with a
bias correction is used ([GP24] eq. (8)).

**Isn't OLS with HAC standard errors safer?**
Only if the regressors are exogenous — otherwise OLS is inconsistent. Even then, GLS can
be far more precise and HAC intervals tend to under-cover with persistent errors.

**What if there is no serial correlation or heteroskedasticity?**
BIC selects $k^* = 0$ and/or the Lasso selects no covariate; you get OLS with valid
standard errors.

**Can I use panel data?**
Not yet: a `MultiIndex` raises a `TimeIndexError`.

**Which warnings and errors can I get, and what do I do about them?**

| Warning or error | What it means | What to do |
|---|---|---|
| `TimeIndexError` | The index is unsorted, has duplicates, has gaps (dates or periods), or is a panel `MultiIndex`. | Sort it and remove duplicates. For trading-day data pass `allow_gaps=True` (lags are then positional). Panels are not supported. |
| `MissingDataError` | Missing or infinite values. | `missing="drop"` drops incomplete rows (for time series only leading and trailing ones, since interior gaps would corrupt the lags). Infinite values always raise. |
| `CollinearityError` | `X` is rank deficient; the message names the columns. | Drop or combine them (for example a constant plus a full set of dummies). |
| `InsufficientDataError` | Too few observations for the model. | Use a longer sample or a smaller model: fewer regressors, a lower `kmax` or `ar_order`, fewer `cv_folds`. |
| `InputError` | Other invalid input: shapes, non-numeric data, unknown options. | The message names the argument. |
| `NonStationaryWarning` | A root of the estimated AR filter is on or near the unit circle. | The errors may have a unit root: screen with `check_stationarity=True`, difference the data or respecify the model. |
| `CollinearityWarning` | Columns were dropped where that is safe (lagged constants and trends in the Durbin regression). | Usually nothing; the summary notes list them. |
| `LaggedDependentWarning` | A regressor equals a lag of `y`. | Declare it: `lagged_dep_vars=["ylag"]`. |
| `SpecificationWarning` | An unusual specification, such as no intercept. | Add a column of ones, unless the omission is intended. |
| `SampleSizeWarning` | The sample is small relative to the parameters, lags or folds. | Expect noisier lag or covariate selection; a smaller `kmax`, `z` or `cv_folds` helps. |
| `WeightWarning` | The FGLS weights are extreme (a very large max/min variance ratio). | Look for outliers; `weight_clip=(0.01, 0.99)` winsorises the fitted log-variance. |
| `ConvergenceWarning` | An iterative routine did not converge: the Lasso path, the MA(1) likelihood of GMA, or iterated Cochrane–Orcutt. | Check the scaling of the data and the specification before relying on the fit. |
| `ExperimentalWarning` | The feature is experimental, such as `test_predetermined`. | Results and the API may change. |

**How do I silence the warnings?**
All warnings derive from `pyfgls.exceptions.PyFGLSWarning`:
`warnings.simplefilter("ignore", pf.exceptions.PyFGLSWarning)`.
