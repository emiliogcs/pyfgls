# Reading `summary()`

`print(res.summary())` prints three blocks: a header that describes the fit, the
coefficient table, and numbered notes. `res.summary(alpha=0.10)` gives 90% intervals.
Quick example 1 (serial correlation) and Quick example 2 (heteroskedasticity) show
typical output.

## Header

| Line | Meaning |
|---|---|
| `Dep. variable` | Name of `y`: the Series name, the left-hand side of a formula, or `y` for arrays. |
| `Method` | `FGLS (serial)` ([PG26]), `GMA (MA(1))` ([PG26] §5), `FGLS (hetero)` ([GP24]), `F-CH (serial+hetero)` (both); `OLS` when every correction is switched off. |
| `Observations` | Rows used, after any `missing="drop"`. |
| `Effective obs.` | Rows in the final regression. Quasi-differencing with an AR(k\*) filter loses the first k\* rows (there is no Prais–Winsten step), so this is `Observations − k*`; variances are scaled by it. GMA transforms all rows. |
| `Covariance` | The `cov_type` used; `robust` by default. |
| `Inference` | `normal` critical values, or `Student-t` with `n_eff − q` degrees of freedom when you fit with `use_t=True`. |
| `AR order k*` | The selected AR order and how it was chosen: `[BIC (kmax=15)]` means BIC over the orders 0 to 15, all estimated on a common sample; `fixed` when you passed `ar_order`. |
| `AR coefs rho` | The estimated filter ρ̂₁, …, ρ̂ₖ\*, with A(L) = 1 − ρ₁L − … − ρₖ\*L^k\*; `-` when k\* = 0 (no serial correlation detected, so OLS with valid standard errors). |
| `Non-exog. share` | How large the [PG26] eq. (29) correction is relative to the naive variance: near 0 the regressors behave as exogenous. See [Interpreting `nonexogeneity_share`](nonexogeneity.md). |
| `Lagged dep. vars` | Regressors you declared as lags of `y` (`lagged_dep_vars`). |
| `MA coef theta`, `Durbin order k_T` | GMA only: the estimated MA(1) coefficient and the number of lags in the Durbin regression used to estimate it. |
| `Skedastic fit` | How the variance function was estimated: `lasso` (default), `svr`, `wls_s1`, `wls_s2`, `known`. |
| `Selected z` | Candidate covariates with a non-zero coefficient in the skedastic fit. `(none)` means constant weights, so the estimate is OLS. Long lists are cut with `...`; the full list is `res.hetero.selected_z`. |
| `df_hat` | Degrees of freedom of the skedastic fit (non-zero coefficients, intercept included), used by the bias correction of [GP24] eq. (8). |
| `Var. ratio max/min` | Largest over smallest fitted variance: how unequal the GLS weights are (1 means equal weights). |

## Coefficient table

| Column | Meaning |
|---|---|
| `coef` | The FGLS estimate, `res.params`. |
| `std err` | Square root of the diagonal of `res.cov_params()`, `res.bse`. |
| `z` (or `t`) | `coef / std err`, `res.tvalues`. |
| `P>\|z\|` | Two-sided p-value for the hypothesis that the coefficient is 0, `res.pvalues`. |
| `0.025`, `0.975` | Bounds of the 95% confidence interval, `res.conf_int()`; other levels with `alpha`. |

`res.to_frame()` returns the whole table as a DataFrame, and `res.to_dict()` and
`res.to_json()` serialise the fit.

## Notes

The notes say which variance formula was used, with its equation in the paper, and
report what the estimator did on its own: columns of the Durbin regression dropped
because they are collinear (the lag of the constant, of trends, of seasonal dummies, or a
lag that duplicates another regressor), or a declared assumption that the data contradict
(`exog="exogenous"` with a large `nonexogeneity_share`). Problems found while fitting are
raised as warnings at fit time (see the FAQ).

## Comparing fits

| Call | Table |
|---|---|
| `res.compare_ols()` | This fit next to OLS on the same data, with HAC standard errors (Andrews quadratic-spectral kernel, automatic bandwidth) when the fit corrects serial correlation and HC3 otherwise; override with `cov_type=`. |
| `res.compare_stages()` | A combined fit next to its serial-only stage: F-C versus F-CH. |
| `pf.compare([res1, res2, ...], names=[...])` | Any fits side by side, including the benchmarks in `pf.benchmarks` (OLS, Cochrane–Orcutt, the Durbin estimator, infeasible GLS). |

Each table has one row block per coefficient with its estimate, standard error and
confidence-interval width, and one column per fit.
