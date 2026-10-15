# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-15

First release.

### Added
- Project skeleton: packaging (hatchling, PEP 621), CI, pre-commit, typing marker.
- Input validation (time index, gaps, missing values, rank checks), QR-based linear
  algebra, warning/exception hierarchy, lag utilities.
- `SerialFGLS`: serial-correlation FGLS of Perron & González-Coya (2026) — Durbin
  regression with BIC/AIC lag selection on a common sample, quasi-differenced OLS,
  naive (eq. 27) and exogeneity-robust (eqs. 28–29) variances, HC variants, Wallis
  recovery for lagged dependent variables, AR-root checks, `nonexogeneity_share`.
- `FGLSResults` with summary, coefficient table, Wald tests, JSON export; `compare()`.
- `pyfgls.simulate`: data-generating processes of both papers.
- Monte Carlo acceptance tests replicating [PG26] Table 1 and Tables 2–5 cells.
- `LassoFGLS`: heteroskedasticity FGLS of González-Coya & Perron (2024) — candidate
  dictionary, adaptive Lasso (ridge first step, joint CV over lambda and psi with
  common folds, exact LARS path), SVR and Romano–Wolf WLS-S1/S2 skedastic fits,
  known variances, eq. (6)–(8) bias-corrected sandwich; unified `FGLS` class.
- Combined serial + heteroskedasticity estimator (F-C / F-CH, [PG26] Supplement
  S-4) with an optional first-stage correction; wild and sieve bootstrap covariances.
- Benchmarks: OLS with HAC (Andrews QS, Newey–West) or HC0–HC3, Cochrane–Orcutt,
  Durbin, infeasible GLS, hybrid GLS; `compare_ols()`; statsmodels cross-checks.
- Diagnostics: pre-determinedness LM test of [PG26] Remark 9 (experimental), ADF
  screening (`check_stationarity=True`), weight-concentration diagnostics.
- GMA estimator for MA(1) errors ([PG26] Section 5): `serial="ma1"` / `ma1=True`.
- Formula interface `from_formula` (optional `formula` extra, formulaic).
- `pyfgls.sklearn.FGLSRegressor` (scikit-learn estimator), `pyfgls.plotting`
  (`plot_diagnostics`, `plot_intervals`) and `FGLSResults.plot_diagnostics()`.
- `ic_penalty="params"` for the lag selection (penalise every Durbin coefficient a lag
  adds) and a warning when the selected order hits `kmax`; with many regressors the
  paper's order-based penalty tends to select `kmax`.
- `benchmarks/performance.py`: T = 10,000, q = 10, d = 100 runs in a few seconds.
- Documentation site (mkdocs-material): quickstart, guidance for practitioners (from the
  papers' concluding remarks), theory pages, how-tos, replication tables, API reference;
  four executed example notebooks; replication scripts in `benchmarks/`.

### Fixed
- [PG26] Remark 9 test: variable-addition form on the innovation scale (the n R^2
  version over-rejected with pre-determined but non-exogenous regressors).
