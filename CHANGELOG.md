# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

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
