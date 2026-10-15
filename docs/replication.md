# Replication

The `tests/statistical` suite reproduces the papers' simulations with fewer replications
(`pytest --runslow tests/statistical`, run nightly in CI). The scripts in `benchmarks/`
produce full tables.

```bash
python benchmarks/replicate_pg26_tables.py --table 1 --reps 10000
python benchmarks/replicate_pg26_tables.py --table general --reps 2000
python benchmarks/replicate_gp24_tables.py --reps 2000 --T 100 200 400
```

Runtimes on one core of an Apple M1 Pro: about 1.5 minutes for Table 1 with 10,000
replications, 3–4 minutes for the general case with 2,000, and about 20 minutes for the
[GP24] tables with 2,000 replications at T = 100, 200, 400. The scripts use every core by
default (`--jobs -1`), which divides these times roughly by the number of cores; `--seed`
fixes the draws.

## [PG26] Table 1 (T = 500, AR(1) 0.8, non-exogenous regressor)

| Estimator | paper RMSE | pyfgls RMSE (5,000 reps) | paper bias | pyfgls bias |
|---|---|---|---|---|
| OLS | 0.399 | 0.398 | 0.395 | 0.394 |
| Durbin | 0.036 | 0.037 | 0.002 | 0.001 |
| Infeasible GLS | 0.025 | 0.024 | 0.000 | 0.000 |
| FGLS | 0.025 | 0.026 | 0.001 | 0.001 |
| Cochrane–Orcutt (two-step) | 0.041 | 0.041 | 0.031 | 0.031 |
| ρ̂ FGLS ($\sum_j\hat\rho_j$) | 0.034 | 0.034 | −0.008 | −0.008 |
| ρ̂ Cochrane–Orcutt | 0.174 | 0.174 | −0.171 | −0.170 |

## [PG26] general case (T = 200, $\rho_x$ = 0.8)

AR(2) errors (1.34, −0.42), exogenous regressor: MSE×100 of OLS 10.7 and FGLS 0.405
(paper: 11.5 and 0.41). Robust 95% intervals cover between 0.93 and 0.95 for AR(1) and
AR(2) errors with $\gamma \in \{0, 0.25, 0.5\}$; the naive intervals under-cover when
$\gamma \ne 0$.

## [GP24] Table 1 (slope eMSE of Lasso-FGLS relative to OLS)

pyfgls (2,000 replications per cell) / paper (10,000 replications):

| v(x) | T = 100 | T = 200 | T = 400 |
|---|---|---|---|
| 1 | 1.02–1.04 / 1.01 | 1.04 / 1.00 | 1.00 / 1.00 |
| x | 0.97 / 0.92 | 0.97 / 0.92 | 0.92 / 0.92 |
| x² | 0.77 / 0.70 | 0.75 / 0.70 | 0.70 / 0.70 |
| (log x)² | 0.58 / 0.54 | 0.52 / 0.46 | 0.46 / 0.46 |
| exp(0.2x + 0.2x²) | 0.51 / 0.49 | 0.49 / 0.45 | 0.45 / 0.45 |
| step | 0.80 / 0.74 | 0.78 / 0.74 | 0.74 / 0.73 |

At T = 400 every ratio matches the paper; at T = 100–200 the pyfgls ratios are 0.03–0.06
higher. The tuning-free benchmarks agree with the paper at T = 100 (WLS-S1: x 0.92 vs
0.91, x² 0.71 vs 0.68, (log x)² 0.52 vs 0.54; infeasible GLS within 0.02), and so do the
interval lengths of Table 2 (Lasso: x² 1.12 vs 1.10, (log x)² 0.32 vs 0.32, exp 1.35 vs
1.34), so the small-sample difference comes from the tuning of the adaptive Lasso (an
open item). The 95% intervals from eqs. (6)+(8) cover 0.94–0.96 throughout.

## [PG26] Supplement S-4: serial correlation and heteroskedasticity (T = 200)

Slope MSE × 100 and coverage of 95% intervals, 1,000 replications, candidate covariates
$W = (w, w^2, \cos w, \cos 2w)$.

| errors, v(·), driver | F-C | F-CH | Lasso only | OLS | coverage F-C / F-CH |
|---|---|---|---|---|---|
| AR(1) 0.5, x², w = x | 3.99 | 2.84 | 6.13 | 6.67 | 0.962 / 0.956 |
| AR(1) 0.5, (log x)², w = x | 0.47 | 0.22 | 0.65 | 0.79 | 0.960 / 0.960 |
| AR(1) 0.5, exp, w = x | 8.44 | 3.82 | 9.75 | 13.65 | 0.962 / 0.956 |
| AR(2) (1.34, −0.42), x² | 1.65 | 1.08 | 51.3 | 50.1 | 0.958 / 0.953 |
| MA(1) 0.5, x² | 3.95 | 2.92 | 5.41 | 6.21 | 0.958 / 0.947 |
| ARMA(1,1) (0.8, −0.4), x² | 4.29 | 3.24 | 6.52 | 7.15 | 0.958 / 0.956 |
| AR(1) 0.5, x², mixture driver | 3.90 | 2.40 | 5.17 | 6.19 | 0.957 / 0.956 |
| AR(1) 0.5, homoskedastic | 0.534 | 0.542 | 0.941 | 0.923 | 0.954 / 0.951 |

Non-exogenous regressor ($x_t + \gamma e_{t-1}$): coverage of the F-CH interval with and
without the first-stage correction is 0.942 / 0.777 (AR(1) 0.5, γ = 0.5) and 0.923 / 0.860
(AR(1) 0.8, γ = 1). The correction is therefore on by default for pre-determined
regressors.
