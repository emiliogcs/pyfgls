# FAQ

**`pip install pyfgls` says "No matching distribution found". Why?**
`pyfgls` requires Python 3.10 or newer, and `pip` skips releases that do not support
the Python it runs on. Older versions of `pip` do not say that this is the reason. On
macOS, the built-in `python3` is 3.9. Check with `python3 --version`, install Python 3.10+
(python.org or `brew install python@3.12`) and use a virtual environment:
`python3.12 -m venv .venv && source .venv/bin/activate && python -m pip install pyfgls`.

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

**How do I silence the warnings?**
All warnings derive from `pyfgls.exceptions.PyFGLSWarning`:
`warnings.simplefilter("ignore", pf.exceptions.PyFGLSWarning)`.
