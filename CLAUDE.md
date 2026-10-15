# CLAUDE.md — Build `pyfgls`: Feasible GLS for Python

> Save this file as `CLAUDE.md` in an empty directory, start Claude Code there, and say:
> **"Read CLAUDE.md fully, then execute Phase 0 through Phase 8 in order. Commit after every phase. Stop and report only if you hit a blocker listed under 'Stop conditions'."**

---

## 0. Mission

Build an open-source, pip-installable Python library, **`pyfgls`**, implementing the Feasible Generalized Least Squares (FGLS) methods of two papers by Emilio González-Coya and Pierre Perron:

- **[PG26]** Perron & González-Coya (2026), *Feasible GLS for time series regression*, Journal of Econometrics 258, 106346. → treatment of **serial correlation** (Durbin-regression-based FGLS, BIC-selected autoregressive filter, exogeneity-robust variance estimator).
- **[GP24]** González-Coya & Perron (2024), *Estimation in the Presence of Heteroskedasticity of Unknown Form: A Lasso-based Approach*, Journal of Econometric Methods 13(1), 29–48. → treatment of **heteroskedasticity** (adaptive-Lasso estimate of the skedastic function, bias-corrected robust variance).

The library must (a) implement each procedure faithfully, (b) combine them into one estimator for time series with both problems, (c) be robust to messy real-world inputs, and (d) be pleasant to use (statsmodels-like results, pandas-aware, readable summaries, helpful errors).

Quality bar: this should look like a serious econometrics package — typed, tested against the papers' own simulation tables, documented, CI-green, ready for PyPI.

### Where it lives
- Create a **new repository in the user's personal GitHub account** (not an organization). Use the `gh` CLI.
- Resolve the owner with `gh api user --jq .login`. If `gh auth status` fails, **stop** and ask the user to run `gh auth login` (see Stop conditions).
- Repo name `pyfgls`, **private at first** (`gh repo create pyfgls --private --source=. --remote=origin --push`). Tell the user at the end how to make it public (`gh repo edit --visibility public --accept-visibility-change-consequences`).
- **Do not commit the paper PDFs** (copyright). Cite them in `README`, `CITATION.cff` and docstrings.
- License: MIT. Author: Emilio González-Coya (use the git identity already configured; do not invent an email).

---

## 1. Notation (use consistently in code comments and docs)

Model: `y_t = x_t'β + u_t`, `t = 1..T`, `x_t ∈ R^q`.

**Serial correlation [PG26].** `u_t = C(L) e_t = Σ_{j≥0} c_j e_{t-j}`, `c_0 = 1`, invertible, so `A(L) u_t = e_t` with `A(L) = 1 − Σ_{j≥1} ρ_j L^j`. Innovations `e_t` are a martingale difference sequence.
- *Pre-determined* regressors: `E[x_t (e_t, e_{t+1}, …)] = 0` (the only condition FGLS needs).
- *Exogenous* regressors: additionally `E[x_t (e_{t-1}, e_{t-2}, …)] = 0`.
- Key facts the library must respect: OLS is inconsistent with non-exogenous regressors; FGLS is consistent with merely pre-determined regressors; **Cochrane–Orcutt is NOT valid with non-exogenous regressors** (it estimates ρ from OLS residuals); the **Durbin regression** gives consistent ρ̂ either way.

**Heteroskedasticity [GP24].** `Var(u_t | ·) = v_t(η)`, unknown form, `η` unknown covariates; candidate covariates `z_t ∈ R^d` (may exceed T). Diagonal `W = diag(v_t)`.

---

## 2. Algorithms to implement (the spec)

Implement exactly as below. Equation numbers refer to the papers. Where a detail is ambiguous, follow the "Defaults & decisions" in §9 and log the choice in the docstring.

### 2.1 Serial-correlation FGLS — [PG26] §4 (the core)

**Step 1 — Durbin regression, lag selection by BIC (eq. 14).**
For each `k ∈ {0,…,k_max}` run OLS of

```
y_t = Σ_{j=1..k} ρ_j y_{t-j} + x_t'β − Σ_{j=1..k} x_{t-j}'δ_j + e_{kt}
```

on the **common sample** `t = k_max+1,…,T` (Ng–Perron 2005 modified criterion so models are comparable):

```
BIC(k) = ln(σ̂²_k) + k · ln(T − k_max) / (T − k_max),   σ̂²_k = (T − k_max)^{-1} Σ ê²_{kt}
k* = argmin_k BIC(k)
```
Re-estimate with `k = k*` on its own effective sample (`t = k*+1..T`) and keep `ρ̂^D_1..ρ̂^D_{k*}` (coefficients on the lagged `y`). Only these are used downstream; `β̂^D` and `δ̂` are by-products.

**Robustness requirements for this step (important):**
- The lagged-`x` block must **drop columns that are collinear** with the contemporaneous block or each other: lagged constants, lagged deterministic trends, seasonal/dummy variables that are periodic, etc. Use a rank-revealing QR (or `scipy.linalg.qr(pivoting=True)`) to detect and drop them. Never solve normal equations; use `lstsq`/QR.
- Cap `k_max` so that the number of Durbin regressors is comfortably below the effective sample (e.g. ≤ `0.5 · (T − k_max)`); warn when the cap binds.
- `k_max` default: `int(12 * (T/200)**0.25)` (equals 12 at T=200, as in the [PG26] simulations); user-overridable. Also offer `kmax="theory"` = `⌊(T/ln T)^{1/3}⌋` (the rate used in Supplement Lemma A.2; much smaller in practice). Optionally apply the adjusted order `max(k*, ⌈(ln T)^{3/2}⌉)` via `min_lags="adjusted"` (Lemma A.2: restores the tail condition (23) when errors are a genuine AR(∞)); default off.
- `k* = 0` is a legitimate outcome → FGLS reduces to OLS (but still returns the robust variance).
- Allow `ar_order=<int>` to bypass selection, and `ic="bic" | "aic"`.

**Step 2 — Quasi-differenced regression (eq. 15).**
```
(y_t − Σ_j ρ̂_j y_{t-j}) = (x_t − Σ_j ρ̂_j x_{t-j})'β + e_{kt},   t = k*+1..T
```
OLS on the filtered data gives `β̂_FGLS`. Filter **all** columns including the constant (the filtered constant column is `1−Σρ̂_j`, so the coefficient is the original intercept — report it as such). Drop the first `k*` observations (no Prais–Winsten in the papers; may be added later as an option, off by default).

**Variance estimators (Theorem 2).**
Let `x̃̂_t = x_t − Σ_j ρ̂_j x_{t-j}` (filtered regressors), `û_t = y_t − x_t'β̂_FGLS` (**unfiltered** FGLS residuals), `ê_t` the residuals from step 2, effective sample `t = k+1..T`, `n_eff = T − k`.

- *Naive (eq. 27)*: `AVar = σ̂²_{e,k} [ (n_eff)^{-1} Σ x̃̂ x̃̂' ]^{-1}`, with `σ̂²_{e,k} = n_eff^{-1} Σ ê²`. Valid only under exogeneity (condition 25). Provide as `cov_type="naive"` for comparison.
- *Corrected, exogeneity-robust (eq. 28–29)* — **the default** (`cov_type="robust"`):
  1. `ĝ_j = n_eff^{-1} Σ_t x̃̂_t û_{t-j}`, `Ĝ_k = [ĝ_1,…,ĝ_k]` (q×k). Pair with the **raw** lagged residual `û_{t-j}`, not a projected one.
  2. `ψ̂_{t,k}` = residual vector (k×1) from the multivariate OLS regression of `(y_{t-1},…,y_{t-k})'` on `W_{t,k} = (1, w_t', w_{t-1}', …, w_{t-k}')'` (the stochastic regressors and their lags — the first-stage regression underlying eq. 14; handle collinear/deterministic columns as above). `Σ̂_k = n_eff^{-1} Σ ψ̂ψ̂'`.
  3. `Q̂_k = n_eff^{-1} Σ x̃̂ x̃̂'`.
  4. `AVar_R = σ̂²_{e,k} [ Q̂^{-1} + Q̂^{-1} Ĝ Σ̂^{-1} Ĝ' Q̂^{-1} ]` (second term is 0 when `k = 0`).
  5. `Var(β̂) = AVar_R / n_eff` (use `T` or `n_eff` consistently; default `n_eff`; document it).
  6. No kernel/bandwidth/truncation beyond the already-selected `k`. The correction is PSD by construction.
- *Diagnostic*: expose `result.nonexogeneity_share` = size of the correction relative to the naive term (e.g. `trace(Q̂^{-1}ĜΣ̂^{-1}Ĝ'Q̂^{-1}) / trace(Q̂^{-1})`). Values near 0 mean regressors behave as exogenous ([PG26], discussion after Thm 2).
- Wald CI: `c'β̂ ± z_{1−α/2} (c' AVar_R c / n_eff)^{1/2}`; also t/z p-values (normal by default; `use_t=True` option with `n_eff − q` dof).

**Lagged dependent variables as regressors (§4.2, eq. 30).**
If the model contains `p_y` lags of `y` among `x` (user declares them via `lagged_dep_vars=[...]`, since Assumption 5(ii) fails otherwise), run the Durbin regression with order `k* + p_y`, and recover `ρ̂_j = δ̂_{j,i} / β̂_i` for a chosen non-lagged-dep regressor `i` (Wallis 1967). Provide `wallis_regressor="auto"` (the regressor with largest |t| among eligible ones) | name | `"median"` (median of ratios across eligible regressors); warn if |β̂_i| is tiny. Mark as less-tested; cover by a simulation test.

**Diagnostics for pre-determinedness (Remark 9).**
`pyfgls.diagnostics.test_predetermined(y, X, max_lag)`:
(a) Durbin regression → `β̂^D`; (b) `û^D = y − Xβ̂^D`; (c) demean; (d) LM variable-addition test (`LM = n·R²` from regressing demeaned `û^D` on `X` plus candidate lags of the stochastic regressors, sequentially for lags 1,2,…), χ² p-values. Return a tidy result with advice ("add lag(s) of x_j as regressors, then refit"). Flag as **experimental** in the docs.

### 2.2 Heteroskedasticity FGLS with Lasso — [GP24] §§2–4

Setup: `y = Xβ + u`, `E[u|X]=0`, `Var(u_t) = v_t`.

**Steps (Section 2):**
1. OLS → `û_t`.
2. Fit the skedastic function by regressing the response `log(max(û_t², δ²))` with `δ = 0.1` (floor avoids log 0; make `delta` configurable) on candidate covariates `z_t`.
3. `ū_t² = exp(fitted value)` → `W̄ = diag(ū_t²)`; `β̂_FGLS = (X'W̄^{-1}X)^{-1} X'W̄^{-1}y`. (Solve as OLS on rows scaled by `ū_t^{-1}`. Normalize weights to mean 1; this does not change β̂.)

**Skedastic-function estimators (`skedastic=`):**
- `"lasso"` (default, recommended by [GP24]) — **Adaptive Lasso** (Zou 2006), eq. (4):
  1. *Ridge first step*: `γ̂^ridge` with λ^r chosen by K-fold CV (works for `d > T`).
  2. Weights `ŵ_j = |γ̂_j^ridge|^{-ψ}`.
  3. Solve `min_γ ½ Σ_t (log û²_t − γ_0 − Σ_j z_tj γ_j)² + λ Σ_j ŵ_j |γ_j|`; intercept unpenalized.
  4. **Tune (λ, ψ) jointly by K-fold CV**, `K = 10`, `ψ ∈ {0, 0.25, 0.5, 0.75, 1, 2}` (L = 6), the **same fold assignment across all ψ** so CV errors are comparable; choose the (λ*, ψ*) minimizing the CV mean squared prediction error of `log(û²)`.
  - Implementation: standardize `z` columns (as glmnet does) and rescale column j by `1/ŵ_j` so that an ordinary `sklearn.linear_model.lasso_path`/`Lasso(coordinate descent)` solves the adaptive problem; back-transform coefficients. Columns with ŵ = ∞ (ridge coefficient exactly 0) are dropped. `ψ = 0` is the plain Lasso.
  - If all slope coefficients are zero → constant weights → FGLS ≡ OLS. **This must work gracefully** (it is the paper's "no loss under homoskedasticity" property).
  - Options: `cv_folds=10`, `psi_grid=(0,.25,.5,.75,1,2)`, `lambda_rule="min" | "1se"` (default `"min"`, as in the paper), `random_state`, `fold_scheme="random" | "blocked"`.
- `"svr"` — ε-insensitive SVR with Gaussian kernel, (ε, λ) chosen by CV (`sklearn.svm.SVR`). Second-tier; the paper finds it dominated by Lasso. `df̂ = |ε₊| + |ε₋|` = number of observations with residual at or beyond the ε-tube (≈ number of support vectors).
- `"wls_s1"` / `"wls_s2"` — Romano & Wolf (2017) baselines, eq. (2)–(3): `v̄(z) = exp(c̃ + γ̃' [log z_1,…,log z_d])` (S1) and `exp(c̃ + γ̃' z)` (S2), fitted by OLS of `log û²` on `log z` / `z`. Require positive `z` for S1 (validate).
- `"none"` → plain OLS (+ HR variance).
- `"known"` / callable / array → user-supplied weights or variance function (also used for the infeasible-GLS benchmark).

**Candidate covariates `z_t` (`z=`):**
- User-supplied matrix/DataFrame/column names, or
- `z="auto"` (default): a flexible dictionary built from the non-constant regressors in `X`: `{x, x², log(x) [only if x>0], cos(x), cos(2x)}` per regressor — the specification used in [GP24] §5 (`z = (1, x, log(x)², x², cos x, cos 2x)`; note the paper uses `[log x]²`; include **both** `log(x)` and `log(x)²` when x>0). Skip transforms that are undefined/constant, de-duplicate, and warn if `d` is very large. Also provide `pyfgls.hetero.features.make_dictionary(X, transforms=("poly2","log","trig"))` so users can extend.
- Sparse/dummy columns are passed through untransformed.

**Variance / confidence intervals (§4, eq. 5–8)** — essential, because the skedastic fit is only an approximation:
- `Var(β̂_FGLS) = (T^{-1}X'Ŵ^{-1}X)^{-1} Ω̂ (T^{-1}X'Ŵ^{-1}X)^{-1} / T`, `Ω̂ = T^{-1} X'Σ̂X`, `Σ̂` diagonal (eq. 6).
- **Lasso/SVR (eq. 8, Miller–Startz bias correction):**
  ```
  Σ̂_tt = (û²_{t,FGLS} / (ū_t²)²) · ( 1/(1 − h_{t,FGLS})² + 4 · (h_{t,OLS}/k) · d̂f )
  ```
  with `h_{t,FGLS} = [X (X'Ŵ^{-1}X)^{-1} X'Ŵ^{-1}]_tt`, `h_{t,OLS} = [X(X'X)^{-1}X']_tt`, `k` = number of regressors, `û_{FGLS} = y − Xβ̂_FGLS`, `d̂f` = number of nonzero Lasso coefficients (Zou–Hastie–Tibshirani 2007; **include the intercept, document it**) or the SVR df above.
- **WLS-S1/S2 (eq. 7, Romano–Wolf)**: `Ω̂₁ = T^{-1} Σ ẽ_t² x_t x_t' / (ū_t²)²` with the HC3-type OLS residual `ẽ_t² = û_t² / (1 − h_{t,OLS})²`. (If the printed definition looks ambiguous, use HC3 on the OLS residuals and note it.)
- Also expose plain `cov_type="HC0".."HC3"` and the OLS comparator with HC3 SEs.
- Wald CIs `β̂_k ± z_{1−α/2} SE`.

### 2.3 Combined estimator: serial correlation **and** heteroskedasticity

**Source: [PG26 Supplement], Section S-4, eqs. (A.2)–(A.7)** ("F-C" = corrects serial correlation only; "F-CH" = corrects both). Implement exactly:

1. **(A.3)** Run §2.1 → `ρ̂^D_1..ρ̂^D_{k*}`, `k*`; OLS on the quasi-differenced regression gives residuals `ê_{tk}`, `t = k*+1..T`. (This is `β̂_{F-C}`.)
2. **Skedastic fit**: Adaptive Lasso (§2.2: ridge → weights → joint CV over (λ, ψ), K=10, ψ-grid as above) of `log(max{ê²_{tk}, δ²})` (`δ = 0.1`) on `z_t`, where `z_t` may contain some or all elements of the (raw, **unfiltered**) `x_t` and transformations of them. Default `z="auto"` dictionary built from the raw non-constant regressors (supplement uses `W = (1, w, w², cos w, cos 2w)`), aligned to `t = k*+1..T`. Predicted values `ṽ_t ≡ ẽ²_{tk}` (exp of the fitted log-variance).
3. **WLS** on the quasi-differenced regression with weights `1/ẽ²_{tk}` → `β̂_{F-CH}` (OLS on `y*_t = (y_t − Σρ̂_j y_{t-j})/ẽ_{tk}`, `x*_t = (x_t − Σρ̂_j x_{t-j})/ẽ_{tk}`; eqs. A.6–A.7).
4. **Variance (A.4)–(A.5)** — the same Miller–Startz/Rothenberg bias-corrected sandwich as §2.2 eq. (6)/(8), applied to the **filtered** regressors:
   ```
   Var(β̂_{F-CH}) = (T⁻¹ X'W̃⁻¹X)⁻¹ Ω̂ (T⁻¹ X'W̃⁻¹X)⁻¹ / T,   Ω̂ = T⁻¹ X'Σ̂X   (X = filtered regressors, t=k*+1..T)
   Σ̂_tt = ( ê²_{t,F-CH} / (ẽ²_{tk})² ) · ( 1/(1 − h_{t,F-CH})² + 4 · (h_{t,F-C} / k) · d̂f )
   ```
   where `ê_{t,F-CH} = y*_t − x*_t'β̂_{F-CH}` are the FGLS residuals of the weighted filtered regression, `h_{t,F-CH}` = leverage of the weighted filtered regression, `h_{t,F-C}` = leverage of the **unweighted** filtered (F-C) regression (the analogue of `h_{t,OLS}` in eq. 8), `k` = number of regressors, `d̂f` = number of nonzero Lasso coefficients (Zou–Hastie–Tibshirani 2007). CIs: `β̂_k ± z_{1−α/2} SE`.
   - The supplement states that F-C confidence intervals are built "the same way", i.e. with the same remaining-heteroskedasticity correction (use unit weights, `h_{t,F-C}` in both places).
   - The supplement does **not** write out the first-stage (Durbin) correction of eq. (29) for the combined case. Provide it as an option `first_stage_correction=True|False` (default `True` when `exog="predetermined"`, i.e. add the eq. 29 term computed from weighted filtered quantities; default `False` when the user declares `exog="exogenous"`), and **validate by simulation with γ ≠ 0** (see §6). If the default-True version over-covers noticeably, report numbers rather than silently switching.
   - Always provide `cov_type="bootstrap"` (below) as an independent cross-check.
5. `FGLS(serial="auto", hetero="lasso")` is the headline call; `serial=None` → pure §2.2 estimator, `hetero=None` → pure §2.1 estimator. Expose both stage results (`res.stage_serial`, `res.stage_hetero`) so users can see F-C vs F-CH side by side (`res.compare_stages()`).
6. Under homoskedasticity F-CH must be essentially equivalent to F-C (no loss), as shown in the supplement.

**Bootstrap cross-check (`cov_type="bootstrap"`)**: wild bootstrap (Rademacher or Mammen) on the filtered innovations for the heteroskedasticity-only case; recursive/residual-based sieve bootstrap using the estimated AR(k*) for the serial case; `n_boot=999`, seeded, optional `n_jobs`. Used mainly as a validation tool and for small samples.

### 2.4 Benchmarks / comparators (needed for tests, docs and `compare()`)
- OLS with **HAC** (Andrews 1991 quadratic-spectral kernel, automatic bandwidth from AR(1) approximation: `w(ℓ) = (3/z²)(sin z / z − cos z)`, `z = 6πℓ/5`) and Newey–West.
- **Cochrane–Orcutt** (iterated optionally) — with a documented warning that it is invalid for non-exogenous regressors.
- **Durbin** estimator (β̂^D from eq. 14 directly).
- **Infeasible GLS** given a known `Ω` (AR coefficients or full matrix) — for simulation benchmarks only (`pyfgls.benchmarks`).
- **GMA** for MA(1)-type errors ([PG26] §5): `y*_t = x*_t'β + e_t` with `y*_t = Σ_{j=0}^{t-1} (−θ̂)^j y_{t-j}`, `x*_t = Σ_{j=0}^{t-1} (−θ̂)^j x_{t-j}`, `θ̂` the MLE of `ū_t = e_t + θ e_{t-1}` on OLS residuals; `Var = σ̂²_e (X*'X*)^{-1}`; `k_T = int(4 (T/100)^{2/9})`. Lower priority; implement in Phase 6.

---

## 3. Public API (design target — keep it this simple)

```python
import pyfgls as pf

# arrays or pandas; names preserved
mod = pf.FGLS(y, X, serial="auto", hetero="lasso")      # defaults shown
res = mod.fit()                                          # cov_type="robust"
print(res.summary())

# formula interface (optional extra: pip install pyfgls[formula])
res = pf.FGLS.from_formula("y ~ x1 + x2", data=df, serial="auto", hetero=None).fit()

# single-purpose shortcuts
pf.SerialFGLS(y, X, ar_order=None, ic="bic", kmax=None)         # paper PG26
pf.LassoFGLS(y, X, z="auto", skedastic="lasso")                 # paper GP24 (cross-section)

# results
res.params, res.bse, res.tvalues, res.pvalues, res.conf_int(alpha=0.05), res.cov_params()
res.resid, res.fittedvalues, res.predict(X_new)
res.ar_order, res.rho, res.ar_roots, res.bic_path            # serial block
res.skedastic_coef, res.selected_z, res.weights, res.df_hat  # hetero block
res.nonexogeneity_share                                       # PG26 diagnostic
res.wald_test("x1 = x2"), res.compare_ols()                   # tidy comparison vs OLS-HAC
res.plot_diagnostics()                                        # optional (matplotlib extra)
res.to_frame()                                                # coef table as DataFrame

# comparisons
pf.compare([ols_res, fgls_res])                               # side-by-side table
```

Design rules:
- Parameters are explicit keyword arguments with documented defaults; no magic globals.
- `serial ∈ {"auto", "fixed", "ma1", None}`, `hetero ∈ {"lasso","svr","wls_s1","wls_s2","known", None}`, `cov_type ∈ {"robust","naive","HC0-3","bootstrap"}`, `exog ∈ {"predetermined" (default), "exogenous"}` (the latter only affects diagnostics/warnings; the robust variance is valid under both).
- Results objects are immutable dataclasses with `__repr__`, `summary()` (monospace table: coef, SE, z, p, CI; header with n_eff, k*, ρ̂, selected z's, df̂, variance type) and `to_dict()` / `to_json()`.
- Also ship a **scikit-learn-compatible** wrapper `pyfgls.sklearn.FGLSRegressor` (`fit/predict/get_params/set_params`, `coef_`, `intercept_`) — nice-to-have in Phase 7.
- Public API frozen with `__all__`; everything else underscore-private.

---

## 4. Robustness checklist (build all of it)

**Input handling**
- Accept `ndarray`, `Series`, `DataFrame`, lists; convert to `float64`; keep column names and index; `y` 1-D (squeeze (T,1)).
- Time ordering: for a `DatetimeIndex`/`PeriodIndex` require monotone increasing, no duplicates; detect and raise on **gaps** unless `allow_gaps=True` (then warn that lags are positional). Panels are out of scope (clear error).
- Missing values: `missing="raise"` (default) | `"drop"` (leading/trailing only for time series; interior NaN → raise with location) .
- Constant/collinear regressors: rank check via QR with clear message naming offending columns; automatic drop only where safe and warned (lagged-block in Durbin).
- Detect an intercept; warn if absent (still works).
- Detect obvious lagged dependent variable among `X` (corr(y_t−1, x) ≈ 1 or equal) and warn to declare `lagged_dep_vars`.
- `T` too small relative to `q`, `k_max`, folds → informative errors/warnings, never silent nonsense.

**Statistical sanity**
- Check estimated AR polynomial: roots of `1 − Σρ̂_j z^j` must be outside the unit circle (Assumption 2); warn if min modulus < 1.05 (near unit root; suggest differencing/unit-root tests). Never silently "fix" ρ̂.
- Stationarity screening (optional `check_stationarity=True`: ADF on `y` and regressors, warn only).
- Weight diagnostics: warn if `max(w)/min(w) > 1e6`; optional `weight_clip=(lo, hi)` quantile winsorization of the fitted log-variance (default `None`, so default behaviour matches the papers).
- Heteroskedasticity floor `delta`: default `0.1` as in the paper (document that it is in the units of `y`); expose a scale-aware opt-in `delta_scale=True` (floor = `0.1·σ̂_OLS`).
- Covariance matrices: symmetrize, check PSD (eigen-clip with a warning), use `scipy.linalg.solve`/Cholesky with fallback to `pinv`; report condition numbers on request.
- Numerical: QR/`lstsq` everywhere; no explicit inverses of X'X except small q×q for variance (via solve); float64 only; deterministic with `random_state`.

**Engineering**
- Python ≥ 3.10; dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`; extras: `[formula]` (formulaic), `[plot]` (matplotlib), `[stats]` (statsmodels, used only in tests/benchmarks), `[dev]`, `[docs]`.
- Fully type-hinted, `py.typed`, `mypy --strict` clean, `ruff` (lint+format), `pre-commit`.
- Custom warnings/exceptions in `pyfgls.exceptions` (`PyFGLSWarning`, `NonStationaryWarning`, `CollinearityWarning`, `ConvergenceWarning`, `InputError`…) so users can filter them.
- Logging via `logging` (no prints); `verbose` flag.
- Vectorized implementations; the lag-selection loop and CV must run for T=10,000, q=10, d=100 in seconds. Provide `n_jobs` for CV/bootstrapping (joblib).
- Reproducible: every stochastic routine takes `random_state` (int | Generator).

---

## 5. Repository layout

```
pyfgls/
├─ pyproject.toml            # hatchling, PEP 621, extras, ruff/mypy/pytest config
├─ README.md  LICENSE  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md  CLAUDE.md
├─ .github/workflows/{ci.yml,release.yml,docs.yml}
├─ .pre-commit-config.yaml
├─ src/pyfgls/
│  ├─ __init__.py  _version.py  py.typed  exceptions.py
│  ├─ models.py            # FGLS, SerialFGLS, LassoFGLS (user-facing)
│  ├─ results.py           # FGLSResults, summary, compare
│  ├─ serial/              # durbin.py, lagselect.py, filter.py, variance.py (eq 27, 28–29), ma.py, wallis.py
│  ├─ hetero/              # features.py, adaptive_lasso.py, svr.py, wls.py, weights.py, variance.py (eq 6–8)
│  ├─ combined.py          # §2.3
│  ├─ covariance/          # hac.py (Andrews QS), hc.py, bootstrap.py
│  ├─ benchmarks.py        # OLS, CO, Durbin, infeasible GLS
│  ├─ diagnostics.py       # test_predetermined, ar_roots, nonexogeneity_share, weight checks
│  ├─ simulate.py          # DGPs from both papers (see §6)
│  ├─ sklearn.py           # FGLSRegressor wrapper
│  └─ _validation.py  _linalg.py  _utils.py
├─ tests/                  # unit/, property/, statistical/ (marked slow)
├─ docs/                   # mkdocs-material: quickstart, theory, API ref, replication notebooks
├─ examples/               # 01_serial_correlation.ipynb, 02_heteroskedasticity.ipynb, 03_both.ipynb, 04_nonexogenous.ipynb
└─ benchmarks/             # replicate_pg26_tables.py, replicate_gp24_tables.py
```

---

## 6. Validation against the papers (acceptance tests)

Implement `pyfgls.simulate` with the papers' DGPs, then write **statistical tests** (`@pytest.mark.slow`, run nightly + on demand; use fewer replications, with tolerances covering Monte-Carlo noise; fixed seeds).

**[PG26] serial correlation**
- *AR(1), non-exogenous regressor (their Table 1):* `y_t = x_t'β + u_t`, `u_t = ρ u_{t-1} + e_t`, `x_t = (1, w_t)'`, `w_t = v_t + e_{t-1}`, `v_t, e_t ~ iid N(0,1)` independent, `β = (1,1)'`, `ρ = 0.8`, `T = 500`, `u_0 = 0`. Paper values for the **slope** (10,000 reps): OLS RMSE 0.399 / bias 0.395; Durbin RMSE 0.036; infeasible GLS 0.025; **FGLS RMSE 0.025, bias ≈ 0.001**; Cochrane–Orcutt FGLS RMSE 0.041, bias 0.031; ρ̂ from FGLS RMSE 0.034, bias −0.008; CO's ρ̂ bias −0.171. Test with ~1,000 reps and ±15% relative tolerance on RMSE, absolute ±0.01 on biases.
- *General case (their Tables 2–5):* `x_{1t} = ρ_x x_{1,t-1} + v_t + γ e_{t-1}`, `ρ_x = 0.8`, `γ ∈ {0, 0.25, 0.5}` (γ ≠ 0 ⇒ non-exogenous), AR(1) `ρ_u ∈ {−0.5,0,0.2,0.5,0.8}`, AR(2) `(1.34,−0.42)`, `(0.5,−0.3)`, etc., MA(1) θ∈{−0.7,−0.4,0.5}, `T ∈ {200,500}`, `k_max = 12`. Assertions: (i) FGLS MSE ≤ OLS MSE in essentially all cells and close to infeasible GLS; (ii) **coverage of the robust (eq. 29) 95% CI lies in [0.92, 0.96]** for AR(1)/AR(2) with γ ∈ {0, .25, .5}; (iii) the **naive (eq. 27) variance under-covers when γ ≠ 0** (coverage clearly < robust) and ≈ robust when γ = 0; (iv) OLS+HAC under-covers with strong persistence; (v) AR(2) (1.34, −0.42), γ=0, T=200: OLS MSE ×100 ≈ 11.5 vs FGLS ≈ 0.41 (≈ 96% reduction).
- Median selected `k*` should be 1 for AR(1) and 2 for AR(2) (paper's `Med. k` column).

**[GP24] heteroskedasticity**
- DGP (eq. 9): `y_t = α + x_t β + √v(x_t) ε_t`, `x_t ~ U[1,4]`, `ε ~ N(0,1)`, `(α,β) = (1,1)`, `T ∈ {100,200,400}`. Skedastic functions: `x^γ` (γ = 0,1,2), `[log x]²`, `exp(0.2x + 0.2x²)`, step (1 on [1,2), 4 on [2,3), 9 on [3,4)). Candidate `z = (1, x, log(x)², x², cos x, cos 2x)`.
- Target eMSE of Lasso-FGLS **relative to OLS** (their Table 1): v=1 → ≈1.00–1.01; v=x → ≈0.92; v=x² → ≈0.70; v=[log x]² → ≈0.46–0.54; v=exp(.) → ≈0.45–0.49; step → ≈0.73–0.74. Tolerance ±0.05 with ≥2,000 reps. Also assert Lasso-FGLS never exceeds OLS eMSE by more than ~0.03 under homoskedasticity.
- Coverage of eq. (6)+(8) CIs ≈ 0.95 (±0.015), and average length < OLS-HC length under heteroskedasticity (e.g. v=[log x]²: ≈0.31–0.32 vs 0.44 at T=100).
- Extended robustness (their Table 3/Cases 1–5): add `k_w = 20` extra covariates `w_j ~ U(1,4)` correlated or uncorrelated with x, `z` = {x, w, cos(w), cos(2w), …} (up to ~60–63 covariates, T=200): Lasso-FGLS must not be noticeably worse than OLS (ratio ≤ ~1.06) even when the true driver is excluded (Case 5), and must beat OLS for strong heteroskedasticity when the driver (or a correlated proxy) is included.

**[PG26 Supplement] combined serial + heteroskedasticity (F-C vs F-CH, Table S.9)**
- DGP: `u_t` AR(1) `0.5`; AR(2) `(1.34, −0.42)`; MA(1) `θ = 0.5`; ARMA(1,1) `(0.8, −0.4)`, with `e_t = √v_t(z) ε_t`, `ε_t ~ iid N(0,1)`; `v(x) = x²`, `[log x]²`, `exp(0.2x + 0.2x²)`; `T = 200`; `W = (1, w, w², cos w, cos 2w)` with (a) `w = x` and (b) `w = φx + (1−φ)q`, `q ~ U(1,4)`, `φ ~ Bernoulli(0.75)`.
- Assertions: F-CH MSE ≤ F-C MSE (gain larger for stronger heteroskedasticity, smaller in case b but still positive); all estimators ≈ unbiased (gain is variance reduction); F-CH coverage ≈ 0.95 (±0.02); F-CH CI shorter than OLS-HAC and F-C; under homoskedastic errors F-CH ≈ F-C (no loss).
- Repeat with `γ ≠ 0` (non-exogenous regressor) to decide the `first_stage_correction` default (§2.3).
- *Predictive regression (Supplement S-3):* `y_{t+2} = α + β x_t + u_{t+2}`, `u_t = e_t + θ e_{t-1}`, `x_t = ρ_x x_{t-1} + v_t + γ e_{t-1}`, `ρ_x = 0`, `γ ∈ {0, 0.5}`, `θ ∈ {−0.7, −0.4, 0.5}`, `T ∈ {200, 500}`: FGLS and GMA MSE ≪ OLS; known weakness is under-coverage (≈0.84–0.87) at `γ = 0.5, θ = −0.7` for FGLS (BIC truncation) while GMA stays near 0.95 — reproduce and document this as a known limitation, not a bug.
- *Robustness of quasi-differencing (Supplement S-1):* with exogenous regressors, GLS with a *wrong* AR(1) ρ of the same sign as `cor_u(1)` has lower MSE than OLS (Corollary A.1: for AR(1) errors and iid regressors, the asymptotic MSE ratio is < 1 iff `ρ(1+ρ_u²) − 2ρ_u(1+ρ²) < 0`; check by simulation with `ρ_u = 0.5`, `ρ ∈ (0.1, 0.9)`, `T = 200`, `x ~ N(0,1)`). Optional helper `pyfgls.benchmarks.hybrid_gls` (Godfrey LM test; if rejected quasi-difference with any ρ ~ U(0.1,0.9)/U(−0.9,−0.1) matching the sign of the sample first-order autocorrelation of OLS residuals) — **exogenous regressors only**; low priority, Phase 5.

**Exact-algebra unit tests**
- AR(1) quasi-differencing equals explicit `D` matrix GLS (eq. 8 of PG26 form) up to numerical tolerance for small T.
- FGLS with a known ρ and fixed weights equals the closed-form GLS `(X'Ω⁻¹X)⁻¹X'Ω⁻¹y`.
- BIC path picks `k=0` for white-noise errors with high probability; picks true order for AR(p) with large T.
- Rescaling invariance: scaling a column of X by c scales its coefficient by 1/c and leaves t-stats unchanged; scaling y scales β̂ and SEs, leaves t-stats and selected `k*` unchanged (the `delta` floor aside — test with `delta` scaled accordingly).
- Adaptive-Lasso solver matches a reference implementation (e.g. `glmnet`-style brute force or `sklearn` path + rescaling) on small problems; `ψ=0` equals sklearn Lasso.
- Lagged-block collinearity: constant + trend + dummy regressors don't crash Durbin and don't change `ρ̂`.
- Property tests (`hypothesis`): row-permutation is **not** allowed (time order) — assert the API rejects unsorted index; equivariance under affine transformations of y; robust variance PSD.
- Cross-checks vs `statsmodels` where possible (HAC variance, OLS, `GLSAR` for the exogenous AR(1) case within tolerance).

If the author's original Python Monte-Carlo scripts/tables exist in the working folder or are supplied (e.g. a `TablesJan2026/code/` directory), **use them as the reference implementation for numerical comparison** rather than re-deriving; do not copy proprietary data into the repo without asking.

---

## 7. Documentation & UX

- `README.md`: 30-second pitch, install, 3 code examples (serial, hetero, both) with printed `summary()` output, a plot of OLS-HAC vs FGLS intervals, "When should I use this?" bullets (when OLS is inconsistent; when GLS beats OLS; non-exogenous regressors; unknown-form heteroskedasticity), citations, badges (CI, coverage, PyPI, license).
- Docs site (mkdocs-material + mkdocstrings): Quickstart; **Theory** pages summarizing each paper's logic with equation references (own words, no copied text); **How-to** (lagged dependent variables, many candidate z's, interpreting `nonexogeneity_share`); API reference; FAQ ("Why not Cochrane–Orcutt?", "Why robust variance?"); replication pages showing the paper tables regenerated.
- Docstrings in NumPy style with equation numbers and a runnable example each (doctests on).
- Error messages: say what is wrong, where (column/index), and how to fix.
- `CITATION.cff` with both papers; `README` BibTeX.
- Four example notebooks under `examples/` executing top-to-bottom in CI (nbmake or papermill).

---

## 8. Phased work plan (commit at the end of each; run `ruff`, `mypy`, `pytest` before committing)

| Phase | Deliverable | Done when |
|---|---|---|
| **0** | Repo bootstrap: git init, `gh` auth check, `pyproject.toml`, src layout, CI skeleton, pre-commit, LICENSE, empty package imports; create private GitHub repo and push | `pip install -e .[dev]` works; CI green on first push |
| **1** | `_validation`, `_linalg`, `exceptions`, lag-matrix utilities, QR rank handling | unit tests pass |
| **2** | `serial/`: Durbin regression, BIC selection, filter, FGLS β̂, naive & robust variance, lagged-dep-var (Wallis) path, `SerialFGLS` + results/summary | PG26 Table-1 and AR(2) acceptance tests pass |
| **3** | `hetero/`: feature dictionary, adaptive Lasso (ridge → weights → CV over λ,ψ), SVR, WLS-S1/S2, weights, eq. 6–8 variance, `LassoFGLS` | GP24 Table-1/Cases acceptance tests pass |
| **4** | `combined.py` + `FGLS` unified class + bootstrap cross-check | combined simulation: coverage within [0.91, 0.97], MSE ≤ either single-purpose estimator within tolerance |
| **5** | Benchmarks module (OLS-HAC Andrews QS, CO, Durbin, infeasible GLS), `compare()`, `compare_ols()`, `wald_test` | cross-checks vs statsmodels pass |
| **6** | Diagnostics (`test_predetermined`, roots, weight checks), GMA/MA(1) option, formula interface | tests pass |
| **7** | sklearn wrapper, plotting extra, performance pass (T=10k benchmark), `n_jobs` | benchmark script documented in README |
| **8** | Docs site, notebooks, README polish, CITATION, CHANGELOG, `v0.1.0` tag; TestPyPI dry-run (do **not** publish to PyPI without the user's go-ahead) | `mkdocs build --strict`, notebooks execute, CI green on 3.10–3.13 × Linux/macOS/Windows |

At the end print: repo URL, how to run the replication scripts, and a short list of open items (§9).

---

## 9. Defaults & decisions (where the papers leave room) — and open items for the author

Defaults chosen (document each in the relevant docstring, make all overridable):
1. `k_max = int(12·(T/200)^{1/4})` (12 at T=200). [PG26] says only that `k_max` grows with T and uses 12 at T=200.
2. BIC as in the papers (Ng–Perron effective-sample version); `min_lags=None` (adjusted-order option available).
3. Lasso `df̂` = number of nonzero coefficients **including the intercept**; SVR `df̂` = number of obs at/outside the ε-tube.
4. HC3 for the Romano–Wolf variance (eq. 7).
5. Variance scaling by `n_eff = T − k*`.
6. Heteroskedasticity floor `δ = 0.1` (paper default; scale-aware option opt-in).
7. Adaptive-Lasso CV: random K-fold (K=10), `λ` at CV minimum.
8. Drop the first `k*` observations (no Prais–Winsten).
9. Normal critical values by default (the papers' asymptotic CIs); t option available.

**Open items — implement a reasonable version, flag clearly, and list in the final report (the user will confirm against the Supplement):**
- **A. Combined variance (§2.3 step 4)**: the supplement (A.4)–(A.5) gives the sandwich with the Miller–Startz correction but no explicit first-stage (Durbin) term for the combined case. Implemented as an option (`first_stage_correction`); decide the default from the simulations and confirm.
- **B. Exact form of `ẽ_t`** in the printed eq. (7) (HC3 residual definition) — use HC3, confirm.
- **C. Multi-regressor Wallis recovery** (`ρ̂_j = δ̂_{j,i}/β̂_i`): which regressor to use; implemented `"auto"`, `"median"`, named; confirm preferred default.
- **D. LM variable-addition test details (Remark 9)**: implemented as described; confirm the intended auxiliary regression.
- **E. Whether to add Prais–Winsten initial observations** (not in the papers).

---

## 10. Stop conditions (otherwise keep going autonomously)

Stop and ask the user **only** if: (1) `gh` is unauthenticated or the owner/repo name is taken; (2) a required package cannot be installed; (3) an acceptance test in §6 fails after two principled fix attempts and the discrepancy appears to be an error in the spec rather than the code (report the numbers); (4) you are about to publish anything outside the private GitHub repo (PyPI, public visibility).

## 11. Working style

- Plan with a task list; commit small, conventional-commit messages (`feat:`, `fix:`, `test:`, `docs:`).
- Write tests first for the numerical core; never loosen a test tolerance to make it pass without noting why.
- Prefer clarity over cleverness; keep equation numbers in comments (`# PG26 eq. (29)`).
- Do not copy text from the papers; paraphrase theory in docs and cite.
- When in doubt about statistics, simulate it.
