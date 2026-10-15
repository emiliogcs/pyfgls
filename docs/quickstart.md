# Quickstart

## Install

```bash
pip install pyfgls                    # core
pip install "pyfgls[formula,plot]"    # formula interface and diagnostic plots
```

## Serial correlation

```python
import pyfgls as pf
from pyfgls.simulate import serial_dgp

sim = serial_dgp(500, random_state=0)  # AR(1) errors, non-exogenous regressor
res = pf.SerialFGLS(sim.y, sim.X).fit()
print(res.summary())
res.ar_order, res.rho, res.bic_path  # selected order, AR coefficients, BIC path
res.nonexogeneity_share  # size of the eq. (29) correction
res.compare_ols()  # side by side with OLS + HAC
```

Inputs can be NumPy arrays or pandas objects; with pandas, names and the (time) index
are carried through to the results. The index must be sorted and, for dates and periods,
free of gaps (pass `allow_gaps=True` for trading-day data).

## Heteroskedasticity

```python
from pyfgls.simulate import hetero_dgp

sim = hetero_dgp(400, skedastic="x2", random_state=1)
res = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit()
res.selected_z, res.skedastic_coef, res.df_hat
```

By default the candidate covariates are `{x, x², log x, (log x)², cos x, cos 2x}` for each
non-constant regressor; pass `z=` a matrix, a DataFrame or regressor names to use your own.

## Both

```python
res = pf.FGLS(y, X, random_state=0).fit()  # F-CH
res.compare_stages()  # F-C (serial only) vs F-CH
```

## Formulas

```python
res = pf.FGLS.from_formula("y ~ x + lag(x)", data=df, missing="drop").fit()
```

## Inference options

| `cov_type` | Serial (`SerialFGLS`) | Heteroskedastic (`LassoFGLS`) | Combined (`FGLS`) |
|---|---|---|---|
| `"robust"` (default) | [PG26] eq. (29) | [GP24] eq. (8) / (7) | Suppl. (A.4)–(A.5) |
| `"naive"` / `"nonrobust"` | eq. (27) | model-based | — |
| `"HC0"`–`"HC3"` | eq. (29) with HC weights | White-type | HC0 / HC3 |
| `"bootstrap"` | sieve | wild | sieve (wild) |

Use `use_t=True` for Student-t critical values with `n_eff − q` degrees of freedom.
