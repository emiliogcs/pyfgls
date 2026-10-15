# Lagged dependent variables

When lags of $y$ are regressors (e.g. a dynamic model $y_t = \alpha y_{t-1} + \gamma w_t + u_t$
with AR errors), OLS is inconsistent and the Durbin-regression coefficients on lagged
$y$ mix $\alpha$ with the AR coefficients of the errors. Declare the lagged dependent
variables so `pyfgls` can recover the AR coefficients from the lagged *stochastic*
regressors (the Wallis ratio, [PG26] Section 4.2):

```python
X = pd.DataFrame({"const": 1.0, "ylag": y.shift(1), "w": w}).iloc[1:]
res = pf.SerialFGLS(y.iloc[1:], X, lagged_dep_vars=["ylag"]).fit()
res.serial.wallis_regressors  # regressor(s) used for the ratio
```

* `wallis_regressor="auto"` (default) uses the regressor whose contemporaneous Durbin
  coefficient has the largest $|t|$; `"median"` takes the median of the ratios over all
  eligible regressors; or name one.
* At least one stochastic regressor that is not a lag of $y$ is required.
* If you forget to declare a regressor that equals a lag of $y$, `pyfgls` warns
  (`LaggedDependentWarning`) and refuses to run the standard algorithm.

This path is less tested than the standard one; the test suite checks it by simulation
(bias and coverage of $\hat\alpha$ and $\hat\gamma$).
