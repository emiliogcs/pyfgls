# Interpreting `nonexogeneity_share`

The robust variance of [PG26] eq. (29) is the naive variance (eq. (27)) plus a term that
accounts for the estimation of the AR filter. That term is proportional to
$g_j = E[\tilde x_t u_{t-j}]$, which is zero when the regressors are exogenous.

`res.nonexogeneity_share` reports the average, over coefficients, of the correction
relative to the naive variance, and `res.serial.nonexogeneity_by_coef` gives it per
coefficient:

* **≈ 0** — the regressors behave as exogenous; naive and robust intervals coincide.
* **> 0.1** — the first-stage correction matters; the naive intervals would under-cover.

The robust variance is valid in both cases, so it is always the default. Declaring
`exog="exogenous"` does not change the estimates; it only adds a note to the summary when
the share is large.
