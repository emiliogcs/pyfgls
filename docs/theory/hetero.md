# Heteroskedasticity of unknown form ([GP24])

Model: $y = X\beta + u$, $E[u\mid X] = 0$, $\operatorname{Var}(u_t) = v_t$ with an unknown
variance function.

## Estimating the skedastic function

GLS weights each observation by $1/v_t$. [GP24] estimate $\log v_t$ by regressing
$\log\max(\hat u_t^2,\delta^2)$ — the log of the squared OLS residuals, floored to avoid
$\log 0$ — on a set of candidate covariates $z_t$ with an **adaptive Lasso**:

1. a ridge regression gives preliminary coefficients;
2. each candidate is penalised in inverse proportion to $|\hat\gamma^{\text{ridge}}_j|^{\psi}$;
3. the penalty level $\lambda$ and the exponent $\psi$ are chosen jointly by
   cross-validation.

With $\ell_t = \log\max(\hat u_t^2,\delta^2)$,

$$
\hat\gamma = \arg\min_{\gamma_0,\gamma}\ \sum_t\big(\ell_t - \gamma_0 - z_t'\gamma\big)^2
+ \lambda\sum_j \frac{|\gamma_j|}{|\hat\gamma^{\text{ridge}}_j|^{\psi}},
\qquad \hat v_t = \exp(\hat\gamma_0 + z_t'\hat\gamma),
$$

and the estimator is weighted least squares with weights $1/\hat v_t$:
$\hat\beta = (X'\hat V^{-1}X)^{-1}X'\hat V^{-1}y$ with $\hat V = \operatorname{diag}(\hat v_t)$.

The Lasso can handle many candidates, even more than observations, and selects a
parsimonious model. If nothing is selected the weights are constant and FGLS reduces to
OLS, so nothing is lost under homoskedasticity.

## Inference

Because the estimated variance function is only an approximation, `pyfgls` uses a
heteroskedasticity-robust sandwich for the weighted estimator (eq. (6)), with a
finite-sample bias correction that inflates the squared residuals by the leverage of
the weighted regression and by the degrees of freedom of the skedastic fit (eq. (8)).
The resulting intervals have close to nominal coverage and are much shorter than OLS
intervals under heteroskedasticity.

The parametric alternatives of Romano and Wolf (2017) (`"wls_s1"`, `"wls_s2"`) and a
support-vector regression (`"svr"`) are also available.
