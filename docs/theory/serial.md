# Serial correlation ([PG26])

Model: $y_t = x_t'\beta + u_t$ with $u_t = C(L)e_t$ invertible, so that
$A(L)u_t = e_t$ with $A(L) = 1 - \sum_{j\ge1}\rho_j L^j$ and $e_t$ a martingale
difference sequence.

## Exogenous versus pre-determined regressors

The regressors are *pre-determined* when $x_t$ is uncorrelated with current and future
innovations, and *exogenous* when, in addition, it is uncorrelated with past innovations.
Many time-series regressors are only pre-determined: they react to past shocks.

Under pre-determinedness alone, OLS is generally **inconsistent** — its error $u_t$
contains past innovations that are correlated with $x_t$. GLS, which works with the
innovations $e_t$ themselves, remains consistent. This reverses the usual advice that
OLS is the "robust" choice ([PG26] Section 3).

## Estimating the filter

The feasible version needs an estimate of $A(L)$ that is itself consistent under
pre-determinedness. Estimating it from OLS residuals (Cochrane–Orcutt) fails because the
residuals inherit the OLS inconsistency. Instead [PG26] use the **Durbin regression**
(eq. (14))

$$
y_t = \sum_{j=1}^k \rho_j y_{t-j} + x_t'\beta - \sum_{j=1}^k x_{t-j}'\delta_j + e_{kt},
$$

whose regressors are all uncorrelated with $e_t$. The order $k$ is chosen by BIC, with
every candidate order estimated on the same sample $t = k_{\max}+1,\dots,T$ so that the
criteria are comparable:

$$
\mathrm{BIC}(k) = \ln\hat\sigma_k^2 + k\,\frac{\ln(T-k_{\max})}{T-k_{\max}},
\qquad k = 0,\dots,k_{\max},
$$

with $\hat\sigma_k^2$ the mean squared residual of the order-$k$ Durbin regression. Lagged deterministic terms (constant, trends, seasonal dummies)
are collinear with their contemporaneous values and are dropped automatically.

## FGLS and its variance

The data are quasi-differenced with $\hat\rho_1,\dots,\hat\rho_{k^*}$ and OLS is applied
(eq. (15)):

$$
\tilde y_t = y_t - \sum_{j=1}^{k^*}\hat\rho_j y_{t-j}, \qquad
\tilde x_t = x_t - \sum_{j=1}^{k^*}\hat\rho_j x_{t-j}, \qquad
\hat\beta = \Big(\sum_{t>k^*}\tilde x_t\tilde x_t'\Big)^{-1}\sum_{t>k^*}\tilde x_t\tilde y_t .
$$
 Because the filter is estimated, the variance of $\hat\beta$ has two parts
(Theorem 2): the usual GLS term and a correction for the first-stage estimation of
$\rho$, which involves $g_j = E[\tilde x_t u_{t-j}]$. Under exogeneity $g_j = 0$ and the
correction vanishes (the naive variance, eq. (27)); otherwise it does not, and the
**robust** variance of eqs. (28)–(29) — the default in `pyfgls` — must be used. Its
relative size is reported as `nonexogeneity_share`.

## Lagged dependent variables

If lags of $y$ are among the regressors, the coefficients on lagged $y$ in the Durbin
regression no longer estimate $\rho$. Declare them (`lagged_dep_vars=[...]`) and `pyfgls`
recovers $\rho_j$ from the ratio of the coefficients of a lagged and a contemporaneous
stochastic regressor (Wallis, 1967; [PG26] Section 4.2).
