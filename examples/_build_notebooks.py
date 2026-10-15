"""Generate the example notebooks from this script (keeps them reviewable as code).

Run ``python examples/_build_notebooks.py`` and then execute them, e.g.
``pytest --nbmake examples/``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """\
import warnings

import numpy as np
import pandas as pd

import pyfgls as pf

warnings.simplefilter("ignore", pf.exceptions.PyFGLSWarning)
"""


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, src in cells:
        nb.cells.append(
            nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        )
    return nb


NOTEBOOKS = {
    "01_serial_correlation.ipynb": [
        (
            "md",
            "# Serial correlation: FGLS of Perron & González-Coya (2026)\n\n"
            "We simulate the design of [PG26] Table 1: AR(1) errors with $\\rho = 0.8$ and a "
            "regressor $w_t = v_t + e_{t-1}$ that reacts to past shocks (pre-determined but "
            "not exogenous). OLS is inconsistent here; FGLS based on the Durbin regression is not.",
        ),
        ("code", SETUP),
        (
            "code",
            "from pyfgls.simulate import serial_dgp\n\n"
            "sim = serial_dgp(500, random_state=0)\n"
            "idx = pd.period_range('1980-01', periods=500, freq='M')\n"
            "y = pd.Series(sim.y, index=idx, name='y')\n"
            "X = pd.DataFrame(sim.X, index=idx, columns=['const', 'w'])\n"
            "res = pf.SerialFGLS(y, X).fit()\n"
            "print(res.summary())",
        ),
        ("md", "The BIC path over AR orders (all orders estimated on a common sample):"),
        ("code", "res.bic_path.round(3)"),
        (
            "md",
            "Compare with OLS (HAC standard errors) and Cochrane–Orcutt, which estimates the "
            "filter from the (inconsistent) OLS residuals:",
        ),
        (
            "code",
            "co = pf.benchmarks.cochrane_orcutt(y, X)\n"
            "pf.compare([pf.benchmarks.ols(y, X), co, res]).round(4)",
        ),
        ("md", "A small Monte Carlo makes the point (the paper uses 10,000 replications):"),
        (
            "code",
            "from pyfgls.serial import estimate_serial\n"
            "from pyfgls._linalg import lstsq\n\n"
            "rng = np.random.default_rng(1)\n"
            "err = {'OLS': [], 'Cochrane-Orcutt': [], 'FGLS': []}\n"
            "for _ in range(200):\n"
            "    s = serial_dgp(500, random_state=rng)\n"
            "    err['OLS'].append(lstsq(s.X, s.y)[1] - 1)\n"
            "    err['Cochrane-Orcutt'].append(pf.benchmarks.cochrane_orcutt(s.y, s.X).params.iloc[1] - 1)\n"
            "    err['FGLS'].append(estimate_serial(s.y, s.X, ['c', 'w'], kmax=12).beta[1] - 1)\n"
            "pd.DataFrame({k: {'bias': np.mean(v), 'RMSE': np.sqrt(np.mean(np.square(v)))}\n"
            "              for k, v in err.items()}).T.round(3)",
        ),
        ("code", "res.plot_diagnostics();"),
    ],
    "02_heteroskedasticity.ipynb": [
        (
            "md",
            "# Heteroskedasticity of unknown form: Lasso-FGLS (González-Coya & Perron, 2024)\n\n"
            "$y = 1 + x + \\sqrt{v(x)}\\,\\varepsilon$ with $x \\sim U[1,4]$ and $v(x) = x^2$.",
        ),
        ("code", SETUP),
        (
            "code",
            "from pyfgls.simulate import hetero_dgp\n\n"
            "sim = hetero_dgp(400, skedastic='x2', random_state=1)\n"
            "df = pd.DataFrame({'y': sim.y, 'x': sim.X[:, 1]})\n"
            "res = pf.LassoFGLS.from_formula('y ~ x', data=df, random_state=0).fit()\n"
            "print(res.summary())",
        ),
        (
            "md",
            "The adaptive Lasso picks among $\\{x, x^2, \\log x, (\\log x)^2, \\cos x, \\cos 2x\\}$:",
        ),
        ("code", "res.skedastic_coef.round(4)"),
        ("md", "Against OLS with HC3 standard errors, and the alternative skedastic estimators:"),
        (
            "code",
            "ols = pf.LassoFGLS.from_formula('y ~ x', data=df, skedastic='none').fit()\n"
            "svr = pf.LassoFGLS.from_formula('y ~ x', data=df, skedastic='svr', random_state=0).fit()\n"
            "wls = pf.LassoFGLS.from_formula('y ~ x', data=df, skedastic='wls_s1', z=['x']).fit()\n"
            "pf.compare([ols, wls, svr, res], names=['OLS', 'WLS-S1', 'SVR', 'Lasso']).round(4)",
        ),
        (
            "code",
            "from pyfgls.plotting import plot_intervals\n\n"
            "plot_intervals([ols, res], labels=['OLS (HC3)', 'Lasso-FGLS']);",
        ),
        ("md", "Under homoskedasticity the Lasso typically selects nothing and FGLS equals OLS:"),
        (
            "code",
            "h = hetero_dgp(400, skedastic='homoskedastic', random_state=2)\n"
            "r0 = pf.LassoFGLS(h.y, h.X, random_state=0).fit()\n"
            "r0.selected_z, r0.df_hat",
        ),
    ],
    "03_both.ipynb": [
        (
            "md",
            "# Serial correlation and heteroskedasticity together\n\n"
            "[PG26] Supplement S-4: AR(1) errors whose innovations have variance $x_t^2$. "
            "F-C corrects for serial correlation only; F-CH also weights by the "
            "Lasso-estimated variances.",
        ),
        ("code", SETUP),
        (
            "code",
            "from pyfgls.simulate import combined_dgp\n\n"
            "sim = combined_dgp(300, ar=(0.5,), skedastic='x2', random_state=2)\n"
            "res = pf.FGLS(sim.y, sim.X, random_state=0).fit()\n"
            "print(res.summary())",
        ),
        ("code", "res.compare_stages().round(4)"),
        (
            "md",
            "An independent check of the analytic standard errors with the sieve bootstrap "
            "(few replications here to keep the notebook fast):",
        ),
        (
            "code",
            "boot = pf.FGLS(sim.y, sim.X, random_state=0, cv_folds=5).fit(cov_type='bootstrap', n_boot=49)\n"
            "pd.DataFrame({'analytic': res.bse, 'bootstrap': boot.bse}).round(4)",
        ),
    ],
    "04_nonexogenous.ipynb": [
        (
            "md",
            "# Non-exogenous regressors: why the robust variance matters\n\n"
            "With a regressor that responds to past shocks, the naive GLS variance ([PG26] "
            "eq. (27)) ignores the estimation of the AR filter; the robust variance (eqs. "
            "(28)–(29)) does not.",
        ),
        ("code", SETUP),
        (
            "code",
            "from pyfgls.simulate import serial_dgp\n\n"
            "sim = serial_dgp(200, ar=(0.5,), rho_x=0.8, gamma=1.0, burn=100, random_state=3)\n"
            "robust = pf.SerialFGLS(sim.y, sim.X).fit()\n"
            "naive = pf.SerialFGLS(sim.y, sim.X).fit(cov_type='naive')\n"
            "print('nonexogeneity share:', round(robust.nonexogeneity_share, 3))\n"
            "pf.compare([naive, robust]).round(4)",
        ),
        ("md", "Coverage of the two 95% intervals in a small simulation:"),
        (
            "code",
            "from pyfgls.serial import estimate_serial\n\n"
            "rng = np.random.default_rng(4)\n"
            "cov_r = cov_n = 0\n"
            "R = 400\n"
            "for _ in range(R):\n"
            "    s = serial_dgp(200, ar=(0.5,), rho_x=0.8, gamma=1.0, burn=100, random_state=rng)\n"
            "    e = estimate_serial(s.y, s.X, ['c', 'x'], kmax=12)\n"
            "    b = e.beta[1]\n"
            "    cov_r += abs(b - 1) < 1.96 * np.sqrt(e.cov[1, 1])\n"
            "    cov_n += abs(b - 1) < 1.96 * np.sqrt(e.variance.naive_avar[1, 1] / e.variance.n)\n"
            "print(f'robust: {cov_r / R:.3f}   naive: {cov_n / R:.3f}')",
        ),
        (
            "md",
            "## Lagged dependent variables\n\nDeclare them; the AR filter is then recovered by "
            "the Wallis ratio ([PG26] Section 4.2).",
        ),
        (
            "code",
            "from pyfgls.simulate import arma_errors\n\n"
            "rng = np.random.default_rng(5)\n"
            "T = 500\n"
            "w = rng.normal(size=T + 1)\n"
            "u = arma_errors(rng.normal(size=T + 1), (0.5,))\n"
            "yy = np.zeros(T + 1)\n"
            "for t in range(1, T + 1):\n"
            "    yy[t] = 0.5 * yy[t - 1] + 1.0 + w[t] + u[t]\n"
            "Xd = pd.DataFrame({'const': 1.0, 'ylag': yy[:-1], 'w': w[1:]})\n"
            "dyn = pf.SerialFGLS(yy[1:], Xd, lagged_dep_vars=['ylag']).fit()\n"
            "dyn.compare_ols().round(3)",
        ),
        (
            "md",
            "## Is the model dynamically complete? (experimental)\n\n"
            "[PG26] Remark 9: test whether lags of the regressors are missing.",
        ),
        (
            "code",
            "from pyfgls.diagnostics import test_predetermined\n\n"
            "test_predetermined(sim.y, sim.X, max_lag=3)",
        ),
        (
            "md",
            "Now a model that omits $x_{t-1}$: the true equation is $y_t = 1 + x_t + 0.4x_{t-1} + u_t$, "
            "so $x_t$ is correlated with the composite error. The test rejects, and adding the lag fixes it.",
        ),
        (
            "code",
            "rng = np.random.default_rng(7)\n"
            "x = rng.normal(size=401)\n"
            "u = arma_errors(rng.normal(size=400), (0.5,))\n"
            "y2 = 1.0 + x[1:] + 0.4 * x[:-1] + u\n"
            "X_short = pd.DataFrame({'const': 1.0, 'x': x[1:]})\n"
            "print(test_predetermined(y2, X_short, max_lag=2))\n"
            "X_long = X_short.assign(x_lag1=x[:-1])\n"
            "print(test_predetermined(y2, X_long, max_lag=2, regressors=['x']))\n"
            "pf.SerialFGLS(y2, X_long).fit().params.round(3)",
        ),
    ],
}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nbf.write(notebook(cells), HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
