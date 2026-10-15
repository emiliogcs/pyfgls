"""Acceptance tests: replicate the [PG26] simulation results (serial correlation).

Replication counts are smaller than the paper's (10,000); tolerances cover the Monte
Carlo noise at these counts (relative +-15% on RMSE/MSE, absolute +-0.01 on biases).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from pyfgls._linalg import lstsq
from pyfgls._utils import ar_filter
from pyfgls.serial import estimate_serial
from pyfgls.simulate import serial_dgp

from ._mc import covered, mse, rmse, run

pytestmark = pytest.mark.slow
NAMES = ["const", "x"]


# ------------------------------------------------------------------------- Table 1
@pytest.fixture(scope="module")
def table1():
    def one(rng):
        s = serial_dgp(500, random_state=rng)  # AR(1) 0.8, w_t = v_t + e_{t-1}, u_0 = 0
        est = estimate_serial(s.y, s.X, NAMES, kmax=12)
        gls = lstsq(ar_filter(s.X, [0.8]), ar_filter(s.y, [0.8]))
        return {
            "ols": lstsq(s.X, s.y)[1] - 1,
            "fgls": est.beta[1] - 1,
            "durbin": est.durbin.beta[1] - 1,
            "gls": gls[1] - 1,
            "rho": est.rho.sum() - 0.8,
            "k": est.order,
        }

    return run(2000, 20260101, one)


def test_table1_ols_is_inconsistent(table1):
    assert rmse(table1["ols"]) == pytest.approx(0.399, rel=0.15)
    assert table1["ols"].mean() == pytest.approx(0.395, abs=0.01)


def test_table1_fgls_matches_infeasible_gls(table1):
    assert rmse(table1["fgls"]) == pytest.approx(0.025, rel=0.15)
    assert table1["fgls"].mean() == pytest.approx(0.001, abs=0.01)
    assert rmse(table1["gls"]) == pytest.approx(0.025, rel=0.15)
    assert rmse(table1["durbin"]) == pytest.approx(0.036, rel=0.15)
    assert rmse(table1["fgls"]) < rmse(table1["durbin"])


def test_table1_rho_estimates(table1):
    # the paper reports the persistence sum_j rho_j of the selected AR(k*) filter
    assert rmse(table1["rho"]) == pytest.approx(0.034, rel=0.15)
    assert table1["rho"].mean() == pytest.approx(-0.008, abs=0.01)
    assert np.median(table1["k"]) == 1


# ------------------------------------------------------------- Tables 2-5 (T = 200)
CELLS = {
    "AR1(0.5)": {"ar": (0.5,)},
    "AR1(0.8)": {"ar": (0.8,)},
    "AR2(1.34,-0.42)": {"ar": (1.34, -0.42)},
    "AR2(0.5,-0.3)": {"ar": (0.5, -0.3)},
}


def _general(cell: str, gamma: float, R: int = 1000, T: int = 200):
    ar = CELLS[cell]["ar"]

    def one(rng):
        s = serial_dgp(T, ar=ar, rho_x=0.8, gamma=gamma, burn=100, random_state=rng)
        est = estimate_serial(s.y, s.X, NAMES, kmax=12)
        b = est.beta[1]
        se_r = np.sqrt(est.cov[1, 1])
        se_n = np.sqrt(est.variance.naive_avar[1, 1] / est.variance.n)
        return {
            "ols": lstsq(s.X, s.y)[1] - 1,
            "fgls": b - 1,
            "gls": lstsq(ar_filter(s.X, ar), ar_filter(s.y, ar))[1] - 1,
            "cov_r": covered(b, se_r, 1.0),
            "cov_n": covered(b, se_n, 1.0),
            "k": est.order,
        }

    return run(R, 4242, one)


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("cell", list(CELLS))
def test_general_case_coverage_and_mse(cell, gamma):
    r = _general(cell, gamma)
    cov_r, cov_n = r["cov_r"].mean(), r["cov_n"].mean()
    # (ii) the robust eq. (29) interval has close to nominal coverage
    assert 0.92 <= cov_r <= 0.96, cov_r
    # (iii) the naive eq. (27) interval under-covers when the regressor is not exogenous
    if gamma == 0.5:
        assert cov_r - cov_n >= 0.01, (cov_r, cov_n)
    elif gamma == 0.0:
        assert abs(cov_r - cov_n) <= 0.01, (cov_r, cov_n)
    else:
        assert cov_r >= cov_n
    # (i) FGLS beats OLS and is close to the infeasible GLS
    assert mse(r["fgls"]) <= mse(r["ols"])
    assert mse(r["fgls"]) <= 1.35 * mse(r["gls"])
    # selected order: 1 for AR(1), 2 for AR(2) ([PG26] "Med. k")
    assert np.median(r["k"]) == len(CELLS[cell]["ar"])


def test_ar2_persistent_errors_mse_reduction():
    r = _general("AR2(1.34,-0.42)", 0.0, R=1000)
    ols, fgls = 100 * mse(r["ols"]), 100 * mse(r["fgls"])
    assert ols == pytest.approx(11.5, rel=0.15)
    assert fgls == pytest.approx(0.41, rel=0.15)
    assert fgls / ols < 0.05  # about a 96% reduction


@pytest.mark.parametrize("theta", [-0.4, 0.5])
def test_ma1_errors_fgls_beats_ols(theta):
    def one(rng):
        s = serial_dgp(200, ar=(), ma=(theta,), rho_x=0.8, gamma=0.25, burn=100, random_state=rng)
        est = estimate_serial(s.y, s.X, NAMES, kmax=12)
        return {"ols": lstsq(s.X, s.y)[1] - 1, "fgls": est.beta[1] - 1}

    r = run(500, 77, one)
    assert mse(r["fgls"]) < mse(r["ols"])


# --------------------------------------------- lagged dependent variable (Wallis)
def test_lagged_dependent_variable_wallis_path():
    T, alpha, rho = 500, 0.5, 0.5

    def one(rng):
        n = T + 201
        e = rng.standard_normal(n)
        w = scipy.signal.lfilter([1.0], [1.0, -0.5], rng.standard_normal(n))
        u = scipy.signal.lfilter([1.0], [1.0, -rho], e)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = alpha * y[t - 1] + 1.0 + w[t] + u[t]
        Y = y[201:]
        X = np.column_stack([np.ones(T), y[200:-1], w[201:]])
        est = estimate_serial(Y, X, ["c", "ylag", "w"], kmax=12, lagged_dep_idx=[1])
        se = np.sqrt(np.diag(est.cov))
        return {
            "a": est.beta[1] - alpha,
            "b": est.beta[2] - 1.0,
            "cov_a": covered(est.beta[1], se[1], alpha),
            "cov_b": covered(est.beta[2], se[2], 1.0),
            "ols_a": lstsq(X, Y)[1] - alpha,
            "k": est.order,
        }

    r = run(600, 99, one)
    assert abs(r["a"].mean()) < 0.01
    assert abs(r["b"].mean()) < 0.01
    assert r["ols_a"].mean() > 0.1  # OLS is inconsistent with AR errors and a lagged y
    assert 0.92 <= r["cov_a"].mean() <= 0.97
    assert 0.92 <= r["cov_b"].mean() <= 0.97
    assert np.median(r["k"]) == 1
