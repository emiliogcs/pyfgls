"""Acceptance tests: replicate the [GP24] simulation results (heteroskedasticity).

DGP (eq. (9)): ``y = 1 + x + sqrt(v(x)) eps``, ``x ~ U[1, 4]``, candidate covariates
``z = (x, [log x]^2, x^2, cos x, cos 2x)`` plus the unpenalised intercept. The
efficiency measure is the empirical MSE of the slope relative to OLS ([GP24] Table 1).

[GP24] Table 1, Lasso column (T = 100 / 200 / 400): v = 1: 1.01/1.00/1.00; x: 0.92 (all);
x^2: 0.70 (all); [log x]^2: 0.54/0.46/0.46; exp: 0.49/0.45/0.45; step: 0.74/0.74/0.73.
pyfgls matches the T = 400 column within 0.01. At T = 100-200 its ratios are 0.03-0.06
higher (x^2: 0.75-0.77), although the tuning-free WLS-S1/S2 and infeasible-GLS ratios
agree with the paper within about 0.02 and the interval lengths match Table 2 — an
open item on the details of the adaptive-Lasso tuning (see the project report).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfgls._linalg import hat_diag, lstsq
from pyfgls.hetero import estimate_hetero
from pyfgls.simulate import hetero_dgp

from ._mc import Z95, run

pytestmark = pytest.mark.slow


def paper_z(x):
    return np.column_stack([x, np.log(x) ** 2, x**2, np.cos(x), np.cos(2 * x)])


def ols_hc3(X, y):
    b = lstsq(X, y)
    u = y - X @ b
    h = hat_diag(X)
    A = np.linalg.inv(X.T @ X)
    V = A @ (X * (u**2 / (1 - h) ** 2)[:, None]).T @ X @ A
    return b, np.sqrt(np.diag(V))


def _table1(skedastic: str, T: int, R: int, seed: int):
    def one(rng):
        s = hetero_dgp(T, skedastic=skedastic, random_state=rng)
        x = s.X[:, 1]
        est = estimate_hetero(
            s.y, s.X, paper_z(x), list("abcde"), random_state=int(rng.integers(2**31))
        )
        se = np.sqrt(np.diag(est.cov))
        bo, so = ols_hc3(s.X, s.y)
        return {
            "fgls": est.beta - s.beta,
            "ols": bo - s.beta,
            "cover": np.abs(est.beta - s.beta) <= Z95 * se,
            "len": 2 * Z95 * se,
            "len_ols": 2 * Z95 * so,
        }

    return run(R, seed, one)


def emse_ratio(r, coef: int = 1) -> float:
    """Empirical MSE of FGLS relative to OLS for one coefficient (slope by default)."""
    return float(np.mean(r["fgls"][:, coef] ** 2) / np.mean(r["ols"][:, coef] ** 2))


#: [GP24] Table 1 targets (Lasso-FGLS slope eMSE relative to OLS), matched at T = 400.
TABLE1_T400 = {
    "homoskedastic": 1.00,
    "x": 0.92,
    "x2": 0.70,
    "log2": 0.46,
    "exp": 0.45,
    "step": 0.735,
}


@pytest.mark.parametrize("skedastic", list(TABLE1_T400))
def test_table1_efficiency_T400(skedastic):
    r = _table1(skedastic, 400, 1000, 2024)
    ratio = emse_ratio(r)
    assert ratio == pytest.approx(TABLE1_T400[skedastic], abs=0.05), ratio
    # eqs. (6)+(8) intervals: close to nominal coverage
    assert r["cover"][:, 1].mean() == pytest.approx(0.95, abs=0.02)


@pytest.mark.parametrize("skedastic", ["homoskedastic", "x2", "log2", "exp", "step"])
def test_table1_coverage_and_length_T100(skedastic):
    r = _table1(skedastic, 100, 1500, 99)
    assert r["cover"][:, 1].mean() == pytest.approx(0.95, abs=0.015)
    ratio = emse_ratio(r)
    if skedastic == "homoskedastic":
        assert ratio <= 1.05  # little loss even at T = 100
    else:
        assert ratio < 0.85
        # shorter intervals than OLS-HC3 under heteroskedasticity
        assert r["len"][:, 1].mean() < r["len_ols"][:, 1].mean()


def test_no_loss_under_homoskedasticity_larger_samples():
    r = _table1("homoskedastic", 800, 500, 7)
    assert emse_ratio(r) <= 1.02


def test_interval_lengths_log2_T100():
    r = _table1("log2", 100, 1000, 11)
    assert r["len"][:, 1].mean() == pytest.approx(0.315, abs=0.03)
    assert r["len_ols"][:, 1].mean() == pytest.approx(0.44, abs=0.03)


# ------------------------------------------------------- extended robustness designs
def _extended(case: str, R: int = 400, T: int = 200, k_w: int = 20):
    """[GP24] Section 5 robustness: many irrelevant / proxy candidate covariates."""

    def one(rng):
        corr = 0.5 if case == "proxy" else 0.0
        sk = "homoskedastic" if case == "homoskedastic" else "x2"
        s = hetero_dgp(T, skedastic=sk, n_extra=k_w, extra_corr=corr, random_state=rng)
        x = s.X[:, 1]
        W = s.info["W"]
        blocks = [W, np.cos(W), np.cos(2 * W)]
        if case in ("driver", "homoskedastic"):
            blocks = [np.column_stack([x, x**2, np.cos(x), np.cos(2 * x)]), *blocks]
        Z = np.column_stack(blocks)
        est = estimate_hetero(
            s.y, s.X, Z, [f"z{j}" for j in range(Z.shape[1])], random_state=int(rng.integers(2**31))
        )
        return {"fgls": est.beta - s.beta, "ols": lstsq(s.X, s.y) - s.beta}

    return run(R, 99, one)


def test_extended_driver_included_beats_ols():
    assert emse_ratio(_extended("driver")) < 0.85


def test_extended_correlated_proxy_beats_ols():
    assert emse_ratio(_extended("proxy")) < 1.0


@pytest.mark.parametrize("case", ["excluded", "homoskedastic"])
def test_extended_no_noticeable_loss(case):
    assert emse_ratio(_extended(case)) <= 1.06
