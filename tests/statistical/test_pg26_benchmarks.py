"""Acceptance tests for the benchmark estimators against [PG26] (Table 1, Section 5, S-1)."""

from __future__ import annotations

import numpy as np
import pytest

from pyfgls._linalg import lstsq
from pyfgls._utils import ar_filter
from pyfgls.benchmarks import cochrane_orcutt, ols
from pyfgls.serial import estimate_serial
from pyfgls.simulate import arma_errors, serial_dgp

from ._mc import covered, mse, rmse, run

pytestmark = pytest.mark.slow


def test_table1_cochrane_orcutt_is_biased_with_nonexogenous_regressor():
    def one(rng):
        s = serial_dgp(500, random_state=rng)
        co = cochrane_orcutt(s.y, s.X)  # two-step "CO-FGLS" of [PG26] Table 1
        return {"b": co.params.iloc[1] - 1, "rho": co.rho.iloc[0] - 0.8}

    r = run(1000, 20260101, one)
    assert rmse(r["b"]) == pytest.approx(0.041, rel=0.15)
    assert r["b"].mean() == pytest.approx(0.031, abs=0.01)
    assert r["rho"].mean() == pytest.approx(-0.171, abs=0.02)
    assert rmse(r["rho"]) == pytest.approx(0.174, rel=0.15)


def test_ols_hac_undercovers_with_persistent_errors_while_fgls_does_not():
    def one(rng):
        s = serial_dgp(200, ar=(0.8,), rho_x=0.8, gamma=0.0, burn=100, random_state=rng)
        o = ols(s.y, s.X, cov_type="HAC")
        est = estimate_serial(s.y, s.X, ["c", "x"], kmax=12)
        return {
            "ols": covered(o.params.iloc[1], o.bse.iloc[1], 1.0),
            "fgls": covered(est.beta[1], np.sqrt(est.cov[1, 1]), 1.0),
            "len_ols": o.bse.iloc[1],
            "len_fgls": np.sqrt(est.cov[1, 1]),
        }

    r = run(800, 5, one)
    assert r["ols"].mean() < 0.90
    assert r["fgls"].mean() > r["ols"].mean() + 0.03
    assert r["len_fgls"].mean() < r["len_ols"].mean()


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_supplement_s1_wrong_rho_quasi_differencing_beats_ols(rho):
    """Corollary A.1: rho_u = 0.5, iid N(0,1) regressor -> any rho in (0, 1) helps."""
    rho_u = 0.5
    assert rho * (1 + rho_u**2) - 2 * rho_u * (1 + rho**2) < 0

    def one(rng):
        T = 200
        X = np.column_stack([np.ones(T), rng.standard_normal(T)])
        u = arma_errors(rng.standard_normal(T + 100), (rho_u,))[100:]
        y = X @ [1.0, 1.0] + u
        return {
            "ols": lstsq(X, y)[1] - 1,
            "gls": lstsq(ar_filter(X, [rho]), ar_filter(y, [rho]))[1] - 1,
        }

    r = run(2000, 8, one)
    assert mse(r["gls"]) < mse(r["ols"])
