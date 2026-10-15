"""Acceptance tests: MA(1) errors and the GMA estimator ([PG26] Section 5, Table 4).

``x_t = 0.8 x_{t-1} + v_t + gamma e_{t-1}``, ``u_t = e_t + theta e_{t-1}``, ``T = 200``.
[PG26] Table 4 (T = 200): theta = -0.7, gamma = 0.5 -> MSE x 100 OLS 1.36, GMA 0.06,
FGLS 0.08; coverage GMA 0.91, FGLS 0.87. The FGLS under-coverage there is a known
limitation (the BIC truncates a slowly decaying AR(inf) approximation), not a bug; the
GMA transformation avoids it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfgls._linalg import lstsq
from pyfgls.serial import estimate_serial
from pyfgls.serial.ma import estimate_gma
from pyfgls.simulate import serial_dgp

from ._mc import covered, mse, run

pytestmark = pytest.mark.slow


def _cell(theta: float, gamma: float, R: int = 800):
    def one(rng):
        s = serial_dgp(200, ar=(), ma=(theta,), rho_x=0.8, gamma=gamma, burn=100, random_state=rng)
        f = estimate_serial(s.y, s.X, ["c", "x"], kmax=12)
        g = estimate_gma(s.y, s.X)
        return {
            "ols": lstsq(s.X, s.y)[1] - 1,
            "fgls": f.beta[1] - 1,
            "gma": g.beta[1] - 1,
            "cov_fgls": covered(f.beta[1], np.sqrt(f.cov[1, 1]), 1.0),
            "cov_gma": covered(g.beta[1], np.sqrt(g.cov[1, 1]), 1.0),
        }

    return run(R, 4, one)


@pytest.mark.parametrize("theta", [-0.7, 0.5])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_gma_and_fgls_beat_ols(theta, gamma):
    r = _cell(theta, gamma)
    assert mse(r["gma"]) < mse(r["ols"])
    assert mse(r["fgls"]) < mse(r["ols"])
    assert 0.89 <= r["cov_gma"].mean() <= 0.97


def test_known_limitation_large_negative_ma_with_nonexogenous_regressor():
    r = _cell(-0.7, 0.5, R=1000)
    assert 100 * mse(r["gma"]) == pytest.approx(0.06, abs=0.02)
    assert 100 * mse(r["fgls"]) == pytest.approx(0.08, abs=0.02)
    assert 0.82 <= r["cov_fgls"].mean() <= 0.90  # paper: 0.87
    assert r["cov_gma"].mean() > r["cov_fgls"].mean()
