"""Acceptance tests: [PG26] Supplement S-4 (serial correlation + heteroskedasticity).

``y = 1 + x + u``, ``x ~ U[1, 4]``, ``u`` ARMA with innovations
``e_t = sqrt(v(w_t)) eps_t``; candidate covariates ``W = (w, w^2, cos w, cos 2w)``
with ``w = x`` (case a) or a Bernoulli(0.75) mixture of ``x`` and an independent
``U[1, 4]`` (case b); ``T = 200``, ``k_max = 12``. F-C corrects for serial correlation
only, F-CH for both.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfgls._linalg import hat_diag, lstsq
from pyfgls.combined import combined_covariance, estimate_combined
from pyfgls.hetero import estimate_hetero
from pyfgls.simulate import combined_dgp

from ._mc import covered, mse, run

pytestmark = pytest.mark.slow


def _cell(ar=(0.5,), ma=(), skedastic="x2", driver="x", gamma=0.0, R=600, seed=31):
    def one(rng):
        s = combined_dgp(
            200, ar=ar, ma=ma, skedastic=skedastic, driver=driver, gamma=gamma, random_state=rng
        )
        w = s.info["w"]
        W = np.column_stack([w, w**2, np.cos(w), np.cos(2 * w)])
        sd = int(rng.integers(2**31))
        est = estimate_combined(s.y, s.X, ["c", "x"], W, list("abcd"), kmax=12, random_state=sd)
        fr, het, k = est.serial.filtered, est.hetero, est.order
        cov_nc = combined_covariance(
            fr.Xf, fr.yf - fr.Xf @ het.beta, s.y - s.X @ het.beta, het.v,
            start=k, C=None, h_fc=hat_diag(fr.Xf), df=het.df,
        )  # fmt: skip
        lasso_only = estimate_hetero(s.y, s.X, W, list("abcd"), random_state=sd)
        se_fc, se_fch = np.sqrt(est.cov_fc[1, 1]), np.sqrt(est.cov_fch[1, 1])
        return {
            "fc": est.beta_fc[1] - 1,
            "fch": est.beta_fch[1] - 1,
            "lasso_only": lasso_only.beta[1] - 1,
            "ols": lstsq(s.X, s.y)[1] - 1,
            "cov_fc": covered(est.beta_fc[1], se_fc, 1.0),
            "cov_fch": covered(est.beta_fch[1], se_fch, 1.0),
            "cov_fch_nofs": covered(est.beta_fch[1], np.sqrt(cov_nc[1, 1]), 1.0),
            "len_fc": se_fc,
            "len_fch": se_fch,
        }

    return run(R, seed, one)


CELLS = {
    "AR1-x2": {"ar": (0.5,), "skedastic": "x2"},
    "AR1-log2": {"ar": (0.5,), "skedastic": "log2"},
    "AR1-exp": {"ar": (0.5,), "skedastic": "exp"},
    "AR2-x2": {"ar": (1.34, -0.42), "skedastic": "x2"},
    "MA1-x2": {"ar": (), "ma": (0.5,), "skedastic": "x2"},
    "ARMA-x2": {"ar": (0.8,), "ma": (-0.4,), "skedastic": "x2"},
    "AR1-x2-mixture": {"ar": (0.5,), "skedastic": "x2", "driver": "mixture"},
}


@pytest.mark.parametrize("cell", list(CELLS))
def test_fch_dominates_fc_and_covers(cell):
    r = _cell(**CELLS[cell])
    # efficiency: correcting heteroskedasticity on top of serial correlation helps
    assert mse(r["fch"]) <= mse(r["fc"])
    # ... and beats each single-purpose estimator (within Monte Carlo tolerance)
    assert mse(r["fch"]) <= 1.05 * mse(r["lasso_only"])
    assert mse(r["fch"]) < mse(r["ols"])
    # the gain is variance reduction: both are essentially unbiased
    assert abs(r["fch"].mean()) < 0.02
    # (A.4)-(A.5) intervals: close to nominal coverage, shorter than F-C intervals
    assert 0.93 <= r["cov_fch"].mean() <= 0.97
    # F-C intervals carry the same df correction ("built the same way" in the
    # Supplement) without weights, which makes them slightly conservative (0.96-0.97).
    assert 0.91 <= r["cov_fc"].mean() <= 0.98
    assert r["len_fch"].mean() < r["len_fc"].mean()


def test_no_loss_under_homoskedastic_errors():
    r = _cell(ar=(0.5,), skedastic="homoskedastic")
    assert mse(r["fch"]) <= 1.05 * mse(r["fc"])
    assert 0.93 <= r["cov_fch"].mean() <= 0.97


@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_nonexogenous_regressor_first_stage_correction(gamma):
    """Open item A: coverage with and without the first-stage (Durbin) correction."""
    r = _cell(ar=(0.8,), skedastic="x2", gamma=gamma, R=800, seed=77)
    with_fs, without_fs = r["cov_fch"].mean(), r["cov_fch_nofs"].mean()
    # Monte Carlo (1,000 reps): with/without the correction 0.942/0.777 (AR(1) 0.5,
    # gamma = 0.5) and 0.923/0.860 (AR(1) 0.8, gamma = 1): the correction is needed.
    assert with_fs >= without_fs + 0.03
    assert 0.91 <= with_fs <= 0.97  # Phase-4 acceptance band of the project brief
