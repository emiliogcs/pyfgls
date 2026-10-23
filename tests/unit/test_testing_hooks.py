"""Hooks used by cross-implementation tests: explicit folds, penalty grid, export_fit."""

import json

import numpy as np
import pytest

import pyfgls as pf
from pyfgls.exceptions import InputError
from pyfgls.hetero.adaptive_lasso import adaptive_lasso, check_foldid, make_folds
from pyfgls.simulate import combined_dgp, hetero_dgp, serial_dgp
from pyfgls.testing import export_fit, export_fit_json


def _z(x):
    return np.column_stack([x, np.log(x) ** 2, x**2, np.cos(x), np.cos(2 * x)])


def test_foldid_reproduces_random_state_folds():
    sim = hetero_dgp(200, skedastic="log2", random_state=1)
    x = sim.X[:, 1]
    r = np.log(
        np.maximum((sim.y - sim.X @ np.linalg.lstsq(sim.X, sim.y, rcond=None)[0]) ** 2, 0.01)
    )
    a = adaptive_lasso(r, _z(x), random_state=7)
    folds = make_folds(200, 10, "random", 7)
    b = adaptive_lasso(r, _z(x), foldid=folds + 5)  # labels need not start at 0
    np.testing.assert_allclose(a.fitted, b.fitted)
    assert a.tuning["psi"] == b.tuning["psi"]
    np.testing.assert_array_equal(b.tuning["foldid"], folds)


def test_lambda_grid_is_used_for_every_psi():
    sim = hetero_dgp(150, skedastic="x2", random_state=2)
    x = sim.X[:, 1]
    r = np.log(np.maximum(sim.u**2, 0.01))
    grid = np.logspace(0, -3, 20)
    fit = adaptive_lasso(r, _z(x), random_state=0, lambda_grid=grid)
    for _psi, (alphas, _) in fit.tuning["cv_path"].items():
        np.testing.assert_allclose(alphas, grid)
    assert fit.tuning["lambda"] in grid
    with pytest.raises(InputError):
        adaptive_lasso(r, _z(x), lambda_grid=[0.1, -1.0])


def test_check_foldid_validation():
    with pytest.raises(InputError):
        check_foldid([0, 1, 0], 4)
    with pytest.raises(InputError):
        check_foldid([3, 3, 3], 3)
    np.testing.assert_array_equal(check_foldid([7, 7, 9, 9], 4), [0, 0, 1, 1])


def test_models_accept_foldid():
    sim = hetero_dgp(200, skedastic="x2", random_state=3)
    folds = make_folds(200, 10, "random", 11)
    a = pf.LassoFGLS(sim.y, sim.X, random_state=11).fit()
    b = pf.LassoFGLS(sim.y, sim.X, foldid=folds).fit()
    np.testing.assert_allclose(a.params, b.params)
    c = combined_dgp(200, ar=(0.5,), skedastic="x2", random_state=4)
    res = pf.FGLS(c.y, c.X, random_state=5).fit()
    n_eff = res.n_eff
    f2 = make_folds(n_eff, 10, "random", 5)
    res2 = pf.FGLS(c.y, c.X, foldid=f2).fit()
    np.testing.assert_allclose(res.params, res2.params)


def test_export_fit_serial_hetero_combined_gma():
    s = serial_dgp(300, random_state=0)
    out = export_fit(pf.SerialFGLS(s.y, s.X).fit())
    for key in ("kstar", "bic_path", "rho_hat", "Ghat", "Sigma_psi", "AVar_R", "vcov", "ar_roots"):
        assert key in out
    h = hetero_dgp(200, random_state=1)
    oh = export_fit(pf.LassoFGLS(h.y, h.X, random_state=0).fit())
    assert oh["cv_error_matrix"].shape == (6, 100)
    assert oh["ridge_coef"] is not None
    c = combined_dgp(200, random_state=2)
    oc = export_fit(pf.FGLS(c.y, c.X, random_state=0).fit())
    assert {"beta_FC", "beta_FCH", "vcov_FC", "vcov_FCH", "serial", "hetero"} <= set(oc)
    g = export_fit(pf.SerialFGLS(s.y, s.X, ma1=True).fit())
    assert "theta" in g
    json.loads(export_fit_json(pf.SerialFGLS(s.y, s.X).fit()))
    boot = pf.SerialFGLS(s.y, s.X).fit(cov_type="bootstrap", n_boot=5, boot_random_state=1)
    assert export_fit(boot)["kstar"] == out["kstar"]
    with pytest.raises(InputError):
        export_fit(pf.benchmarks.ols(s.y, s.X))
