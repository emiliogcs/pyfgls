import json
import warnings

import numpy as np
import pandas as pd
import pytest

import pyfgls as pf
from pyfgls._linalg import lstsq
from pyfgls._utils import ar_filter
from pyfgls.exceptions import (
    InputError,
    LaggedDependentWarning,
    NonStationaryWarning,
    SampleSizeWarning,
)
from pyfgls.serial import (
    durbin_design,
    estimate_serial,
    filtered_regression,
    fit_durbin,
    select_ar_order,
    serial_variance,
)
from pyfgls.serial.variance import g_matrix, psi_residuals
from pyfgls.simulate import arma_errors, serial_dgp


def ar_data(T, ar, *, seed=0, gamma=0.0, rho_x=0.5, burn=100):
    return serial_dgp(
        T, ar=ar, rho_x=rho_x, gamma=gamma, burn=burn, random_state=np.random.default_rng(seed)
    )


# ------------------------------------------------------------------- exact algebra
def test_ar1_quasi_differencing_equals_D_matrix_gls():
    sim = ar_data(30, (0.6,), seed=1)
    rho = 0.6
    T = 30
    D = np.zeros((T - 1, T))
    for i in range(T - 1):
        D[i, i] = -rho
        D[i, i + 1] = 1.0
    X, y = sim.X, sim.y
    gls = np.linalg.solve(X.T @ D.T @ D @ X, X.T @ D.T @ D @ y)
    fr = filtered_regression(y, X, [rho])
    np.testing.assert_allclose(fr.beta, gls, rtol=1e-10)


def test_known_ar2_filter_equals_closed_form_gls():
    sim = ar_data(40, (0.5, -0.3), seed=2)
    rho = np.array([0.5, -0.3])
    T = 40
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i : i + 3] = [-rho[1], -rho[0], 1.0]
    Omega_inv = D.T @ D  # conditional on the first two observations
    X, y = sim.X, sim.y
    gls = np.linalg.solve(X.T @ Omega_inv @ X, X.T @ Omega_inv @ y)
    np.testing.assert_allclose(filtered_regression(y, X, rho).beta, gls, rtol=1e-10)


def test_filtered_constant_keeps_intercept_scale():
    sim = ar_data(2000, (0.7,), seed=3)
    res = pf.SerialFGLS(sim.y, sim.X).fit()
    assert res.params["const"] == pytest.approx(1.0, abs=0.25)


def test_durbin_design_is_nested_and_drops_lagged_constant():
    sim = ar_data(100, (0.5,), seed=4)
    d3 = durbin_design(sim.y, sim.X, ["const", "x"], 3, start=3)
    assert d3.dropped == ("L1.const", "L2.const", "L3.const")
    assert d3.names[:4] == ("const", "x", "L1.y", "L1.x")
    d2 = durbin_design(sim.y, sim.X, ["const", "x"], 2, start=3)
    np.testing.assert_array_equal(d3.prefix(2), d2.Z)
    assert d3.block_end == (2, 4, 6, 8)


def test_lag_selection_rss_matches_direct_regressions():
    sim = ar_data(150, (0.5, 0.2), seed=5)
    sel = select_ar_order(sim.y, sim.X, ["const", "x"], 6)
    for k in range(7):
        d = durbin_design(sim.y, sim.X, ["const", "x"], k, start=6)
        resid = d.y - d.Z @ lstsq(d.Z, d.y)
        assert sel.sigma2[k] == pytest.approx(resid @ resid / d.nobs, rel=1e-10)
    n = 150 - 6
    np.testing.assert_allclose(
        sel.values, np.log(sel.sigma2) + np.arange(7) * np.log(n) / n, rtol=1e-12
    )


def test_deterministic_regressors_do_not_crash_or_change_rho():
    rng = np.random.default_rng(6)
    T = 240
    t = np.arange(1.0, T + 1)
    quarter = np.arange(T) % 4
    D = np.column_stack([(quarter == j).astype(float) for j in (1, 2, 3)])
    x = rng.normal(size=T)
    X = np.column_stack([np.ones(T), t, D, x])
    u = arma_errors(rng.normal(size=T), (0.6,))
    y = X @ np.r_[1.0, 0.01, 0.5, -0.5, 0.2, 1.0] + u
    names = ["const", "trend", "q2", "q3", "q4", "x"]
    d = durbin_design(y, X, names, 2)
    assert {"L1.const", "L1.trend", "L1.q2", "L1.q3", "L1.q4"} <= set(d.dropped)
    rho = fit_durbin(d).coef[d.ylag_columns()]
    # rank-deficient regression with all lagged columns (min-norm lstsq) gives the same rho
    full = np.column_stack(
        [X[2:], y[1:-1], X[1:-1], y[:-2], X[:-2]]
    )  # [x_t, y_{t-1}, x_{t-1}, y_{t-2}, x_{t-2}]
    coef_full = np.linalg.lstsq(full, y[2:], rcond=None)[0]
    np.testing.assert_allclose(rho, coef_full[[6, 13]], rtol=1e-6)
    res = pf.SerialFGLS(y, pd.DataFrame(X, columns=names), ar_order=2).fit()
    assert np.isfinite(res.bse).all()


# --------------------------------------------------------------------- variances
def _eq29_by_hand(y, X, k):
    """Independent implementation of [PG26] eqs. (28)-(29)."""
    T = X.shape[0]
    # Durbin regression of order k on t = k..T-1
    rows = np.arange(k, T)
    Zy = np.column_stack([y[rows - j] for j in range(1, k + 1)])
    W = np.column_stack(
        [X[rows]] + [X[rows - j][:, 1:] for j in range(1, k + 1)]
    )  # drop lagged const
    coef = np.linalg.lstsq(np.column_stack([Zy, W]), y[rows], rcond=None)[0]
    rho = coef[:k]
    Xf = X[rows] - sum(rho[j - 1] * X[rows - j] for j in range(1, k + 1))
    yf = y[rows] - sum(rho[j - 1] * y[rows - j] for j in range(1, k + 1))
    beta = np.linalg.solve(Xf.T @ Xf, Xf.T @ yf)
    e = yf - Xf @ beta
    u = y - X @ beta
    n = T - k
    s2 = e @ e / n
    Q = Xf.T @ Xf / n
    G = np.column_stack([Xf.T @ u[rows - j] / n for j in range(1, k + 1)])
    psi = Zy - W @ np.linalg.lstsq(W, Zy, rcond=None)[0]
    Sig = psi.T @ psi / n
    Qi = np.linalg.inv(Q)
    avar = s2 * (Qi + Qi @ G @ np.linalg.inv(Sig) @ G.T @ Qi)
    return beta, avar / n, s2 * Qi / n


@pytest.mark.parametrize("k", [1, 2, 3])
def test_robust_variance_matches_hand_computation(k):
    sim = ar_data(300, (0.6, -0.2), seed=7, gamma=0.5)
    est = estimate_serial(sim.y, sim.X, ["const", "x"], ar_order=k)
    beta, V, Vn = _eq29_by_hand(sim.y, sim.X, k)
    np.testing.assert_allclose(est.beta, beta, rtol=1e-8)
    np.testing.assert_allclose(est.cov, V, rtol=1e-7)
    np.testing.assert_allclose(est.variance.naive_avar / est.variance.n, Vn, rtol=1e-8)


def test_influence_form_equals_eq29_in_standard_case():
    sim = ar_data(250, (0.5, 0.2), seed=8, gamma=0.5)
    est = estimate_serial(sim.y, sim.X, ["const", "x"], ar_order=2)
    d = est.design
    J = np.zeros((2, d.Z.shape[1]))
    for j, col in enumerate(d.ylag_columns()):
        J[j, col] = 1.0
    fr = est.filtered
    v_general = serial_variance(fr.Xf, fr.ehat, fr.uhat, d, J=J)
    np.testing.assert_allclose(v_general.cov, est.cov, rtol=1e-8)


def test_robust_variance_is_psd_and_correction_nonnegative():
    sim = ar_data(200, (0.8,), seed=9, gamma=0.5, rho_x=0.8)
    est = estimate_serial(sim.y, sim.X, ["const", "x"])
    assert np.linalg.eigvalsh(est.cov).min() >= 0
    assert np.linalg.eigvalsh(est.variance.correction).min() >= -1e-12
    assert est.variance.nonexogeneity_share > 0


def test_g_matrix_and_psi_shapes():
    sim = ar_data(120, (0.5,), seed=10)
    est = estimate_serial(sim.y, sim.X, ["const", "x"], ar_order=3)
    fr = est.filtered
    G = g_matrix(fr.Xf, fr.uhat, 3, 3)
    assert G.shape == (2, 3)
    psi = psi_residuals(est.design, 3)
    assert psi.shape == (117, 3)
    # psi is orthogonal to the other Durbin regressors
    other = [c for c in range(est.design.Z.shape[1]) if c not in est.design.ylag_columns()]
    np.testing.assert_allclose(est.design.Z[:, other].T @ psi, 0.0, atol=1e-8)


def test_k_zero_reduces_to_ols():
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(100), rng.normal(size=100)])
    y = X @ [1.0, 2.0] + rng.normal(size=100)
    res = pf.SerialFGLS(y, X, ar_order=0).fit()
    beta = lstsq(X, y)
    np.testing.assert_allclose(res.params, beta)
    e = y - X @ beta
    V = (e @ e / 100) * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(res.cov, V, rtol=1e-10)
    assert res.ar_order == 0
    assert res.nonexogeneity_share == 0.0


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_hc_variants_close_to_robust_under_homoskedasticity(cov_type):
    sim = ar_data(3000, (0.5,), seed=12, gamma=0.5)
    r = pf.SerialFGLS(sim.y, sim.X).fit()
    h = pf.SerialFGLS(sim.y, sim.X).fit(cov_type=cov_type)
    np.testing.assert_allclose(h.bse, r.bse, rtol=0.1)
    np.testing.assert_allclose(h.params, r.params)


# ---------------------------------------------------------------- lag selection
def test_bic_picks_zero_for_white_noise():
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(100 + s)
        X = np.column_stack([np.ones(300), rng.normal(size=300)])
        y = X @ [1.0, 1.0] + rng.normal(size=300)
        hits += estimate_serial(y, X, ["c", "x"], kmax=8).order == 0
    assert hits >= 54


def test_bic_picks_true_order_in_large_samples():
    hits = 0
    for s in range(20):
        sim = ar_data(3000, (0.5, -0.3), seed=200 + s)
        hits += estimate_serial(sim.y, sim.X, ["c", "x"]).order == 2
    assert hits >= 18


def test_aic_and_min_lags_and_fixed_order():
    sim = ar_data(400, (0.5,), seed=13)
    assert pf.SerialFGLS(sim.y, sim.X, ic="aic").fit().ar_order >= 1
    r = pf.SerialFGLS(sim.y, sim.X, min_lags="adjusted").fit()
    assert r.ar_order == int(np.ceil(np.log(400) ** 1.5)) == 15
    assert r.serial.kmax == 14
    assert r.serial.method == "bic"
    assert pf.SerialFGLS(sim.y, sim.X, ar_order=3).fit().ar_order == 3
    assert pf.SerialFGLS(sim.y, sim.X, ar_order=3).fit().bic_path is None
    assert pf.SerialFGLS(sim.y, sim.X, kmax="theory").fit().serial.kmax == 4
    with pytest.raises(InputError):
        pf.SerialFGLS(sim.y, sim.X, ar_order=300).fit()
    with pytest.raises(InputError):
        pf.SerialFGLS(sim.y, sim.X, ic="hqic")


def test_kmax_cap_warns_for_short_samples():
    rng = np.random.default_rng(14)
    T, q = 60, 6
    X = np.column_stack([np.ones(T), rng.normal(size=(T, q - 1))])
    y = X.sum(axis=1) + rng.normal(size=T)
    with pytest.warns(SampleSizeWarning, match="kmax was reduced"):
        res = pf.SerialFGLS(y, X, kmax=12).fit()
    assert res.serial.kmax < 12


# ----------------------------------------------------------- robustness / checks
def test_scaling_invariance():
    sim = ar_data(300, (0.6,), seed=15, gamma=0.5)
    base = pf.SerialFGLS(sim.y, sim.X).fit()
    Xs = sim.X.copy()
    Xs[:, 1] *= 10.0
    r1 = pf.SerialFGLS(sim.y, Xs).fit()
    assert r1.params.iloc[1] == pytest.approx(base.params.iloc[1] / 10, rel=1e-8)
    np.testing.assert_allclose(r1.tvalues, base.tvalues, rtol=1e-6)
    r2 = pf.SerialFGLS(5.0 * sim.y, sim.X).fit()
    assert r2.ar_order == base.ar_order
    np.testing.assert_allclose(r2.params, 5 * base.params, rtol=1e-8)
    np.testing.assert_allclose(r2.bse, 5 * base.bse, rtol=1e-6)
    np.testing.assert_allclose(r2.tvalues, base.tvalues, rtol=1e-6)


def test_affine_equivariance_in_y():
    sim = ar_data(300, (0.6,), seed=16)
    base = pf.SerialFGLS(sim.y, sim.X).fit()
    r = pf.SerialFGLS(2.0 * sim.y + 3.0 * sim.X[:, 1], sim.X).fit()
    np.testing.assert_allclose(r.params, 2.0 * base.params + [0.0, 3.0], rtol=1e-8)
    np.testing.assert_allclose(r.bse, 2.0 * base.bse, rtol=1e-6)


def test_near_unit_root_warns():
    rng = np.random.default_rng(17)
    T = 400
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    u = arma_errors(rng.normal(size=T), (0.995,))
    with pytest.warns(NonStationaryWarning):
        pf.SerialFGLS(X @ [1.0, 1.0] + u, X).fit()


def test_undeclared_lagged_dependent_variable():
    sim = ar_data(200, (0.5,), seed=18)
    y = sim.y
    X = np.column_stack([sim.X[1:], y[:-1]])
    with pytest.warns(LaggedDependentWarning), pytest.raises(InputError, match="lagged_dep_vars"):
        pf.SerialFGLS(y[1:], X).fit()


def test_lagged_dependent_variable_path_runs():
    rng = np.random.default_rng(19)
    T = 2000
    w = rng.normal(size=T + 1)
    u = arma_errors(rng.normal(size=T + 1), (0.5,))
    y = np.zeros(T + 1)
    for t in range(1, T + 1):
        y[t] = 0.5 * y[t - 1] + 1.0 + w[t] + u[t]
    X = pd.DataFrame({"const": 1.0, "ylag": y[:-1], "w": w[1:]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", LaggedDependentWarning)
        res = pf.SerialFGLS(pd.Series(y[1:], name="y"), X, lagged_dep_vars=["ylag"]).fit()
    assert res.ar_order == 1
    assert abs(res.params["ylag"] - 0.5) < 3 * res.bse["ylag"]
    assert abs(res.params["w"] - 1.0) < 3 * res.bse["w"]
    assert res.serial.wallis_regressors == ("w",)
    assert res.rho.iloc[0] == pytest.approx(0.5, abs=0.1)
    # OLS is badly biased here
    assert lstsq(X.to_numpy(), y[1:])[1] > 0.6
    with pytest.raises(InputError, match="not eligible"):
        pf.SerialFGLS(y[1:], X, lagged_dep_vars=["ylag"], wallis_regressor="const").fit()


def test_lagged_dep_without_eligible_regressor():
    rng = np.random.default_rng(20)
    y = np.cumsum(rng.normal(size=201)) * 0.1 + rng.normal(size=201)
    X = np.column_stack([np.ones(200), y[:-1]])
    with pytest.raises(InputError, match="stochastic regressor"):
        pf.SerialFGLS(y[1:], X, lagged_dep_vars=[1], ar_order=1).fit()


# ---------------------------------------------------------------- results object
@pytest.fixture
def fitted():
    sim = ar_data(300, (0.6,), seed=21, gamma=0.5)
    idx = pd.period_range("1990-01", periods=300, freq="M")
    y = pd.Series(sim.y, index=idx, name="growth")
    X = pd.DataFrame(sim.X, index=idx, columns=["const", "spread"])
    return pf.SerialFGLS(y, X).fit()


def test_results_are_labelled(fitted):
    assert list(fitted.params.index) == ["const", "spread"]
    assert fitted.resid.index[0] == pd.Period("1990-01", freq="M")
    assert len(fitted.resid) == 300
    assert len(fitted.resid_final) == 300 - fitted.ar_order
    assert fitted.resid_final.index[0] == fitted.resid.index[fitted.ar_order]
    np.testing.assert_allclose(fitted.fittedvalues + fitted.resid, fitted.data.y, rtol=1e-12)
    assert fitted.conf_int().shape == (2, 2)
    assert fitted.cov_params().shape == (2, 2)
    assert fitted.n_eff == 300 - fitted.ar_order
    assert fitted.df_resid == fitted.n_eff - 2


def test_summary_and_serialisation(fitted):
    text = str(fitted.summary())
    for token in ("FGLS Regression Results", "growth", "spread", "AR order k*", "robust"):
        assert token in text
    d = json.loads(fitted.to_json())
    assert d["serial"]["ar_order"] == fitted.ar_order
    assert set(d["params"]) == {"const", "spread"}
    assert "spread" in repr(fitted)
    assert "<pre>" in fitted.summary()._repr_html_()


def test_use_t_changes_pvalues_and_intervals(fitted):
    sim = ar_data(300, (0.6,), seed=21, gamma=0.5)
    rt = pf.SerialFGLS(sim.y, sim.X).fit(use_t=True)
    rz = pf.SerialFGLS(sim.y, sim.X).fit()
    assert (rt.pvalues > rz.pvalues).all()
    wt = (rt.conf_int().iloc[:, 1] - rt.conf_int().iloc[:, 0]).to_numpy()
    wz = (rz.conf_int().iloc[:, 1] - rz.conf_int().iloc[:, 0]).to_numpy()
    assert (wt > wz).all()
    assert "P>|t|" in rt.to_frame().columns


def test_wald_tests(fitted):
    w1 = fitted.wald_test("spread = 1")
    t = (fitted.params["spread"] - 1) / fitted.bse["spread"]
    assert w1.statistic == pytest.approx(t**2, rel=1e-8)
    w2 = fitted.wald_test(np.eye(2), [1.0, 1.0])
    w3 = fitted.wald_test(["const = 1", "spread = 1"])
    assert w2.statistic == pytest.approx(w3.statistic, rel=1e-10)
    assert w3.df == 2
    w4 = fitted.wald_test((np.array([[0.0, 1.0]]), np.array([1.0])))
    assert w4.statistic == pytest.approx(w1.statistic)
    assert "chi2(2)" in repr(w3)


def test_predict(fitted):
    Xn = pd.DataFrame({"spread": [0.0, 1.0], "const": [1.0, 1.0]})
    p = fitted.predict(Xn)
    np.testing.assert_allclose(p, [fitted.params["const"], fitted.params.sum()])
    np.testing.assert_allclose(fitted.predict([1.0, 2.0]), fitted.params @ [1.0, 2.0])
    with pytest.raises(InputError):
        fitted.predict(np.ones((2, 3)))


def test_compare_table(fitted):
    sim = ar_data(300, (0.6,), seed=21, gamma=0.5)
    naive = pf.SerialFGLS(sim.y, sim.X).fit(cov_type="naive")
    robust = pf.SerialFGLS(sim.y, sim.X).fit()
    tab = pf.compare([naive, robust])
    assert tab.shape == (6, 2)
    assert (tab.xs("std err", level=1).iloc[:, 1] >= tab.xs("std err", level=1).iloc[:, 0]).all()


def test_ar_roots_reported(fitted):
    roots = fitted.ar_roots
    assert np.all(np.abs(roots) > 1)
    np.testing.assert_allclose(np.abs(roots[0]), 1 / fitted.rho.iloc[0], rtol=1e-8)


def test_exogenous_declaration_adds_note_when_violated():
    sim = ar_data(1000, (0.8,), seed=22, gamma=1.0, rho_x=0.0)
    r = pf.SerialFGLS(sim.y, sim.X, exog="exogenous").fit()
    assert r.nonexogeneity_share > 0
    sim0 = ar_data(500, (0.5,), seed=23, gamma=0.0)
    r0 = pf.SerialFGLS(sim0.y, sim0.X).fit()
    assert abs(r0.nonexogeneity_share) < 0.05


def test_filtered_innovations_match_quasi_difference(fitted):
    k = fitted.ar_order
    rho = fitted.rho.to_numpy()
    yf = ar_filter(fitted.data.y, rho)
    Xf = ar_filter(fitted.data.X, rho)
    np.testing.assert_allclose(fitted.resid_final, yf - Xf @ fitted.params.to_numpy())
    assert len(yf) == 300 - k
