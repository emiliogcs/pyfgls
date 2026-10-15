import warnings

import numpy as np
import pandas as pd
import pytest

import pyfgls as pf
from pyfgls.diagnostics import (
    adf_test,
    ar_roots,
    check_stationarity,
    nonexogeneity_share,
    test_predetermined,
    weight_diagnostics,
)
from pyfgls.exceptions import ExperimentalWarning, NonStationaryWarning
from pyfgls.serial.ma import estimate_gma, gma_durbin_order, inverse_ma_filter, ma1_mle
from pyfgls.simulate import arma_errors, hetero_dgp, serial_dgp


# ---------------------------------------------------------------- predetermined
def test_predetermined_test_detects_missing_regressor_lag():
    rng = np.random.default_rng(0)
    T = 600
    x = rng.normal(size=T + 1)
    u = arma_errors(rng.normal(size=T), (0.5,))
    y = 1.0 + x[1:] + 0.8 * x[:-1] + u  # x_{t-1} omitted from the model below
    X = np.column_stack([np.ones(T), x[1:]])
    with pytest.warns(ExperimentalWarning):
        t = test_predetermined(y, X, max_lag=2)
    assert t.rejected
    assert "Add lag" in t.advice
    assert list(t.table.columns) == ["LM", "df", "p-value"]
    assert "Remark 9" in repr(t)


def test_predetermined_test_accepts_correct_model():
    sim = serial_dgp(600, ar=(0.5,), rho_x=0.5, gamma=0.0, burn=100, random_state=1)
    with pytest.warns(ExperimentalWarning):
        t = test_predetermined(sim.y, sim.X, max_lag=2, regressors=["x1"])
    assert t.table["p-value"].min() > 0.01
    assert t.regressors == ("x1",)


# ------------------------------------------------------------------------- ADF
def test_adf_distinguishes_random_walk_from_stationary():
    rng = np.random.default_rng(2)
    rw = np.cumsum(rng.normal(size=400))
    st = arma_errors(rng.normal(size=400), (0.5,))
    assert not adf_test(rw).unit_root_rejected
    assert adf_test(st).unit_root_rejected


def test_check_stationarity_warns():
    rng = np.random.default_rng(3)
    T = 300
    x = np.cumsum(rng.normal(size=T))
    y = x + rng.normal(size=T)
    from pyfgls._validation import prepare_data

    d = prepare_data(y, np.column_stack([np.ones(T), x]))
    with pytest.warns(NonStationaryWarning, match="unit root"):
        out = check_stationarity(d)
    assert set(out) == {"y", "x1"}
    with pytest.warns(NonStationaryWarning):
        pf.SerialFGLS(y, np.column_stack([np.ones(T), x]), check_stationarity=True)


# --------------------------------------------------------------------- weights
def test_weight_diagnostics_and_accessors():
    sim = hetero_dgp(300, skedastic="x2", random_state=4)
    res = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit()
    wd = weight_diagnostics(res)
    assert wd["variance_ratio"] >= 1.0
    assert 0 < wd["effective_n_share"] <= 1.0
    assert 0 < wd["top1pct_weight_share"] < 1.0
    with pytest.raises(ValueError):
        ar_roots(res)
    s = serial_dgp(300, random_state=5)
    rs = pf.SerialFGLS(s.y, s.X).fit()
    np.testing.assert_allclose(ar_roots(rs), rs.ar_roots)
    assert nonexogeneity_share(rs) == rs.nonexogeneity_share
    with pytest.raises(ValueError):
        weight_diagnostics(rs)


# ------------------------------------------------------------------------- GMA
def test_ma1_mle_recovers_theta():
    rng = np.random.default_rng(6)
    e = rng.normal(size=3001)
    u = e[1:] + 0.6 * e[:-1]
    assert ma1_mle(u) == pytest.approx(0.6, abs=0.04)
    assert ma1_mle(e[1:] - 0.4 * e[:-1]) == pytest.approx(-0.4, abs=0.04)


def test_inverse_ma_filter_matches_explicit_sum():
    a = np.arange(1.0, 7.0)
    th = 0.5
    expected = [sum((-th) ** j * a[t - j] for j in range(t + 1)) for t in range(6)]
    np.testing.assert_allclose(inverse_ma_filter(a, th), expected)
    assert gma_durbin_order(100) == 4


def test_gma_estimator_and_model_option():
    sim = serial_dgp(500, ar=(), ma=(0.7,), rho_x=0.5, gamma=0.0, burn=50, random_state=7)
    est = estimate_gma(sim.y, sim.X)
    assert est.theta == pytest.approx(0.7, abs=0.1)
    assert est.n_eff == 500
    assert est.durbin_order == gma_durbin_order(500)
    res = pf.SerialFGLS(sim.y, sim.X, ma1=True).fit()
    assert res.method == "GMA (MA(1))"
    assert res.serial.extra["MA coef theta"] == pytest.approx(est.theta)
    assert abs(res.params["x1"] - 1.0) < 4 * res.bse["x1"]
    assert "MA coef theta" in str(res.summary())
    with pytest.raises(pf.exceptions.InputError):
        pf.FGLS(sim.y, sim.X, serial="ma1").fit()


# ---------------------------------------------------------------------- formula
def test_formula_interface():
    sim = serial_dgp(300, ar=(0.5,), rho_x=0.5, gamma=0.0, burn=100, random_state=8)
    df = pd.DataFrame({"growth": sim.y, "spread": sim.X[:, 1]})
    df.index = pd.period_range("1990-01", periods=300, freq="M")
    res = pf.SerialFGLS.from_formula("growth ~ spread", data=df).fit()
    ref = pf.SerialFGLS(df["growth"], sim.X).fit()
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy())
    assert list(res.params.index) == ["const", "spread"]
    assert res.data.y_name == "growth"
    lagged = pf.FGLS.from_formula(
        "growth ~ spread + lag(spread)", data=df, hetero=None, missing="drop"
    ).fit()
    assert "lag(spread)" in lagged.params.index
    assert lagged.nobs == 299
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = pf.LassoFGLS.from_formula("growth ~ spread", data=df, random_state=0).fit()
    assert h.hetero is not None
