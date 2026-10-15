import numpy as np
import pytest
import statsmodels.api as sm

import pyfgls as pf
from pyfgls._linalg import lstsq
from pyfgls._utils import ar_filter
from pyfgls.benchmarks import cochrane_orcutt, durbin, hybrid_gls, infeasible_gls, ols
from pyfgls.covariance.hac import (
    andrews_bandwidth,
    bartlett_kernel,
    hac_covariance,
    long_run_covariance,
    newey_west_lags,
    qs_kernel,
)
from pyfgls.covariance.hc import hc_covariance
from pyfgls.exceptions import InputError
from pyfgls.serial import estimate_serial
from pyfgls.simulate import arma_errors, serial_dgp


@pytest.fixture
def ts():
    return serial_dgp(300, ar=(0.6,), rho_x=0.5, gamma=0.0, burn=100, random_state=3)


# ------------------------------------------------------------------- kernels
def test_qs_kernel_values():
    assert qs_kernel(np.array([0.0]))[0] == 1.0
    z = 6 * np.pi / 5
    assert qs_kernel(np.array([1.0]))[0] == pytest.approx(3 / z**2 * (np.sin(z) / z - np.cos(z)))
    x = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(qs_kernel(x), qs_kernel(-x))
    np.testing.assert_allclose(bartlett_kernel(np.array([0.0, 0.5, 2.0])), [1.0, 0.5, 0.0])


def test_andrews_bandwidth_formula():
    rng = np.random.default_rng(0)
    s = arma_errors(rng.normal(size=500), (0.5,))[:, None]
    bw = andrews_bandwidth(s, "qs")
    r = float(s[:-1, 0] @ s[1:, 0] / (s[:-1, 0] @ s[:-1, 0]))
    alpha = 4 * r**2 / (1 - r) ** 4
    assert bw == pytest.approx(1.3221 * (alpha * 500) ** 0.2, rel=1e-10)
    assert newey_west_lags(100) == 4


def test_long_run_variance_of_white_noise_is_variance():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(20000, 1))
    S, _ = long_run_covariance(s, kernel="qs")
    assert S[0, 0] == pytest.approx(1.0, abs=0.05)
    with pytest.raises(InputError):
        long_run_covariance(s, kernel="parzen")


# ------------------------------------------------------ cross-checks: statsmodels
def test_newey_west_matches_statsmodels(ts):
    L = 5
    b = lstsq(ts.X, ts.y)
    V, used_bw = hac_covariance(ts.X, ts.y - ts.X @ b, kernel="bartlett", bandwidth=L + 1)
    assert used_bw == L + 1
    smr = sm.OLS(ts.y, ts.X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
    np.testing.assert_allclose(V, smr.cov_params(), rtol=1e-10)
    r = ols(ts.y, ts.X, cov_type="NW")
    smr2 = sm.OLS(ts.y, ts.X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(300), "use_correction": False}
    )
    np.testing.assert_allclose(r.cov, smr2.cov_params(), rtol=1e-10)


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3", "nonrobust"])
def test_hc_matches_statsmodels(ts, cov_type):
    b = lstsq(ts.X, ts.y)
    V = hc_covariance(ts.X, ts.y - ts.X @ b, cov_type)
    smr = sm.OLS(ts.y, ts.X).fit(cov_type=cov_type)
    np.testing.assert_allclose(V, smr.cov_params(), rtol=1e-10)
    np.testing.assert_allclose(ols(ts.y, ts.X, cov_type=cov_type).params, smr.params)


def test_infeasible_gls_with_omega_matches_statsmodels_gls(ts):
    rho, T = 0.6, 300
    idx = np.arange(T)
    Om = rho ** np.abs(idx[:, None] - idx[None, :]) / (1 - rho**2)
    r = infeasible_gls(ts.y, ts.X, omega=Om)
    smr = sm.GLS(ts.y, ts.X, sigma=Om).fit()
    np.testing.assert_allclose(r.params, smr.params, rtol=1e-10)
    np.testing.assert_allclose(r.cov, smr.cov_params(), rtol=1e-8)
    # conditional (filter) version drops the first observation only
    r2 = infeasible_gls(ts.y, ts.X, rho=[rho])
    np.testing.assert_allclose(r2.params, lstsq(ar_filter(ts.X, [rho]), ar_filter(ts.y, [rho])))
    assert r2.n_eff == T - 1
    with pytest.raises(InputError):
        infeasible_gls(ts.y, ts.X)


def test_fgls_close_to_statsmodels_glsar_with_exogenous_regressors(ts):
    ours = pf.SerialFGLS(ts.y, ts.X, ar_order=1).fit()
    glsar = sm.GLSAR(ts.y, ts.X, rho=1).iterative_fit(maxiter=50)
    np.testing.assert_allclose(ours.params, glsar.params, rtol=0.02, atol=0.02)
    np.testing.assert_allclose(ours.rho.iloc[0], glsar.model.rho[0], atol=0.03)
    np.testing.assert_allclose(ours.bse, glsar.bse, rtol=0.1)


# ------------------------------------------------------------------ estimators
def test_cochrane_orcutt_two_step_by_hand(ts):
    r = cochrane_orcutt(ts.y, ts.X, iterate=False)
    b0 = lstsq(ts.X, ts.y)
    u = ts.y - ts.X @ b0
    rho = float(u[:-1] @ u[1:] / (u[:-1] @ u[:-1]))
    np.testing.assert_allclose(r.rho.iloc[0], rho)
    np.testing.assert_allclose(r.params, lstsq(ar_filter(ts.X, [rho]), ar_filter(ts.y, [rho])))
    assert r.method == "Cochrane-Orcutt (two-step)"
    assert "not valid with non-exogenous" in " ".join(r.notes)
    it = cochrane_orcutt(ts.y, ts.X, iterate=True)
    assert it.serial.extra["iterations"] > 1
    assert it.method == "Cochrane-Orcutt (iterated)"
    assert cochrane_orcutt(ts.y, ts.X, ar_order=2).ar_order == 2
    with pytest.raises(InputError):
        cochrane_orcutt(ts.y, ts.X, ar_order=0)


def test_durbin_benchmark_matches_internal(ts):
    est = estimate_serial(ts.y, ts.X, ["const", "x1"])
    d = durbin(ts.y, ts.X)
    np.testing.assert_allclose(d.params, est.durbin.beta)
    assert d.method == "Durbin"
    assert durbin(ts.y, ts.X, ar_order=2).ar_order == 2


def test_hybrid_gls_paths():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    white = hybrid_gls(X @ [1.0, 1.0] + rng.normal(size=200), X, random_state=0)
    assert white.method in ("Hybrid (OLS)", "Hybrid (GLS)")
    u = arma_errors(rng.normal(size=200), (0.7,))
    ar = hybrid_gls(X @ [1.0, 1.0] + u, X, random_state=0)
    assert ar.method == "Hybrid (GLS)"
    assert 0.1 <= ar.rho.iloc[0] <= 0.9


def test_ols_defaults_and_compare_ols(ts):
    r = ols(ts.y, ts.X)
    assert r.cov_type.startswith("HAC (QS")
    res = pf.SerialFGLS(ts.y, ts.X).fit()
    tab = res.compare_ols()
    assert tab.shape == (6, 2)
    assert tab.columns[0] == "OLS"
    hres = pf.LassoFGLS(ts.y, ts.X, random_state=0).fit()
    assert hres.compare_ols().shape == (6, 2)
    with pytest.raises(InputError):
        ols(ts.y, ts.X, cov_type="HC4")
    with pytest.raises(InputError):
        ols(ts.y)
