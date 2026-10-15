import numpy as np
import pytest

import pyfgls as pf
from pyfgls._linalg import hat_diag
from pyfgls.combined import combined_covariance, estimate_combined
from pyfgls.covariance import bootstrap_covariance, sieve_sample, wild_multipliers, wild_sample
from pyfgls.exceptions import InputError
from pyfgls.hetero.variance import hetero_covariance
from pyfgls.serial import estimate_serial
from pyfgls.serial.variance import first_stage_influence, serial_variance
from pyfgls.simulate import combined_dgp


@pytest.fixture
def sim():
    return combined_dgp(250, ar=(0.5,), skedastic="x2", random_state=np.random.default_rng(0))


def paper_w(x):
    return np.column_stack([x, x**2, np.cos(x), np.cos(2 * x)])


def test_unit_weights_make_fch_equal_fc(sim):
    est = estimate_combined(
        sim.y,
        sim.X,
        ["const", "x"],
        np.zeros((250, 0)),
        (),
        method="known",
        known=np.ones(250),
    )
    np.testing.assert_allclose(est.beta_fch, est.beta_fc, rtol=1e-10)
    np.testing.assert_allclose(est.cov_fch, est.cov_fc, rtol=1e-10)


def test_combined_variance_without_correction_is_gp24_eq8_on_filtered_data(sim):
    x = sim.X[:, 1]
    est = estimate_combined(
        sim.y, sim.X, ["const", "x"], paper_w(x), list("abcd"), random_state=0,
        first_stage_correction=False,
    )  # fmt: skip
    fr = est.serial.filtered
    het = est.hetero
    V = hetero_covariance(
        "robust",
        X=fr.Xf,
        v=het.v,
        u_fgls=fr.yf - fr.Xf @ het.beta,
        u_ols=fr.ehat,
        h_ols=hat_diag(fr.Xf),
        df=het.df,
        method="lasso",
    )
    np.testing.assert_allclose(est.cov_fch, V, rtol=1e-8)


def test_first_stage_correction_with_unit_weights_matches_serial_hc0(sim):
    ser = estimate_serial(sim.y, sim.X, ["const", "x"], ar_order=2)
    fr = ser.filtered
    C = first_stage_influence(ser.design)
    V = combined_covariance(
        fr.Xf, fr.ehat, fr.uhat, np.ones(fr.n_eff), start=2, C=C,
        h_fc=hat_diag(fr.Xf), df=1.0, cov_type="HC0",
    )  # fmt: skip
    ref = serial_variance(fr.Xf, fr.ehat, fr.uhat, ser.design, cov_type="HC0")
    np.testing.assert_allclose(V, ref.cov, rtol=1e-8)


def test_first_stage_correction_increases_variance_with_nonexogenous_regressor():
    sim = combined_dgp(
        400, ar=(0.8,), skedastic="x2", gamma=1.0, random_state=np.random.default_rng(1)
    )
    a = pf.FGLS(sim.y, sim.X, random_state=0).fit()
    b = pf.FGLS(sim.y, sim.X, random_state=0, first_stage_correction=False).fit()
    np.testing.assert_allclose(a.params, b.params)
    assert a.bse["x1"] > b.bse["x1"]
    c = pf.FGLS(sim.y, sim.X, random_state=0, exog="exogenous").fit()
    np.testing.assert_allclose(c.bse, b.bse)


def test_fgls_results_structure(sim):
    res = pf.FGLS(sim.y, sim.X, random_state=0).fit()
    assert res.method == "F-CH (serial+hetero)"
    assert res.ar_order == 1
    assert res.hetero.skedastic == "lasso"
    assert res.stage_serial.method == "F-C (serial)"
    assert res.stage_hetero is res
    tab = res.compare_stages()
    assert list(tab.columns) == ["F-C (serial)", "F-CH (serial+hetero)"]
    assert res.n_eff == 249
    assert len(res.resid_final) == 249
    text = str(res.summary())
    assert "Skedastic fit" in text
    assert "AR order" in text
    for cov_type in ("HC0", "HC3"):
        assert np.isfinite(pf.FGLS(sim.y, sim.X, random_state=0).fit(cov_type=cov_type).bse).all()


def test_fgls_dispatch(sim):
    s = pf.FGLS(sim.y, sim.X, hetero=None).fit()
    assert s.method == "FGLS (serial)"
    h = pf.FGLS(sim.y, sim.X, serial=None, random_state=0).fit()
    assert h.method == "FGLS (hetero)"
    o = pf.FGLS(sim.y, sim.X, serial=None, hetero=None).fit()
    assert o.method == "OLS"
    f = pf.FGLS(sim.y, sim.X, serial="fixed", ar_order=2, hetero=None).fit()
    assert f.ar_order == 2
    k = pf.FGLS(sim.y, sim.X, hetero="known", variance=sim.info["v"]).fit()
    assert k.hetero.skedastic == "known"
    with pytest.raises(InputError, match="ar_order"):
        pf.FGLS(sim.y, sim.X, serial="fixed")
    with pytest.raises(InputError):
        pf.FGLS(sim.y, sim.X, serial="arma")
    with pytest.raises(InputError):
        pf.FGLS(sim.y, sim.X, hetero="kernel")
    with pytest.raises(InputError):
        pf.FGLS(sim.y, sim.X).fit(cov_type="naive")


def test_wild_multipliers_moments():
    rng = np.random.default_rng(2)
    r = wild_multipliers(200_000, "rademacher", rng)
    assert set(np.unique(r)) == {-1.0, 1.0}
    m = wild_multipliers(200_000, "mammen", rng)
    assert m.mean() == pytest.approx(0.0, abs=0.01)
    assert (m**2).mean() == pytest.approx(1.0, abs=0.01)
    assert (m**3).mean() == pytest.approx(1.0, abs=0.03)
    with pytest.raises(InputError):
        wild_multipliers(3, "normal", rng)


def test_sieve_sample_reproduces_ar_recursion():
    rng = np.random.default_rng(3)
    fitted = np.zeros(6)
    uhat = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    innov = np.array([0.0, 0.0, 0.0, 0.0])  # centred zeros: u follows the recursion
    out = sieve_sample(fitted, uhat, innov, np.array([0.5, 0.25]), "iid", "rademacher", rng)
    expected = [1.0, 2.0, 1.25, 1.125, 0.875, 0.71875]
    np.testing.assert_allclose(out, expected)
    w = wild_sample(np.ones(4), np.array([1.0, -1.0, 2.0, 0.0]), "rademacher", rng)
    assert np.allclose(np.abs(w - 1), [1, 1, 2, 0])


def test_bootstrap_reproducible_and_n_jobs_invariant(sim):
    def refit(y):
        return np.array([y.mean(), y.std()])

    def draw(rng):
        return rng.normal(size=50)

    a = bootstrap_covariance(refit, draw, n_boot=40, random_state=5)
    b = bootstrap_covariance(refit, draw, n_boot=40, random_state=5, n_jobs=2)
    np.testing.assert_allclose(a.cov, b.cov)
    np.testing.assert_allclose(a.draws, b.draws)
    with pytest.raises(InputError):
        bootstrap_covariance(refit, draw, n_boot=1)


def test_bootstrap_fit_paths(sim):
    s = pf.SerialFGLS(sim.y, sim.X).fit(cov_type="bootstrap", n_boot=60, boot_random_state=1)
    assert s.cov_type.startswith("bootstrap (sieve AR(1)")
    assert s.options["n_boot"] == 60
    h = pf.LassoFGLS(sim.y, sim.X, random_state=0, cv_folds=5).fit(
        cov_type="bootstrap", n_boot=20, wild_dist="mammen"
    )
    assert h.cov_type == "bootstrap (wild, mammen)"
    c = pf.FGLS(sim.y, sim.X, random_state=0, cv_folds=5).fit(cov_type="bootstrap", n_boot=20)
    assert np.isfinite(c.bse).all()
    assert c.stage_serial is not None
    # analytic and bootstrap standard errors agree in order of magnitude
    r = pf.SerialFGLS(sim.y, sim.X).fit()
    assert 0.5 < s.bse["x1"] / r.bse["x1"] < 2.0
