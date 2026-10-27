import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

import pyfgls as pf
from pyfgls._linalg import hat_diag, lstsq
from pyfgls.exceptions import InputError, WeightWarning
from pyfgls.hetero import (
    adaptive_lasso,
    estimate_hetero,
    hetero_covariance,
    make_dictionary,
    ridge_cv,
    svr_skedastic,
    variances_from_log,
    wls_skedastic,
)
from pyfgls.hetero.adaptive_lasso import lasso_coefs, make_folds, standardize
from pyfgls.hetero.features import dedupe_columns
from pyfgls.hetero.weights import log_squared_residuals
from pyfgls.simulate import hetero_dgp


def paper_z(x):
    return np.column_stack([x, np.log(x) ** 2, x**2, np.cos(x), np.cos(2 * x)])


@pytest.fixture
def sim():
    return hetero_dgp(300, skedastic="x2", random_state=np.random.default_rng(0))


# -------------------------------------------------------------------- dictionary
def test_dictionary_transforms_and_dummies():
    x = np.linspace(1, 4, 20)
    d = (np.arange(20) % 2).astype(float)
    neg = np.sin(np.arange(20.0))  # takes negative values; not affine in x
    X = np.column_stack([np.ones(20), x, d, neg])
    Z, names = make_dictionary(X, ["const", "x", "d", "w"])
    assert names == (
        "x", "x^2", "log(x)", "log(x)^2", "cos(x)", "cos(2x)", "d", "w", "w^2", "cos(w)", "cos(2w)",
    )  # fmt: skip
    np.testing.assert_allclose(Z[:, 3], np.log(x) ** 2)
    _Z2, n2 = make_dictionary(X, ["const", "x", "d", "w"], transforms=("linear",))
    assert n2 == ("x", "d", "w")
    _Z3, n3 = make_dictionary(X, ["const", "x", "d", "w"], skip=[1])
    assert "x" not in n3
    with pytest.raises(InputError):
        make_dictionary(X, ["const", "x", "d", "w"], transforms=("cubic",))


def test_dictionary_dedupes_and_handles_constant_only():
    Z = np.column_stack([np.arange(5.0), 2 * np.arange(5.0) + 1, np.ones(5)])
    _Zd, kept = dedupe_columns(Z)
    assert kept == [0]
    Z0, n0 = make_dictionary(np.ones((5, 1)), ["const"])
    assert Z0.shape == (5, 0)
    assert n0 == ()


# ---------------------------------------------------------------- adaptive lasso
def test_lasso_coefs_match_sklearn():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 4, 120)
    Zs, *_ = standardize(paper_z(x))
    r = np.log(np.maximum((x * rng.normal(size=120)) ** 2, 0.01))
    rc = r - r.mean()
    alphas = np.array([0.5, 0.1, 0.02, 0.005])
    C = lasso_coefs(Zs, rc, alphas)
    for j, a in enumerate(alphas):
        sk = Lasso(alpha=a, fit_intercept=False, tol=1e-12, max_iter=1_000_000).fit(Zs, rc)
        np.testing.assert_allclose(C[:, j], sk.coef_, atol=1e-6)


def test_adaptive_lasso_psi_zero_is_plain_lasso():
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 4, 200)
    Z = paper_z(x)
    r = np.log(np.maximum((x * rng.normal(size=200)) ** 2, 0.01))
    fit = adaptive_lasso(r, Z, psi_grid=(0.0,), random_state=0)
    lam = fit.tuning["lambda"]
    Zs, _mu, sd, _ = standardize(Z)
    sk = Lasso(alpha=lam, tol=1e-12, max_iter=1_000_000).fit(Zs, r)
    np.testing.assert_allclose(fit.coef, sk.coef_ / sd, atol=1e-6)
    np.testing.assert_allclose(fit.fitted, sk.predict(Zs), atol=1e-6)


def test_adaptive_lasso_matches_reweighted_brute_force():
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 4, 150)
    Z = paper_z(x)
    r = 0.8 * np.log(x) + rng.normal(size=150)
    fit = adaptive_lasso(r, Z, psi_grid=(1.0,), random_state=0)
    Zs, _mu, sd, _ = standardize(Z)
    g_ridge, _ = ridge_cv(Zs, r, make_folds(150, 10, "random", 0))
    w = 1.0 / np.abs(g_ridge)
    # solve min (2n)^-1 ||r - g0 - Zs g||^2 + lam sum w_j |g_j| directly by rescaling
    sk = Lasso(alpha=fit.tuning["lambda"], tol=1e-12, max_iter=1_000_000).fit(Zs / w, r)
    np.testing.assert_allclose(fit.coef, sk.coef_ / w / sd, atol=1e-6)


def test_df_counts_intercept_and_selection():
    rng = np.random.default_rng(4)
    x = rng.uniform(1, 4, 400)
    r = np.log(np.maximum((x**1.5 * rng.normal(size=400)) ** 2, 0.01))
    fit = adaptive_lasso(r, paper_z(x), ["x", "lx2", "x2", "c1", "c2"], random_state=0)
    assert fit.df == np.count_nonzero(fit.coef) + 1
    assert set(fit.selected) <= {"x", "lx2", "x2", "c1", "c2"}
    assert len(fit.selected) >= 1


def test_constant_response_gives_constant_fit():
    Z = np.random.default_rng(5).normal(size=(50, 3))
    fit = adaptive_lasso(np.full(50, 2.0), Z, random_state=0)
    np.testing.assert_allclose(fit.fitted, 2.0)
    assert fit.df == 1.0
    assert adaptive_lasso(np.ones(30), np.zeros((30, 0))).df == 1.0


def test_ridge_handles_more_covariates_than_observations():
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(40, 80))
    r = Z[:, 0] + rng.normal(size=40)
    fit = adaptive_lasso(r, Z, random_state=0)
    assert np.isfinite(fit.fitted).all()
    assert fit.coef.shape == (80,)


def test_folds():
    f = make_folds(23, 5, "blocked")
    assert np.all(np.diff(f) >= 0)
    assert set(f) == set(range(5))
    a = make_folds(23, 5, "random", 1)
    b = make_folds(23, 5, "random", 1)
    np.testing.assert_array_equal(a, b)
    assert np.bincount(a).max() - np.bincount(a).min() <= 1
    with pytest.raises(InputError):
        make_folds(3, 5)
    with pytest.raises(InputError):
        make_folds(10, 1)


def test_lambda_1se_is_sparser():
    rng = np.random.default_rng(7)
    x = rng.uniform(1, 4, 300)
    r = np.log(np.maximum((x * rng.normal(size=300)) ** 2, 0.01))
    a = adaptive_lasso(r, paper_z(x), random_state=0)
    b = adaptive_lasso(r, paper_z(x), random_state=0, lambda_rule="1se")
    assert b.tuning["lambda"] >= a.tuning["lambda"] or b.tuning["psi"] != a.tuning["psi"]
    assert np.count_nonzero(b.coef) <= np.count_nonzero(a.coef) + 1


# ---------------------------------------------------------- other skedastic fits
def test_wls_s1_s2():
    rng = np.random.default_rng(8)
    x = rng.uniform(1, 4, 200)
    r = 2 * np.log(x) + rng.normal(size=200) * 0.01
    f1 = wls_skedastic(r, x[:, None], ["x"], kind="wls_s1")
    assert f1.coef[0] == pytest.approx(2.0, abs=0.01)
    f2 = wls_skedastic(r, x[:, None], ["x"], kind="wls_s2")
    assert f2.df == 2.0
    with pytest.raises(InputError, match="non-positive"):
        wls_skedastic(r, (x - 2)[:, None], ["x"], kind="wls_s1")


def test_svr_runs_and_counts_df():
    rng = np.random.default_rng(9)
    x = rng.uniform(1, 4, 150)
    r = np.log(np.maximum((x * rng.normal(size=150)) ** 2, 0.01))
    fit = svr_skedastic(r, paper_z(x), random_state=0, C_grid=(1.0,), eps_grid=(0.25, 0.5))
    assert 0 < fit.df <= 150
    assert fit.df == np.sum(np.abs(r - fit.fitted) >= fit.tuning["epsilon"] - 1e-8)


# ------------------------------------------------------------------- weights
def test_variances_normalised_and_clipped():
    h = np.array([0.0, 1.0, 2.0, 3.0])
    v = variances_from_log(h)
    assert v.mean() == pytest.approx(1.0)
    with pytest.warns(WeightWarning):
        variances_from_log(np.array([0.0, 20.0, 1.0]))
    vc = variances_from_log(np.linspace(0, 10, 101), clip=(0.1, 0.9))
    assert vc.max() / vc.min() == pytest.approx(np.exp(8.0), rel=1e-6)
    with pytest.raises(InputError):
        variances_from_log(h, clip=(0.9, 0.1))
    with pytest.raises(InputError):
        log_squared_residuals(np.ones(3), 0.0)


# ------------------------------------------------------------------ estimator
def test_none_equals_ols(sim):
    est = estimate_hetero(sim.y, sim.X, np.zeros((300, 0)), (), method="none")
    np.testing.assert_allclose(est.beta, lstsq(sim.X, sim.y))
    np.testing.assert_allclose(est.v, 1.0)


def test_known_variances_equal_closed_form_gls(sim):
    v = sim.info["v"]
    est = estimate_hetero(sim.y, sim.X, np.zeros((300, 0)), (), method="known", known=v)
    Wi = np.diag(1 / v)
    gls = np.linalg.solve(sim.X.T @ Wi @ sim.X, sim.X.T @ Wi @ sim.y)
    np.testing.assert_allclose(est.beta, gls, rtol=1e-10)
    est2 = estimate_hetero(
        sim.y, sim.X, np.zeros((300, 0)), (), method="known", known=lambda X: X[:, 1] ** 2
    )
    np.testing.assert_allclose(est2.beta, gls, rtol=1e-10)


def test_lasso_fgls_falls_back_to_ols_when_nothing_selected(sim):
    # a response that does not depend on z: the Lasso keeps the intercept only
    rng = np.random.default_rng(10)
    Z = rng.normal(size=(300, 4))
    est = estimate_hetero(sim.y, sim.X, Z, ["a", "b", "c", "d"], random_state=0, lambda_rule="1se")
    if est.fit is not None and not np.any(est.fit.coef):
        np.testing.assert_allclose(est.beta, lstsq(sim.X, sim.y), rtol=1e-10)
        np.testing.assert_allclose(est.v, 1.0)


def _eq8_by_hand(X, y, v, u_ols, df):
    T, k = X.shape
    W = np.diag(1 / v)
    A = np.linalg.inv(X.T @ W @ X)
    b = A @ X.T @ W @ y
    u = y - X @ b
    hf = np.diag(X @ A @ X.T @ W)
    ho = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    sig = u**2 / v**2 * (1 / (1 - hf) ** 2 + 4 * ho / k * df)
    M = np.linalg.inv(X.T @ W @ X / T)
    Om = X.T @ np.diag(sig) @ X / T
    return b, M @ Om @ M / T


def test_eq8_variance_by_hand(sim):
    x = sim.X[:, 1]
    est = estimate_hetero(sim.y, sim.X, paper_z(x), list("abcde"), random_state=0)
    b, V = _eq8_by_hand(sim.X, sim.y, est.v, est.u_unweighted, est.df)
    np.testing.assert_allclose(est.beta, b, rtol=1e-10)
    np.testing.assert_allclose(est.cov, V, rtol=1e-8)


def test_eq7_variance_by_hand(sim):
    x = sim.X[:, 1]
    est = estimate_hetero(sim.y, sim.X, x[:, None], ["x"], method="wls_s1")
    X, v = sim.X, est.v
    T = 300
    u0 = est.u_unweighted
    ho = hat_diag(X)
    sig = (u0**2 / (1 - ho) ** 2) / v**2
    M = np.linalg.inv((X / v[:, None]).T @ X / T)
    V = M @ (X.T @ np.diag(sig) @ X / T) @ M / T
    np.testing.assert_allclose(est.cov, V, rtol=1e-8)


def test_variance_invariant_to_weight_normalisation(sim):
    v = sim.info["v"]
    kw = {"X": sim.X, "u_ols": sim.u, "h_ols": hat_diag(sim.X), "df": 3.0, "method": "lasso"}
    u = sim.y - sim.X @ lstsq(sim.X / np.sqrt(v)[:, None], sim.y / np.sqrt(v))
    V1 = hetero_covariance("robust", v=v, u_fgls=u, **kw)
    V2 = hetero_covariance("robust", v=7.0 * v, u_fgls=u, **kw)
    np.testing.assert_allclose(V1, V2, rtol=1e-10)


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_other_cov_types(sim, cov_type):
    res = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit(cov_type=cov_type)
    assert np.isfinite(res.bse).all()
    assert res.cov_type == cov_type


# ------------------------------------------------------------------ model class
def test_lasso_fgls_end_to_end(sim):
    df = pd.DataFrame({"const": 1.0, "size": sim.X[:, 1]})
    y = pd.Series(sim.y, name="wage")
    res = pf.LassoFGLS(y, df, random_state=1).fit()
    ols = pf.LassoFGLS(y, df, skedastic="none").fit()
    assert res.hetero.skedastic == "lasso"
    assert res.method == "FGLS (hetero)"
    assert ols.method == "OLS"
    assert res.bse["size"] < ols.bse["size"]
    assert res.selected_z
    assert res.skedastic_coef.index[0] == "intercept"
    assert res.weights.shape == (300,)
    assert res.df_hat == len(res.selected_z) + 1
    text = str(res.summary())
    assert "Skedastic fit" in text
    assert "wage" in text
    assert res.n_eff == 300
    # reproducible with a fixed seed
    res2 = pf.LassoFGLS(y, df, random_state=1).fit()
    np.testing.assert_allclose(res.params, res2.params)


def test_user_supplied_z_matrix_and_names(sim):
    x = sim.X[:, 1]
    Zdf = pd.DataFrame({"x": x, "x2": x**2})
    res = pf.LassoFGLS(sim.y, sim.X, z=Zdf, random_state=0).fit()
    assert set(res.skedastic_coef.index) == {"intercept", "x", "x2"}
    res2 = pf.LassoFGLS(sim.y, sim.X, z=["x1"], skedastic="wls_s1").fit()
    assert list(res2.skedastic_coef.index) == ["intercept", "x1"]
    with pytest.raises(InputError):
        pf.LassoFGLS(sim.y, sim.X, z="everything")


def test_missing_rows_dropped_jointly_with_z(sim):
    y = sim.y.copy()
    y[5] = np.nan
    Z = np.column_stack([sim.X[:, 1], sim.X[:, 1] ** 2])
    Z[7, 0] = np.nan
    m = pf.LassoFGLS(y, sim.X, z=Z, missing="drop", random_state=0)
    assert m.data.nobs == 298
    assert m.Z.shape == (298, 2)


def test_known_and_callable_skedastic(sim):
    r1 = pf.LassoFGLS(sim.y, sim.X, skedastic=sim.info["v"]).fit()
    r2 = pf.LassoFGLS(sim.y, sim.X, skedastic="known", variance=sim.info["v"]).fit()
    r3 = pf.LassoFGLS(sim.y, sim.X, skedastic=lambda X: X[:, 1] ** 2).fit()
    np.testing.assert_allclose(r1.params, r2.params)
    np.testing.assert_allclose(r1.params, r3.params)
    assert r1.method == "GLS (known variances)"
    with pytest.raises(InputError):
        pf.LassoFGLS(sim.y, sim.X, skedastic="known").fit()


def test_svr_and_wls_models(sim):
    r = pf.LassoFGLS(sim.y, sim.X, skedastic="svr", random_state=0).fit()
    assert r.skedastic_coef is None
    assert r.df_hat > 1
    w = pf.LassoFGLS(sim.y, sim.X, z=["x1"], skedastic="wls_s2").fit()
    assert "eq. (7)" in " ".join(w.notes)


def test_delta_scale_and_weight_clip(sim):
    r = pf.LassoFGLS(10 * sim.y, sim.X, delta_scale=True, random_state=0).fit()
    assert r.hetero.tuning["delta"] > 0.1
    c = pf.LassoFGLS(sim.y, sim.X, weight_clip=(0.05, 0.95), random_state=0).fit()
    assert c.hetero.variance.max() / c.hetero.variance.min() < np.inf


def test_scale_invariance_with_scaled_delta(sim):
    a = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit()
    b = pf.LassoFGLS(4.0 * sim.y, sim.X, delta=0.4, random_state=0).fit()
    np.testing.assert_allclose(b.params, 4 * a.params, rtol=1e-6)
    np.testing.assert_allclose(b.tvalues, a.tvalues, rtol=1e-6)


def test_cross_section_ignores_row_order(sim):
    perm = np.random.default_rng(11).permutation(300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.LassoFGLS(pd.Series(sim.y).iloc[perm], pd.DataFrame(sim.X).iloc[perm], random_state=0)


def test_invalid_options(sim):
    with pytest.raises(InputError):
        pf.LassoFGLS(sim.y, sim.X, skedastic="kernel")
    with pytest.raises(InputError):
        pf.LassoFGLS(sim.y, sim.X, lambda_rule="max")
    with pytest.raises(InputError):
        pf.LassoFGLS(sim.y, sim.X).fit(cov_type="robustest")


def test_lasso_coefs_are_scale_equivariant_for_tiny_penalties():
    """sklearn's LARS stops below float32 eps (~1.2e-7); lasso_coefs must not."""
    from pyfgls.hetero.adaptive_lasso import lasso_coefs

    rng = np.random.default_rng(42)
    n = 120
    Z = rng.normal(size=(n, 6))
    Z[:, 1] = Z[:, 0] + 0.1 * rng.normal(size=n)
    r = Z[:, :3] @ [1.0, -1.0, 0.5] + rng.normal(size=n)
    Zc, rc = Z - Z.mean(0), r - r.mean()
    alphas = np.max(np.abs(Zc.T @ rc)) / n * np.logspace(0, -4, 50)
    c = 1e-6
    ref = lasso_coefs(Zc, rc, alphas)
    small = lasso_coefs(c * Zc, rc, c * alphas)
    np.testing.assert_allclose(small * c, ref, rtol=1e-8, atol=1e-10)


def test_round_off_coefficients_are_not_selected(monkeypatch):
    """A coefficient of order 1e-17 (LARS interpolation at a knot) is not a selection."""
    import importlib

    from pyfgls.hetero.adaptive_lasso import adaptive_lasso

    # the package re-exports the function adaptive_lasso under the submodule's name
    al_mod = importlib.import_module("pyfgls.hetero.adaptive_lasso")

    sim = hetero_dgp(200, skedastic="homoskedastic", random_state=3)
    x = sim.X[:, 1]
    Z = np.column_stack([x, x**2, np.cos(x)])
    r = np.log(np.maximum(sim.u**2, 0.01))
    ref = adaptive_lasso(r, Z, random_state=0)
    original = al_mod.lasso_coefs

    def noisy(Zc, rc, alphas):
        out = original(Zc, rc, alphas)
        return np.where(out == 0.0, 1e-17, out)

    monkeypatch.setattr(al_mod, "lasso_coefs", noisy)
    fit = adaptive_lasso(r, Z, random_state=0)
    assert fit.selected == ref.selected
    assert fit.df == ref.df


def test_lasso_coefs_use_the_exact_path_for_tiny_covariates(monkeypatch):
    """sklearn's LARS flags pivots below 1e-7 (absolute) as degenerate; with tiny
    covariates that sent the fit to the slow coordinate-descent fallback."""
    import importlib

    al_mod = importlib.import_module("pyfgls.hetero.adaptive_lasso")
    rng = np.random.default_rng(5)
    n = 100
    x = rng.uniform(1, 4, size=n)
    Z = np.column_stack([x, np.log(x) ** 2, x**2, np.cos(x), np.cos(2 * x)])
    r = np.log(np.maximum(rng.normal(size=n) ** 2, 0.01))
    Zc, rc = Z - Z.mean(0), r - r.mean()
    alphas = np.max(np.abs(Zc.T @ rc)) / n * np.logspace(0, -4, 100)
    ref = lasso_coefs(Zc, rc, alphas)

    def no_fallback(*args, **kwargs):
        raise AssertionError("coordinate-descent fallback used")

    monkeypatch.setattr(al_mod, "lasso_path", no_fallback)
    c = 1e-6
    small = lasso_coefs(c * Zc, rc, c * alphas)
    np.testing.assert_allclose(small * c, ref, rtol=1e-8, atol=1e-10)
