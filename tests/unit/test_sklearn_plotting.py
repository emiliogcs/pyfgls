import matplotlib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pyfgls as pf
from pyfgls.plotting import acf, plot_diagnostics, plot_intervals
from pyfgls.simulate import combined_dgp, hetero_dgp, serial_dgp
from pyfgls.sklearn import FGLSRegressor

matplotlib.use("Agg")


def test_regressor_fit_predict_and_params():
    sim = hetero_dgp(200, skedastic="x2", random_state=0)
    reg = FGLSRegressor(serial=None, random_state=0).fit(sim.X[:, 1:], sim.y)
    assert reg.n_features_in_ == 1
    assert reg.coef_.shape == (1,)
    ref = pf.LassoFGLS(sim.y, sim.X, random_state=0).fit()
    assert reg.intercept_ == pytest.approx(ref.params.iloc[0])
    assert reg.coef_[0] == pytest.approx(ref.params.iloc[1])
    np.testing.assert_allclose(reg.predict(sim.X[:, 1:]), sim.X @ ref.params.to_numpy())
    assert reg.score(sim.X[:, 1:], sim.y) > 0
    params = reg.get_params()
    assert params["serial"] is None
    reg2 = clone(reg).set_params(hetero=None)
    assert reg2.hetero is None
    with pytest.raises(ValueError):
        reg.predict(np.ones((3, 2)))


def test_regressor_in_pipeline_and_cross_validation():
    sim = serial_dgp(300, ar=(0.5,), rho_x=0.5, gamma=0.0, burn=100, random_state=1)
    pipe = make_pipeline(StandardScaler(), FGLSRegressor(hetero=None))
    scores = cross_val_score(pipe, sim.X[:, 1:], sim.y, cv=TimeSeriesSplit(3))
    assert np.all(np.isfinite(scores))
    no_int = FGLSRegressor(hetero=None, fit_intercept=False).fit(sim.X, sim.y)
    assert no_int.intercept_ == 0.0
    assert no_int.coef_.shape == (2,)


def test_acf_white_noise():
    rng = np.random.default_rng(2)
    r = acf(rng.normal(size=5000), 5)
    assert r.shape == (5,)
    assert np.all(np.abs(r) < 0.05)


@pytest.mark.parametrize("kind", ["serial", "hetero", "combined"])
def test_plots_render(kind):
    if kind == "serial":
        s = serial_dgp(200, random_state=3)
        res = pf.SerialFGLS(s.y, s.X).fit()
    elif kind == "hetero":
        s = hetero_dgp(200, random_state=3)
        res = pf.LassoFGLS(s.y, s.X, random_state=0).fit()
    else:
        s = combined_dgp(200, random_state=3)
        res = pf.FGLS(s.y, s.X, random_state=0).fit()
    fig = plot_diagnostics(res)
    assert len(fig.axes) == 4
    fig2 = plot_intervals([res], params=["x1"])
    assert fig2.axes
    with pytest.raises(pf.exceptions.InputError):
        plot_intervals([])
