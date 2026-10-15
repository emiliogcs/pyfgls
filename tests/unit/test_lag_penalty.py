import numpy as np
import pytest

import pyfgls as pf
from pyfgls.exceptions import InputError, SpecificationWarning
from pyfgls.serial import select_ar_order
from pyfgls.simulate import arma_errors


def many_regressors(T=1000, q=10, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(T), rng.normal(size=(T, q - 1))])
    u = arma_errors(rng.normal(size=T + 100), (0.5, 0.2))[100:]
    return X.sum(axis=1) + u, X


def test_params_penalty_counts_all_added_coefficients():
    y, X = many_regressors()
    names = [f"x{j}" for j in range(X.shape[1])]
    a = select_ar_order(y, X, names, 8)
    b = select_ar_order(y, X, names, 8, ic_penalty="params")
    n = a.n_common
    # each lag adds 1 + 9 Durbin coefficients (lagged constant dropped)
    np.testing.assert_allclose(b.values - a.values, 9 * np.arange(9) * np.log(n) / n)
    assert b.order <= a.order
    with pytest.raises(InputError):
        select_ar_order(y, X, names, 8, ic_penalty="hqic")


def test_order_penalty_overselects_with_many_regressors_and_warns():
    y, X = many_regressors()
    with pytest.warns(SpecificationWarning, match="ic_penalty='params'"):
        r = pf.SerialFGLS(y, X, kmax=8).fit()
    assert r.ar_order == 8
    r2 = pf.SerialFGLS(y, X, kmax=8, ic_penalty="params").fit()
    assert 1 <= r2.ar_order <= 2  # heavier penalty: no runaway to kmax (may miss a weak lag)
    assert r2.options["ic_penalty"] == "params"
