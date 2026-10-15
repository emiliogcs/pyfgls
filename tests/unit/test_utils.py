import numpy as np
import pytest

from pyfgls._utils import (
    adjusted_min_lags,
    ar_filter,
    ar_roots,
    check_random_state,
    critical_value,
    default_kmax,
    lag_matrix,
    theory_kmax,
    two_sided_pvalues,
)


def test_lag_matrix_vector():
    x = np.arange(1.0, 7.0)
    L = lag_matrix(x, 2)
    np.testing.assert_array_equal(L, [[2, 1], [3, 2], [4, 3], [5, 4]])


def test_lag_matrix_matrix_lag_major_and_start():
    x = np.column_stack([np.arange(6.0), 10 + np.arange(6.0)])
    L = lag_matrix(x, 2, start=3)
    # rows t = 3, 4, 5 ; columns [a_{t-1}, b_{t-1}, a_{t-2}, b_{t-2}]
    np.testing.assert_array_equal(L[0], [2, 12, 1, 11])
    assert L.shape == (3, 4)
    assert lag_matrix(x, 0, start=2).shape == (4, 0)
    with pytest.raises(ValueError):
        lag_matrix(x, 3, start=2)


def test_ar_filter_matches_explicit_loop():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    rho = np.array([0.5, -0.2, 0.1])
    out = ar_filter(x, rho)
    expected = np.array([x[t] - sum(rho[j] * x[t - j - 1] for j in range(3)) for t in range(3, 30)])
    np.testing.assert_allclose(out, expected)
    # the constant column filters to 1 - sum(rho)
    np.testing.assert_allclose(ar_filter(np.ones(10), rho), 1 - rho.sum())
    # empty filter is the identity
    np.testing.assert_allclose(ar_filter(x, []), x)


def test_ar_roots():
    np.testing.assert_allclose(np.abs(ar_roots([0.5])), [2.0])
    r = ar_roots([1.34, -0.42])  # roots of 1 - 1.34 z + 0.42 z^2
    np.testing.assert_allclose(sorted(np.abs(r)), sorted([1 / 0.84, 1 / 0.5]), rtol=1e-8)
    assert ar_roots([]).size == 0
    assert ar_roots([0.0, 0.0]).size == 0
    np.testing.assert_allclose(np.abs(ar_roots([0.0, 0.25])), [2.0, 2.0])


def test_lag_rules():
    assert default_kmax(200) == 12
    assert default_kmax(500) == 15
    assert theory_kmax(200) == 3
    assert adjusted_min_lags(200) == 13


def test_random_state():
    g = np.random.default_rng(3)
    assert check_random_state(g) is g
    a = check_random_state(5).normal()
    b = check_random_state(5).normal()
    assert a == b


def test_critical_values_and_pvalues():
    assert critical_value(0.05) == pytest.approx(1.959964, rel=1e-6)
    assert critical_value(0.05, df=10) == pytest.approx(2.228139, rel=1e-6)
    with pytest.raises(ValueError):
        critical_value(1.5)
    p = two_sided_pvalues(np.array([1.959964, 0.0]))
    np.testing.assert_allclose(p, [0.05, 1.0], rtol=1e-5)
    assert two_sided_pvalues(np.array([2.228139]), df=10)[0] == pytest.approx(0.05, rel=1e-5)
