import numpy as np
import pytest

from pyfgls._linalg import (
    condition_number,
    hat_diag,
    independent_columns,
    inv_psd,
    lstsq,
    make_psd,
    matrix_rank,
    ols,
    solve_psd,
)
from pyfgls.exceptions import CollinearityWarning


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_lstsq_matches_normal_equations(rng):
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=50)
    coef = lstsq(X, y)
    np.testing.assert_allclose(coef, np.linalg.solve(X.T @ X, X.T @ y), rtol=1e-10)
    coef2, resid = ols(X, y)
    np.testing.assert_allclose(coef2, coef)
    np.testing.assert_allclose(X.T @ resid, 0.0, atol=1e-10)


def test_lstsq_multiple_rhs(rng):
    X = rng.normal(size=(40, 2))
    Y = rng.normal(size=(40, 3))
    np.testing.assert_allclose(lstsq(X, Y), np.linalg.lstsq(X, Y, rcond=None)[0], rtol=1e-10)


def test_independent_columns_drops_lagged_constant_and_trend():
    T = 100
    t = np.arange(1.0, T + 1)
    base = np.column_stack([np.ones(T - 1), t[1:]])
    # lagged constant (== constant) and lagged trend (== trend - 1) are dependent
    lagged = np.column_stack([np.ones(T - 1), t[:-1], np.sin(t[:-1])])
    keep, drop = independent_columns(lagged, base)
    assert keep.tolist() == [2]
    assert drop.tolist() == [0, 1]


def test_independent_columns_is_order_preserving(rng):
    a = rng.normal(size=60)
    b = rng.normal(size=60)
    A = np.column_stack([a, b, a + 2 * b, rng.normal(size=60)])
    keep, drop = independent_columns(A)
    assert keep.tolist() == [0, 1, 3]
    assert drop.tolist() == [2]


def test_independent_columns_scale_invariant(rng):
    a = rng.normal(size=80)
    A = np.column_stack([1e6 * a, 1e-6 * a, rng.normal(size=80)])
    keep, drop = independent_columns(A)
    assert keep.tolist() == [0, 2]
    assert drop.tolist() == [1]


def test_independent_columns_periodic_dummies():
    T = 48
    s = np.arange(T) % 4
    D = np.column_stack([(s == j).astype(float) for j in range(1, 4)])  # 3 quarterly dummies
    base = np.column_stack([np.ones(T - 1), D[1:]])
    keep, drop = independent_columns(D[:-1], base)  # lagged dummies
    assert keep.size == 0
    assert drop.tolist() == [0, 1, 2]


def test_matrix_rank(rng):
    X = rng.normal(size=(30, 3))
    assert matrix_rank(X) == 3
    assert matrix_rank(np.column_stack([X, X[:, 0] - X[:, 1]])) == 3
    assert matrix_rank(np.zeros((5, 0))) == 0


def test_hat_diag(rng):
    X = rng.normal(size=(25, 3))
    H = X @ np.linalg.solve(X.T @ X, X.T)
    np.testing.assert_allclose(hat_diag(X), np.diag(H), rtol=1e-10)
    # rank-deficient X: leverages of the column space
    Xd = np.column_stack([X, X[:, 0]])
    np.testing.assert_allclose(hat_diag(Xd), np.diag(H), rtol=1e-8)
    assert np.isclose(hat_diag(X).sum(), 3.0)


def test_solve_psd_and_inverse(rng):
    A = rng.normal(size=(20, 4))
    S = A.T @ A
    b = rng.normal(size=4)
    np.testing.assert_allclose(solve_psd(S, b), np.linalg.solve(S, b), rtol=1e-10)
    np.testing.assert_allclose(inv_psd(S) @ S, np.eye(4), atol=1e-10)


def test_solve_psd_singular_falls_back_with_warning():
    S = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.warns(CollinearityWarning, match="pseudo-inverse"):
        x = solve_psd(S, np.array([2.0, 2.0]))
    np.testing.assert_allclose(S @ x, [2.0, 2.0])


def test_make_psd():
    M = np.array([[1.0, 0.0], [0.0, -0.5]])
    with pytest.warns(CollinearityWarning, match="positive semidefinite"):
        P = make_psd(M)
    np.testing.assert_allclose(P, np.diag([1.0, 0.0]), atol=1e-12)
    # rounding-level negativity is clipped silently
    N = np.array([[1.0, 0.0], [0.0, -1e-14]])
    P2 = make_psd(N)
    assert np.linalg.eigvalsh(P2).min() >= 0.0
    # asymmetric input is symmetrised
    A = np.array([[2.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(make_psd(A), [[2.0, 0.5], [0.5, 2.0]])


def test_condition_number():
    assert condition_number(np.diag([1.0, 10.0])) == pytest.approx(10.0)
    assert condition_number(np.zeros((2, 2))) == np.inf
