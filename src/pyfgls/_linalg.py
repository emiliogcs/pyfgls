"""Numerically careful linear-algebra helpers.

Rules followed throughout pyfgls:

* regressions are solved by QR / ``lstsq``, never through the normal equations;
* rank decisions use a rank-revealing (pivoted) QR on column-normalised matrices so they
  do not depend on the units of the regressors;
* only small (q x q) symmetric positive definite systems are solved directly, by
  Cholesky with a pseudo-inverse fallback.
"""

from __future__ import annotations

import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pyfgls.exceptions import CollinearityWarning

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.intp]

_EPS = float(np.finfo(np.float64).eps)


def default_rank_tol(n: int, m: int) -> float:
    """Relative tolerance used to declare a column linearly dependent.

    ``max(n, m) * eps * 100`` applied to column-normalised data: roughly ``1e-11`` for
    a few hundred observations. Exactly collinear columns (lagged constants, lagged
    trends, periodic dummies) have relative residual norms near machine precision.
    """
    return max(n, m, 1) * _EPS * 100.0


def lstsq(X: FloatArray, y: FloatArray) -> FloatArray:
    """Least-squares coefficients via LAPACK ``gelsy`` (complete orthogonal factorisation).

    ``y`` may be a vector or a matrix of several right-hand sides.
    """
    sol = scipy.linalg.lstsq(X, y, lapack_driver="gelsy", check_finite=False)
    coef: FloatArray = np.asarray(sol[0], dtype=np.float64)
    return coef


def ols(X: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ``(coef, resid)`` of the OLS regression of ``y`` on ``X``."""
    coef = lstsq(X, y)
    return coef, y - X @ coef


def column_norms(X: FloatArray) -> FloatArray:
    """Euclidean norm of each column, with zeros replaced by one (safe for scaling)."""
    norms: FloatArray = np.sqrt(np.einsum("ij,ij->j", X, X))
    norms[norms == 0.0] = 1.0
    return norms


def independent_columns(
    A: FloatArray,
    base: FloatArray | None = None,
    *,
    tol: float | None = None,
) -> tuple[IntArray, IntArray]:
    """Split the columns of ``A`` into linearly independent and dependent ones.

    Columns are processed **in order** (modified Gram--Schmidt with
    re-orthogonalisation on unit-norm columns), so that when a column is a linear
    combination of ``base`` and of earlier columns of ``A`` it is the *later* column that
    is reported as dependent. This keeps the decision interpretable ("the lag of the
    constant is dropped") and makes it stable under truncation: the selection for the
    first ``j`` columns does not depend on columns after ``j``.

    Parameters
    ----------
    A : ndarray, shape (n, m)
        Candidate columns.
    base : ndarray, shape (n, p), optional
        Columns that are always kept (assumed to have full column rank). Columns of
        ``A`` in the span of ``base`` are dependent.
    tol : float, optional
        Relative residual-norm threshold; default :func:`default_rank_tol`.

    Returns
    -------
    keep, drop : ndarray of int
        Indices (into the columns of ``A``) of the independent and dependent columns.
    """
    n, m = A.shape
    if tol is None:
        tol = default_rank_tol(n, m + (0 if base is None else base.shape[1]))
    if m == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty

    if base is not None and base.shape[1] > 0:
        Qb, _ = np.linalg.qr(base / column_norms(base))
    else:
        Qb = np.zeros((n, 0))

    Q = np.empty((n, min(n, m)))
    r = 0
    keep: list[int] = []
    drop: list[int] = []
    norms = column_norms(A)
    for j in range(m):
        v = A[:, j] / norms[j]
        if not np.any(A[:, j]):
            drop.append(j)
            continue
        for _ in range(2):  # twice is enough (Kahan--Parlett)
            if Qb.shape[1]:
                v = v - Qb @ (Qb.T @ v)
            if r:
                v = v - Q[:, :r] @ (Q[:, :r].T @ v)
        nv = float(np.linalg.norm(v))
        if nv <= tol or r >= n:
            drop.append(j)
        else:
            Q[:, r] = v / nv
            r += 1
            keep.append(j)
    return np.asarray(keep, dtype=np.intp), np.asarray(drop, dtype=np.intp)


def matrix_rank(X: FloatArray, *, tol: float | None = None) -> int:
    """Numerical rank of ``X`` from a pivoted QR of the column-normalised matrix."""
    n, m = X.shape
    if m == 0:
        return 0
    if tol is None:
        tol = default_rank_tol(n, m)
    R = scipy.linalg.qr(X / column_norms(X), mode="r", pivoting=True, check_finite=False)[0]
    d = np.abs(np.diag(R))
    if d.size == 0 or d[0] == 0.0:
        return 0
    return int(np.sum(d > tol * d[0]))


def hat_diag(X: FloatArray) -> FloatArray:
    """Diagonal of the projection matrix ``X (X'X)^- X'`` (the leverages).

    Computed as the squared row norms of the orthonormal factor of a pivoted QR,
    restricted to the numerical rank, so it is valid for rank-deficient ``X`` too.
    """
    n, m = X.shape
    if m == 0:
        return np.zeros(n)
    Q, R, _ = scipy.linalg.qr(X, mode="economic", pivoting=True, check_finite=False)
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > default_rank_tol(n, m) * d[0])) if d.size and d[0] > 0 else 0
    Qr = Q[:, :rank]
    return np.asarray(np.einsum("ij,ij->i", Qr, Qr), dtype=np.float64)


def symmetrize(A: FloatArray) -> FloatArray:
    """Return ``(A + A') / 2``."""
    return np.asarray(0.5 * (A + A.T), dtype=np.float64)


def solve_psd(A: FloatArray, B: FloatArray, *, name: str = "matrix") -> FloatArray:
    """Solve ``A X = B`` for symmetric positive (semi)definite ``A``.

    Uses a Cholesky factorisation; if ``A`` is not numerically positive definite,
    warns with :class:`~pyfgls.exceptions.CollinearityWarning` and falls back to the
    Moore--Penrose pseudo-inverse.
    """
    A = symmetrize(A)
    try:
        c = scipy.linalg.cho_factor(A, check_finite=False)
        return np.asarray(scipy.linalg.cho_solve(c, B, check_finite=False), dtype=np.float64)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"The {name} is not positive definite (it is singular or nearly so); "
            "a pseudo-inverse is used instead. Check the regressors for collinearity.",
            CollinearityWarning,
            stacklevel=3,
        )
        return np.asarray(np.linalg.pinv(A, hermitian=True) @ B, dtype=np.float64)


def inv_psd(A: FloatArray, *, name: str = "matrix") -> FloatArray:
    """Inverse of a small symmetric positive definite matrix (see :func:`solve_psd`)."""
    return symmetrize(solve_psd(A, np.eye(A.shape[0]), name=name))


def make_psd(
    A: FloatArray,
    *,
    name: str = "covariance matrix",
    rtol: float = 1e-8,
) -> FloatArray:
    """Symmetrise ``A`` and clip negative eigenvalues to zero.

    Negative eigenvalues smaller than ``rtol * max|eigenvalue|`` in magnitude are
    rounding noise and are clipped silently; larger ones trigger a warning.
    """
    A = symmetrize(A)
    if A.size == 0:
        return A
    w, V = np.linalg.eigh(A)
    if w.min() >= 0.0:
        return A
    scale = max(float(np.abs(w).max()), np.finfo(np.float64).tiny)
    if w.min() < -rtol * scale:
        warnings.warn(
            f"The {name} was not positive semidefinite (smallest eigenvalue "
            f"{w.min():.3g}); negative eigenvalues were set to zero.",
            CollinearityWarning,
            stacklevel=3,
        )
    w = np.clip(w, 0.0, None)
    return symmetrize((V * w) @ V.T)


def condition_number(A: FloatArray) -> float:
    """2-norm condition number of ``A`` (``inf`` if singular)."""
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0 or s[-1] == 0.0:
        return float("inf")
    return float(s[0] / s[-1])
