"""Candidate covariates ``z_t`` for the skedastic function ([GP24] Section 5).

The default dictionary applies, to every non-constant regressor ``x``,

    x, x^2, log(x), [log(x)]^2 (only if x > 0), cos(x), cos(2x),

which contains the specification ``z = (1, x, [log x]^2, x^2, cos x, cos 2x)`` used in
the [GP24] simulations (the intercept of the skedastic regression is always included
and never penalised). Binary/dummy columns are passed through untransformed, undefined
or constant transforms are skipped, and duplicated columns are removed.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls.exceptions import InputError, SampleSizeWarning

#: Transform families understood by :func:`make_dictionary`.
TRANSFORMS = ("linear", "poly2", "log", "trig")

#: Warn when the dictionary has more columns than this many per observation.
MANY_COLUMNS_RATIO = 1.0


def _is_dummy(x: FloatArray) -> bool:
    u = np.unique(x)
    return u.size <= 2


def make_dictionary(
    X: FloatArray,
    names: Sequence[str],
    *,
    transforms: Sequence[str] = ("linear", "poly2", "log", "trig"),
    skip: Sequence[int] = (),
) -> tuple[FloatArray, tuple[str, ...]]:
    """Build the candidate-covariate dictionary from regressors.

    Parameters
    ----------
    X : ndarray (T, q)
        Regressors (constant columns are ignored).
    names : sequence of str
        Regressor names.
    transforms : sequence of {"linear", "poly2", "log", "trig"}
        ``"linear"``: ``x``; ``"poly2"``: ``x^2``; ``"log"``: ``log x`` and
        ``(log x)^2`` when ``x > 0``; ``"trig"``: ``cos x`` and ``cos 2x``.
    skip : sequence of int
        Regressor positions to leave out entirely (e.g. lagged dependent variables).

    Returns
    -------
    Z : ndarray (T, d)
    z_names : tuple of str

    Examples
    --------
    >>> X = np.column_stack([np.ones(10), np.arange(1.0, 11.0), np.arange(10) % 2])
    >>> Z, zn = make_dictionary(X, ["const", "x", "d"])
    >>> zn
    ('x', 'x^2', 'log(x)', 'log(x)^2', 'cos(x)', 'cos(2x)', 'd')
    """
    bad = [t for t in transforms if t not in TRANSFORMS]
    if bad:
        raise InputError(f"Unknown transforms {bad}; choose from {list(TRANSFORMS)}.")
    T = X.shape[0]
    cols: list[FloatArray] = []
    out_names: list[str] = []

    def add(v: FloatArray, nm: str) -> None:
        if not np.all(np.isfinite(v)) or np.ptp(v) <= 1e-12 * max(1.0, float(np.abs(v).max())):
            return
        cols.append(v)
        out_names.append(nm)

    for j, nm in enumerate(names):
        if j in set(skip):
            continue
        x = X[:, j]
        if np.ptp(x) == 0.0:
            continue  # constant
        if _is_dummy(x):
            add(x, nm)
            continue
        if "linear" in transforms:
            add(x, nm)
        if "poly2" in transforms:
            add(x**2, f"{nm}^2")
        if "log" in transforms and np.all(x > 0):
            lx = np.log(x)
            add(lx, f"log({nm})")
            add(lx**2, f"log({nm})^2")
        if "trig" in transforms:
            add(np.cos(x), f"cos({nm})")
            add(np.cos(2.0 * x), f"cos(2{nm})")

    if not cols:
        return np.zeros((T, 0)), ()
    Z = np.column_stack(cols)
    Z, kept = dedupe_columns(Z)
    z_names = tuple(out_names[i] for i in kept)
    if Z.shape[1] > MANY_COLUMNS_RATIO * T:
        warnings.warn(
            f"The candidate dictionary has {Z.shape[1]} columns for {T} observations; the "
            "adaptive Lasso can handle d > T, but cross-validation will be slow and noisy.",
            SampleSizeWarning,
            stacklevel=2,
        )
    return Z, z_names


def dedupe_columns(Z: FloatArray, *, tol: float = 1e-10) -> tuple[FloatArray, list[int]]:
    """Remove columns that are exact duplicates (up to scale) of earlier ones."""
    keep: list[int] = []
    normed: list[FloatArray] = []
    for j in range(Z.shape[1]):
        v = Z[:, j] - Z[:, j].mean()
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            continue
        v = v / nv
        if any(abs(float(v @ w)) > 1.0 - tol for w in normed):
            continue
        keep.append(j)
        normed.append(v)
    return Z[:, keep], keep
