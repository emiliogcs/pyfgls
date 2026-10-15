"""Small shared utilities: lags, autoregressive filters, lag-order rules, RNGs."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.stats

from pyfgls._linalg import FloatArray

RandomState: TypeAlias = int | np.random.Generator | np.random.SeedSequence | None


def check_random_state(random_state: RandomState) -> np.random.Generator:
    """Turn ``None``, an int, a ``SeedSequence`` or a ``Generator`` into a ``Generator``."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def as_2d(x: npt.ArrayLike) -> FloatArray:
    """View a vector as a one-column matrix; leave matrices unchanged."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a[:, None]
    return a


def lag_matrix(x: npt.ArrayLike, nlags: int, *, start: int | None = None) -> FloatArray:
    """Matrix of lags ``[x_{t-1}, ..., x_{t-nlags}]`` for ``t = start, ..., T-1``.

    Parameters
    ----------
    x : array_like, shape (T,) or (T, m)
        Series (columns are variables, rows are time).
    nlags : int
        Number of lags.
    start : int, optional
        First row (0-based) of the output sample; default ``nlags``. Must be
        ``>= nlags``.

    Returns
    -------
    ndarray, shape (T - start, m * nlags)
        Lag-major column order: all variables at lag 1, then all at lag 2, ...

    Examples
    --------
    >>> lag_matrix([1.0, 2.0, 3.0, 4.0], 2)
    array([[2., 1.],
           [3., 2.]])
    """
    a = as_2d(x)
    T = a.shape[0]
    if start is None:
        start = nlags
    if nlags < 0 or start < nlags:
        raise ValueError(f"need 0 <= nlags <= start, got nlags={nlags}, start={start}")
    if nlags == 0:
        return np.zeros((T - start, 0))
    return np.hstack([a[start - j : T - j] for j in range(1, nlags + 1)])


def ar_filter(x: npt.ArrayLike, rho: npt.ArrayLike, *, start: int | None = None) -> FloatArray:
    """Quasi-difference ``x_t - sum_j rho_j x_{t-j}`` for ``t = start, ..., T-1``.

    This is the filter ``A(L) = 1 - rho_1 L - ... - rho_k L^k`` of [PG26] eq. (15),
    applied to every column of ``x`` (including a constant, whose filtered value is
    ``1 - sum_j rho_j``). The first ``k`` observations are dropped (no Prais--Winsten).

    Examples
    --------
    >>> ar_filter([1.0, 2.0, 4.0, 8.0], [0.5])
    array([1.5, 3. , 6. ])
    """
    a = np.asarray(x, dtype=np.float64)
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    k = r.size
    if start is None:
        start = k
    T = a.shape[0]
    out: FloatArray = a[start:T].copy()
    for j in range(1, k + 1):
        out -= r[j - 1] * a[start - j : T - j]
    return out


def ar_roots(rho: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Roots of the lag polynomial ``1 - rho_1 z - ... - rho_k z^k``.

    The filter is invertible (stationary errors, [PG26] Assumption 2) when all roots
    lie strictly outside the unit circle.

    Examples
    --------
    >>> np.abs(ar_roots([0.5]))
    array([2.])
    """
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    if r.size == 0:
        return np.zeros(0, dtype=np.complex128)
    # np.roots wants the highest power first: -rho_k z^k - ... - rho_1 z + 1
    coefs = np.concatenate([-r[::-1], [1.0]])
    nz = np.flatnonzero(coefs)
    if nz.size == 0 or nz[0] == coefs.size - 1:
        return np.zeros(0, dtype=np.complex128)
    return np.asarray(np.roots(coefs[nz[0] :]), dtype=np.complex128)


def default_kmax(T: int) -> int:
    """Default maximal lag order ``int(12 * (T / 200) ** 0.25)`` (12 at ``T = 200``).

    [PG26] only requires ``k_max`` to grow with ``T`` and uses 12 in its simulations
    with ``T = 200``; this rule scales that choice like Schwert's (1989) rule.
    """
    return int(12 * (T / 200.0) ** 0.25)


def theory_kmax(T: int) -> int:
    """Rate-based maximal order ``floor((T / ln T) ** (1/3))`` ([PG26] Suppl. Lemma A.2)."""
    return math.floor(math.pow(T / math.log(T), 1.0 / 3.0))


def adjusted_min_lags(T: int) -> int:
    """Lower bound ``ceil((ln T) ** 1.5)`` on the AR order ([PG26] Suppl. Lemma A.2)."""
    return math.ceil(math.pow(math.log(T), 1.5))


def critical_value(alpha: float, *, df: float | None = None) -> float:
    """Two-sided critical value: standard normal, or Student-t when ``df`` is given."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if df is None:
        return float(scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    return float(scipy.stats.t.ppf(1.0 - alpha / 2.0, df))


def two_sided_pvalues(stat: FloatArray, *, df: float | None = None) -> FloatArray:
    """Two-sided p-values of t/z statistics (normal, or Student-t with ``df``)."""
    a = np.abs(stat)
    if df is None:
        return np.asarray(2.0 * scipy.stats.norm.sf(a), dtype=np.float64)
    return np.asarray(2.0 * scipy.stats.t.sf(a, df), dtype=np.float64)
