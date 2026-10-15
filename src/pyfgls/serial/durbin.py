"""The Durbin (1960) regression underlying the AR filter of [PG26] eq. (14).

For an AR order ``k`` the Durbin regression is

    y_t = sum_{j=1..k} rho_j y_{t-j} + x_t' beta - sum_{j=1..k} x_{t-j}' delta_j + e_{kt},

estimated by OLS. Its coefficients on the lagged dependent variable estimate the AR
coefficients of the errors consistently whether the regressors are exogenous or only
pre-determined, because every regressor is uncorrelated with the innovation ``e_t``.

The design matrix is laid out lag by lag::

    [x_t | y_{t-1}, x_{t-1} | y_{t-2}, x_{t-2} | ... | y_{t-k}, x_{t-k}]

and lagged columns that are linear combinations of the contemporaneous block or of
earlier columns (lagged constants, lagged trends, periodic dummies, or a lag of ``y``
that is already a regressor) are dropped *in order*. Hence the design for any order
``k' < k`` is a prefix of the design for ``k`` on the same sample, which lets the lag
selection reuse one QR factorisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyfgls._linalg import FloatArray, default_rank_tol, independent_columns, lstsq

#: Relative residual-norm threshold for dropping lagged columns. Lagged deterministic
#: terms are collinear up to rounding (~1e-15); genuine stochastic regressors are far
#: above this.
LAG_BLOCK_TOL = 1e-9


@dataclass(frozen=True)
class DurbinDesign:
    """Design matrix of a Durbin regression on the sample ``t = start, ..., T-1``.

    Attributes
    ----------
    Z : ndarray, shape (n, p)
        Regressors (first ``n_x`` columns: the contemporaneous ``x_t``).
    y : ndarray, shape (n,)
        Dependent variable on the same sample.
    names : tuple of str
        Column names (``"L2.y"``, ``"L1.x1"``, ...).
    n_x : int
        Number of contemporaneous regressors.
    ylag_pos : dict
        ``lag -> column of Z`` for the kept lags of ``y``.
    xlag_pos : dict
        ``(lag, regressor index) -> column of Z`` for the kept lags of ``x``.
    block_end : tuple of int
        ``block_end[k]`` is the number of columns of the order-``k`` design.
    dropped : tuple of str
        Names of the lagged columns dropped as collinear.
    start : int
        First (0-based) row of the sample.
    order : int
        AR order ``k``.
    """

    Z: FloatArray
    y: FloatArray
    names: tuple[str, ...]
    n_x: int
    ylag_pos: dict[int, int]
    xlag_pos: dict[tuple[int, int], int]
    block_end: tuple[int, ...]
    dropped: tuple[str, ...]
    start: int
    order: int

    @property
    def nobs(self) -> int:
        """Number of observations in the regression."""
        return int(self.y.shape[0])

    def prefix(self, k: int) -> FloatArray:
        """Design of the order-``k`` regression (a prefix of the columns)."""
        return self.Z[:, : self.block_end[k]]

    def ylag_columns(self, k: int | None = None) -> list[int]:
        """Positions of the kept ``y`` lags of order ``<= k``."""
        k = self.order if k is None else k
        return [self.ylag_pos[j] for j in range(1, k + 1) if j in self.ylag_pos]


def durbin_design(
    y: FloatArray,
    X: FloatArray,
    x_names: Sequence[str],
    order: int,
    *,
    start: int | None = None,
    y_name: str = "y",
    tol: float = LAG_BLOCK_TOL,
) -> DurbinDesign:
    """Build the Durbin-regression design of order ``order`` ([PG26] eq. (14)).

    Parameters
    ----------
    y : ndarray, shape (T,)
    X : ndarray, shape (T, q)
        Regressors (full rank, validated).
    x_names : sequence of str
    order : int
        AR order ``k``.
    start : int, optional
        First 0-based row of the sample; default ``order``. Use ``start = k_max`` to
        put every order on the common sample of the lag selection.
    y_name : str
    tol : float
        Relative tolerance for dropping collinear lagged columns.

    Returns
    -------
    DurbinDesign
    """
    T, q = X.shape
    if start is None:
        start = order
    if start < order:
        raise ValueError("start must be at least the AR order")
    rows = slice(start, T)
    base = X[rows]

    parts: list[FloatArray] = []
    meta: list[tuple[int, int]] = []  # (lag, regressor index or -1 for y)
    for j in range(1, order + 1):
        parts.append(y[start - j : T - j, None])
        meta.append((j, -1))
        parts.append(X[start - j : T - j])
        meta.extend((j, i) for i in range(q))
    if parts:
        A = np.hstack(parts)
        keep, drop = independent_columns(A, base, tol=max(tol, default_rank_tol(*A.shape)))
    else:
        A = np.zeros((T - start, 0))
        keep = drop = np.zeros(0, dtype=np.intp)

    def _name(m: tuple[int, int]) -> str:
        lag, i = m
        return f"L{lag}.{y_name}" if i < 0 else f"L{lag}.{x_names[i]}"

    names = list(x_names) + [_name(meta[c]) for c in keep]
    ylag_pos: dict[int, int] = {}
    xlag_pos: dict[tuple[int, int], int] = {}
    counts = np.zeros(order + 1, dtype=int)
    for pos, c in enumerate(keep, start=q):
        lag, i = meta[c]
        counts[lag] += 1
        if i < 0:
            ylag_pos[lag] = pos
        else:
            xlag_pos[(lag, i)] = pos
    block_end = tuple(int(q + counts[: k + 1].sum()) for k in range(order + 1))
    Z = np.hstack([base, A[:, keep]]) if keep.size else base.copy()
    return DurbinDesign(
        Z=np.ascontiguousarray(Z),
        y=y[rows].copy(),
        names=tuple(names),
        n_x=q,
        ylag_pos=ylag_pos,
        xlag_pos=xlag_pos,
        block_end=block_end,
        dropped=tuple(_name(meta[c]) for c in drop),
        start=start,
        order=order,
    )


@dataclass(frozen=True)
class DurbinFit:
    """OLS fit of a Durbin regression."""

    design: DurbinDesign
    coef: FloatArray
    resid: FloatArray
    sigma2: float
    bse: FloatArray

    @property
    def tvalues(self) -> FloatArray:
        """Conventional t statistics of the Durbin coefficients."""
        out: FloatArray = self.coef / self.bse
        return out

    @property
    def beta(self) -> FloatArray:
        """Coefficients on the contemporaneous regressors (``beta^D``)."""
        return self.coef[: self.design.n_x]


def fit_durbin(design: DurbinDesign) -> DurbinFit:
    """OLS of the Durbin regression with conventional standard errors."""
    Z, yv = design.Z, design.y
    coef = lstsq(Z, yv)
    resid = yv - Z @ coef
    n, p = Z.shape
    sigma2 = float(resid @ resid / n)
    R = np.linalg.qr(Z, mode="r")
    Rinv = np.linalg.solve(R, np.eye(p)) if p else np.zeros((0, 0))
    s2_dof = float(resid @ resid / max(n - p, 1))
    bse = np.sqrt(s2_dof * np.einsum("ij,ij->i", Rinv, Rinv))
    return DurbinFit(design=design, coef=coef, resid=resid, sigma2=sigma2, bse=bse)
