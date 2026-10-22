"""Adaptive Lasso estimate of the skedastic function ([GP24] eq. (4)).

Given the response ``r_t = log(max(u_t^2, delta^2))`` (log squared residuals) and
candidate covariates ``z_t`` (d may exceed T), solve

    min_{g0, g}  1/2 sum_t (r_t - g0 - z_t' g)^2 + lambda sum_j w_j |g_j|,
    w_j = |g_j^ridge|^(-psi),

with an unpenalised intercept (Zou, 2006). Implementation:

1. standardise the columns of ``z`` (mean 0, variance 1, as glmnet does);
2. ridge first step, its penalty chosen by K-fold cross-validation (closed form through
   the SVD, so ``d > T`` is fine);
3. for each ``psi`` in the grid, rescale column ``j`` by ``1/w_j = |g_j^ridge|^psi`` so
   that an ordinary Lasso (``sklearn.linear_model.lasso_path``) solves the adaptive
   problem, and back-transform (``psi = 0`` is the plain Lasso);
4. choose ``(lambda, psi)`` jointly by K-fold cross-validation of the mean squared
   prediction error of ``r``, with the **same folds for every psi** so the CV errors
   are comparable ([GP24] Section 3). The adaptive weights are computed once on the
   full sample and held fixed across folds (as with glmnet's ``penalty.factor``).

If every slope is zero at the chosen penalty, the fitted variance is constant and FGLS
reduces to OLS — the "no loss under homoskedasticity" property of [GP24].
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning as SkConvergenceWarning
from sklearn.linear_model import lars_path_gram, lasso_path

from pyfgls._linalg import FloatArray, IntArray
from pyfgls._utils import RandomState, check_random_state
from pyfgls.exceptions import ConvergenceWarning, InputError, SampleSizeWarning

#: Default grid for the adaptive-weight exponent psi ([GP24]: L = 6 values).
PSI_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)

LambdaRule = Literal["min", "1se"]
CVWeights = Literal["fold", "full"]
RidgeScale = Literal["standardized", "original"]
FoldScheme = Literal["random", "blocked"]


@dataclass(frozen=True)
class SkedasticFit:
    """A fitted model for the log-variance ``log v_t``.

    Attributes
    ----------
    fitted : ndarray (T,)
        Fitted log-variance ``h_t``; the FGLS variances are ``exp(h_t)``.
    intercept : float
    coef : ndarray (d,)
        Slopes on the original (unstandardised) scale of ``z``.
    names : tuple of str
    df : float
        Degrees of freedom used in the [GP24] eq. (8) bias correction.
    method : str
    tuning : dict
        Tuning parameters and cross-validation output.
    """

    fitted: FloatArray
    intercept: float
    coef: FloatArray
    names: tuple[str, ...]
    df: float
    method: str
    tuning: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> tuple[str, ...]:
        """Names of the covariates with non-zero coefficients."""
        return tuple(n for n, c in zip(self.names, self.coef, strict=True) if c != 0.0)


def check_foldid(foldid: npt.ArrayLike, n: int) -> IntArray:
    """Validate user-supplied fold labels and recode them as ``0..K-1``."""
    f = np.asarray(foldid).reshape(-1)
    if f.shape[0] != n:
        raise InputError(
            f"foldid has {f.shape[0]} entries; the skedastic fit uses {n} observations."
        )
    _, codes = np.unique(f, return_inverse=True)
    if int(np.max(codes)) < 1:
        raise InputError("foldid must define at least two folds.")
    return np.asarray(codes, dtype=np.intp)


def make_folds(
    n: int, K: int, scheme: FoldScheme = "random", random_state: RandomState = None
) -> IntArray:
    """Fold labels ``0..K-1`` for ``n`` observations (random or contiguous blocks)."""
    if K < 2:
        raise InputError(f"cv_folds must be at least 2, got {K}.")
    if n < K:
        raise InputError(f"cv_folds={K} exceeds the number of observations ({n}).")
    if scheme == "random":
        rng = check_random_state(random_state)
        return np.asarray(rng.permutation(np.arange(n) % K), dtype=np.intp)
    if scheme == "blocked":
        return np.asarray(np.arange(n) * K // n, dtype=np.intp)
    raise InputError(f"fold_scheme must be 'random' or 'blocked', got {scheme!r}.")


def standardize(Z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Standardise columns; returns ``(Zs, mean, sd, keep)`` dropping constant columns."""
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    keep = np.flatnonzero(sd > 1e-12 * np.maximum(1.0, np.abs(mu)))
    Zs = (Z[:, keep] - mu[keep]) / sd[keep]
    return Zs, mu, sd, keep.astype(np.intp)


def ridge_cv(
    Zs: FloatArray, r: FloatArray, folds: IntArray, *, n_grid: int = 50
) -> tuple[FloatArray, float]:
    """Ridge regression of ``r`` on standardised ``Zs`` with the penalty chosen by CV.

    Objective ``||r - g0 - Zs g||^2 + lam ||g||^2``; grid ``lam = n 10^[-4, 4]``.
    """
    n, _d = Zs.shape
    grid = n * np.logspace(-4, 4, n_grid)
    K = int(np.max(folds)) + 1
    sse = np.zeros(n_grid)
    for k in range(K):
        te = folds == k
        tr = ~te
        Ztr = Zs[tr]
        zm = Ztr.mean(axis=0)
        rm = r[tr].mean()
        U, s, Vt = np.linalg.svd(Ztr - zm, full_matrices=False)
        Ur = U.T @ (r[tr] - rm)
        shrink = s[None, :] / (s[None, :] ** 2 + grid[:, None])  # (grid, rank)
        coefs = (shrink * Ur[None, :]) @ Vt  # (grid, d)
        pred = (Zs[te] - zm) @ coefs.T + rm
        sse += ((r[te][:, None] - pred) ** 2).sum(axis=0)
    lam = float(grid[int(np.argmin(sse))])
    return ridge_fit(Zs, r, lam), lam


def ridge_fit(Zs: FloatArray, r: FloatArray, lam: float) -> FloatArray:
    """Ridge slopes ``argmin ||r - g0 - Zs g||^2 + lam ||g||^2`` (intercept unpenalised)."""
    U, s, Vt = np.linalg.svd(Zs - Zs.mean(axis=0), full_matrices=False)
    g = Vt.T @ ((s / (s**2 + lam)) * (U.T @ (r - r.mean())))
    return np.asarray(g, dtype=np.float64)


def adaptive_scale(g_ridge: FloatArray, psi: float) -> FloatArray:
    """Column scales ``1 / w_j = |g_j^ridge|^psi`` (zero where the ridge coefficient is 0)."""
    a = np.abs(g_ridge)
    return np.where(a > 0, a**psi, 0.0)


def _cv_errors(
    Zs: FloatArray,
    r: FloatArray,
    folds: IntArray,
    psis: FloatArray,
    alphas: list[FloatArray],
    g_full: FloatArray,
    lam_ridge: float,
    fold_weights: bool,
    n_jobs: int | None,
    units: FloatArray | None = None,
) -> list[FloatArray]:
    """Fold-by-fold mean squared prediction errors: one (K, L) array per psi."""
    K = int(np.max(folds)) + 1

    def one_fold(k: int) -> list[FloatArray]:
        te = folds == k
        tr = ~te
        g = ridge_fit(Zs[tr], r[tr], lam_ridge) if fold_weights else g_full
        if units is not None:
            g = g * units
        out = []
        for i, psi in enumerate(psis):
            Ztil = Zs * adaptive_scale(g, float(psi))
            coefs, zm, rm = _path(Ztil[tr], r[tr], alphas[i])
            pred = (Ztil[te] - zm) @ coefs + rm
            out.append(((r[te][:, None] - pred) ** 2).mean(axis=0))
        return out

    if n_jobs is not None and n_jobs != 1:
        per_fold = Parallel(n_jobs=n_jobs)(delayed(one_fold)(k) for k in range(K))
    else:
        per_fold = [one_fold(k) for k in range(K)]
    return [np.vstack([pf[i] for pf in per_fold]) for i in range(psis.size)]


def lasso_coefs(Zc: FloatArray, rc: FloatArray, alphas: FloatArray) -> FloatArray:
    """Lasso coefficients of centred data at the penalties ``alphas`` (shape (d, L)).

    Objective ``(2n)^{-1} ||rc - Zc g||^2 + alpha ||g||_1``. The exact piecewise-linear
    solution path is computed by LARS with the Lasso modification (Efron et al., 2004)
    and interpolated at ``alphas``; coordinate descent converges very slowly on the
    highly correlated candidate dictionaries typical here. Falls back to coordinate
    descent if LARS reports a degenerate active set.
    """
    n, d = Zc.shape
    G = Zc.T @ Zc
    Xy = Zc.T @ rc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SkConvergenceWarning)
            knots, _, path = lars_path_gram(
                Xy, G, n_samples=n, method="lasso", max_iter=max(500, 8 * d)
            )
    except (SkConvergenceWarning, np.linalg.LinAlgError, ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SkConvergenceWarning)
            _, coefs, _ = lasso_path(
                Zc, rc, alphas=alphas, precompute=G, Xy=Xy, max_iter=100_000, tol=1e-10
            )
        return np.asarray(coefs, dtype=np.float64)
    kn = np.asarray(knots[::-1], dtype=np.float64)  # increasing
    P = np.asarray(path[:, ::-1], dtype=np.float64)
    out = np.empty((d, alphas.size))
    for j in range(d):
        out[j] = np.interp(alphas, kn, P[j], left=P[j, 0], right=0.0)
    return out


def _path(
    Ztil: FloatArray, r: FloatArray, alphas: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Lasso path on centred data; returns ``(coefs (d, L), column means, r mean)``."""
    zm = Ztil.mean(axis=0)
    rm = float(r.mean())
    return lasso_coefs(Ztil - zm, r - rm, alphas), zm, rm


def adaptive_lasso(
    r: FloatArray,
    Z: FloatArray,
    names: Sequence[str] | None = None,
    *,
    psi_grid: Sequence[float] = PSI_GRID,
    cv_folds: int = 10,
    lambda_rule: LambdaRule = "min",
    fold_scheme: FoldScheme = "random",
    n_lambda: int = 100,
    lambda_min_ratio: float | None = None,
    random_state: RandomState = None,
    n_jobs: int | None = None,
    cv_weights: CVWeights = "fold",
    ridge_scale: RidgeScale = "standardized",
    foldid: npt.ArrayLike | None = None,
    lambda_grid: Sequence[float] | None = None,
) -> SkedasticFit:
    """Adaptive Lasso of ``r`` on ``Z`` with ``(lambda, psi)`` tuned jointly by CV.

    Parameters
    ----------
    r : ndarray (T,)
        Response, typically ``log(max(u_t^2, delta^2))``.
    Z : ndarray (T, d)
        Candidate covariates (without a constant).
    names : sequence of str, optional
    psi_grid : sequence of float
        Exponents of the adaptive weights; ``0`` is the plain Lasso.
    cv_folds : int
        Number of folds ``K`` (default 10).
    lambda_rule : {"min", "1se"}
        Penalty at the CV minimum (default, as in [GP24]) or the largest penalty within
        one standard error of it.
    fold_scheme : {"random", "blocked"}
        Random folds (default) or contiguous blocks (for dependent data).
    n_lambda : int
        Length of the penalty grid.
    lambda_min_ratio : float, optional
        Smallest penalty relative to the largest (default ``1e-4`` if ``T > d``, else
        ``1e-2``, as in glmnet).
    random_state : int or Generator, optional
        Seeds the fold assignment.
    n_jobs : int, optional
        Parallel jobs over the CV folds (joblib).
    cv_weights : {"fold", "full"}
        ``"fold"`` (default) re-estimates the ridge first step (and hence the adaptive
        weights) on each training fold, so the CV error is not flattered by weights
        that have seen the test fold; ``"full"`` computes the weights once on the
        full sample (glmnet ``penalty.factor`` practice).
    ridge_scale : {"standardized", "original"}
        Scale on which the ridge coefficients define the adaptive weights:
        standardised covariates (default, unit-free) or the original units of ``z``
        (what one gets from glmnet's reported ridge coefficients).
    foldid : array_like of int, optional
        Explicit fold label of every observation; overrides ``cv_folds``,
        ``fold_scheme`` and ``random_state`` (e.g. to reproduce the folds of another
        implementation).
    lambda_grid : sequence of float, optional
        Explicit Lasso penalty grid (objective ``(2n)^{-1} RSS + lambda ||g||_1`` on
        the rescaled covariates), used for every ``psi`` instead of the default grid
        ``lambda_max(psi) * logspace(0, log10(lambda_min_ratio), n_lambda)``.

    Returns
    -------
    SkedasticFit
        ``df`` is the number of non-zero coefficients **including the intercept**
        (Zou, Hastie and Tibshirani, 2007).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(1, 4, 400)
    >>> u = x * rng.standard_normal(400)
    >>> Z = np.column_stack([x, x**2, np.cos(x)])
    >>> fit = adaptive_lasso(np.log(np.maximum(u**2, 0.01)), Z, ["x", "x2", "cosx"], random_state=0)
    >>> "x" in fit.selected or "x2" in fit.selected
    True
    """
    T = r.shape[0]
    d = Z.shape[1]
    names = tuple(names) if names is not None else tuple(f"z{j + 1}" for j in range(d))
    if len(names) != d:
        raise InputError("names must have one entry per column of Z.")
    psis = np.asarray(sorted({float(p) for p in psi_grid}), dtype=np.float64)
    if psis.size == 0 or (psis < 0).any():
        raise InputError("psi_grid must contain non-negative values.")
    if lambda_rule not in ("min", "1se"):
        raise InputError(f"lambda_rule must be 'min' or '1se', got {lambda_rule!r}.")
    rbar = float(r.mean())

    def constant_fit(reason: str, extra: dict[str, Any] | None = None) -> SkedasticFit:
        tun: dict[str, Any] = {"reason": reason}
        tun.update(extra or {})
        return SkedasticFit(
            fitted=np.full(T, rbar),
            intercept=rbar,
            coef=np.zeros(d),
            names=names,
            df=1.0,
            method="lasso",
            tuning=tun,
        )

    if d == 0:
        return constant_fit("no candidate covariates")
    if foldid is None:
        K = min(cv_folds, T)
        folds = make_folds(T, K, fold_scheme, random_state)
    else:
        folds = check_foldid(foldid, T)
        K = int(np.max(folds)) + 1
    if T < 2 * K:
        warnings.warn(
            f"Only {T} observations for {K}-fold cross-validation; the skedastic fit is noisy.",
            SampleSizeWarning,
            stacklevel=2,
        )
    Zs, mu, sd, keep = standardize(Z)
    if Zs.shape[1] == 0:
        return constant_fit("all candidate covariates are constant")

    g_ridge, lam_ridge = ridge_cv(Zs, r, folds)
    units = 1.0 / sd[keep] if ridge_scale == "original" else None
    g_w = g_ridge if units is None else g_ridge * units
    ratio = lambda_min_ratio if lambda_min_ratio is not None else (1e-4 if T > d else 1e-2)
    rc_full = r - rbar
    grids: list[FloatArray] = []
    if lambda_grid is not None:
        user_grid = np.sort(np.asarray(lambda_grid, dtype=np.float64))[::-1]
        if user_grid.size == 0 or not np.all(user_grid > 0):
            raise InputError("lambda_grid must contain positive penalties.")
    for psi in psis:
        if lambda_grid is not None:
            grids.append(user_grid)
            continue
        Ztil = Zs * adaptive_scale(g_w, float(psi))
        lam_max = float(np.max(np.abs((Ztil - Ztil.mean(axis=0)).T @ rc_full))) / T
        lam_max = lam_max if lam_max > 0 else 1.0
        grids.append(lam_max * np.logspace(0.0, np.log10(ratio), n_lambda))
    curves_all = _cv_errors(
        Zs, r, folds, psis, grids, g_ridge, lam_ridge, cv_weights == "fold", n_jobs, units
    )
    means = [c.mean(axis=0) for c in curves_all]
    best = [(float(np.min(m)), i, int(np.argmin(m))) for i, m in enumerate(means)]
    cv_min, i_psi, i_lam = min(best)
    alphas, curves = grids[i_psi], curves_all[i_psi]
    scale = adaptive_scale(g_w, float(psis[i_psi]))
    if lambda_rule == "1se":
        se = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        thresh = means[i_psi][i_lam] + se[i_lam]
        ok = np.flatnonzero(means[i_psi] <= thresh)
        i_lam = int(np.min(ok))  # alphas are decreasing: smallest index = largest penalty
    psi_star = float(psis[i_psi])
    lam_star = float(alphas[i_lam])

    coefs, zm, rm = _path(Zs * scale, r, alphas[: i_lam + 1])
    gtil = coefs[:, -1]
    if not np.all(np.isfinite(gtil)):
        warnings.warn("The Lasso path did not converge.", ConvergenceWarning, stacklevel=2)
        gtil = np.nan_to_num(gtil)
    g_std = gtil * scale  # coefficients on standardised z
    coef = np.zeros(d)
    coef[keep] = g_std / sd[keep]
    intercept = float(rm - zm @ gtil)
    intercept_orig = intercept - float(coef[keep] @ mu[keep])
    fitted = intercept_orig + Z @ coef
    tuning: dict[str, Any] = {
        "lambda": lam_star,
        "psi": psi_star,
        "cv_mse": cv_min,
        "lambda_ridge": lam_ridge,
        "lambda_rule": lambda_rule,
        "cv_folds": K,
        "fold_scheme": fold_scheme,
        "cv_weights": cv_weights,
        "foldid": folds,
        "lambda_ridge_coef": g_ridge,
        "cv_path": {float(p): (a, m) for p, a, m in zip(psis, grids, means, strict=True)},
    }
    return SkedasticFit(
        fitted=np.asarray(fitted, dtype=np.float64),
        intercept=intercept_orig,
        coef=coef,
        names=names,
        df=float(np.count_nonzero(coef) + 1),
        method="lasso",
        tuning=tuning,
    )
