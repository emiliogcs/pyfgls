"""Support-vector-regression estimate of the skedastic function ([GP24] Section 3).

epsilon-insensitive SVR with a Gaussian (RBF) kernel of ``r_t = log(max(u_t^2, delta^2))``
on the standardised candidate covariates, with ``(epsilon, C)`` chosen by K-fold
cross-validation. [GP24] find it dominated by the adaptive Lasso; it is offered as a
nonparametric alternative.

Degrees of freedom for the bias correction: ``df = |eps+| + |eps-|``, the number of
observations whose residual lies on or outside the epsilon-tube (approximately the
number of support vectors).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVR

from pyfgls._linalg import FloatArray
from pyfgls._utils import RandomState
from pyfgls.exceptions import SampleSizeWarning
from pyfgls.hetero.adaptive_lasso import (
    FoldScheme,
    SkedasticFit,
    check_foldid,
    make_folds,
    standardize,
)

#: Default grids, relative to the standard deviation of the response.
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
EPS_GRID: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0)


def svr_skedastic(
    r: FloatArray,
    Z: FloatArray,
    names: Sequence[str] | None = None,
    *,
    C_grid: Sequence[float] = C_GRID,
    eps_grid: Sequence[float] = EPS_GRID,
    cv_folds: int = 10,
    fold_scheme: FoldScheme = "random",
    random_state: RandomState = None,
    n_jobs: int | None = None,
    foldid: npt.ArrayLike | None = None,
) -> SkedasticFit:
    """Fit ``r`` on ``Z`` by RBF-kernel SVR with ``(epsilon, C)`` tuned by K-fold CV.

    ``C`` and ``epsilon`` grids are multiplied by the standard deviation of ``r``.
    """
    T, d = Z.shape
    names = tuple(names) if names is not None else tuple(f"z{j + 1}" for j in range(d))
    if T > 5000:
        warnings.warn(
            f"SVR scales quadratically or worse in T (= {T}); this may be slow. "
            "Consider skedastic='lasso'.",
            SampleSizeWarning,
            stacklevel=2,
        )
    Zs, _, _, _ = standardize(Z)
    sr = float(r.std()) or 1.0
    if Zs.shape[1] == 0:
        rbar = float(r.mean())
        return SkedasticFit(
            fitted=np.full(T, rbar),
            intercept=rbar,
            coef=np.zeros(d),
            names=names,
            df=1.0,
            method="svr",
            tuning={"reason": "no non-constant covariates"},
        )
    folds = (
        make_folds(T, min(cv_folds, T), fold_scheme, random_state)
        if foldid is None
        else check_foldid(foldid, T)
    )
    grid = {
        "C": [c * sr for c in C_grid],
        "epsilon": [e * sr for e in eps_grid],
    }
    search = GridSearchCV(
        SVR(kernel="rbf", gamma="scale"),
        grid,
        cv=PredefinedSplit(folds),
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(Zs, r)
    model = search.best_estimator_
    fitted = np.asarray(model.predict(Zs), dtype=np.float64)
    eps = float(model.epsilon)
    df = float(np.sum(np.abs(r - fitted) >= eps - 1e-8))
    return SkedasticFit(
        fitted=fitted,
        intercept=float(np.asarray(model.intercept_).ravel()[0]),
        coef=np.zeros(d),
        names=names,
        df=df,
        method="svr",
        tuning={
            "C": float(model.C),
            "epsilon": eps,
            "cv_mse": float(-search.best_score_),
            "n_support": len(model.support_),
            "foldid": folds,
        },
    )
