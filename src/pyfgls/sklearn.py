"""scikit-learn compatible wrapper: :class:`FGLSRegressor`.

Use it inside pipelines, grid searches or ``cross_val_score``. Remember that
cross-validation with shuffled folds breaks the time order that the serial-correlation
correction relies on; use ``TimeSeriesSplit`` (or ``serial=None``) in that case.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pyfgls._linalg import FloatArray
from pyfgls.models import FGLS
from pyfgls.results import FGLSResults

__all__ = ["FGLSRegressor"]


class FGLSRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """scikit-learn estimator wrapping :class:`pyfgls.FGLS`.

    Parameters
    ----------
    serial : {"auto", "fixed", "ma1", None}
        Serial-correlation correction (see :class:`pyfgls.FGLS`).
    hetero : {"lasso", "svr", "wls_s1", "wls_s2", None}
        Heteroskedasticity correction.
    fit_intercept : bool
        Add a constant column.
    ar_order, kmax, ic, z, cv_folds, random_state, n_jobs
        Passed to :class:`pyfgls.FGLS`.
    cov_type : str
        Covariance used for ``results_``.

    Attributes
    ----------
    coef_ : ndarray
    intercept_ : float
    results_ : FGLSResults
    n_features_in_ : int

    Examples
    --------
    >>> from pyfgls.simulate import hetero_dgp
    >>> sim = hetero_dgp(200, random_state=0)
    >>> reg = FGLSRegressor(serial=None, random_state=0).fit(sim.X[:, 1:], sim.y)
    >>> reg.coef_.shape
    (1,)
    """

    def __init__(
        self,
        serial: str | None = "auto",
        hetero: str | None = "lasso",
        *,
        fit_intercept: bool = True,
        ar_order: int | None = None,
        kmax: Any = None,
        ic: str = "bic",
        z: Any = "auto",
        cv_folds: int = 10,
        random_state: Any = None,
        n_jobs: int | None = None,
        cov_type: str = "robust",
    ) -> None:
        self.serial = serial
        self.hetero = hetero
        self.fit_intercept = fit_intercept
        self.ar_order = ar_order
        self.kmax = kmax
        self.ic = ic
        self.z = z
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cov_type = cov_type

    def fit(self, X: Any, y: Any) -> FGLSRegressor:
        """Fit the FGLS model to ``(X, y)``."""
        Xa, ya = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        self.n_features_in_ = Xa.shape[1]
        Xd = np.column_stack([np.ones(Xa.shape[0]), Xa]) if self.fit_intercept else Xa
        model = FGLS(
            ya,
            Xd,
            serial=self.serial,  # type: ignore[arg-type]
            hetero=self.hetero,  # type: ignore[arg-type]
            ar_order=self.ar_order,
            kmax=self.kmax,
            ic=self.ic,  # type: ignore[arg-type]
            z=self.z,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.results_: FGLSResults = model.fit(cov_type=self.cov_type)
        params = self.results_.params.to_numpy()
        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = params.copy()
        return self

    def predict(self, X: Any) -> FloatArray:
        """Predict ``intercept_ + X coef_``."""
        check_is_fitted(self, "coef_")
        Xa = check_array(X, dtype=np.float64)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Xa.shape[1]} features, but FGLSRegressor was fitted with "
                f"{self.n_features_in_}."
            )
        out: FloatArray = Xa @ self.coef_ + self.intercept_
        return out
