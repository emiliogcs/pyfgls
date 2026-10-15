"""Warnings and exceptions raised by pyfgls.

Every warning derives from :class:`PyFGLSWarning` and every error from
:class:`PyFGLSError`, so users can filter or catch them as a group::

    import warnings
    from pyfgls.exceptions import PyFGLSWarning

    warnings.simplefilter("ignore", PyFGLSWarning)
"""

from __future__ import annotations

__all__ = [
    "CollinearityError",
    "CollinearityWarning",
    "ConvergenceWarning",
    "ExperimentalWarning",
    "InputError",
    "InsufficientDataError",
    "LaggedDependentWarning",
    "MissingDataError",
    "NonStationaryWarning",
    "PyFGLSError",
    "PyFGLSWarning",
    "SampleSizeWarning",
    "SpecificationWarning",
    "TimeIndexError",
    "WeightWarning",
]


# ----------------------------------------------------------------------------- warnings
class PyFGLSWarning(UserWarning):
    """Base class for all warnings emitted by pyfgls."""


class NonStationaryWarning(PyFGLSWarning):
    """The estimated autoregressive filter is (close to) non-invertible.

    Raised when a root of ``1 - rho_1 z - ... - rho_k z^k`` lies inside, on, or very
    close to the unit circle. FGLS assumes stationary, invertible errors
    ([PG26] Assumption 2); consider differencing or a unit-root test.
    """


class CollinearityWarning(PyFGLSWarning):
    """Columns were dropped because they are (nearly) collinear with others.

    pyfgls only drops columns automatically where it is safe, e.g. lagged constants or
    lagged trends in the lagged-regressor block of the Durbin regression.
    """


class ConvergenceWarning(PyFGLSWarning):
    """An iterative routine (Lasso, SVR, MLE) did not fully converge."""


class WeightWarning(PyFGLSWarning):
    """The estimated FGLS weights are extreme (very large max/min ratio)."""


class SampleSizeWarning(PyFGLSWarning):
    """The sample is small relative to the number of parameters, lags, or folds."""


class SpecificationWarning(PyFGLSWarning):
    """The model specification looks unusual (e.g. no intercept)."""


class LaggedDependentWarning(SpecificationWarning):
    """A regressor looks like a lag of the dependent variable.

    Declare such regressors with ``lagged_dep_vars=[...]`` so the Durbin regression is
    run with the appropriate order ([PG26] Section 4.2).
    """


class ExperimentalWarning(PyFGLSWarning):
    """The feature is experimental; results and API may change."""


# --------------------------------------------------------------------------- exceptions
class PyFGLSError(Exception):
    """Base class for all errors raised by pyfgls."""


class InputError(PyFGLSError, ValueError):
    """The inputs are invalid (wrong shape, non-numeric, inconsistent, ...)."""


class MissingDataError(InputError):
    """The inputs contain missing or non-finite values that cannot be handled."""


class TimeIndexError(InputError):
    """The time index is unsorted, duplicated, has gaps, or is a panel index."""


class CollinearityError(InputError):
    """The regressor matrix is rank deficient."""


class InsufficientDataError(InputError):
    """There are too few observations for the requested model."""
