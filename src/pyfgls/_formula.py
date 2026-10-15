"""Formula interface (optional ``formula`` extra: ``pip install pyfgls[formula]``)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pyfgls.exceptions import InputError


def design_from_formula(formula: str, data: Any) -> tuple[pd.Series, pd.DataFrame]:
    """Build ``(y, X)`` from a Wilkinson formula with :mod:`formulaic`.

    Missing values are *not* dropped here (``na_action="ignore"``) so that the
    estimator's ``missing`` policy applies — dropping interior rows silently would
    break the time order. ``formulaic``'s ``Intercept`` column is renamed ``const``.
    Lags can be written ``lag(x)`` / ``lag(x, 2)``; combine with ``missing="drop"``.
    """
    try:
        from formulaic import model_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The formula interface needs formulaic: pip install 'pyfgls[formula]'."
        ) from exc
    if "~" not in formula:
        raise InputError("The formula needs a left-hand side: 'y ~ x1 + x2'.")
    mm = model_matrix(formula, data, na_action="ignore")
    lhs = mm.lhs
    if lhs.shape[1] != 1:
        raise InputError("The formula must have exactly one dependent variable.")
    y = lhs.iloc[:, 0].astype(float)
    y.name = str(lhs.columns[0])
    X = pd.DataFrame(mm.rhs).astype(float).rename(columns={"Intercept": "const"})
    return y, X
