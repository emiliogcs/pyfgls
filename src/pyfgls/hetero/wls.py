"""Romano and Wolf (2017) parametric skedastic functions ([GP24] eqs. (2)--(3)).

* ``"wls_s1"``: ``v(z) = exp(c + g' [log z_1, ..., log z_d])`` — requires ``z > 0``;
* ``"wls_s2"``: ``v(z) = exp(c + g' z)``;

both fitted by OLS of ``r_t = log(max(u_t^2, delta^2))`` on the transformed covariates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from pyfgls._linalg import FloatArray, independent_columns, lstsq
from pyfgls.exceptions import InputError
from pyfgls.hetero.adaptive_lasso import SkedasticFit

WLSKind = Literal["wls_s1", "wls_s2"]


def wls_skedastic(
    r: FloatArray, Z: FloatArray, names: Sequence[str] | None = None, *, kind: WLSKind = "wls_s1"
) -> SkedasticFit:
    """OLS fit of the log-variance on ``log z`` (S1) or ``z`` (S2)."""
    T, d = Z.shape
    names = tuple(names) if names is not None else tuple(f"z{j + 1}" for j in range(d))
    if kind == "wls_s1":
        if np.any(Z <= 0):
            bad = [names[j] for j in range(d) if np.any(Z[:, j] <= 0)]
            raise InputError(
                f"skedastic='wls_s1' takes logs of the covariates, but {bad} have "
                "non-positive values. Use 'wls_s2' or positive covariates."
            )
        F = np.log(Z)
    elif kind == "wls_s2":
        F = Z.copy()
    else:
        raise InputError(f"kind must be 'wls_s1' or 'wls_s2', got {kind!r}.")
    keep, _ = independent_columns(F, np.ones((T, 1)))
    D = np.column_stack([np.ones(T), F[:, keep]])
    b = lstsq(D, r)
    coef = np.zeros(d)
    coef[keep] = b[1:]
    return SkedasticFit(
        fitted=np.asarray(D @ b, dtype=np.float64),
        intercept=float(b[0]),
        coef=coef,
        names=names,
        df=float(D.shape[1]),
        method=kind,
        tuning={"transform": "log" if kind == "wls_s1" else "level"},
    )
