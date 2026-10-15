"""AR coefficients from the Durbin regression when lagged dependent variables are regressors.

With ``p_y`` lags of ``y`` among the regressors, the coefficients on the lagged ``y`` in
the Durbin regression mix the AR coefficients of the errors with those of the model,
so they no longer estimate ``rho`` ([PG26] Section 4.2, eq. (30)). Under the common
factor structure, however, the coefficient on ``x_{i,t-j}`` is ``-rho_j beta_i`` for
every (non-lagged-dependent) regressor ``x_i``, so

    rho_j = delta_{j,i} / beta_i

(Wallis, 1967), with ``delta_{j,i}`` minus the coefficient on ``x_{i,t-j}``. The
Durbin design of :mod:`pyfgls.serial.durbin` automatically contains the lags of ``y``
up to order ``k + p_y`` (the lags of the declared lagged-dependent columns supply the
extra ones).

Choice of ``i`` (``wallis_regressor``):

* ``"auto"`` (default): the eligible regressor whose contemporaneous Durbin coefficient
  has the largest ``|t|`` (best identified ratio);
* ``"median"``: the median of the ratios across eligible regressors, lag by lag;
* a regressor name.

This path is less tested than the standard one; see the simulation test in
``tests/statistical``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls.exceptions import InputError, SpecificationWarning
from pyfgls.serial.durbin import DurbinFit

#: Below this |t| of beta_i the ratio delta/beta_i is weakly identified.
WEAK_T = 2.0


@dataclass(frozen=True)
class WallisResult:
    """AR coefficients recovered by the Wallis ratio.

    Attributes
    ----------
    rho : ndarray (k,)
    J : ndarray (k, p)
        Jacobian of ``rho`` with respect to the Durbin coefficients.
    used : tuple of str
        Regressor(s) whose ratios were used.
    ratios : dict
        ``name -> rho`` implied by each eligible regressor.
    """

    rho: FloatArray
    J: FloatArray
    used: tuple[str, ...]
    ratios: dict[str, FloatArray]


def eligible_regressors(fit: DurbinFit, k: int, exclude: Sequence[int]) -> list[int]:
    """Regressors whose lags ``1..k`` are all in the Durbin design (and not excluded)."""
    d = fit.design
    return [
        i
        for i in range(d.n_x)
        if i not in set(exclude) and all((j, i) in d.xlag_pos for j in range(1, k + 1))
    ]


def _single(fit: DurbinFit, k: int, i: int) -> tuple[FloatArray, FloatArray]:
    b = fit.coef
    d = fit.design
    bi = b[i]
    rho = np.array([-b[d.xlag_pos[(j, i)]] / bi for j in range(1, k + 1)])
    J = np.zeros((k, b.size))
    for j in range(1, k + 1):
        J[j - 1, d.xlag_pos[(j, i)]] = -1.0 / bi
        J[j - 1, i] = -rho[j - 1] / bi
    return rho, J


def wallis_rho(
    fit: DurbinFit,
    k: int,
    x_names: Sequence[str],
    lagged_dep_idx: Sequence[int],
    choice: str = "auto",
) -> WallisResult:
    """Recover ``rho_1..rho_k`` from the Durbin regression by the Wallis ratio.

    Parameters
    ----------
    fit : DurbinFit
        Durbin regression of order ``k``.
    k : int
    x_names : sequence of str
    lagged_dep_idx : sequence of int
        Positions of the lagged dependent variables among the regressors.
    choice : {"auto", "median"} or str
        Regressor selection rule, or a regressor name.
    """
    p = fit.coef.size
    if k == 0:
        return WallisResult(rho=np.zeros(0), J=np.zeros((0, p)), used=(), ratios={})
    elig = eligible_regressors(fit, k, lagged_dep_idx)
    if not elig:
        raise InputError(
            "Cannot recover the AR coefficients with lagged dependent variables: the "
            "Wallis ratio needs at least one stochastic regressor that is not a lag of y "
            "(deterministic terms such as the constant are not eligible)."
        )
    ratios = {x_names[i]: _single(fit, k, i)[0] for i in elig}
    tvals = np.abs(fit.tvalues[elig])

    if choice == "auto":
        pick = [elig[int(np.argmax(tvals))]]
    elif choice == "median":
        pick = elig
    else:
        if choice not in x_names:
            raise InputError(f"wallis_regressor {choice!r} is not a regressor name.")
        i = list(x_names).index(choice)
        if i not in elig:
            raise InputError(
                f"wallis_regressor {choice!r} is not eligible (it is a lagged dependent "
                "variable or a deterministic term whose lags are collinear)."
            )
        pick = [i]

    weak = [x_names[i] for i in pick if abs(fit.tvalues[i]) < WEAK_T]
    if weak:
        warnings.warn(
            f"The Wallis ratio uses regressor(s) {weak} whose Durbin coefficient has |t| < "
            f"{WEAK_T}; the recovered AR coefficients are weakly identified. Consider "
            "wallis_regressor='median' or another regressor.",
            SpecificationWarning,
            stacklevel=3,
        )

    if len(pick) == 1:
        rho, J = _single(fit, k, pick[0])
        return WallisResult(rho=rho, J=J, used=(x_names[pick[0]],), ratios=ratios)

    singles = [_single(fit, k, i) for i in pick]
    R = np.vstack([s[0] for s in singles])  # (m, k)
    rho = np.median(R, axis=0)
    J = np.zeros((k, p))
    for j in range(k):
        order = np.argsort(R[:, j], kind="stable")
        m = order.size
        mids = [order[m // 2]] if m % 2 else [order[m // 2 - 1], order[m // 2]]
        for s in mids:
            J[j] += singles[s][1][j] / len(mids)
    return WallisResult(rho=rho, J=J, used=tuple(x_names[i] for i in pick), ratios=ratios)
