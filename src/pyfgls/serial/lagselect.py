"""Selection of the autoregressive order by an information criterion ([PG26] eq. (14)).

Every candidate order ``k = 0, ..., k_max`` is estimated on the **common sample**
``t = k_max + 1, ..., T`` so the criteria are comparable (the modification of Ng and
Perron, 2005):

    BIC(k) = ln(sigma2_k) + k * ln(T - k_max) / (T - k_max),
    AIC(k) = ln(sigma2_k) + 2 k / (T - k_max),

with ``sigma2_k`` the mean squared residual of the order-``k`` Durbin regression. The
penalty counts the AR order ``k``, as in the paper (``ic_penalty="order"``).

Each extra lag adds ``1 + m`` coefficients to the Durbin regression (``m`` stochastic
regressors). With one regressor, as in the [PG26] simulations, this hardly matters; with
many regressors the order-based penalty is too weak in finite samples and the
criterion tends to pick ``k_max``. ``ic_penalty="params"`` penalises every added
coefficient instead (``p_k ln(n)/n`` with ``p_k`` the number of lagged columns).

Because the order-``k`` design is a prefix of the order-``k_max`` design, a single QR
factorisation gives every residual sum of squares: with ``Z = QR`` and ``r = Q'y``,
``RSS_k = RSS_kmax + sum_{i >= p_k} r_i^2``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls._utils import adjusted_min_lags, default_kmax, theory_kmax
from pyfgls.exceptions import InputError, InsufficientDataError, SampleSizeWarning
from pyfgls.serial.durbin import durbin_design

Criterion = Literal["bic", "aic"]
ICPenalty = Literal["order", "params"]
KmaxRule = int | Literal["default", "theory"] | None

#: Upper bound on the number of Durbin regressors relative to the common sample.
MAX_REGRESSOR_SHARE = 0.5


@dataclass(frozen=True)
class LagSelection:
    """Outcome of the AR-order selection.

    Attributes
    ----------
    order : int
        Selected order ``k*`` (after any ``min_lags`` adjustment).
    selected_by_ic : int
        Order that minimises the criterion (before adjustment).
    kmax : int
        Largest order considered.
    criterion : str
        ``"bic"``, ``"aic"`` or ``"fixed"``.
    values : ndarray
        Criterion for ``k = 0, ..., kmax`` (empty for a fixed order).
    sigma2 : ndarray
        Mean squared residuals on the common sample.
    n_common : int
        Size of the common sample ``T - kmax``.
    """

    order: int
    selected_by_ic: int
    kmax: int
    criterion: str
    values: FloatArray
    sigma2: FloatArray
    n_common: int


def n_durbin_regressors(q: int, k: int, has_const: bool) -> int:
    """Conservative count of Durbin regressors for order ``k`` (lagged constant dropped)."""
    return q + k * (1 + q - int(has_const))


def resolve_kmax(
    kmax: KmaxRule,
    T: int,
    q: int,
    *,
    has_const: bool,
    extra_lags: int = 0,
) -> int:
    """Turn the ``kmax`` option into an order that the sample can support.

    The default is ``int(12 * (T/200)**0.25)``; ``"theory"`` gives
    ``floor((T / ln T)**(1/3))``. The order is then capped so that the Durbin regression
    has at most ``0.5 * (T - kmax)`` regressors; a :class:`SampleSizeWarning` is issued
    when the cap binds.
    """
    if kmax is None or kmax == "default":
        k = default_kmax(T)
    elif kmax == "theory":
        k = theory_kmax(T)
    elif isinstance(kmax, int | np.integer) and not isinstance(kmax, bool):
        k = int(kmax)
        if k < 0:
            raise InputError(f"kmax must be non-negative, got {kmax}.")
    else:
        raise InputError(f"kmax must be an int, 'default' or 'theory'; got {kmax!r}.")

    cap = k
    while cap > 0 and (
        n_durbin_regressors(q, cap + extra_lags, has_const)
        > MAX_REGRESSOR_SHARE * (T - cap - extra_lags)
    ):
        cap -= 1
    if n_durbin_regressors(q, extra_lags, has_const) > MAX_REGRESSOR_SHARE * (T - extra_lags):
        raise InsufficientDataError(
            f"T = {T} observations are too few for {q} regressors: the Durbin regression "
            f"needs fewer than {MAX_REGRESSOR_SHARE:.0%} of the sample as regressors."
        )
    if cap < k:
        warnings.warn(
            f"kmax was reduced from {k} to {cap} so that the Durbin regression has at most "
            f"{MAX_REGRESSOR_SHARE:.0%} of the {T} observations as regressors.",
            SampleSizeWarning,
            stacklevel=3,
        )
    return cap


def select_ar_order(
    y: FloatArray,
    X: FloatArray,
    x_names: Sequence[str],
    kmax: int,
    *,
    ic: Criterion = "bic",
    min_lags: int | None = None,
    y_name: str = "y",
    ic_penalty: ICPenalty = "order",
) -> LagSelection:
    """Choose the AR order minimising BIC/AIC over ``k = 0..kmax`` (common sample).

    Parameters
    ----------
    y, X : ndarray
        Data (``X`` validated, full column rank).
    x_names : sequence of str
    kmax : int
        Largest order (see :func:`resolve_kmax`).
    ic : {"bic", "aic"}
    min_lags : int, optional
        Lower bound on the returned order (e.g. ``ceil((ln T)^1.5)``,
        [PG26] Suppl. Lemma A.2), applied after the minimisation. It may exceed
        ``kmax``; the caller caps it by what the sample supports.
    ic_penalty : {"order", "params"}
        Count the AR order (paper) or all added Durbin coefficients in the penalty.

    Returns
    -------
    LagSelection
    """
    if ic not in ("bic", "aic"):
        raise InputError(f"ic must be 'bic' or 'aic', got {ic!r}.")
    design = durbin_design(y, X, x_names, kmax, start=kmax, y_name=y_name)
    n = design.nobs
    Q, _ = np.linalg.qr(design.Z)
    r = Q.T @ design.y
    resid_full = design.y - Q @ r
    rss_full = float(resid_full @ resid_full)
    # tail[p] = sum_{i >= p} r_i^2
    tail = np.concatenate([np.cumsum((r**2)[::-1])[::-1], [0.0]])
    rss = np.array([rss_full + tail[design.block_end[k]] for k in range(kmax + 1)])
    sigma2 = rss / n
    if ic_penalty == "order":
        npar = np.arange(kmax + 1, dtype=np.float64)
    elif ic_penalty == "params":
        npar = np.array(
            [design.block_end[k] - design.block_end[0] for k in range(kmax + 1)], dtype=np.float64
        )
    else:
        raise InputError(f"ic_penalty must be 'order' or 'params', got {ic_penalty!r}.")
    penalty = math.log(n) / n if ic == "bic" else 2.0 / n
    with np.errstate(divide="ignore"):
        values = np.log(sigma2) + npar * penalty
    k_ic = int(np.argmin(values))
    k = k_ic
    if min_lags is not None:
        k = max(k, min_lags)
    return LagSelection(
        order=k,
        selected_by_ic=k_ic,
        kmax=kmax,
        criterion=ic,
        values=values,
        sigma2=sigma2,
        n_common=n,
    )


def resolve_min_lags(min_lags: int | Literal["adjusted"] | None, T: int) -> int | None:
    """Turn the ``min_lags`` option into an integer bound (or ``None``)."""
    if min_lags is None:
        return None
    if min_lags == "adjusted":
        return adjusted_min_lags(T)
    if isinstance(min_lags, int | np.integer) and not isinstance(min_lags, bool) and min_lags >= 0:
        return int(min_lags)
    raise InputError(f"min_lags must be None, 'adjusted' or a non-negative int; got {min_lags!r}.")
