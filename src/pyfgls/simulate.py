"""Data-generating processes from the simulation studies of the two papers.

All functions return a :class:`Simulation` with ``y`` (T,), ``X`` (T, q) including a
constant column, and the latent errors, so the tests can compare estimators with the
infeasible GLS benchmark.

* :func:`serial_dgp` — [PG26] Section 5: ``y_t = b0 + b1 x_t + u_t`` with AR/MA errors
  and a regressor ``x_t = rho_x x_{t-1} + v_t + gamma e_{t-1}`` that is not exogenous
  when ``gamma != 0`` (Table 1 is ``rho_x = 0``, ``gamma = 1``, AR(1) ``0.8``,
  ``u_0 = 0``).
* :func:`hetero_dgp` — [GP24] eq. (9): ``y_t = a + b x_t + sqrt(v(x_t)) eps_t`` with
  ``x_t ~ U[1, 4]``.
* :func:`combined_dgp` — [PG26] Supplement S-4: serially correlated errors whose
  innovations are heteroskedastic, ``e_t = sqrt(v(w_t)) eps_t``.
* :func:`predictive_dgp` — [PG26] Supplement S-3: predictive regression with MA(1)
  errors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal

from pyfgls._linalg import FloatArray
from pyfgls._utils import RandomState, check_random_state

__all__ = [
    "SKEDASTIC_FUNCTIONS",
    "Simulation",
    "arma_errors",
    "combined_dgp",
    "hetero_dgp",
    "predictive_dgp",
    "serial_dgp",
]


@dataclass(frozen=True)
class Simulation:
    """Simulated regression data.

    Attributes
    ----------
    y : ndarray (T,)
    X : ndarray (T, q)
        Regressors, first column the constant.
    beta : ndarray (q,)
        True coefficients.
    u : ndarray (T,)
        Regression errors.
    e : ndarray (T,)
        Innovations of the error process.
    info : dict
        DGP parameters (AR/MA coefficients, variance function values, ...).
    """

    y: FloatArray
    X: FloatArray
    beta: FloatArray
    u: FloatArray
    e: FloatArray
    info: dict[str, Any] = field(default_factory=dict)


def arma_errors(
    e: npt.ArrayLike,
    ar: Sequence[float] = (),
    ma: Sequence[float] = (),
) -> FloatArray:
    """``u_t = sum_i ar_i u_{t-i} + e_t + sum_i ma_i e_{t-i}`` driven by innovations ``e``.

    Pre-sample values of ``u`` and ``e`` are zero (the ``u_0 = 0`` initialisation of
    [PG26] Table 1).
    """
    ev = np.asarray(e, dtype=np.float64)
    b = np.concatenate([[1.0], np.asarray(ma, dtype=np.float64)])
    a = np.concatenate([[1.0], -np.asarray(ar, dtype=np.float64)])
    return np.asarray(scipy.signal.lfilter(b, a, ev), dtype=np.float64)


def serial_dgp(
    T: int,
    *,
    ar: Sequence[float] = (0.8,),
    ma: Sequence[float] = (),
    rho_x: float = 0.0,
    gamma: float = 1.0,
    beta: Sequence[float] = (1.0, 1.0),
    burn: int = 0,
    random_state: RandomState = None,
) -> Simulation:
    """[PG26] Section 5 DGP with a possibly non-exogenous regressor.

    ``y_t = b0 + b1 x_t + u_t``, ``u_t`` ARMA with N(0,1) innovations ``e_t``,
    ``x_t = rho_x x_{t-1} + v_t + gamma e_{t-1}``, ``v_t ~ N(0,1)`` independent of
    ``e``. The regressor is pre-determined; it is exogenous iff ``gamma = 0``.

    With the defaults (``rho_x = 0``, ``gamma = 1``, AR(1) ``0.8``, ``burn = 0``) this is
    the [PG26] Table 1 design with ``u_0 = 0``. Use ``burn > 0`` to start from
    (approximately) the stationary distribution.

    Examples
    --------
    >>> sim = serial_dgp(500, random_state=0)
    >>> sim.X.shape
    (500, 2)
    """
    rng = check_random_state(random_state)
    n = T + burn
    e_all = rng.standard_normal(n + 1)  # e_0, ..., e_n
    v = rng.standard_normal(n)
    e = e_all[1:]
    u = arma_errors(e, ar, ma)
    x = scipy.signal.lfilter([1.0], [1.0, -rho_x], v + gamma * e_all[:-1])
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    X = np.column_stack([np.ones(n), x])[burn:]
    y = X @ b + u[burn:]
    return Simulation(
        y=y,
        X=X,
        beta=b,
        u=u[burn:],
        e=e[burn:],
        info={"ar": tuple(ar), "ma": tuple(ma), "rho_x": rho_x, "gamma": gamma},
    )


def _step(x: FloatArray) -> FloatArray:
    return np.where(x < 2.0, 1.0, np.where(x < 3.0, 4.0, 9.0))


#: Skedastic functions of [GP24] Section 5 and the [PG26] Supplement.
SKEDASTIC_FUNCTIONS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "homoskedastic": lambda x: np.ones_like(x),
    "x": lambda x: x,
    "x2": lambda x: x**2,
    "log2": lambda x: np.log(x) ** 2,
    "exp": lambda x: np.exp(0.2 * x + 0.2 * x**2),
    "step": _step,
}


def _skedastic(v: str | Callable[[FloatArray], FloatArray]) -> Callable[[FloatArray], FloatArray]:
    if callable(v):
        return v
    try:
        return SKEDASTIC_FUNCTIONS[v]
    except KeyError:
        raise ValueError(
            f"unknown skedastic function {v!r}; choose from {sorted(SKEDASTIC_FUNCTIONS)}"
        ) from None


def hetero_dgp(
    T: int,
    *,
    skedastic: str | Callable[[FloatArray], FloatArray] = "x2",
    beta: Sequence[float] = (1.0, 1.0),
    n_extra: int = 0,
    extra_corr: float = 0.0,
    random_state: RandomState = None,
) -> Simulation:
    """[GP24] eq. (9): ``y_t = a + b x_t + sqrt(v(x_t)) eps_t``, ``x_t ~ U[1, 4]``.

    Parameters
    ----------
    T : int
    skedastic : str or callable
        Key of :data:`SKEDASTIC_FUNCTIONS` (``"homoskedastic"``, ``"x"``, ``"x2"``,
        ``"log2"``, ``"exp"``, ``"step"``) or a function of ``x``.
    beta : (a, b)
    n_extra : int
        Number of extra candidate covariates ``w_j ~ U[1, 4]`` (not regressors) for the
        robustness designs of [GP24] Section 5 (returned in ``info["W"]``).
    extra_corr : float
        Correlation-inducing weight: ``w_j = c x + (1 - c) q_j`` with ``q_j ~ U[1, 4]``.
    random_state : int or Generator, optional
    """
    rng = check_random_state(random_state)
    x = rng.uniform(1.0, 4.0, T)
    eps = rng.standard_normal(T)
    v = _skedastic(skedastic)(x)
    u = np.sqrt(v) * eps
    b = np.asarray(beta, dtype=np.float64)
    X = np.column_stack([np.ones(T), x])
    info: dict[str, Any] = {"v": v, "skedastic": skedastic}
    if n_extra:
        Qx = rng.uniform(1.0, 4.0, (T, n_extra))
        info["W"] = extra_corr * x[:, None] + (1.0 - extra_corr) * Qx
    return Simulation(y=X @ b + u, X=X, beta=b, u=u, e=eps, info=info)


def combined_dgp(
    T: int,
    *,
    ar: Sequence[float] = (0.5,),
    ma: Sequence[float] = (),
    skedastic: str | Callable[[FloatArray], FloatArray] = "x2",
    beta: Sequence[float] = (1.0, 1.0),
    driver: str = "x",
    phi_prob: float = 0.75,
    gamma: float = 0.0,
    burn: int = 100,
    random_state: RandomState = None,
) -> Simulation:
    """[PG26] Supplement S-4: ARMA errors with heteroskedastic innovations.

    ``y_t = b0 + b1 x_t + u_t``, ``x_t ~ U[1, 4]`` iid, ``u_t`` ARMA driven by
    ``e_t = sqrt(v(w_t)) eps_t``. The variance driver is ``w_t = x_t`` (``driver="x"``,
    case (a)) or ``w_t = phi_t x_t + (1 - phi_t) q_t`` with ``phi_t ~ Bernoulli(phi_prob)``
    and ``q_t ~ U[1, 4]`` (``driver="mixture"``, case (b)). With ``gamma != 0`` the
    regressor becomes ``x_t + gamma e_{t-1}`` (pre-determined, not exogenous).
    The candidate covariates ``w_t`` are returned in ``info["w"]``.
    """
    rng = check_random_state(random_state)
    n = T + burn
    x = rng.uniform(1.0, 4.0, n)
    if driver == "x":
        w = x
    elif driver == "mixture":
        phi = rng.binomial(1, phi_prob, n).astype(np.float64)
        w = phi * x + (1.0 - phi) * rng.uniform(1.0, 4.0, n)
    else:
        raise ValueError("driver must be 'x' or 'mixture'")
    v = _skedastic(skedastic)(w)
    e = np.sqrt(v) * rng.standard_normal(n)
    u = arma_errors(e, ar, ma)
    xr = x.copy()
    if gamma:
        xr[1:] += gamma * e[:-1]
    b = np.asarray(beta, dtype=np.float64)
    X = np.column_stack([np.ones(n), xr])[burn:]
    return Simulation(
        y=X @ b + u[burn:],
        X=X,
        beta=b,
        u=u[burn:],
        e=e[burn:],
        info={"v": v[burn:], "w": w[burn:], "ar": tuple(ar), "ma": tuple(ma), "gamma": gamma},
    )


def predictive_dgp(
    T: int,
    *,
    theta: float = 0.5,
    rho_x: float = 0.0,
    gamma: float = 0.0,
    beta: Sequence[float] = (1.0, 1.0),
    burn: int = 100,
    random_state: RandomState = None,
) -> Simulation:
    """[PG26] Supplement S-3 predictive regression ``y_{t+2} = a + b x_t + u_{t+2}``.

    ``u_t = e_t + theta e_{t-1}``, ``x_t = rho_x x_{t-1} + v_t + gamma e_{t-1}``.
    Returned aligned: ``y[t]`` is ``y_{t+2}`` and ``X[t] = (1, x_t)``.
    """
    rng = check_random_state(random_state)
    n = T + burn + 2
    e = rng.standard_normal(n + 1)
    v = rng.standard_normal(n)
    x = np.asarray(scipy.signal.lfilter([1.0], [1.0, -rho_x], v + gamma * e[:-1]), dtype=np.float64)
    u = e[1:] + theta * e[:-1]  # u_t for t = 0..n-1
    b = np.asarray(beta, dtype=np.float64)
    X = np.column_stack([np.ones(n - 2), x[:-2]])
    y = X @ b + u[2:]
    return Simulation(
        y=y[burn:],
        X=X[burn:],
        beta=b,
        u=u[2:][burn:],
        e=e[1:][2:][burn:],
        info={"theta": theta, "rho_x": rho_x, "gamma": gamma},
    )
