"""Bootstrap covariance matrices — an independent cross-check of the analytic variances.

* Heteroskedasticity only (cross section): **wild bootstrap**
  ``y*_t = x_t' b + u_t eta_t`` with Rademacher or Mammen multipliers.
* Serial correlation (with or without heteroskedasticity): **residual-based sieve
  bootstrap** using the estimated AR(k*) filter,
  ``u*_t = sum_j rho_j u*_{t-j} + e*_t``, ``y*_t = x_t' b + u*_t``, with innovations
  ``e*_t = e_t eta_t`` (wild, default; keeps any heteroskedasticity) or resampled
  with replacement (``"iid"``). The first ``k*`` errors are set to the observed
  residuals.

The whole estimator (lag selection, skedastic fit with cross-validation) is re-run on
every bootstrap sample, so the bootstrap reflects all estimation steps. The regressors
are held fixed: with non-exogenous regressors the bootstrap therefore does not
reproduce the feedback from past errors to the regressors and is best read as a check
of the heteroskedasticity / small-sample part of the analytic variance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from pyfgls._linalg import FloatArray
from pyfgls._utils import RandomState
from pyfgls.exceptions import InputError

WildDist = Literal["rademacher", "mammen"]
Scheme = Literal["wild", "iid"]

_SQ5 = float(np.sqrt(5.0))


def wild_multipliers(n: int, dist: WildDist, rng: np.random.Generator) -> FloatArray:
    """Mean-zero, unit-variance multipliers for the wild bootstrap."""
    if dist == "rademacher":
        return np.asarray(rng.choice([-1.0, 1.0], size=n), dtype=np.float64)
    if dist == "mammen":
        p = (_SQ5 + 1.0) / (2.0 * _SQ5)
        lo, hi = (1.0 - _SQ5) / 2.0, (1.0 + _SQ5) / 2.0
        return np.where(rng.random(n) < p, lo, hi).astype(np.float64)
    raise InputError(f"wild_dist must be 'rademacher' or 'mammen', got {dist!r}.")


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap draws of the coefficients and their covariance."""

    cov: FloatArray
    draws: FloatArray
    n_boot: int
    scheme: str


def wild_sample(
    fitted: FloatArray, resid: FloatArray, dist: WildDist, rng: np.random.Generator
) -> FloatArray:
    """One wild-bootstrap sample ``fitted + resid * eta``."""
    return np.asarray(fitted + resid * wild_multipliers(resid.size, dist, rng), dtype=np.float64)


def sieve_sample(
    fitted: FloatArray,
    uhat: FloatArray,
    innovations: FloatArray,
    rho: FloatArray,
    scheme: Scheme,
    dist: WildDist,
    rng: np.random.Generator,
) -> FloatArray:
    """One sieve-bootstrap sample with AR(k) errors built from ``innovations``.

    ``innovations`` are the residuals of the filtered regression (length ``T - k``).
    """
    T = fitted.size
    k = rho.size
    e = innovations - innovations.mean()
    if scheme == "wild":
        e_star = e * wild_multipliers(e.size, dist, rng)
    elif scheme == "iid":
        e_star = rng.choice(e, size=e.size, replace=True)
    else:
        raise InputError(f"bootstrap scheme must be 'wild' or 'iid', got {scheme!r}.")
    u = np.empty(T)
    u[:k] = uhat[:k]
    for t in range(k, T):
        u[t] = e_star[t - k] + (rho @ u[t - k : t][::-1] if k else 0.0)
    return np.asarray(fitted + u, dtype=np.float64)


def bootstrap_covariance(
    refit: Callable[[FloatArray], FloatArray],
    draw: Callable[[np.random.Generator], FloatArray],
    *,
    n_boot: int = 999,
    random_state: RandomState = None,
    n_jobs: int | None = None,
    scheme: str = "wild",
) -> BootstrapResult:
    """Bootstrap covariance of ``refit(y*)`` over ``n_boot`` samples ``y* = draw(rng)``.

    Each replication gets its own child seed (``SeedSequence.spawn``), so results do
    not depend on ``n_jobs``.
    """
    if n_boot < 2:
        raise InputError("n_boot must be at least 2.")
    if isinstance(random_state, np.random.Generator):
        seed_seq = np.random.SeedSequence(int(random_state.integers(2**63)))
    elif isinstance(random_state, np.random.SeedSequence):
        seed_seq = random_state
    else:
        seed_seq = np.random.SeedSequence(random_state)
    children = seed_seq.spawn(n_boot)

    def one(ss: np.random.SeedSequence) -> FloatArray:
        return refit(draw(np.random.default_rng(ss)))

    if n_jobs is not None and n_jobs != 1:
        draws = Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in children)
    else:
        draws = [one(c) for c in children]
    D = np.vstack(draws)
    return BootstrapResult(
        cov=np.atleast_2d(np.cov(D, rowvar=False)), draws=D, n_boot=n_boot, scheme=scheme
    )
