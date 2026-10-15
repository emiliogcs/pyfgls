"""Small Monte Carlo helpers shared by the acceptance tests."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

Z95 = 1.959963984540054


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def mse(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def run(
    R: int,
    seed: int,
    one: Callable[[np.random.Generator], dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Run ``one(rng)`` R times with a single seeded generator and stack the outputs."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(R):
            for k, v in one(rng).items():
                out.setdefault(k, []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def covered(est: float, se: float, true: float) -> bool:
    return abs(est - true) <= Z95 * se
