"""Timing of the main estimators on a large problem (T = 10,000, q = 10, d = 100).

Usage::

    python benchmarks/performance.py [--T 10000] [--q 10] [--d 100] [--jobs 1]

Prints wall-clock seconds for the lag selection + serial FGLS, the adaptive-Lasso
FGLS with ``d`` candidate covariates, and the combined estimator.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import pyfgls as pf
from pyfgls.simulate import arma_errors


def make_data(T: int, q: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 4.0, size=(T, q - 1))
    X = np.column_stack([np.ones(T), x])
    Z = np.column_stack([x, rng.uniform(1.0, 4.0, size=(T, d - (q - 1)))])
    e = np.sqrt(x[:, 0] ** 2) * rng.standard_normal(T)
    u = arma_errors(e, (0.5, 0.2))
    y = X @ np.ones(q) + u
    return y, X, Z


def timed(label: str, fn) -> None:
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    extra = f"k*={res.ar_order}" if res.serial is not None else ""
    if res.hetero is not None:
        extra += f" selected={len(res.selected_z)}"
    print(f"{label:<40s} {dt:7.2f} s   {extra}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--T", type=int, default=10_000)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    warnings.simplefilter("ignore", pf.exceptions.PyFGLSWarning)
    y, X, Z = make_data(args.T, args.q, args.d)
    print(f"T={args.T}, q={args.q}, d={args.d}, n_jobs={args.jobs}")
    timed(
        "SerialFGLS (BIC, ic_penalty=params)",
        lambda: pf.SerialFGLS(y, X, ic_penalty="params").fit(),
    )
    timed(
        f"LassoFGLS (d={args.d}, 10-fold CV, 6 psi)",
        lambda: pf.LassoFGLS(y, X, z=Z, random_state=0, n_jobs=args.jobs).fit(),
    )
    timed(
        "FGLS combined (F-CH)",
        lambda: pf.FGLS(y, X, z=Z, ic_penalty="params", random_state=0, n_jobs=args.jobs).fit(),
    )


if __name__ == "__main__":
    main()
