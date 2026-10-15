"""Replicate the Monte Carlo design of González-Coya & Perron (2024), Table 1.

``y = 1 + x + sqrt(v(x)) eps``, ``x ~ U[1, 4]``, candidate covariates
``z = (x, [log x]^2, x^2, cos x, cos 2x)``. For each skedastic function and sample size
the script reports, for the slope, the empirical MSE of Lasso-FGLS relative to OLS,
the coverage of the 95% interval from eqs. (6)+(8), and the average interval length
next to OLS with HC3 standard errors.

Usage::

    python benchmarks/replicate_gp24_tables.py --reps 2000 --T 100 200 400 --jobs 4
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SKEDASTIC = ["homoskedastic", "x", "x2", "log2", "exp", "step"]
Z95 = 1.959963984540054


def _chunk(T: int, skedastic: str, seed: int, n: int) -> np.ndarray:
    warnings.simplefilter("ignore")
    from pyfgls._linalg import hat_diag, lstsq
    from pyfgls.hetero import estimate_hetero
    from pyfgls.simulate import hetero_dgp

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = hetero_dgp(T, skedastic=skedastic, random_state=rng)
        x = s.X[:, 1]
        Z = np.column_stack([x, np.log(x) ** 2, x**2, np.cos(x), np.cos(2 * x)])
        est = estimate_hetero(s.y, s.X, Z, list("abcde"), random_state=int(rng.integers(2**31)))
        se = float(np.sqrt(est.cov[1, 1]))
        b = lstsq(s.X, s.y)
        u = s.y - s.X @ b
        h = hat_diag(s.X)
        A = np.linalg.inv(s.X.T @ s.X)
        V = A @ (s.X * (u**2 / (1 - h) ** 2)[:, None]).T @ s.X @ A
        out.append(
            [est.beta[1] - 1, b[1] - 1, abs(est.beta[1] - 1) <= Z95 * se, 2 * Z95 * se,
             2 * Z95 * np.sqrt(V[1, 1])]
        )  # fmt: skip
    return np.asarray(out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--T", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--skedastic", nargs="+", default=SKEDASTIC)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", type=str, default=None, help="optional CSV output path")
    args = ap.parse_args()
    rows = []
    chunks = 20
    for T in args.T:
        for sk in args.skedastic:
            t0 = time.time()
            parts = Parallel(n_jobs=args.jobs)(
                delayed(_chunk)(T, sk, args.seed + 1000 * T + i, args.reps // chunks)
                for i in range(chunks)
            )
            a = np.vstack(parts)
            rows.append(
                {
                    "T": T,
                    "v(x)": sk,
                    "eMSE FGLS/OLS": np.mean(a[:, 0] ** 2) / np.mean(a[:, 1] ** 2),
                    "coverage": a[:, 2].mean(),
                    "length FGLS": a[:, 3].mean(),
                    "length OLS-HC3": a[:, 4].mean(),
                }
            )
            print(f"T={T} {sk}: done in {time.time() - t0:.0f}s", flush=True)
    table = pd.DataFrame(rows).set_index(["T", "v(x)"])
    print(table.round(3).to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
