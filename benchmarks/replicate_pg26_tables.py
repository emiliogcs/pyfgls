"""Replicate the Monte Carlo designs of Perron & González-Coya (2026).

Table 1 (``--table 1``): ``y = 1 + w + u``, ``u`` AR(1) 0.8 with ``u_0 = 0``,
``w_t = v_t + e_{t-1}`` (not exogenous), ``T = 500``. Reports RMSE and bias of the
slope for OLS, Durbin, infeasible GLS, FGLS and Cochrane--Orcutt, and of the AR
persistence ``sum_j rho_j``.

General case (``--table general``): ``x_t = 0.8 x_{t-1} + v_t + gamma e_{t-1}``, AR/MA
errors, ``T = 200``, ``k_max = 12``. Reports MSE x 100 of OLS, FGLS and infeasible GLS,
coverage of the robust (eq. 29), naive (eq. 27) and OLS-HAC 95% intervals, and the
median selected order.

Usage::

    python benchmarks/replicate_pg26_tables.py --table 1 --reps 10000
    python benchmarks/replicate_pg26_tables.py --table general --reps 2000 --jobs 4
"""

from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

Z95 = 1.959963984540054

GENERAL_CELLS = {
    "AR(1) -0.5": {"ar": (-0.5,)},
    "AR(1) 0": {"ar": (0.0,)},
    "AR(1) 0.2": {"ar": (0.2,)},
    "AR(1) 0.5": {"ar": (0.5,)},
    "AR(1) 0.8": {"ar": (0.8,)},
    "AR(2) 1.34,-0.42": {"ar": (1.34, -0.42)},
    "AR(2) 0.5,-0.3": {"ar": (0.5, -0.3)},
    "MA(1) -0.7": {"ar": (), "ma": (-0.7,)},
    "MA(1) -0.4": {"ar": (), "ma": (-0.4,)},
    "MA(1) 0.5": {"ar": (), "ma": (0.5,)},
}


def _table1_chunk(seed: int, n: int) -> np.ndarray:
    warnings.simplefilter("ignore")
    from pyfgls._linalg import lstsq
    from pyfgls._utils import ar_filter
    from pyfgls.benchmarks import cochrane_orcutt
    from pyfgls.serial import estimate_serial
    from pyfgls.simulate import serial_dgp

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = serial_dgp(500, random_state=rng)
        est = estimate_serial(s.y, s.X, ["c", "x"], kmax=12)
        co = cochrane_orcutt(s.y, s.X)
        out.append(
            [
                lstsq(s.X, s.y)[1] - 1,
                est.durbin.beta[1] - 1,
                lstsq(ar_filter(s.X, [0.8]), ar_filter(s.y, [0.8]))[1] - 1,
                est.beta[1] - 1,
                co.params.iloc[1] - 1,
                est.rho.sum() - 0.8,
                co.rho.iloc[0] - 0.8,
            ]
        )
    return np.asarray(out)


def _general_chunk(cell: dict, gamma: float, seed: int, n: int) -> np.ndarray:
    warnings.simplefilter("ignore")
    from pyfgls._linalg import lstsq
    from pyfgls._utils import ar_filter
    from pyfgls.benchmarks import ols
    from pyfgls.serial import estimate_serial
    from pyfgls.simulate import serial_dgp

    rng = np.random.default_rng(seed)
    ar = cell.get("ar", ())
    out = []
    for _ in range(n):
        s = serial_dgp(200, rho_x=0.8, gamma=gamma, burn=100, random_state=rng, **cell)
        est = estimate_serial(s.y, s.X, ["c", "x"], kmax=12)
        o = ols(s.y, s.X, cov_type="HAC")
        b = est.beta[1]
        se_r = np.sqrt(est.cov[1, 1])
        se_n = np.sqrt(est.variance.naive_avar[1, 1] / est.variance.n)
        gls = lstsq(ar_filter(s.X, ar), ar_filter(s.y, ar))[1] - 1 if ar else np.nan
        out.append(
            [
                o.params.iloc[1] - 1,
                b - 1,
                gls,
                abs(b - 1) <= Z95 * se_r,
                abs(b - 1) <= Z95 * se_n,
                abs(o.params.iloc[1] - 1) <= Z95 * o.bse.iloc[1],
                est.order,
            ]
        )
    return np.asarray(out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", choices=["1", "general"], default="1")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    chunks = 20
    if args.table == "1":
        parts = Parallel(n_jobs=args.jobs)(
            delayed(_table1_chunk)(args.seed + i, args.reps // chunks) for i in range(chunks)
        )
        a = np.vstack(parts)
        names = [
            "OLS",
            "Durbin",
            "GLS (infeasible)",
            "FGLS",
            "Cochrane-Orcutt",
            "rho FGLS",
            "rho CO",
        ]
        table = pd.DataFrame(
            {"RMSE": np.sqrt(np.mean(a**2, axis=0)), "bias": a.mean(axis=0)}, index=names
        )
    else:
        rows = []
        for label, cell in GENERAL_CELLS.items():
            for g in args.gamma:
                parts = Parallel(n_jobs=args.jobs)(
                    delayed(_general_chunk)(cell, g, args.seed + i, args.reps // chunks)
                    for i in range(chunks)
                )
                a = np.vstack(parts)
                rows.append(
                    {
                        "errors": label,
                        "gamma": g,
                        "MSEx100 OLS": 100 * np.mean(a[:, 0] ** 2),
                        "MSEx100 FGLS": 100 * np.mean(a[:, 1] ** 2),
                        "MSEx100 GLS": 100 * np.nanmean(a[:, 2] ** 2),
                        "cover robust": a[:, 3].mean(),
                        "cover naive": a[:, 4].mean(),
                        "cover OLS-HAC": a[:, 5].mean(),
                        "med k": np.median(a[:, 6]),
                    }
                )
                print(f"{label} gamma={g}: done", flush=True)
        table = pd.DataFrame(rows).set_index(["errors", "gamma"])
    print(table.round(4).to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
