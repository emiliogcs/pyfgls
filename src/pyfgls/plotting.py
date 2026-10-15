"""Diagnostic plots (optional ``plot`` extra: ``pip install pyfgls[plot]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls.exceptions import InputError
from pyfgls.results import FGLSResults

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["acf", "plot_diagnostics", "plot_intervals"]


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting needs matplotlib: pip install 'pyfgls[plot]'.") from exc
    return plt


def acf(x: FloatArray, nlags: int = 20) -> FloatArray:
    """Sample autocorrelations ``r_1..r_nlags``."""
    v = np.asarray(x, dtype=np.float64) - np.mean(x)
    d = float(v @ v)
    n = v.size
    out = np.array([float(v[j:] @ v[: n - j]) / d for j in range(1, min(nlags, n - 1) + 1)])
    return out


def plot_intervals(
    results: Sequence[FGLSResults],
    *,
    params: Sequence[str] | None = None,
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Point estimates with confidence intervals for several fits, side by side."""
    plt = _plt()
    if not results:
        raise InputError("No results to plot.")
    names = list(params) if params is not None else list(results[0].params.index)
    labels = list(labels) if labels is not None else [r.method for r in results]
    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.9 * len(names)))
    else:
        fig = cast("Figure", ax.figure)
    m = len(results)
    offsets = np.linspace(-0.25, 0.25, m) if m > 1 else np.zeros(1)
    for i, (r, lab) in enumerate(zip(results, labels, strict=True)):
        ci = r.conf_int(alpha).loc[names]
        est = r.params.loc[names].to_numpy()
        ypos = np.arange(len(names)) + offsets[i]
        ax.errorbar(
            est,
            ypos,
            xerr=[est - ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy() - est],
            fmt="o",
            capsize=3,
            label=lab,
        )
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel(f"estimate and {1 - alpha:.0%} interval")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def plot_diagnostics(
    res: FGLSResults, *, nlags: int = 20, figsize: tuple[float, float] = (10.0, 7.0)
) -> Figure:
    """Four diagnostic panels for an FGLS fit.

    1. final-stage residuals (estimated innovations) in time order;
    2. their autocorrelations with ``+-1.96/sqrt(n)`` bands (should look white);
    3. the BIC path over AR orders (serial fits) or the fitted variances against the
       fitted values (heteroskedasticity fits);
    4. coefficients and intervals of this fit next to OLS (HAC or HC3).
    """
    plt = _plt()
    fig: Figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    e = res.resid_final.to_numpy()
    ax = axes[0, 0]
    ax.plot(np.arange(e.size), e, lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_title("Final-stage residuals")
    ax = axes[0, 1]
    r = acf(e, nlags)
    ax.bar(np.arange(1, r.size + 1), r, width=0.6)
    band = 1.96 / np.sqrt(e.size)
    ax.axhline(band, color="grey", ls="--", lw=0.8)
    ax.axhline(-band, color="grey", ls="--", lw=0.8)
    ax.set_title("Residual autocorrelations")
    ax.set_xlabel("lag")
    ax = axes[1, 0]
    if res.serial is not None and res.serial.ic_path is not None:
        path = res.serial.ic_path
        ax.plot(path.index, path.to_numpy(), marker="o", ms=3)
        ax.axvline(res.serial.ar_order, color="C3", ls="--", lw=0.8)
        ax.set_title(f"{str(path.name).upper()} by AR order (k* = {res.serial.ar_order})")
        ax.set_xlabel("k")
    elif res.hetero is not None:
        fitted = res.fittedvalues.to_numpy()[res.nobs - res.hetero.variance.size :]
        ax.scatter(fitted, np.log(res.hetero.variance), s=6)
        ax.set_title("Fitted log-variance")
        ax.set_xlabel("fitted value")
    else:
        ax.axis("off")
    from pyfgls.benchmarks import ols

    base = ols(res.data, cov_type="HAC" if res.serial is not None else "HC3")
    plot_intervals([base, res], labels=["OLS", res.method], ax=axes[1, 1])
    axes[1, 1].set_title("Coefficients")
    fig.tight_layout()
    return fig
