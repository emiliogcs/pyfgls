"""Hooks for cross-implementation testing (e.g. against the R package ``fgls``).

:func:`export_fit` returns the intermediates of a fit as a dictionary of NumPy
arrays and scalars, with the names used by the cross-language golden fixtures:

* serial correlation ([PG26]): ``kstar``, ``bic_path``, ``rho_hat``, ``beta_durbin``,
  ``beta_fgls``, ``e_hat``, ``u_hat``, ``Ghat``, ``Sigma_psi``, ``Qhat``, ``sigma2_e``,
  ``AVar_naive``, ``AVar_R``, ``vcov``, ``nonexogeneity_share``, ``ar_roots``,
  ``dropped_lags``;
* heteroskedasticity ([GP24]): ``ridge_coef``, ``psi_grid``, ``cv_lambda``,
  ``cv_error_matrix``, ``psi_star``, ``lambda_star``, ``gamma_hat``, ``nonzero_set``,
  ``df_hat``, ``fitted_logvar``, ``weights``, ``beta_fgls``, ``vcov_eq6_8``, ``h_ols``,
  ``h_fgls``, ``foldid`` (0-based);
* combined (Supplement S-4): ``beta_FC``, ``beta_FCH``, ``vcov_FC``, ``vcov_FCH`` and the
  two blocks above (``serial`` and ``hetero``);
* GMA: ``theta``, ``durbin_order``, ``beta``, ``vcov``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pyfgls._utils import ar_roots
from pyfgls.combined import CombinedEstimate
from pyfgls.exceptions import InputError
from pyfgls.hetero.fgls import HeteroEstimate
from pyfgls.hetero.variance import weighted_leverage
from pyfgls.results import FGLSResults
from pyfgls.serial.fgls import SerialEstimate
from pyfgls.serial.ma import GMAEstimate

__all__ = ["export_fit", "export_fit_json"]


def _serial(est: SerialEstimate) -> dict[str, Any]:
    fr, var, sel = est.filtered, est.variance, est.selection
    return {
        "kstar": est.order,
        "kmax": None if sel is None else sel.kmax,
        "bic_path": None if sel is None else sel.values,
        "rho_hat": est.rho,
        "beta_durbin": est.durbin.beta,
        "durbin_names": list(est.design.names),
        "dropped_lags": list(est.design.dropped),
        "beta_fgls": fr.beta,
        "e_hat": fr.ehat,
        "u_hat": fr.uhat,
        "Ghat": var.G,
        "Sigma_psi": var.Sigma,
        "Qhat": fr.Xf.T @ fr.Xf / fr.n_eff,
        "sigma2_e": var.sigma2,
        "AVar_naive": var.naive_avar,
        "AVar_R": var.avar,
        "vcov": var.cov,
        "nonexogeneity_share": var.nonexogeneity_share,
        "ar_roots": ar_roots(est.rho),
    }


def _hetero(est: HeteroEstimate, X: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": est.method,
        "beta_ols": est.beta_unweighted,
        "h_ols": est.h_unweighted,
        "weights": est.v,
        "h_fgls": weighted_leverage(X, est.v),
        "beta_fgls": est.beta,
        "vcov_eq6_8": est.cov,
        "df_hat": est.df,
        "delta": est.delta,
    }
    fit = est.fit
    if fit is None:
        return out
    tun = fit.tuning
    out.update(
        {
            "fitted_logvar": fit.fitted,
            "intercept": fit.intercept,
            "gamma_hat": fit.coef,
            "nonzero_set": list(fit.selected),
            "z_names": list(fit.names),
            "foldid": tun.get("foldid"),
        }
    )
    if fit.method == "lasso" and "cv_path" in tun:
        psis = sorted(tun["cv_path"])
        out.update(
            {
                "ridge_coef": tun.get("lambda_ridge_coef"),
                "lambda_ridge": tun.get("lambda_ridge"),
                "psi_grid": psis,
                "cv_lambda": np.vstack([tun["cv_path"][p][0] for p in psis]),
                "cv_error_matrix": np.vstack([tun["cv_path"][p][1] for p in psis]),
                "psi_star": tun["psi"],
                "lambda_star": tun["lambda"],
                "cv_mse": tun["cv_mse"],
            }
        )
    return out


def export_fit(result: FGLSResults) -> dict[str, Any]:
    """All intermediates of a fit (see the module docstring for the keys).

    Raises
    ------
    InputError
        If ``result`` does not carry its estimator internals (e.g. a benchmark fit).
    """
    est = result.estimate
    if est is None:
        raise InputError("This result does not carry estimator internals (e.g. a benchmark fit).")
    out: dict[str, Any] = {
        "method": result.method,
        "cov_type": result.cov_type,
        "names": list(result.params.index),
        "params": result.params.to_numpy(),
        "cov": result.cov.to_numpy(),
    }
    if isinstance(est, SerialEstimate):
        out.update(_serial(est))
    elif isinstance(est, HeteroEstimate):
        out.update(_hetero(est, result.data.X))
    elif isinstance(est, CombinedEstimate):
        out["serial"] = _serial(est.serial)
        out["hetero"] = _hetero(est.hetero, est.serial.filtered.Xf)
        out.update(
            {
                "kstar": est.order,
                "beta_FC": est.beta_fc,
                "beta_FCH": est.beta_fch,
                "vcov_FC": est.cov_fc,
                "vcov_FCH": est.cov_fch,
                "first_stage_correction": est.first_stage_correction,
            }
        )
    elif isinstance(est, GMAEstimate):
        out.update(
            {
                "theta": est.theta,
                "durbin_order": est.durbin_order,
                "beta": est.beta,
                "vcov": est.cov,
                "e_hat": est.ehat,
            }
        )
    else:  # pragma: no cover - all estimators are covered above
        raise InputError(f"Unknown estimate type {type(est).__name__}.")
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {"re": x.real.tolist(), "im": x.imag.tolist()}
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def export_fit_json(result: FGLSResults, **kwargs: Any) -> str:
    """:func:`export_fit` serialised to JSON (complex roots as ``{"re", "im"}``)."""
    return json.dumps(_jsonable(export_fit(result)), **kwargs)
