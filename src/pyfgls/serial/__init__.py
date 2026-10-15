"""Serial-correlation FGLS ([PG26]): Durbin regression, lag selection, filter, variance."""

from pyfgls.serial.durbin import DurbinDesign, DurbinFit, durbin_design, fit_durbin
from pyfgls.serial.fgls import SerialEstimate, check_ar_roots, estimate_serial
from pyfgls.serial.filter import FilteredRegression, filtered_regression, quasi_difference
from pyfgls.serial.lagselect import LagSelection, resolve_kmax, select_ar_order
from pyfgls.serial.variance import SerialVariance, serial_variance
from pyfgls.serial.wallis import WallisResult, wallis_rho

__all__ = [
    "DurbinDesign",
    "DurbinFit",
    "FilteredRegression",
    "LagSelection",
    "SerialEstimate",
    "SerialVariance",
    "WallisResult",
    "check_ar_roots",
    "durbin_design",
    "estimate_serial",
    "filtered_regression",
    "fit_durbin",
    "quasi_difference",
    "resolve_kmax",
    "select_ar_order",
    "serial_variance",
    "wallis_rho",
]
