"""pyfgls: Feasible Generalized Least Squares for Python.

Implements the FGLS procedures of

* Perron, P. and González-Coya, E. (2026), "Feasible GLS for time series
  regression", *Journal of Econometrics* 258, 106346 — serial correlation;
* González-Coya, E. and Perron, P. (2024), "Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach", *Journal of
  Econometric Methods* 13(1), 29–48 — heteroskedasticity.
"""

from pyfgls import exceptions, simulate
from pyfgls._version import __version__
from pyfgls.models import FGLS, LassoFGLS, SerialFGLS
from pyfgls.results import FGLSResults, compare

__all__ = [
    "FGLS",
    "FGLSResults",
    "LassoFGLS",
    "SerialFGLS",
    "__version__",
    "compare",
    "exceptions",
    "simulate",
]
