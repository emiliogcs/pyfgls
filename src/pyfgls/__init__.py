"""pyfgls: Feasible Generalized Least Squares for Python.

Implements the FGLS procedures of

* Perron, P. and González-Coya, E. (2026), "Feasible GLS for time series
  regression", *Journal of Econometrics* 258, 106346 — serial correlation;
* González-Coya, E. and Perron, P. (2024), "Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach", *Journal of
  Econometric Methods* 13(1), 29–48 — heteroskedasticity.
"""

from pyfgls._version import __version__

__all__ = ["__version__"]
