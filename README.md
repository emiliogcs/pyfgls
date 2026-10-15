# pyfgls — Feasible GLS for Python

`pyfgls` implements the feasible generalized least squares (FGLS) estimators of
two papers by Emilio González-Coya and Pierre Perron:

- **Perron, P. and González-Coya, E. (2026).** *Feasible GLS for time series regression.*
  Journal of Econometrics 258, 106346. — FGLS for **serially correlated** errors: an
  autoregressive filter estimated from a Durbin regression with BIC lag selection, and a
  variance estimator that stays valid when the regressors are only pre-determined
  (not strictly exogenous).
- **González-Coya, E. and Perron, P. (2024).** *Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach.* Journal of Econometric
  Methods 13(1), 29–48. — FGLS for **heteroskedasticity of unknown form**: the skedastic
  function is estimated by an adaptive Lasso over a dictionary of candidate covariates,
  with a bias-corrected robust variance.

The two procedures can be combined into a single estimator for time series that have
both problems.

> **Status:** under active development (pre-0.1.0). The API below is the design target.

## Install

```bash
pip install -e ".[dev]"     # from a clone of this repository
```

## Usage (target API)

```python
import pyfgls as pf

res = pf.FGLS(y, X, serial="auto", hetero="lasso").fit()
print(res.summary())
```

## Citing

If you use `pyfgls`, please cite the papers that the methods come from:

```bibtex
@article{PerronGonzalezCoya2026,
  author  = {Perron, Pierre and Gonz{\'a}lez-Coya, Emilio},
  title   = {Feasible {GLS} for time series regression},
  journal = {Journal of Econometrics},
  volume  = {258},
  pages   = {106346},
  year    = {2026}
}

@article{GonzalezCoyaPerron2024,
  author  = {Gonz{\'a}lez-Coya, Emilio and Perron, Pierre},
  title   = {Estimation in the presence of heteroskedasticity of unknown form:
             a {L}asso-based approach},
  journal = {Journal of Econometric Methods},
  volume  = {13},
  number  = {1},
  pages   = {29--48},
  year    = {2024}
}
```

## License

MIT © Emilio González-Coya
