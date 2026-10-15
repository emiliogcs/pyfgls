# pyfgls

**Feasible GLS for time series regression and for heteroskedasticity of unknown form.**

`pyfgls` implements the estimators of

* **[PG26]** Perron, P. and González-Coya, E. (2026), *Feasible GLS for time series
  regression*, Journal of Econometrics 258, 106346,
  [doi:10.1016/j.jeconom.2026.106346](https://doi.org/10.1016/j.jeconom.2026.106346) —
  serial correlation;
* **[GP24]** González-Coya, E. and Perron, P. (2024), *Estimation in the presence of
  heteroskedasticity of unknown form: a Lasso-based approach*, Journal of Econometric
  Methods 13(1), 29–48, [doi:10.1515/jem-2023-0007](https://doi.org/10.1515/jem-2023-0007)
  — heteroskedasticity;

and their combination ([PG26] Supplement S-4).

Install it from [PyPI](https://pypi.org/project/pyfgls/) (Python 3.10 or newer):

```bash
pip install pyfgls
```

```python
import pyfgls as pf

res = pf.FGLS(y, X).fit()  # serial correlation + heteroskedasticity
res = pf.SerialFGLS(y, X).fit()  # [PG26] only
res = pf.LassoFGLS(y, X).fit()  # [GP24] only
print(res.summary())
```

Start with the [Quickstart](quickstart.md) and the
[Guidance for practitioners](guidance.md), which explains when to use each estimator
(following the concluding remarks of both papers). The [Theory](theory/serial.md) pages
explain what each estimator does and why, with equation references to the papers.
