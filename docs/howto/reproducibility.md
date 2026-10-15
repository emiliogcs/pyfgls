# Reproducibility

Some steps of the estimators are random. Fix the seeds below and a fit gives the same
numbers every time it is run.

* **Serial correlation only** (`SerialFGLS`, or `FGLS(..., hetero=None)`): the fit is
  deterministic. There is nothing to seed except the bootstrap.
* **Heteroskedasticity** (`LassoFGLS`, `FGLS`): the cross-validation that tunes the
  adaptive Lasso (and SVR) splits the sample into random folds. Without a seed the
  selected covariates, and with them the estimates, can change from run to run. Pass
  `random_state=` (an int or a `numpy.random.Generator`) to the model:

  ```python
  res = pf.LassoFGLS(y, X, random_state=0).fit()
  ```

  `fold_scheme="blocked"` uses contiguous blocks instead of random folds and needs no
  seed.
* **Bootstrap** (`fit(cov_type="bootstrap")`): pass `boot_random_state=`; by default it
  takes the model's `random_state`. Each replication draws from its own child seed
  (`numpy.random.SeedSequence.spawn`), so the result does not depend on `n_jobs`.
* **Simulated data**: every design in `pyfgls.simulate` takes `random_state`.

`n_jobs` only spreads the folds and the bootstrap replications over cores; with a fixed
seed the results are the same for any value.

When you report results, record the versions as well, since numerical libraries can change
the last digits:

```python
import numpy
import pandas
import scipy
import sklearn

import pyfgls

for module in (pyfgls, numpy, scipy, pandas, sklearn):
    print(module.__name__, module.__version__)
```
