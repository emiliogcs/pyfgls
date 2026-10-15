# Many candidate covariates

The adaptive Lasso is designed for large candidate sets:

```python
Z = pd.concat([df[cols], np.log(df[cols]), df[cols] ** 2], axis=1)  # your dictionary
res = pf.LassoFGLS(y, X, z=Z, random_state=0, n_jobs=-1).fit()
res.selected_z
```

Tips:

* Candidates are standardised internally; their units do not matter.
* `z` may have more columns than observations; cross-validation then becomes noisier
  and slower (a `SampleSizeWarning` tells you).
* Use `pyfgls.hetero.make_dictionary(X, names, transforms=("poly2", "log", "trig"))` to
  build the default dictionary and extend it.
* With dependent data use `fold_scheme="blocked"`.
* `lambda_rule="1se"` gives sparser fits; the default `"min"` follows [GP24].
* Extreme weights trigger a `WeightWarning`; `weight_clip=(0.01, 0.99)` winsorises the
  fitted log-variance (off by default to match the paper).
