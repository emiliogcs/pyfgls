# Both problems ([PG26] Supplement S-4)

With serially correlated errors whose innovations are heteroskedastic, the two
corrections are applied in sequence:

1. **F-C**: the serial-correlation FGLS of [PG26];
2. the adaptive-Lasso skedastic fit of [GP24] on the F-C residuals, with candidate
   covariates built from the *unfiltered* regressors;
3. **F-CH**: weighted least squares on the quasi-differenced data.

The variance is the bias-corrected sandwich of [GP24] applied to the filtered regressors.
`pyfgls` optionally adds a first-stage term for the estimated AR filter, the analogue of
[PG26] eq. (29) for heteroskedastic innovations (`first_stage_correction`, on by default
for pre-determined regressors). Under homoskedastic innovations F-CH is essentially
equivalent to F-C.
