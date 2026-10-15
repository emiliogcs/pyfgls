# Contributing to pyfgls

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Checks to run before every commit

```bash
ruff check . && ruff format --check .
mypy                       # strict mode, configured in pyproject.toml
pytest -n auto             # fast unit, property and doctest suite
pytest -n auto --runslow   # adds the Monte Carlo acceptance tests (minutes)
```

The `slow` tests replicate the simulation tables of the two papers with fewer
replications than the papers use. Their tolerances are chosen to cover Monte Carlo
noise at the replication counts in the tests; do not loosen a tolerance without
explaining why in the test.

## Conventions

- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:`).
- NumPy-style docstrings with a runnable example and the equation numbers of the
  paper being implemented (for example `# PG26 eq. (29)`).
- Numerical code uses QR / `lstsq` / `solve`; never invert `X'X` directly.
- Every stochastic routine takes `random_state` (int or `numpy.random.Generator`).
- Do not commit the papers' PDFs; cite them instead.

## Continuous integration

- Every push to `main` runs lint, type checks and the fast tests on Linux for
  Python 3.10–3.13.
- Pull requests, tags and manual runs (`workflow_dispatch`) also run the fast tests on
  macOS and Windows.
- The slow Monte Carlo suite runs nightly on Linux and on demand.
