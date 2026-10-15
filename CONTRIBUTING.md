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

## The documentation page on emiliogcs.com

https://emiliogcs.com/code/ is generated from this repository: `README.md`, `docs/` and
the docstrings (the API reference follows `docs/api.md`). Edit the documentation here,
never the generated page. After a release, or any change to the docs, update it with

```bash
scripts/sync_site.sh path/to/website-checkout   # a clone of emiliogcs/emiliogcs.github.io
```

The script builds the page in a throwaway virtual environment, runs the Quickstart
examples so the page shows their real output, regenerates the API reference, and writes
`code/index.html` (only the part between `<!-- pyfgls:start -->` and
`<!-- pyfgls:end -->`; the head, menu and footer belong to the website) and its images
into the website checkout. It commits nothing: commit there on a branch and open a pull
request. Merging into the website's `main` publishes the page.

`python scripts/build_site_page.py --update-readme` writes the same example outputs into
`README.md`, so the README never shows stale output; the build warns when they differ.

## Continuous integration

- Every push to `main` runs lint, type checks and the fast tests on Linux for
  Python 3.10–3.13.
- Pull requests, tags and manual runs (`workflow_dispatch`) also run the fast tests on
  macOS and Windows.
- The slow Monte Carlo suite runs nightly on Linux and on demand.
