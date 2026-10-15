"""Input validation and conversion.

Everything user-facing goes through :func:`prepare_data`, which turns arrays, lists,
``Series`` and ``DataFrame`` objects into float64 arrays while remembering the names and
the index, and which checks the things that make FGLS results silently wrong: time
order, gaps, missing values, constant and collinear columns.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyfgls._linalg import FloatArray, independent_columns, lstsq
from pyfgls.exceptions import (
    CollinearityError,
    InputError,
    InsufficientDataError,
    LaggedDependentWarning,
    MissingDataError,
    SpecificationWarning,
    TimeIndexError,
)

MissingPolicy = Literal["raise", "drop"]

_MAX_LOCATIONS = 5


@dataclass(frozen=True)
class RegressionData:
    """Validated regression data.

    Attributes
    ----------
    y : ndarray, shape (T,)
        Dependent variable.
    X : ndarray, shape (T, q)
        Regressors.
    y_name : str
        Name of the dependent variable.
    x_names : tuple of str
        Names of the regressors.
    index : pandas.Index or None
        Row labels after any missing-value handling (``None`` for plain arrays).
    const_col : int or None
        Position of an explicit constant column.
    has_implicit_const : bool
        Whether a constant lies in the column span of ``X`` without being a column
        (e.g. a full set of dummies).
    """

    y: FloatArray
    X: FloatArray
    y_name: str
    x_names: tuple[str, ...]
    index: pd.Index | None
    const_col: int | None
    has_implicit_const: bool
    dropped_rows: tuple[Any, ...] = field(default=())

    @property
    def nobs(self) -> int:
        """Number of observations ``T``."""
        return int(self.y.shape[0])

    @property
    def k_vars(self) -> int:
        """Number of regressors ``q``."""
        return int(self.X.shape[1])

    @property
    def has_const(self) -> bool:
        """Whether the regressors contain an explicit or implicit intercept."""
        return self.const_col is not None or self.has_implicit_const


# ------------------------------------------------------------------------- conversions
def _to_float(values: Any, what: str) -> FloatArray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"{what} must be numeric; could not convert it to float64 ({exc})."
        ) from exc
    return arr


def _convert_y(y: Any) -> tuple[FloatArray, str, pd.Index | None]:
    name = "y"
    index: pd.Index | None = None
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise InputError(
                f"y must be one-dimensional, but a DataFrame with {y.shape[1]} columns "
                "was given. Pass a single column, e.g. df['y']."
            )
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        if y.name is not None:
            name = str(y.name)
        index = y.index
        arr = _to_float(y.to_numpy(), "y")
    else:
        arr = _to_float(y, "y")
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise InputError(f"y must be one-dimensional; got an array of shape {arr.shape}.")
    return arr, name, index


def _convert_X(X: Any) -> tuple[FloatArray, list[str] | None, pd.Index | None]:
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        bad = [
            str(c)
            for c in X.columns
            if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))
        ]
        if bad:
            raise InputError(
                f"X has non-numeric columns {bad}. Encode categorical variables as dummies "
                "first (e.g. pandas.get_dummies) or use FGLS.from_formula."
            )
        names = [str(c) for c in X.columns]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise InputError(f"X has duplicated column names {dup}; rename them.")
        return _to_float(X.to_numpy(dtype=np.float64), "X"), names, X.index
    arr = _to_float(X, "X")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise InputError(f"X must be two-dimensional; got an array of shape {arr.shape}.")
    return arr, None, None


def _default_names(X: FloatArray) -> list[str]:
    names: list[str] = []
    j = 0
    for col in X.T:
        if col.size and np.all(col == col[0]) and col[0] != 0.0:
            names.append("const")
        else:
            j += 1
            names.append(f"x{j}")
    if names.count("const") > 1:  # let the collinearity check report it, with unique names
        names = [f"x{i + 1}" for i in range(X.shape[1])]
    return names


def _format_locations(index: pd.Index | None, positions: npt.NDArray[np.intp]) -> str:
    pos = positions[:_MAX_LOCATIONS]
    labels = [str(index[p]) for p in pos] if index is not None else [str(int(p)) for p in pos]
    more = f" (and {positions.size - pos.size} more)" if positions.size > pos.size else ""
    where = "index labels" if index is not None else "row positions"
    return f"{where} {', '.join(labels)}{more}"


# ---------------------------------------------------------------------------- time index
def _asi8(index: pd.Index) -> npt.NDArray[np.int64]:
    """Integer representation of a datetime-like index (ordinals for periods)."""
    return np.asarray(cast(Any, index).asi8, dtype=np.int64)


def _time_index_gaps(index: pd.Index) -> npt.NDArray[np.intp]:
    """Positions ``i`` such that a gap occurs between rows ``i`` and ``i + 1``."""
    empty = np.zeros(0, dtype=np.intp)
    if len(index) < 2:
        return empty
    if isinstance(index, pd.PeriodIndex):
        d = np.diff(_asi8(index))
        return np.flatnonzero(d != 1).astype(np.intp)
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq
        if freq is None and len(index) >= 3:
            try:
                inferred = pd.infer_freq(index)
            except (TypeError, ValueError):
                inferred = None
            freq = pd.tseries.frequencies.to_offset(inferred) if inferred else None
        if freq is not None:
            expected = pd.date_range(index[0], periods=len(index), freq=freq)
            if expected.equals(index):
                return empty
        days = index.normalize().to_numpy().astype("datetime64[D]")
        if index.equals(index.normalize()) and np.all(np.is_busday(days)):
            # business-day data: a gap is a skipped weekday (holiday or missing day)
            d = np.busday_count(days[:-1], days[1:])
            return np.flatnonzero(d != 1).astype(np.intp)
        d = np.diff(_asi8(index))
        vals, counts = np.unique(d, return_counts=True)
        modal = vals[np.argmax(counts)]
        return np.flatnonzero(d > 1.5 * modal).astype(np.intp)
    return empty


def check_time_index(index: pd.Index | None, *, allow_gaps: bool = False) -> None:
    """Check that a time index is sorted, unique, not a panel, and (by default) gap-free.

    Raises
    ------
    TimeIndexError
        If the index is a ``MultiIndex`` (panels are out of scope), is not monotone
        increasing, has duplicates, or has gaps while ``allow_gaps`` is false.
    """
    if index is None:
        return
    if isinstance(index, pd.MultiIndex):
        raise TimeIndexError(
            "A MultiIndex was given; panel data are out of scope for pyfgls. "
            "Estimate one unit at a time or pass a single time series."
        )
    ordered = isinstance(index, pd.DatetimeIndex | pd.PeriodIndex | pd.TimedeltaIndex) or (
        pd.api.types.is_numeric_dtype(index.dtype)
    )
    if not ordered:
        return
    if index.has_duplicates:
        dup = index[index.duplicated()].unique()[:_MAX_LOCATIONS]
        raise TimeIndexError(
            f"The time index has duplicated labels ({', '.join(map(str, dup))}). "
            "Each period must appear once."
        )
    if not index.is_monotonic_increasing:
        if isinstance(index, pd.DatetimeIndex | pd.PeriodIndex | pd.TimedeltaIndex):
            vals = _asi8(index)
        else:
            vals = index.to_numpy(dtype=np.float64)
        pos = np.flatnonzero(np.diff(vals) < 0)
        first = int(pos[0]) + 1 if pos.size else 0
        raise TimeIndexError(
            "The data are not sorted in time (the index is not monotone increasing; "
            f"first out-of-order label near {index[first]!s}). FGLS uses lags, so row "
            "order matters: sort with .sort_index() first."
        )
    gaps = _time_index_gaps(index)
    if gaps.size:
        desc = ", ".join(f"{index[g]!s} -> {index[g + 1]!s}" for g in gaps[:_MAX_LOCATIONS])
        more = f" and {gaps.size - _MAX_LOCATIONS} more" if gaps.size > _MAX_LOCATIONS else ""
        if not allow_gaps:
            raise TimeIndexError(
                f"The time index has {gaps.size} gap(s) ({desc}{more}). Lags would "
                "silently skip the missing periods. Fill or interpolate them, or pass "
                "allow_gaps=True to treat lags as positional (e.g. trading days with "
                "holidays)."
            )
        warnings.warn(
            f"The time index has {gaps.size} gap(s) ({desc}{more}); lags are positional "
            "(allow_gaps=True).",
            SpecificationWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- main entry
def prepare_data(
    y: Any,
    X: Any,
    *,
    missing: MissingPolicy = "raise",
    time_series: bool = True,
    allow_gaps: bool = False,
    warn_no_const: bool = True,
) -> RegressionData:
    """Validate and convert ``(y, X)``.

    Parameters
    ----------
    y : array_like or pandas.Series
        Dependent variable, length ``T``. A ``(T, 1)`` array or one-column
        ``DataFrame`` is squeezed.
    X : array_like, pandas.Series or pandas.DataFrame
        Regressors, ``T`` rows. Include a constant column for an intercept.
    missing : {"raise", "drop"}
        What to do with ``NaN``. ``"drop"`` drops rows with missing values; for time
        series only leading/trailing rows can be dropped (an interior gap would make
        the lags wrong), so interior ``NaN`` still raise.
    time_series : bool
        Whether row order is time order (checks the index, restricts dropping).
    allow_gaps : bool
        Accept a date/period index with gaps (lags are then positional).
    warn_no_const : bool
        Warn when ``X`` has no intercept.

    Returns
    -------
    RegressionData

    Raises
    ------
    InputError, MissingDataError, TimeIndexError, CollinearityError,
    InsufficientDataError
    """
    if missing not in ("raise", "drop"):
        raise InputError(f"missing must be 'raise' or 'drop', got {missing!r}.")
    y_arr, y_name, y_index = _convert_y(y)
    X_arr, x_names, x_index = _convert_X(X)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise InputError(f"y has {y_arr.shape[0]} observations but X has {X_arr.shape[0]} rows.")
    if X_arr.shape[1] == 0:
        raise InputError("X has no columns.")

    index: pd.Index | None
    if y_index is not None and x_index is not None:
        if not y_index.equals(x_index):
            raise InputError(
                "y and X have different indexes. Align them first, e.g. "
                "y, X = y.align(X, join='inner', axis=0)."
            )
        index = y_index
    else:
        index = y_index if y_index is not None else x_index

    if x_names is None:
        x_names = _default_names(X_arr)

    # ---- missing / non-finite values
    if np.isinf(y_arr).any() or np.isinf(X_arr).any():
        bad = np.flatnonzero(np.isinf(y_arr) | np.isinf(X_arr).any(axis=1))
        raise MissingDataError(
            f"Infinite values found at {_format_locations(index, bad)}. Replace them first."
        )
    nan_rows = np.isnan(y_arr) | np.isnan(X_arr).any(axis=1)
    dropped: tuple[Any, ...] = ()
    if nan_rows.any():
        bad = np.flatnonzero(nan_rows)
        if missing == "raise":
            raise MissingDataError(
                f"Missing values (NaN) found at {_format_locations(index, bad)}. Remove or "
                "impute them, or pass missing='drop'."
            )
        if time_series:
            ok = np.flatnonzero(~nan_rows)
            if ok.size == 0:
                raise MissingDataError("All observations have missing values.")
            first, last = int(ok[0]), int(ok[-1])
            interior = np.flatnonzero(nan_rows[first : last + 1]) + first
            if interior.size:
                raise MissingDataError(
                    f"Missing values inside the sample at {_format_locations(index, interior)}. "
                    "With time series only leading/trailing missing values can be dropped, "
                    "because dropping interior rows would make the lags wrong. Impute or "
                    "interpolate them first."
                )
            keep = np.zeros(nan_rows.size, dtype=bool)
            keep[first : last + 1] = True
        else:
            keep = ~nan_rows
        drop_pos = np.flatnonzero(~keep)
        dropped = tuple(index[drop_pos]) if index is not None else tuple(drop_pos.tolist())
        y_arr, X_arr = y_arr[keep], X_arr[keep]
        if index is not None:
            index = index[keep]

    if time_series:
        check_time_index(index, allow_gaps=allow_gaps)

    T, q = X_arr.shape
    if T <= q:
        raise InsufficientDataError(
            f"Only {T} observations for {q} regressors; need more observations than regressors."
        )

    # ---- constants and rank
    const_cols = [j for j in range(q) if np.all(X_arr[:, j] == X_arr[0, j]) and X_arr[0, j] != 0.0]
    zero_cols = [x_names[j] for j in range(q) if not np.any(X_arr[:, j])]
    if zero_cols:
        raise CollinearityError(f"Columns {zero_cols} of X are identically zero; drop them.")
    if len(const_cols) > 1:
        raise CollinearityError(
            f"X has several constant columns {[x_names[j] for j in const_cols]}; keep one."
        )
    _, dependent = independent_columns(X_arr)
    if dependent.size:
        names = [x_names[j] for j in dependent]
        raise CollinearityError(
            f"X is rank deficient: column(s) {names} are linear combinations of the "
            "preceding columns. Drop them (or one column of each set of dummies when an "
            "intercept is included)."
        )
    const_col = const_cols[0] if const_cols else None
    implicit = False
    if const_col is None:
        ones = np.ones(T)
        resid = ones - X_arr @ lstsq(X_arr, ones)
        implicit = bool(np.linalg.norm(resid) <= 1e-8 * np.sqrt(T))
        if not implicit and warn_no_const:
            warnings.warn(
                "X has no intercept (no constant column). Estimates are valid only if the "
                "model truly has no intercept; add a column of ones otherwise.",
                SpecificationWarning,
                stacklevel=3,
            )

    return RegressionData(
        y=np.ascontiguousarray(y_arr),
        X=np.ascontiguousarray(X_arr),
        y_name=y_name,
        x_names=tuple(x_names),
        index=index,
        const_col=const_col,
        has_implicit_const=implicit,
        dropped_rows=dropped,
    )


# ------------------------------------------------------------------- specific checks
def detect_lagged_dependent(
    y: FloatArray,
    X: FloatArray,
    names: Sequence[str],
    *,
    max_lag: int = 4,
    exclude: Sequence[str] = (),
) -> list[tuple[str, int]]:
    """Find regressors that look like lags of the dependent variable.

    A column ``x_j`` is flagged as ``y_{t-p}`` when ``x_{j,t}`` equals ``y_{t-p}`` up to
    rounding, or when their correlation exceeds ``0.99999``.

    Returns
    -------
    list of (name, lag)
    """
    out: list[tuple[str, int]] = []
    T = y.shape[0]
    for j, name in enumerate(names):
        if name in exclude:
            continue
        x = X[:, j]
        if np.all(x == x[0]):
            continue
        for p in range(1, min(max_lag, T - 3) + 1):
            a, b = x[p:], y[:-p]
            if np.allclose(a, b, rtol=1e-10, atol=1e-12 * (np.abs(b).max() + 1.0)):
                out.append((name, p))
                break
            sa, sb = a.std(), b.std()
            if sa > 0 and sb > 0 and np.corrcoef(a, b)[0, 1] > 0.99999:
                out.append((name, p))
                break
    return out


def warn_lagged_dependent(found: list[tuple[str, int]]) -> None:
    """Emit a :class:`LaggedDependentWarning` for undeclared lagged dependent variables."""
    if not found:
        return
    desc = ", ".join(f"{n!r} (looks like y lagged {p})" for n, p in found)
    names = [n for n, _ in found]
    warnings.warn(
        f"Regressor(s) {desc} appear to be lags of the dependent variable. Declare them "
        f"with lagged_dep_vars={names!r} so the AR filter is estimated consistently "
        "([PG26] Section 4.2).",
        LaggedDependentWarning,
        stacklevel=3,
    )


def check_choice(name: str, value: object, choices: Sequence[object]) -> None:
    """Raise :class:`InputError` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise InputError(f"{name} must be one of {list(choices)!r}; got {value!r}.")
