import warnings

import numpy as np
import pandas as pd
import pytest

from pyfgls._validation import (
    check_choice,
    check_time_index,
    detect_lagged_dependent,
    prepare_data,
    warn_lagged_dependent,
)
from pyfgls.exceptions import (
    CollinearityError,
    InputError,
    InsufficientDataError,
    LaggedDependentWarning,
    MissingDataError,
    PyFGLSError,
    SpecificationWarning,
    TimeIndexError,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    T = 40
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=T)
    return y, X


def test_arrays_get_default_names(data):
    y, X = data
    d = prepare_data(y, X)
    assert d.x_names == ("const", "x1", "x2")
    assert d.y_name == "y"
    assert d.const_col == 0
    assert d.has_const
    assert d.index is None
    assert d.nobs == 40
    assert d.k_vars == 3
    assert d.y.dtype == np.float64


def test_lists_and_column_vector_y(data):
    y, X = data
    d = prepare_data(y[:, None].tolist(), X.tolist())
    np.testing.assert_allclose(d.y, y)


def test_pandas_names_and_index(data):
    y, X = data
    idx = pd.period_range("2000Q1", periods=40, freq="Q")
    df = pd.DataFrame(X, columns=["c", "a", "b"], index=idx)
    ys = pd.Series(y, index=idx, name="growth")
    d = prepare_data(ys, df)
    assert d.x_names == ("c", "a", "b")
    assert d.y_name == "growth"
    assert d.index.equals(idx)


def test_one_column_dataframe_y_is_squeezed(data):
    y, X = data
    d = prepare_data(pd.DataFrame({"z": y}), X)
    assert d.y_name == "z"
    with pytest.raises(InputError, match="one-dimensional"):
        prepare_data(pd.DataFrame({"a": y, "b": y}), X)


def test_misaligned_index_raises(data):
    y, X = data
    ys = pd.Series(y, index=np.arange(40))
    Xd = pd.DataFrame(X, index=np.arange(1, 41))
    with pytest.raises(InputError, match="different indexes"):
        prepare_data(ys, Xd)


def test_length_mismatch(data):
    y, X = data
    with pytest.raises(InputError, match="observations"):
        prepare_data(y[:-1], X)


def test_non_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "g": ["u", "v", "u", "v"]})
    with pytest.raises(InputError, match="non-numeric"):
        prepare_data([1.0, 2.0, 3.0, 4.0], df)


def test_missing_raise_and_drop(data):
    y, X = data
    y = y.copy()
    y[0] = np.nan
    y[-1] = np.nan
    with pytest.raises(MissingDataError, match="row positions 0, 39"):
        prepare_data(y, X)
    d = prepare_data(y, X, missing="drop")
    assert d.nobs == 38
    assert d.dropped_rows == (0, 39)
    # interior NaN cannot be dropped for time series
    y2 = data[0].copy()
    y2[10] = np.nan
    with pytest.raises(MissingDataError, match="inside the sample"):
        prepare_data(y2, X, missing="drop")
    # ... but can for cross sections
    assert prepare_data(y2, X, missing="drop", time_series=False).nobs == 39


def test_infinite_values_raise(data):
    y, X = data
    X = X.copy()
    X[3, 1] = np.inf
    with pytest.raises(MissingDataError, match="Infinite"):
        prepare_data(y, X)


def test_rank_deficiency_names_offending_column(data):
    y, X = data
    Xb = pd.DataFrame(np.column_stack([X, X[:, 1] + X[:, 2]]), columns=["c", "a", "b", "s"])
    with pytest.raises(CollinearityError, match=r"\['s'\]"):
        prepare_data(y, Xb)


def test_zero_and_duplicate_constant_columns(data):
    y, X = data
    with pytest.raises(CollinearityError, match="identically zero"):
        prepare_data(y, np.column_stack([X, np.zeros(40)]))
    with pytest.raises(CollinearityError, match="constant columns"):
        prepare_data(y, np.column_stack([X, 2 * np.ones(40)]))


def test_no_intercept_warns_but_implicit_intercept_does_not(data):
    y, X = data
    with pytest.warns(SpecificationWarning, match="no intercept"):
        d = prepare_data(y, X[:, 1:])
    assert not d.has_const
    D = np.zeros((40, 2))
    D[:20, 0] = 1.0
    D[20:, 1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d2 = prepare_data(y, np.column_stack([D, X[:, 1]]))
    assert d2.has_implicit_const
    assert d2.has_const


def test_too_few_observations():
    with pytest.raises(InsufficientDataError):
        prepare_data([1.0, 2.0], [[1.0, 0.0], [1.0, 1.0]])


def test_unsorted_index_rejected(data):
    y, X = data
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    perm = np.random.default_rng(1).permutation(40)
    ys = pd.Series(y, index=idx).iloc[perm]
    Xd = pd.DataFrame(X, index=idx).iloc[perm]
    with pytest.raises(TimeIndexError, match="not sorted"):
        prepare_data(ys, Xd)
    # cross sections do not care about order
    prepare_data(ys, Xd, time_series=False)


def test_unsorted_integer_index_rejected(data):
    y, X = data
    perm = np.random.default_rng(2).permutation(40)
    with pytest.raises(TimeIndexError):
        prepare_data(pd.Series(y).iloc[perm], pd.DataFrame(X).iloc[perm])


def test_duplicate_and_multiindex():
    idx = pd.DatetimeIndex(["2000-01-01", "2000-01-01", "2000-01-02"])
    with pytest.raises(TimeIndexError, match="duplicated"):
        check_time_index(idx)
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]])
    with pytest.raises(TimeIndexError, match="panel"):
        check_time_index(mi)


def test_gap_detection_and_allow_gaps():
    idx = pd.date_range("2000-01-01", periods=12, freq="MS").delete(5)
    with pytest.raises(TimeIndexError, match="gap"):
        check_time_index(idx)
    with pytest.warns(SpecificationWarning, match="positional"):
        check_time_index(idx, allow_gaps=True)
    pidx = pd.period_range("2000Q1", periods=8, freq="Q").delete(3)
    with pytest.raises(TimeIndexError, match="gap"):
        check_time_index(pidx)


def test_regular_indexes_pass():
    check_time_index(pd.date_range("2000-01-03", periods=30, freq="B"))
    check_time_index(pd.date_range("2000-01-31", periods=30, freq="ME"))
    check_time_index(pd.period_range("1990", periods=10, freq="Y"))
    check_time_index(pd.RangeIndex(10))
    # a DatetimeIndex without freq attribute but regular spacing
    check_time_index(pd.DatetimeIndex(pd.date_range("2000", periods=10, freq="D").tolist()))
    # weekday data with a missing weekday is a gap
    bidx = pd.date_range("2000-01-03", periods=20, freq="B").delete(7)
    with pytest.raises(TimeIndexError):
        check_time_index(pd.DatetimeIndex(bidx.tolist()))


def test_detect_lagged_dependent(data):
    y, X = data
    ylag = np.r_[np.nan, y[:-1]]
    Xl = np.column_stack([X[1:], ylag[1:]])
    found = detect_lagged_dependent(y[1:], Xl, ["const", "x1", "x2", "ylag"])
    assert found == [("ylag", 1)]
    assert detect_lagged_dependent(y[1:], Xl, ["const", "x1", "x2", "ylag"], exclude=["ylag"]) == []
    with pytest.warns(LaggedDependentWarning, match="lagged_dep_vars"):
        warn_lagged_dependent(found)


def test_errors_share_base_class():
    assert issubclass(InputError, PyFGLSError)
    assert issubclass(InputError, ValueError)
    with pytest.raises(InputError, match="must be one of"):
        check_choice("ic", "hqic", ["bic", "aic"])
