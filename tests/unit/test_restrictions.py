import numpy as np
import pytest

from pyfgls._restrictions import parse_restrictions
from pyfgls.exceptions import InputError

NAMES = ["const", "x1", "x2", "L1.y"]


def test_simple_equalities():
    R, r, labels = parse_restrictions("x1 = x2", NAMES)
    np.testing.assert_array_equal(R, [[0, 1, -1, 0]])
    np.testing.assert_array_equal(r, [0])
    assert labels == ("x1 = x2",)


def test_scaled_terms_constants_and_missing_equals():
    R, r, _ = parse_restrictions(["2*x1 - x2/2 + 1 = 3", "L1.y"], NAMES)
    np.testing.assert_allclose(R, [[0, 2, -0.5, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(r, [2, 0])


def test_names_are_matched_as_whole_tokens():
    R, _, _ = parse_restrictions("x1 + x2 = 0", ["x1", "x10", "x2"])
    np.testing.assert_array_equal(R, [[1, 0, 1]])


@pytest.mark.parametrize("bad", ["x1 = x2 = 0", "x9 = 0", "x1 * x2 = 0", "1 = 1", ""])
def test_bad_restrictions(bad):
    with pytest.raises(InputError):
        parse_restrictions(bad, NAMES)
