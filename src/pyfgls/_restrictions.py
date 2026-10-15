"""Parsing of linear restrictions such as ``"x1 = x2"`` or ``"x1 + 2*x2 = 1, x3 = 0"``."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

import numpy as np

from pyfgls._linalg import FloatArray
from pyfgls.exceptions import InputError

_Linear = tuple[FloatArray, float]


def _split(restrictions: str | Sequence[str]) -> list[str]:
    items = [restrictions] if isinstance(restrictions, str) else list(restrictions)
    out: list[str] = []
    for item in items:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    if not out:
        raise InputError("No restriction given.")
    return out


def _linear_form(node: ast.AST, n: int, expr: str) -> _Linear:
    """Coefficients and constant of a linear expression AST."""
    if isinstance(node, ast.Expression):
        return _linear_form(node.body, n, expr)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return np.zeros(n), float(node.value)
    if isinstance(node, ast.Name) and node.id.startswith("__p") and node.id.endswith("__"):
        v = np.zeros(n)
        v[int(node.id[3:-2])] = 1.0
        return v, 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        a, c = _linear_form(node.operand, n, expr)
        return (-a, -c) if isinstance(node.op, ast.USub) else (a, c)
    if isinstance(node, ast.BinOp):
        la, lc = _linear_form(node.left, n, expr)
        ra, rc = _linear_form(node.right, n, expr)
        if isinstance(node.op, ast.Add):
            return la + ra, lc + rc
        if isinstance(node.op, ast.Sub):
            return la - ra, lc - rc
        if isinstance(node.op, ast.Mult):
            if not la.any():
                return lc * ra, lc * rc
            if not ra.any():
                return rc * la, rc * lc
        if isinstance(node.op, ast.Div) and not ra.any() and rc != 0:
            return la / rc, lc / rc
    raise InputError(f"Cannot parse the linear restriction {expr!r}.")


def parse_restrictions(
    restrictions: str | Sequence[str], names: Sequence[str]
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Turn restriction strings into ``(R, r, labels)`` with ``R beta = r``.

    Parameters
    ----------
    restrictions : str or sequence of str
        Comma-separated or listed restrictions, each ``"<linear> = <linear>"`` (a
        missing ``=`` means ``= 0``). Coefficients are referred to by name.
    names : sequence of str
        Parameter names.

    Examples
    --------
    >>> R, r, _ = parse_restrictions("x1 = x2, 2*x1 + x2 = 1", ["const", "x1", "x2"])
    >>> R
    array([[ 0.,  1., -1.],
           [ 0.,  2.,  1.]])
    >>> r
    array([0., 1.])
    """
    n = len(names)
    order = sorted(range(n), key=lambda i: -len(names[i]))
    rows: list[FloatArray] = []
    rhs: list[float] = []
    labels: list[str] = []
    for text in _split(restrictions):
        sides = text.split("=")
        if len(sides) > 2:
            raise InputError(f"Restriction {text!r} has more than one '='.")
        if len(sides) == 1:
            sides.append("0")
        forms: list[_Linear] = []
        for side in sides:
            expr = side
            for i in order:
                pattern = rf"(?<![\w.]){re.escape(names[i])}(?![\w.])"
                expr = re.sub(pattern, f" __p{i}__ ", expr)
            try:
                tree = ast.parse(expr.strip() or "0", mode="eval")
            except SyntaxError as exc:
                raise InputError(
                    f"Cannot parse {side.strip()!r} in {text!r}; unknown names? "
                    f"Available: {list(names)}"
                ) from exc
            forms.append(_linear_form(tree, n, text))
        (la, lc), (ra, rc) = forms
        a = la - ra
        if not a.any():
            raise InputError(f"Restriction {text!r} involves no coefficient.")
        rows.append(a)
        rhs.append(rc - lc)
        labels.append(text)
    return np.vstack(rows), np.asarray(rhs), tuple(labels)
