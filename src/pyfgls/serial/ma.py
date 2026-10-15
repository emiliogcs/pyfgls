"""GMA estimator for MA(1) errors ([PG26] Section 5) — implemented in a later phase."""

from __future__ import annotations

from typing import Any

from pyfgls._linalg import FloatArray


def estimate_gma(y: FloatArray, X: FloatArray) -> Any:
    """Raise NotImplementedError (GMA arrives in a later phase)."""
    raise NotImplementedError("serial='ma1' (GMA) is not implemented yet.")


def build_gma_results(data: Any, est: Any, *, use_t: bool, options: dict[str, Any]) -> Any:
    """Raise NotImplementedError (GMA arrives in a later phase)."""
    raise NotImplementedError("serial='ma1' (GMA) is not implemented yet.")
