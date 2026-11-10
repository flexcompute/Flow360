"""
Composable validation functions

Each function implements a single validation criterion and can be composed
to create complex validation logic. These are pure Python apart from the numpy
fast path in `finite_or_raise`, which array-shaped fields depend on to stay
cheap; numpy is already a hard dependency of the package through unyt.
"""

import math
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np


def positive(value: float) -> float:
    """Validate value > 0"""
    # Use "not (value > 0)" to reject NaN, zero, and negative values
    if not (value > 0):
        raise ValueError(f"Value must be positive (>0), got {value}")
    return value


def non_negative(value: float) -> float:
    """Validate value >= 0"""
    # Use "not (value >= 0)" to reject NaN and negative values
    if not (value >= 0):
        raise ValueError(f"Value must be non-negative (>=0), got {value}")
    return value


def _iter_leaf_components(value: Any) -> Iterator[float]:
    """Yield every scalar leaf of a scalar, vector, or nested array value.

    Recursing rather than iterating one level is what lets a shape like `(N, 3)` be
    checked at all: its top-level items are rows, not numbers. Everything reaching here
    has already been through `to_unyt`, so the leaves are numeric.
    """
    if not isinstance(value, Iterable):
        yield float(value)
        return
    try:
        items = tuple(value)
    except TypeError:
        # numpy 0-d arrays register as Iterable but are scalar; treat as one component
        yield float(value)  # type: ignore[arg-type]
        return
    for item in items:
        yield from _iter_leaf_components(item)


def finite_or_raise(value: float | tuple[float, ...], name: str) -> float | tuple[float, ...]:
    """Reject non-finite (NaN/Inf) components in a scalar, vector or nested array value.

    Every dimensioned field passes through here, so the numpy fast path is what keeps a
    large array cheap: walking the leaves in Python costs ~23 ms for a 10k-row (N, 3)
    field against ~0.01 ms for `np.isfinite`.
    """
    if isinstance(value, np.ndarray):
        all_finite = bool(np.isfinite(value).all())
    else:
        all_finite = all(math.isfinite(component) for component in _iter_leaf_components(value))
    if not all_finite:
        raise ValueError(
            f"{name} must be finite (no NaN/Inf); got non-finite component(s): {value}. "
            "A required numeric input was likely left blank or partial."
        )
    return value


def vector3_shape(value: Any) -> tuple[float, float, float]:
    """
    Validate and convert to length-3 tuple.

    Args:
        value: Array-like with 3 elements

    Returns:
        Tuple of 3 floats
    """
    try:
        if len(value) != 3:
            raise ValueError(f"Vector must have exactly 3 components, got {len(value)}")
        # Return fixed-length tuple for type checking
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, AttributeError):
        raise ValueError(f"Value must be array-like with 3 elements, got {type(value)}") from None


def positive_vector(value: tuple[float, ...]) -> tuple[float, ...]:
    """
    Validate all vector components > 0.

    Args:
        value: Tuple of numeric values

    Returns:
        Same tuple if valid
    """
    for x in value:
        if not (x > 0):
            raise ValueError(f"All vector components must be positive (>0), got {x}")
    return value


# ============================================================================
# Vector3 constraint validators
# ============================================================================


def non_null_vector3(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate vector L2 norm > 0."""
    if not (math.hypot(*value) > 0):
        raise ValueError(f"Vector norm must be > 0 (non-null), got {value}")
    return value


# ============================================================================
# Vector2 validators
# ============================================================================


def vector2_shape(value: Any) -> tuple[float, float]:
    """Validate and convert to length-2 tuple."""
    try:
        if len(value) != 2:
            raise ValueError(f"Vector must have exactly 2 components, got {len(value)}")
        return (float(value[0]), float(value[1]))
    except (TypeError, AttributeError):
        raise ValueError(f"Value must be array-like with 2 elements, got {type(value)}") from None


def strictly_increasing(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate v[0] < v[1] (for 2-element range fields)."""
    if not (value[0] < value[1]):
        raise ValueError(f"Values must be strictly increasing (v[0] < v[1]), got {value}")
    return value


def positive_strictly_increasing(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate all elements > 0 AND v[0] < v[1]."""
    for x in value:
        if not (x > 0):
            raise ValueError(f"All values must be positive (>0), got {x}")
    if not (value[0] < value[1]):
        raise ValueError(f"Values must be strictly increasing (v[0] < v[1]), got {value}")
    return value


# ============================================================================
# Array validators
# ============================================================================


def array_shape(value: Any) -> tuple[float, ...]:
    """Validate input is non-empty iterable, convert to variable-length tuple of floats."""
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        raise ValueError(f"Value must be an iterable of numbers, got {type(value)}") from None
    if not result:
        raise ValueError("Array must be non-empty")
    return result


def vector3_array_shape(value: Any) -> Any:
    """Validate a non-empty (N, 3) array of 3-component vectors.

    Composed values always arrive here as a rectangular numpy array: unyt builds the
    array before any validator runs, and a ragged input fails there with numpy's own
    error. So the shape answers the question in one step, and reporting it beats a row
    index that could only ever be 0. Returns `value` unchanged -- the composer discards
    validator return values, so materializing the rows would be a throwaway copy.
    """
    shape = np.shape(value)
    if shape and shape[0] == 0:
        raise ValueError("Array of vectors must be non-empty")
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"Value must be an (N, 3) array of 3-component vectors, got shape {shape}")
    return value


def positive_array(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate all array elements > 0."""
    for x in value:
        if not (x > 0):
            raise ValueError(f"All array elements must be positive (>0), got {x}")
    return value


def non_negative_array(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate all array elements >= 0."""
    for x in value:
        if not (x >= 0):
            raise ValueError(f"All array elements must be non-negative (>=0), got {x}")
    return value


__all__ = [
    "positive",
    "non_negative",
    "finite_or_raise",
    "vector3_shape",
    "positive_vector",
    "vector2_shape",
    "non_null_vector3",
    "strictly_increasing",
    "positive_strictly_increasing",
    "array_shape",
    "vector3_array_shape",
    "positive_array",
    "non_negative_array",
]
