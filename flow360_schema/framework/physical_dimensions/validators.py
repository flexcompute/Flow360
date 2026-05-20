"""
Composable validation functions

Each function implements a single validation criterion and can be composed
to create complex validation logic. All validators are pure Python with no
external dependencies.
"""

import math
from typing import Any


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
    "vector3_shape",
    "positive_vector",
    "vector2_shape",
    "non_null_vector3",
    "strictly_increasing",
    "positive_strictly_increasing",
    "array_shape",
    "positive_array",
    "non_negative_array",
]
