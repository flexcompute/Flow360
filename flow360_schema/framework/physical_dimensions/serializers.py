"""
Serialization utilities

Helper functions for serializing unyt objects to plain JSON values.
"""

from typing import Any

from .dimension_meta import PhysicalDimensionMeta
from .unyt_adapter import from_unyt_array, from_unyt_scalar, is_unyt_quantity


def serialize_scalar(value: Any, dimension_meta: PhysicalDimensionMeta) -> float:
    """
    Serialize scalar value to plain float.

    Args:
        value: unyt quantity or plain number
        dimension_meta: Dimension metadata with SI unit info

    Returns:
        Float value in SI units
    """
    if is_unyt_quantity(value):
        return from_unyt_scalar(value, dimension_meta)
    # Python float is always IEEE 754 double (64-bit) on all platforms.
    # This provides ~15-17 significant digits, sufficient for CFD parameters.
    return float(value)


def serialize_vector3(value: Any, dimension_meta: PhysicalDimensionMeta) -> tuple[float, ...]:
    """
    Serialize Vector3 to plain tuple.

    Args:
        value: unyt array or tuple
        dimension_meta: Dimension metadata with SI unit info

    Returns:
        Tuple of 3 floats in SI units
    """
    if is_unyt_quantity(value):
        return tuple(from_unyt_array(value, dimension_meta))
    # Python float is always IEEE 754 double (64-bit) on all platforms.
    return tuple(float(x) for x in value)


def serialize_raw_scalar(value: Any) -> float:
    """Serialize scalar by extracting .value without unit conversion."""
    if is_unyt_quantity(value):
        return float(value.value)
    return float(value)


def serialize_raw_vector3(value: Any) -> tuple[float, ...]:
    """Serialize vector3 by extracting .value without unit conversion."""
    if is_unyt_quantity(value):
        return tuple(float(x) for x in value.value)
    return tuple(float(x) for x in value)


# ============================================================================
# Vector2 serializers
# ============================================================================


def serialize_vector2(value: Any, dimension_meta: PhysicalDimensionMeta) -> tuple[float, float]:
    """Serialize Vector2 to plain 2-tuple in SI units."""
    if is_unyt_quantity(value):
        arr = from_unyt_array(value, dimension_meta)
        return (float(arr[0]), float(arr[1]))
    return (float(value[0]), float(value[1]))


def serialize_raw_vector2(value: Any) -> tuple[float, float]:
    """Serialize vector2 by extracting .value without unit conversion."""
    raw = value.value if is_unyt_quantity(value) else value
    return (float(raw[0]), float(raw[1]))


# ============================================================================
# Array serializers
# ============================================================================


def serialize_array(value: Any, dimension_meta: PhysicalDimensionMeta) -> list[float]:
    """Serialize variable-length unyt_array to list[float] in SI units."""
    if is_unyt_quantity(value):
        return list(from_unyt_array(value, dimension_meta))
    return [float(x) for x in value]


def serialize_raw_array(value: Any) -> list[float]:
    """Serialize array by extracting .value without unit conversion."""
    if is_unyt_quantity(value):
        return [float(x) for x in value.value]
    return [float(x) for x in value]


__all__ = [
    "serialize_scalar",
    "serialize_vector3",
    "serialize_raw_scalar",
    "serialize_raw_vector3",
    "serialize_vector2",
    "serialize_raw_vector2",
    "serialize_array",
    "serialize_raw_array",
]
