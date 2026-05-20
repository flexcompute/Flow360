"""
Unyt adapter — core workflow functions for converting values to/from unyt quantities.

For utility functions (type checks, unit resolution, etc.), see unyt_utils.py.
"""

from collections.abc import Mapping
from typing import Any

import unyt as u
from unyt import unyt_array

from .dimension_meta import PhysicalDimensionMeta as DimensionMeta
from .unyt_utils import (
    check_dimension,
    coalesce_unyt_quantity_list,
    dsl_to_unyt_unit,
    ensure_float64,
    get_si_unit,
    get_unit_from_unit_system,
    is_bare_numeric_array,
    is_numeric_scalar,
    is_unyt_quantity,
    is_unyt_unit,
    is_zero_value,
)


def parse_display_unit_dict(value: dict[str, Any], dimension_meta: DimensionMeta) -> Any:
    """Parse a display-unit dict ``{"value": ..., "display_unit"?: ...}`` into a
    unyt quantity whose ``.units`` is the user-chosen ``display_unit`` (if
    given) or the field's SI base unit (if not).

    The "value" key always carries the SI magnitude on the wire. When a
    ``display_unit`` is present we convert via ``.to(display_unit)`` so the
    resulting quantity natively reports the user's preferred unit through
    arithmetic; the serializer simply reads ``str(q.units)`` to round-trip.
    """
    extras = set(value) - {"value", "display_unit"}
    if extras:
        raise ValueError(f"Unexpected keys in display-unit dict: {sorted(extras)}")

    si_unit = get_si_unit(dimension_meta)
    raw = value["value"]
    if isinstance(raw, (list, tuple)):
        unyt_q = ensure_float64(unyt_array(raw, si_unit))
    else:
        unyt_q = ensure_float64(float(raw) * si_unit)

    display_unit_dsl = value.get("display_unit")
    if display_unit_dsl is not None:
        target_unit = u.Unit(dsl_to_unyt_unit(display_unit_dsl))
        if target_unit.dimensions != si_unit.dimensions:
            raise ValueError(
                f"display_unit '{display_unit_dsl}' has dimension {target_unit.dimensions} "
                f"but field '{dimension_meta.name}' expects {si_unit.dimensions}"
            )
        unyt_q = unyt_q.to(target_unit)

    return unyt_q


def to_unyt_scalar_with_fallback_info(
    value: float | Any,
    dimension_meta: DimensionMeta,
) -> tuple[Any, bool]:
    """Convert scalar to unyt and return whether SI fallback was used for a bare number."""
    # Bare number: attach unit from active unit system
    if is_numeric_scalar(value):
        if is_zero_value(value) and dimension_meta.allow_zero and not dimension_meta.unit_system_inference:
            unit = get_si_unit(dimension_meta)
            return ensure_float64(float(value) * unit), False
        unit, si_fallback = get_unit_from_unit_system(dimension_meta)
        return ensure_float64(value * unit), si_fallback

    # Raw unit (e.g. `u.m`) -> treat as `1 * unit`
    if is_unyt_unit(value):
        value = 1 * value

    if not is_unyt_quantity(value):
        raise ValueError(f"Unsupported input type for scalar field: {type(value).__name__}")
    if not check_dimension(value, dimension_meta):
        raise ValueError(
            f"Dimension mismatch: expected {dimension_meta.name} ({dimension_meta.si_unit}), "
            f"got {value.units.dimensions}"  # type: ignore[union-attr]
        )
    return ensure_float64(value), False


def to_unyt_array_with_fallback_info(
    value: Any,
    dimension_meta: DimensionMeta,
) -> tuple[Any, bool]:
    """Convert array to unyt and return whether SI fallback was used for a bare number."""

    # Bare numeric array: attach unit from active unit system
    if is_bare_numeric_array(value):
        if is_zero_value(value) and dimension_meta.allow_zero and not dimension_meta.unit_system_inference:
            return ensure_float64(unyt_array(value, get_si_unit(dimension_meta))), False
        unit, si_fallback = get_unit_from_unit_system(dimension_meta)
        return ensure_float64(unyt_array(value, unit)), si_fallback

    # List of unyt_quantity (e.g. from Expression evaluation) -> coalesce into unyt_array
    value = coalesce_unyt_quantity_list(value)

    # Raw unit not supported for arrays
    if is_unyt_unit(value):
        raise ValueError(
            "Raw unit input is only supported for scalar fields as legacy compatibility. "
            "For vector/array fields, pass explicit values with unit (e.g., [1, 2, 3] * u.m)."
        )

    if isinstance(value, Mapping):
        raise ValueError(
            f"Unsupported mapping input for array field: {type(value).__name__}. "
            "Pass a unyt_array, a list of unyt_quantity, or a bare numeric sequence "
            "(plus optional `display_unit` via the dimensioned-type validator)."
        )

    if not is_unyt_quantity(value):
        raise ValueError(f"Unsupported input type for array field: {type(value).__name__}")

    if not check_dimension(value, dimension_meta):
        raise ValueError(
            f"Dimension mismatch: expected {dimension_meta.name} ({dimension_meta.si_unit}), "
            f"got {value.units.dimensions}"
        )
    return ensure_float64(value), False


def from_unyt_scalar(value: Any, dimension_meta: DimensionMeta) -> float:
    """Extract numeric value from unyt quantity, converting to SI units."""
    if not is_unyt_quantity(value):
        raise TypeError(f"Expected unyt quantity, got {type(value).__name__}")

    si_unit = get_si_unit(dimension_meta)
    return float(value.to(si_unit).value)


def from_unyt_array(value: Any, dimension_meta: DimensionMeta) -> tuple[float, ...]:
    """Extract array from unyt array, converting to SI units."""
    if not is_unyt_quantity(value):
        raise TypeError(f"Expected unyt quantity, got {type(value).__name__}")

    si_unit = get_si_unit(dimension_meta)
    converted = value.to(si_unit)
    return tuple(float(x) for x in converted.value)


# Re-export utils that external code imports from this module
__all__ = [
    "dsl_to_unyt_unit",
    "get_si_unit",
    "is_unyt_quantity",
    "to_unyt_scalar_with_fallback_info",
    "to_unyt_array_with_fallback_info",
    "from_unyt_scalar",
    "from_unyt_array",
    "check_dimension",
]
