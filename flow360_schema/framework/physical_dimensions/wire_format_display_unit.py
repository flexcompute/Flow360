"""
Wire format: ``{value: <SI numeric>, display_unit?: <DSL string>}``.

Dormant. Retained as a deletable fallback module; the active wire format is
selected by which ``wire_format_*`` module ``composers.py``,
``data_types.py``, and ``expression/value_or_expression.py`` import from.
"""

from __future__ import annotations

from typing import Any

import unyt as u
from unyt import unyt_array

from .dimension_meta import PhysicalDimensionMeta as DimensionMeta
from .schema_constants import SCHEMA_BASE_URL
from .unyt_utils import (
    dsl_to_unyt_unit,
    ensure_float64,
    get_si_unit,
    is_unyt_quantity,
    units_semantically_equivalent,
    unyt_to_dsl_unit,
)


def is_format_dict(value: Any) -> bool:
    """Detect a ``{value, display_unit?}`` dict.

    Lenient: any dict with a ``value`` key and no ``units`` key counts, so
    that :func:`parse_format_dict` can produce a precise error for malformed
    inputs (extras, wrong dimension, etc.).
    """
    return isinstance(value, dict) and "value" in value and "units" not in value


def parse_format_dict(value: dict[str, Any], dimension_meta: DimensionMeta) -> Any:
    """Parse a ``{value, display_unit?}`` dict into a unyt quantity whose
    ``.units`` is the user-chosen ``display_unit`` (if given) or the field's
    SI base unit (if not).

    The ``value`` key always carries the SI magnitude on the wire. When a
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


def _display_unit_for(unyt_value: Any, dimension_meta: DimensionMeta) -> str | None:
    """Return the DSL display-unit string for ``unyt_value`` (or ``None`` if its
    units are semantically the SI base unit so the caller can omit
    ``display_unit`` from the wire format).
    """
    if not is_unyt_quantity(unyt_value):
        return None

    current_unit = unyt_value.units
    si_unit = get_si_unit(dimension_meta)
    if units_semantically_equivalent(current_unit, si_unit):
        return None
    return unyt_to_dsl_unit(str(current_unit))


def serialize_to_dict(
    unyt_value: Any,
    data_type: Any,
    dimension_meta: DimensionMeta,
) -> dict[str, Any]:
    """Emit the ``{value, display_unit?}`` wire dict for a validated unyt quantity."""
    si_value = data_type.serialize_si(unyt_value, dimension_meta)
    display_unit = _display_unit_for(unyt_value, dimension_meta)
    out: dict[str, Any] = {"value": si_value}
    if display_unit is not None:
        out["display_unit"] = display_unit
    return out


def generate_schema(schema_type_name: str, si_unit: str) -> dict[str, Any]:
    """JSON Schema for a scalar / vector ``{value, display_unit?}`` field.

    Since ``value`` is the SI canonical magnitude in this wire format, the
    inner primitive carries a ``$units`` annotation so downstream consumers
    know the SI unit when ``display_unit`` is omitted.
    """
    return {
        "type": "object",
        "properties": {
            "value": {
                "$ref": f"{SCHEMA_BASE_URL}/{schema_type_name}.json",
                "$units": si_unit,
            },
            "display_unit": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }


def generate_array_schema(element_schema_type_name: str, si_unit: str) -> dict[str, Any]:
    """JSON Schema for a variable-length array ``{value, display_unit?}`` field."""
    return {
        "type": "object",
        "properties": {
            "value": {
                "type": "array",
                "items": {
                    "$ref": f"{SCHEMA_BASE_URL}/{element_schema_type_name}.json",
                    "$units": si_unit,
                },
            },
            "display_unit": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }


__all__ = [
    "is_format_dict",
    "parse_format_dict",
    "serialize_to_dict",
    "generate_schema",
    "generate_array_schema",
    "_display_unit_for",
]
