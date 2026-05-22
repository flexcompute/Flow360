"""
Wire format: ``{value: <numeric in `units`>, units: <unyt expression string>}``.

Matches the 25.9 standard byte-for-byte:

- ``value`` holds the raw numeric in whatever unit the user (or upstream
  pipeline) supplied; it is NOT pre-converted to SI.
- ``units`` carries the raw unyt expression (``str(q.units.expr)``), e.g.
  ``m**2``. No ``^``-DSL substitution is performed.

Both keys are required on the wire. Missing ``units`` is a hard error.
"""

from __future__ import annotations

from typing import Any

import unyt as u
from unyt import unyt_array

from .dimension_meta import PhysicalDimensionMeta as DimensionMeta
from .schema_constants import SCHEMA_BASE_URL
from .unyt_utils import ensure_float64, get_si_unit


def is_format_dict(value: Any) -> bool:
    """Detect the ``{value, units}`` wire shape."""
    return isinstance(value, dict) and "value" in value and "units" in value


def parse_format_dict(value: dict[str, Any], dimension_meta: DimensionMeta) -> Any:
    """Parse a ``{value, units}`` dict into a unyt quantity.

    The ``value`` magnitude is interpreted in ``units`` directly (no SI
    pre-conversion). Missing ``units`` is rejected loudly.
    """
    extras = set(value) - {"value", "units"}
    if extras:
        raise ValueError(f"Unexpected keys in units-format dict: {sorted(extras)}")
    if "units" not in value:
        raise ValueError("Missing required 'units' key in units-format dict")

    target_unit = u.Unit(value["units"])
    si_unit = get_si_unit(dimension_meta)
    if target_unit.dimensions != si_unit.dimensions:
        raise ValueError(
            f"units '{value['units']}' has dimension {target_unit.dimensions} "
            f"but field '{dimension_meta.name}' expects {si_unit.dimensions}"
        )

    raw = value["value"]
    if isinstance(raw, (list, tuple)):
        return ensure_float64(unyt_array(raw, target_unit))
    return ensure_float64(float(raw) * target_unit)


def serialize_to_dict(
    unyt_value: Any,
    data_type: Any,
    dimension_meta: DimensionMeta,  # noqa: ARG001 — kept for contract symmetry
) -> dict[str, Any]:
    """Emit ``{value: <raw>, units: <unyt expr>}`` for a validated unyt quantity."""
    return {
        "value": data_type.serialize_raw(unyt_value),
        "units": str(unyt_value.units.expr),
    }


def generate_schema(schema_type_name: str, si_unit: str) -> dict[str, Any]:  # noqa: ARG001
    """JSON Schema for a scalar / vector ``{value, units}`` field.

    ``si_unit`` is unused for this wire format: the runtime ``units`` key
    carries the user's unit per record, so the static schema has no need to
    annotate the SI unit on the inner primitive.
    """
    return {
        "type": "object",
        "properties": {
            "value": {"$ref": f"{SCHEMA_BASE_URL}/{schema_type_name}.json"},
            "units": {"type": "string"},
        },
        "required": ["value", "units"],
        "additionalProperties": False,
    }


def generate_array_schema(element_schema_type_name: str, si_unit: str) -> dict[str, Any]:  # noqa: ARG001
    """JSON Schema for a variable-length array ``{value, units}`` field."""
    return {
        "type": "object",
        "properties": {
            "value": {
                "type": "array",
                "items": {"$ref": f"{SCHEMA_BASE_URL}/{element_schema_type_name}.json"},
            },
            "units": {"type": "string"},
        },
        "required": ["value", "units"],
        "additionalProperties": False,
    }


__all__ = [
    "is_format_dict",
    "parse_format_dict",
    "serialize_to_dict",
    "generate_schema",
    "generate_array_schema",
]
