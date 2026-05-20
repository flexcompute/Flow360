"""
JSON Schema generators

Functions to generate JSON Schema definitions using $ref to common-schema types.
These functions do NOT depend on unyt or any external validation libraries.
"""

from typing import Any

from .schema_constants import SCHEMA_BASE_URL


def wrap_with_display_unit(inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap an SI value schema in the physical-dimension wire format.

    The wire shape is ``{"value": <inner>, "display_unit"?: <DSL string>}``.
    """
    return {
        "type": "object",
        "properties": {
            "value": inner,
            "display_unit": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }


def generate_schema(schema_type_name: str, si_unit: str) -> dict[str, Any]:
    """
    Generate JSON Schema for a scalar / vector physical-dimension field.

    Args:
        schema_type_name: Common-schema type name (e.g., "Float64", "Vector3Json")
        si_unit: SI unit string (e.g., "m", "m/s", "Pa")

    Returns:
        JSON Schema dict in display-unit wrapped form: an object with a
        ``value`` property holding the SI primitive ($ref + $units) and an
        optional ``display_unit`` string property.
    """
    return wrap_with_display_unit(
        {
            "$ref": f"{SCHEMA_BASE_URL}/{schema_type_name}.json",
            "$units": si_unit,
        }
    )


def generate_array_schema(element_schema_type_name: str, si_unit: str) -> dict[str, Any]:
    """
    Generate JSON Schema for a variable-length array physical-dimension field.

    Args:
        element_schema_type_name: Element's common-schema type name (e.g., "Float64")
        si_unit: SI unit string (e.g., "m", "Pa")

    Returns:
        JSON Schema dict in display-unit wrapped form: an object with a
        ``value`` property holding the array (items referencing the element
        primitive with $units) and an optional ``display_unit`` string.
    """
    return wrap_with_display_unit(
        {
            "type": "array",
            "items": {
                "$ref": f"{SCHEMA_BASE_URL}/{element_schema_type_name}.json",
                "$units": si_unit,
            },
        }
    )


__all__ = [
    "SCHEMA_BASE_URL",
    "generate_schema",
    "generate_array_schema",
    "wrap_with_display_unit",
]
