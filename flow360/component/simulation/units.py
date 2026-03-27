"""
This module is for accessing units and unit systems including flow360 unit system.
"""

import functools

import unyt
from unyt import unit_symbols

from flow360.component.simulation.unit_system import (
    BaseSystemType,
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    imperial_unit_system,
)

# pylint: disable=duplicate-code
__all__ = [
    "BaseSystemType",
    "CGS_unit_system",
    "SI_unit_system",
    "UnitSystem",
    "imperial_unit_system",
]


def import_units(module, namespace):
    """Import Unit objects from a module into a namespace"""
    for key, value in module.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            namespace[key] = value


import_units(unit_symbols, globals())
del import_units


@functools.lru_cache(maxsize=1)
def _get_length_adapter():
    """Lazily build and cache TypeAdapter(Length.Float64) to avoid import-time cost."""
    from flow360_schema.framework.physical_dimensions import (  # pylint: disable=import-outside-toplevel
        Length,
    )
    from pydantic import TypeAdapter  # pylint: disable=import-outside-toplevel

    return TypeAdapter(Length.Float64)


def validate_length(value):
    """Validate a value as Length.Float64 using a cached TypeAdapter.

    Replacement for the old LengthType.validate() pattern.
    Accepts unyt quantities, dicts, and bare numbers (interpreted as SI meters).
    For backward compatibility, plain unit strings are interpreted as 1 * unit.
    """
    if isinstance(value, str):
        value = 1 * unyt.Unit(value)
    return _get_length_adapter().validate_python(value)
