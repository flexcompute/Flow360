"""
This module is for accessing units and unit systems including flow360 unit system.
"""

# pylint: disable=unused-import


import unyt
from unyt import unit_symbols

from .component.flow360_params.unit_system import (
    BaseSystemType,
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    flow360_angular_velocity_unit,
    flow360_area_unit,
    flow360_density_unit,
    flow360_force_unit,
    flow360_length_unit,
    flow360_mass_unit,
    flow360_pressure_unit,
    flow360_temperature_unit,
    flow360_time_unit,
    flow360_unit_system,
    flow360_velocity_unit,
    flow360_viscosity_unit,
    imperial_unit_system,
)


def import_units(module, namespace):
    """Import Unit objects from a module into a namespace"""
    for key, value in module.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            namespace[key] = value


import_units(unit_symbols, globals())
del import_units
