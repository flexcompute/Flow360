"""
This module is for accessing units and unit systems including flow360 unit system.
"""

import unyt
from unyt import unit_symbols

from flow360.component.simulation.unit_system import (
    BaseSystemType,
    CGS_unit_system,
    SI_unit_system,
    UnitSystem,
    flow360_angle_unit,
    flow360_angular_velocity_unit,
    flow360_area_unit,
    flow360_density_unit,
    flow360_force_unit,
    flow360_frequency_unit,
    flow360_kinematic_viscosity_unit,
    flow360_length_unit,
    flow360_mass_flow_rate_unit,
    flow360_mass_unit,
    flow360_pressure_unit,
    flow360_specific_energy_unit,
    flow360_temperature_unit,
    flow360_time_unit,
    flow360_unit_system,
    flow360_velocity_unit,
    flow360_viscosity_unit,
    imperial_unit_system,
)

# pylint: disable=duplicate-code
__all__ = [
    "BaseSystemType",
    "CGS_unit_system",
    "SI_unit_system",
    "UnitSystem",
    "flow360_angular_velocity_unit",
    "flow360_area_unit",
    "flow360_density_unit",
    "flow360_force_unit",
    "flow360_length_unit",
    "flow360_angle_unit",
    "flow360_mass_unit",
    "flow360_pressure_unit",
    "flow360_temperature_unit",
    "flow360_time_unit",
    "flow360_unit_system",
    "flow360_velocity_unit",
    "flow360_viscosity_unit",
    "flow360_kinematic_viscosity_unit",
    "imperial_unit_system",
    "flow360_mass_flow_rate_unit",
    "flow360_specific_energy_unit",
    "flow360_frequency_unit",
]


def import_units(module, namespace):
    """Import Unit objects from a module into a namespace"""
    for key, value in module.__dict__.items():
        if isinstance(value, (unyt.unyt_quantity, unyt.Unit)):
            namespace[key] = value


import_units(unit_symbols, globals())
del import_units
