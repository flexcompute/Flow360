"""
This module provides functions for handling unit conversion into flow360 solver unit system.
"""

# pylint: disable=duplicate-code

import operator
from functools import reduce
from typing import List

from flow360.component.simulation.unit_system import (
    flow360_conversion_unit_system,
    is_flow360_unit,
    u,
)

from ...exceptions import Flow360ConfigurationError

LIQUID_IMAGINARY_FREESTREAM_MACH = 0.05


def get_from_dict_by_key_list(key_list, data_dict):
    """
    Get a value from a nested dictionary using a list of keys.

    Parameters
    ----------
    key_list : List[str]
        List of keys specifying the path to the desired value.
    data_dict : dict
        The dictionary from which to retrieve the value.

    Returns
    -------
    value
        The value located at the specified nested path in the dictionary.
    """

    return reduce(operator.getitem, key_list, data_dict)


def need_conversion(value):
    """
    Check if a value needs conversion to flow360 units.

    Parameters
    ----------
    value : Any
        The value to check for conversion.

    Returns
    -------
    bool
        True if conversion is needed, False otherwise.
    """

    if hasattr(value, "units"):
        return not is_flow360_unit(value)
    return False


def require(required_parameter, required_by, params):
    """
    Ensure that required parameters are present in the provided dictionary.

    Parameters
    ----------
    required_parameter : List[str]
        List of keys specifying the path to the desired parameter that is required.
    required_by : List[str]
        List of keys specifying the path to the parameter that requires required_parameter for unit conversion.
    params : SimulationParams
        The dictionary containing the parameters.

    Raises
    ------
    Flow360ConfigurationError
        Configuration error due to missing parameter.
    """

    required_msg = f'required by {" -> ".join(required_by)} for unit conversion'
    try:
        params_as_dict = params
        if not isinstance(params_as_dict, dict):
            params_as_dict = params.model_dump()
        value = get_from_dict_by_key_list(required_parameter, params_as_dict)
        if value is None:
            raise ValueError

    except Exception as err:
        raise Flow360ConfigurationError(
            message=f'{" -> ".join(required_parameter)} is {required_msg}.',
            field=required_by,
            dependency=required_parameter,
        ) from err

    if hasattr(value, "units") and str(value.units).startswith("flow360"):
        raise Flow360ConfigurationError(
            f'{" -> ".join(required_parameter + ["units"])} must be in physical units ({required_msg}).',
            field=required_by,
            dependency=required_parameter + ["units"],
        )


# pylint: disable=too-many-locals, too-many-return-statements, too-many-statements, too-many-branches
def unit_converter(dimension, params, required_by: List[str] = None) -> u.UnitSystem:
    """

    Returns a flow360 conversion unit system for a given dimension.

    Parameters
    ----------
    dimension : str
        The dimension for which the conversion unit system is needed. e.g., length
    length_unit : unyt_attribute
        Externally provided mesh unit or geometry unit.
    params : SimulationParams or dict
        The parameters needed for unit conversion.
    required_by : List[str], optional
        List of keys specifying the path to the parameter that requires this unit conversion, by default [].

    Returns
    -------
    flow360_conversion_unit_system
        The conversion unit system for the specified dimension. This unit system allows for
        .in_base(unit_system="flow360_v2") conversion.

    Raises
    ------
    ValueError
        The dimension is not recognized.
    """

    if required_by is None:
        required_by = []

    def get_base_length():
        require(["private_attribute_asset_cache", "project_length_unit"], required_by, params)
        base_length = params.private_attribute_asset_cache.project_length_unit.to("m").v.item()
        return base_length

    def get_base_temperature():
        if params.operating_condition.type_name == "LiquidOperatingCondition":
            # Temperature in this condition has no effect because the thermal features will be disabled.
            # Also the viscosity will be constant.
            # pylint:disable = no-member
            return 273 * u.K
        require(["operating_condition", "thermal_state", "temperature"], required_by, params)
        base_temperature = params.operating_condition.thermal_state.temperature.to("K").v.item()
        return base_temperature

    def get_base_velocity():
        if params.operating_condition.type_name == "LiquidOperatingCondition":
            # Provides an imaginary "speed of sound"
            # Resulting in a hardcoded freestream mach of `LIQUID_IMAGINARY_FREESTREAM_MACH`
            # To ensure incompressible range.
            if params.operating_condition.velocity_magnitude.value != 0:
                return (
                    params.operating_condition.velocity_magnitude / LIQUID_IMAGINARY_FREESTREAM_MACH
                ).to("m/s")
            return (
                params.operating_condition.reference_velocity_magnitude
                / LIQUID_IMAGINARY_FREESTREAM_MACH
            ).to("m/s")
        require(["operating_condition", "thermal_state", "temperature"], required_by, params)
        base_velocity = params.operating_condition.thermal_state.speed_of_sound.to("m/s").v.item()
        return base_velocity

    def get_base_time():
        base_length = get_base_length()
        base_velocity = get_base_velocity()
        base_time = base_length / base_velocity
        return base_time

    def get_base_angular_velocity():
        base_time = get_base_time()
        base_angular_velocity = 1 / base_time

        return base_angular_velocity

    def get_base_density():
        if params.operating_condition.type_name == "LiquidOperatingCondition":
            return params.operating_condition.material.density.to("kg/m**3")
        require(["operating_condition", "thermal_state", "density"], required_by, params)
        base_density = params.operating_condition.thermal_state.density.to("kg/m**3").v.item()

        return base_density

    def get_base_viscosity():
        base_density = get_base_density()
        base_length = get_base_length()
        base_velocity = get_base_velocity()
        base_viscosity = base_density * base_velocity * base_length

        return base_viscosity

    def get_base_kinematic_viscosity():
        base_length = get_base_length()
        base_time = get_base_time()
        base_kinematic_viscosity = base_length * base_length / base_time

        return base_kinematic_viscosity

    def get_base_force():
        base_length = get_base_length()
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_force = base_velocity**2 * base_density * base_length**2

        return base_force

    def get_base_moment():
        base_length = get_base_length()
        base_force = get_base_force()
        base_moment = base_force * base_length

        return base_moment

    def get_base_power():
        base_length = get_base_length()
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_power = base_velocity**3 * base_density * base_length**2

        return base_power

    def get_base_heat_flux():
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_heat_flux = base_density * base_velocity**3

        return base_heat_flux

    def get_base_heat_source():
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_length = get_base_length()

        base_heat_source = base_density * base_velocity**3 / base_length

        return base_heat_source

    def get_base_specific_heat_capacity():
        base_velocity = get_base_velocity()
        base_temperature = get_base_temperature()

        base_specific_heat_capacity = base_velocity**2 / base_temperature

        return base_specific_heat_capacity

    def get_base_thermal_conductivity():
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_temperature = get_base_temperature()
        base_length = get_base_length()

        base_thermal_conductivity = base_density * base_velocity**3 * base_length / base_temperature

        return base_thermal_conductivity

    if dimension == u.dimensions.length:
        base_length = get_base_length()
        flow360_conversion_unit_system.base_length = base_length

    elif dimension == u.dimensions.temperature:
        base_temperature = get_base_temperature()
        flow360_conversion_unit_system.base_temperature = base_temperature
        # Flow360 uses absolute temperature for scaling.
        # So the base_delta_temperature and base_temperature can have same scaling.
        flow360_conversion_unit_system.base_delta_temperature = base_temperature

    elif dimension == u.dimensions.area:
        base_length = get_base_length()
        flow360_conversion_unit_system.base_area = base_length**2

    elif dimension == u.dimensions.velocity:
        base_velocity = get_base_velocity()
        flow360_conversion_unit_system.base_velocity = base_velocity

    elif dimension == u.dimensions.time:
        base_time = get_base_time()
        flow360_conversion_unit_system.base_time = base_time

    elif dimension == u.dimensions.angular_velocity:
        base_angular_velocity = get_base_angular_velocity()
        flow360_conversion_unit_system.base_angular_velocity = base_angular_velocity

    elif dimension == u.dimensions.density:
        base_density = get_base_density()
        flow360_conversion_unit_system.base_density = base_density

    elif dimension == u.dimensions.viscosity:
        base_viscosity = get_base_viscosity()
        flow360_conversion_unit_system.base_viscosity = base_viscosity

    elif dimension == u.dimensions.kinematic_viscosity:
        base_kinematic_viscosity = get_base_kinematic_viscosity()
        flow360_conversion_unit_system.base_kinematic_viscosity = base_kinematic_viscosity

    elif dimension == u.dimensions.force:
        base_force = get_base_force()
        flow360_conversion_unit_system.base_force = base_force

    elif dimension == u.dimensions.moment:
        base_moment = get_base_moment()
        flow360_conversion_unit_system.base_moment = base_moment

    elif dimension == u.dimensions.power:
        base_power = get_base_power()
        flow360_conversion_unit_system.base_power = base_power

    elif dimension == u.dimensions.heat_flux:
        base_heat_flux = get_base_heat_flux()
        flow360_conversion_unit_system.base_heat_flux = base_heat_flux

    elif dimension == u.dimensions.specific_heat_capacity:
        base_specific_heat_capacity = get_base_specific_heat_capacity()
        flow360_conversion_unit_system.base_specific_heat_capacity = base_specific_heat_capacity

    elif dimension == u.dimensions.thermal_conductivity:
        base_thermal_conductivity = get_base_thermal_conductivity()
        flow360_conversion_unit_system.base_thermal_conductivity = base_thermal_conductivity

    elif dimension == u.dimensions.inverse_area:
        base_length = get_base_length()
        flow360_conversion_unit_system.base_inverse_area = 1 / base_length**2

    elif dimension == u.dimensions.inverse_length:
        base_length = get_base_length()
        flow360_conversion_unit_system.base_inverse_length = 1 / base_length

    elif dimension == u.dimensions.heat_source:
        base_heat_source = get_base_heat_source()
        flow360_conversion_unit_system.base_heat_source = base_heat_source

    elif dimension == u.dimensions.mass_flow_rate:
        base_density = get_base_density()
        base_length = get_base_length()
        base_time = get_base_time()

        flow360_conversion_unit_system.base_mass_flow_rate = (
            base_density * base_length**3 / base_time
        )

    elif dimension == u.dimensions.specific_energy:
        base_velocity = get_base_velocity()

        flow360_conversion_unit_system.base_specific_energy = base_velocity**2

    elif dimension == u.dimensions.frequency:
        base_time = get_base_time()

        flow360_conversion_unit_system.base_frequency = base_time ** (-1)

    elif dimension == u.dimensions.angle:

        # pylint: disable=no-member
        flow360_conversion_unit_system.base_angle = 1

    elif dimension == u.dimensions.pressure:
        base_force = get_base_force()
        base_length = get_base_length()
        flow360_conversion_unit_system.base_pressure = base_force / (base_length**2)

    else:
        raise ValueError(
            f"Unit converter: not recognized dimension: {dimension}. Conversion for this dimension is not implemented."
        )

    return flow360_conversion_unit_system.conversion_system
