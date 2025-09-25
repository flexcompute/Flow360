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
        base_length = params.base_length.v.item()
        return base_length

    def get_base_temperature():
        if params.operating_condition.type_name != "LiquidOperatingCondition":
            # Temperature in Liquid condition has no effect because the thermal features will be disabled.
            # Also the viscosity will be constant.
            # pylint:disable = no-member
            require(["operating_condition", "thermal_state", "temperature"], required_by, params)
        base_temperature = params.base_temperature.v.item()
        return base_temperature

    def get_base_velocity():
        if params.operating_condition.type_name != "LiquidOperatingCondition":
            require(["operating_condition", "thermal_state", "temperature"], required_by, params)
        base_velocity = params.base_velocity.v.item()
        return base_velocity

    def get_base_time():
        base_time = params.base_time.v.item()
        return base_time

    def get_base_mass():
        base_mass = params.base_mass.v.item()
        return base_mass

    def get_base_angular_velocity():
        base_time = get_base_time()
        base_angular_velocity = 1 / base_time

        return base_angular_velocity

    def get_base_density():
        if params.operating_condition.type_name != "LiquidOperatingCondition":
            require(["operating_condition", "thermal_state", "density"], required_by, params)
        base_density = params.base_density.v.item()
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

    elif dimension == u.dimensions.mass:
        base_mass = get_base_mass()
        flow360_conversion_unit_system.base_mass = base_mass

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


def get_flow360_unit_system_liquid(params, to_flow360_unit: bool = False) -> u.UnitSystem:
    """
    Returns the flow360 unit system when liquid operating condition is used.

    Parameters
    ----------
    params : SimulationParams
        The parameters needed for unit conversion.
    to_flow360_unit : bool, optional
        If True, return the flow360 unit system.

    Returns
    -------
    u.UnitSystem
        The flow360 unit system.

    ##-- When to_flow360_unit is True,
    ##-- time unit should be changed such that it takes into consideration
    ##-- the fact that solver output already multiplied by "velocityScale"
    """

    if to_flow360_unit:
        base_velocity = params.base_velocity
    else:
        # For dimensionalization of Flow360 output
        # The solver output is already re-normalized by `reference velocity` due to "velocityScale"
        # So we need to find the `reference velocity`.
        # `reference_velocity_magnitude` takes precedence, consistent with how "velocityScale" is computed.
        if params.operating_condition.reference_velocity_magnitude is not None:
            base_velocity = (params.operating_condition.reference_velocity_magnitude).to("m/s")
        else:
            base_velocity = params.base_velocity.to("m/s") * LIQUID_IMAGINARY_FREESTREAM_MACH

    time_unit = params.base_length / base_velocity
    return u.UnitSystem(
        name="flow360_liquid",
        length_unit=params.base_length,
        mass_unit=params.base_mass,
        time_unit=time_unit,
        temperature_unit=params.base_temperature,
    )


def compute_udf_dimensionalization_factor(params, requested_unit, using_liquid_op):
    """

    Returns the dimensionalization coefficient and factor given a requested unit

    Parameters
    ----------
    params : SimulationParams
        The parameters needed for unit conversion.
    unit: u.Unit
        The unit to compute the factors.
    using_liquid_op : bool
        If True, compute the factor based on the flow360_liquid unit system.
    Returns
    -------
    coefficient and offset for unit conversion from the requested unit to flow360 unit

    """

    def _compute_coefficient_and_offset(source_unit: u.Unit, target_unit: u.Unit):
        y2 = (2.0 * target_unit).in_units(source_unit).value
        y1 = (1.0 * target_unit).in_units(source_unit).value
        x2 = 2.0
        x1 = 1.0

        coefficient = (y2 - y1) / (x2 - x1)
        offset = y1 / coefficient - x1

        return coefficient, offset

    flow360_unit_system = (
        params.flow360_unit_system
        if not using_liquid_op
        else get_flow360_unit_system_liquid(params=params)
    )
    # Note: Effectively assuming that all the solver vars uses radians and also the expressions expect radians
    flow360_unit_system["angle"] = u.rad  # pylint:disable=no-member
    flow360_unit = flow360_unit_system[requested_unit.dimensions]
    coefficient, offset = _compute_coefficient_and_offset(
        source_unit=requested_unit, target_unit=flow360_unit
    )
    return coefficient, offset
