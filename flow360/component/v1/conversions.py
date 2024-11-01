"""
This module provides functions for handling unit conversion into flow360 solver unit system.
"""

import operator
from functools import reduce
from typing import Callable, List

import pydantic.v1 as pd

from flow360.component.v1.unit_system import (
    flow360_conversion_unit_system,
    is_flow360_unit,
    u,
)
from flow360.exceptions import Flow360ConfigurationError


class ExtraDimensionedProperty(pd.BaseModel):
    """
    Pydantic model representing an extra dimensioned property.

    Parameters
    ----------
    name : str
        The name of the property.
    dependency_list : List[str]
        List of dependencies for the property.
    value_factory : Callable
        A callable function used to calculate the value of the property.
    """

    name: str
    dependency_list: List[str]
    value_factory: Callable


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
    params : Flow360Params
        The dictionary containing the parameters.

    Raises
    ------
    Flow360ConfigurationError
        Configuration error due to missing parameter.
    """

    required_msg = f'required by {" -> ".join(required_by)} for unit conversion'
    try:
        value = get_from_dict_by_key_list(required_parameter, params.dict())
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
def unit_converter(dimension, params, required_by: List[str] = None):
    """

    Returns a flow360 conversion unit system for a given dimension.

    Parameters
    ----------
    dimension : str
        The dimension for which the conversion unit system is needed. e.g., length
    params : Flow360Params
        The parameters needed for unit conversion.
    required_by : List[str], optional
        List of keys specifying the path to the parameter that requires this unit conversion, by default [].

    Returns
    -------
    flow360_conversion_unit_system
        The conversion unit system for the specified dimension. This unit system allows for
        .in_base(unit_system="flow360_v1") conversion.

    Raises
    ------
    ValueError
        The dimension is not recognized.
    """

    if required_by is None:
        required_by = []

    def get_base_length():
        require(["geometry", "mesh_unit"], required_by, params)
        base_length = params.geometry.mesh_unit.to("m").v.item()
        return base_length

    def get_base_angle():
        # pylint: disable=no-member
        return 1 * u.rad

    def get_base_temperature():
        require(["fluid_properties"], required_by, params)
        base_temperature = (
            params.fluid_properties.to_fluid_properties().temperature.to("K").v.item()
        )
        return base_temperature

    def get_base_velocity():
        require(["fluid_properties"], required_by, params)
        base_velocity = params.fluid_properties.speed_of_sound().to("m/s").v.item()
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
        require(["fluid_properties"], required_by, params)
        base_density = params.fluid_properties.to_fluid_properties().density.to("kg/m**3").v.item()

        return base_density

    def get_base_viscosity():
        base_density = get_base_density()
        base_length = get_base_length()
        base_velocity = get_base_velocity()
        base_viscosity = base_density * base_velocity * base_length

        return base_viscosity

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

    def get_base_heat_capacity():
        base_density = get_base_density()
        base_velocity = get_base_velocity()
        base_temperature = get_base_temperature()

        base_heat_capacity = base_density * base_velocity**2 / base_temperature

        return base_heat_capacity

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

    elif dimension == u.dimensions.angle:
        base_angle = get_base_angle()
        flow360_conversion_unit_system.base_angle = base_angle

    elif dimension == u.dimensions.temperature:
        base_temperature = get_base_temperature()
        flow360_conversion_unit_system.base_temperature = base_temperature

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

    elif dimension == u.dimensions.heat_capacity:
        base_heat_capacity = get_base_heat_capacity()
        flow360_conversion_unit_system.base_heat_capacity = base_heat_capacity

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

    else:
        raise ValueError(
            f"Unit converter: not recognised dimension: {dimension}. Conversion for this dimension is not implemented."
        )

    return flow360_conversion_unit_system.conversion_system
