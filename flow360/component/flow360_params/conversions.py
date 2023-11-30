"""
This module provides functions for handling unit conversion into flow360 solver unit system.
"""

import operator
from functools import reduce
from typing import Callable, List

import pydantic as pd

from ...exceptions import Flow360ConfigurationError
from .unit_system import flow360_conversion_unit_system, u


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
        return not str(value.units).startswith("flow360")
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
            f'{" -> ".join(required_parameter)} is {required_msg}.'
        ) from err

    if hasattr(value, "units") and str(value.units).startswith("flow360"):
        raise Flow360ConfigurationError(
            f'{" -> ".join(required_parameter)} must be in physical units ({required_msg}).'
        )


# pylint: disable=too-many-locals, too-many-return-statements
def unit_converter(dimension, params, required_by: List[str] = None):
    """
    Create a flow360 conversion unit system for a given dimension.

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
        .in_base(unit_system="flow360") conversion.

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

    flow360_conv_system = None

    if dimension == u.dimensions.length:
        base_length = get_base_length()
        flow360_conv_system = flow360_conversion_unit_system(base_length=base_length)

    if dimension == u.dimensions.temperature:
        base_temperature = get_base_temperature()
        flow360_conv_system = flow360_conversion_unit_system(base_temperature=base_temperature)

    if dimension == u.dimensions.area:
        base_length = get_base_length()
        flow360_conv_system = flow360_conversion_unit_system(base_area=base_length**2)

    if dimension == u.dimensions.velocity:
        base_velocity = get_base_velocity()
        flow360_conv_system = flow360_conversion_unit_system(base_velocity=base_velocity)

    if dimension == u.dimensions.time:
        base_time = get_base_time()
        flow360_conv_system = flow360_conversion_unit_system(base_time=base_time)

    if dimension == u.dimensions.angular_velocity:
        base_angular_velocity = get_base_angular_velocity()
        flow360_conv_system = flow360_conversion_unit_system(
            base_angular_velocity=base_angular_velocity
        )

    if dimension == u.dimensions.density:
        base_density = get_base_density()
        flow360_conv_system = flow360_conversion_unit_system(base_density=base_density)

    if dimension == u.dimensions.viscosity:
        base_viscosity = get_base_viscosity()
        flow360_conv_system = flow360_conversion_unit_system(base_viscosity=base_viscosity)

    if flow360_conv_system is not None:
        return flow360_conv_system

    raise ValueError(f"Not recognised dimension: {dimension}")