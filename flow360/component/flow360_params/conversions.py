import operator
from functools import reduce
from typing import Callable, List

import pydantic as pd

from ...exceptions import Flow360ConfigurationError
from .unit_system import flow360_conversion_unit_system, u


class ExtraDimensionedProperty(pd.BaseModel):
    name: str
    dependency_list: List[str]
    value_factory: Callable


def get_from_dict_by_key_list(key_list, data_dict):
    return reduce(operator.getitem, key_list, data_dict)


def need_conversion(value):
    if hasattr(value, "units"):
        return not str(value.units).startswith("flow360")
    return False


def require(required_parameters, required_by, params):
    required_msg = f'required by {" -> ".join(required_by)} for unit conversion'
    try:
        value = get_from_dict_by_key_list(required_parameters, params.dict())
        if value is None:
            raise ValueError

    except Exception as err:
        raise Flow360ConfigurationError(
            f'{" -> ".join(required_parameters)} is {required_msg}.'
        ) from err

    if hasattr(value, "units") and str(value.units).startswith("flow360"):
        raise Flow360ConfigurationError(
            f'{" -> ".join(required_parameters)} must be in physical units ({required_msg}).'
        )


def unit_converter(dimension, params, required_by: List[str] = []):
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

    if dimension == u.dimensions.length:
        base_length = get_base_length()
        flow360_conv_system = flow360_conversion_unit_system(base_length=base_length)

        return flow360_conv_system

    if dimension == u.dimensions.temperature:
        base_temperature = get_base_temperature()
        flow360_conv_system = flow360_conversion_unit_system(base_temperature=base_temperature)

        return flow360_conv_system

    if dimension == u.dimensions.area:
        base_length = get_base_length()
        flow360_conv_system = flow360_conversion_unit_system(base_area=base_length**2)

        return flow360_conv_system

    if dimension == u.dimensions.velocity:
        base_velocity = get_base_velocity()
        flow360_conv_system = flow360_conversion_unit_system(base_velocity=base_velocity)

        return flow360_conv_system

    if dimension == u.dimensions.time:
        base_time = get_base_time()
        flow360_conv_system = flow360_conversion_unit_system(base_time=base_time)

        return flow360_conv_system

    if dimension == u.dimensions.angular_velocity:
        base_angular_velocity = get_base_angular_velocity()
        flow360_conv_system = flow360_conversion_unit_system(
            base_angular_velocity=base_angular_velocity
        )

        return flow360_conv_system

    if dimension == u.dimensions.density:
        base_density = get_base_density()
        flow360_conv_system = flow360_conversion_unit_system(base_density=base_density)

        return flow360_conv_system

    if dimension == u.dimensions.viscosity:
        base_viscosity = get_base_viscosity()
        flow360_conv_system = flow360_conversion_unit_system(base_viscosity=base_viscosity)

        return flow360_conv_system

    else:
        raise ValueError(f"Not recognised dimension: {dimension}")
