from functools import reduce
import operator
from .unit_system import flow360_conversion_unit_system, u

def get_from_dict_by_key_list(key_list, data_dict):
    return reduce(operator.getitem, key_list, data_dict)


def need_conversion(value):
    if hasattr(value, 'units'):
        return value.units.registry.unit_system != 'flow360'
    return False


def require(required_parameters, required_by, params):
    try:
        value = get_from_dict_by_key_list(required_parameters, params.dict())
        if value is None:
            raise ValueError
    except Exception:
        print(f'{" -> ".join(required_parameters)} is required by {" -> ".join(required_by)}')
        print(f'but {params.dict()} found.')
        raise



def unit_converter(dimension, required_by, params):

    def get_base_length():
        require(['geometry', 'mesh_unit'], required_by, params)
        base_length = params.geometry.mesh_unit.to('m').v.item()
        return base_length
    
    def get_base_velocity():
        require(['fluid_properties'], required_by, params)
        base_length = params.fluid_properties.speed_of_sound().to('m/s').v.item()
        return base_length


    def get_base_time():
        base_length = get_base_length()
        base_velocity = get_base_velocity()
        base_time = base_length / base_velocity
        return base_time


    if dimension == u.dimensions.length:
        base_length = get_base_length()
        flow360_conv_system = flow360_conversion_unit_system(base_length=base_length)

        return flow360_conv_system


    if dimension == u.dimensions.velocity:
        base_velocity = get_base_velocity()
        flow360_conv_system = flow360_conversion_unit_system(base_velocity=base_velocity)

        return flow360_conv_system


    if dimension == u.dimensions.time:
        base_time = get_base_time()
        flow360_conv_system = flow360_conversion_unit_system(base_time=base_time)

        return flow360_conv_system


    else:
        raise ValueError(f'Not recognised dimension: {dimension}')
