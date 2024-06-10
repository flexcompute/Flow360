from __future__ import annotations

import functools
import json

from flow360.component.simulation.simulation_params import (
    SimulationParams,  # Not required
)


def preprocess_input(func):
    @functools.wraps(func)
    def wrapper(input_params, mesh_unit, *args, **kwargs):
        processed_input = get_simulation_param_dict(input_params, mesh_unit)
        return func(processed_input, mesh_unit, *args, **kwargs)

    return wrapper


def get_simulation_param_dict(input_params: SimulationParams | str | dict, mesh_unit):
    """
    Get the dictionary of `SimulationParams`.
    """
    param_dict = None
    param = None
    if isinstance(input_params, SimulationParams):
        param = input_params
    elif isinstance(input_params, str):
        try:
            # If input is a JSON string
            param_dict = json.loads(input_params)
        except json.JSONDecodeError:
            # If input is a file path
            with open(input_params, "r") as file:
                param_dict = json.load(file)
        if param_dict is None:
            raise ValueError(f"Invalid input <{input_params}> for translator. ")
        param = SimulationParams(**param_dict)
    elif isinstance(input_params, dict):
        param = SimulationParams(**input_params)

    if param is not None:
        return param.preprocess(mesh_unit)
    raise ValueError(f"Invalid input <{input_params.__class__.__name__}> for translator. ")


def get_attribute_from_first_instance(obj_list: list, class_type, attribute_name: str):
    """In a list loop and find the first instance matching the given type and retrive the attribute"""
    for obj in obj_list:
        if isinstance(obj, class_type):
            return getattr(obj, attribute_name)
    return None


def translate_setting_and_apply_to_all_entities(obj_list: list, class_type, translation_func: str):
    """In a list loop and find the all instances matching the given type and apply translation.
    `translation_func` shoud return a dictionary."""
    output_dict = {}
    for obj in obj_list:
        if isinstance(obj, class_type):
            translated_setting = translation_func(obj)
            for entity in obj.entities.stored_entities:
                if output_dict.get(entity.name) is None:
                    output_dict[entity.name] = {}
                output_dict[entity.name].update(translated_setting)
    return output_dict
