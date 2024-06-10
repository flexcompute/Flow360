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


def get_attribute_from_first_instance(
    obj_list: list, class_type, attribute_name: str, check_empty_entities: bool = False
):
    """In a list loop and find the first instance matching the given type and retrive the attribute"""
    for obj in obj_list:
        if isinstance(obj, class_type):
            if check_empty_entities and obj.entities is not None:
                continue
            return getattr(obj, attribute_name)
    return None


def translate_setting_and_apply_to_all_entities(
    obj_list: list,
    class_type,
    translation_func,
    to_list: bool = False,
    entity_injection_func=lambda x: x,
):
    """Translate settings and apply them to all entities of a given type.

    Args:
        obj_list (list): A list of objects to loop through.
        class_type: The type of objects to match.
        translation_func (str): The function to use for translation. This function should return a dictionary.
        to_list (bool, optional): Whether the return is a list which does not differentiate entity name or a
        dict (default).

    Returns:
        dict: A dictionary containing the translated settings applied to all entities.

    """
    if not to_list:
        output = {}
    else:
        output = []

    for obj in obj_list:
        if isinstance(obj, class_type):
            translated_setting = translation_func(obj)
            for entity in obj.entities.stored_entities:
                if not to_list:
                    if output.get(entity.name) is None:
                        output[entity.name] = {}
                    output[entity.name].update(translated_setting)
                else:
                    setting = entity_injection_func(entity)
                    setting.update(translated_setting)
                    output.append(setting)
    return output
