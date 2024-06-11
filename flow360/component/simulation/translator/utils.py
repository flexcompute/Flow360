from __future__ import annotations

import functools
import json
from collections import OrderedDict

from flow360.component.simulation.simulation_params import (
    SimulationParams,  # Not required
)
from flow360.component.simulation.unit_system import LengthType


def preprocess_input(func):
    @functools.wraps(func)
    def wrapper(input_params, mesh_unit, *args, **kwargs):
        validated_mesh_unit = LengthType.validate(mesh_unit)
        processed_input = get_simulation_param_dict(input_params, validated_mesh_unit)
        return func(processed_input, validated_mesh_unit, *args, **kwargs)

    return wrapper


def get_simulation_param_dict(
    input_params: SimulationParams | str | dict, validated_mesh_unit: LengthType
):
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
        return param.preprocess(validated_mesh_unit)
    raise ValueError(f"Invalid input <{input_params.__class__.__name__}> for translator. ")


def remove_units_in_dict(input_dict):
    unit_keys = {"value", "units"}
    if isinstance(input_dict, dict):
        new_dict = {}
        if input_dict.keys() == unit_keys:
            new_dict = input_dict["value"]
            return new_dict
        for key, value in input_dict.items():
            if isinstance(value, dict) and value.keys() == unit_keys:
                new_dict[key] = value["value"]
            else:
                new_dict[key] = remove_units_in_dict(value)
        return new_dict
    elif isinstance(input_dict, list):
        return [remove_units_in_dict(item) for item in input_dict]
    else:
        return input_dict


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


def merge_unique_item_lists(list1: list[str], list2: list[str]) -> list:
    combined = list1 + list2
    return list(OrderedDict.fromkeys(combined))
