"""Utilities for the translator module."""

from __future__ import annotations

import functools
import json
from collections import OrderedDict

from flow360.component.simulation.simulation_params import (
    SimulationParams,  # Not required
)
from flow360.component.simulation.unit_system import LengthType


def preprocess_input(func):
    """Call param preprocess() method before calling the translator."""

    @functools.wraps(func)
    def wrapper(input_params, mesh_unit, *args, **kwargs):
        # pylint: disable=no-member
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
            with open(input_params, "r", encoding="utf-8") as file:
                param_dict = json.load(file)
        if param_dict is None:
            raise ValueError(f"Invalid input <{input_params}> for translator. ")
        param = SimulationParams(**param_dict)
    elif isinstance(input_params, dict):
        param = SimulationParams(**input_params)

    if param is not None:
        return param.preprocess(validated_mesh_unit)
    raise ValueError(f"Invalid input <{input_params.__class__.__name__}> for translator. ")


def replace_dict_key(input_dict: dict, key_to_replace: str, replacement_key: str):
    """Replace a key in a dictionary."""
    if key_to_replace in input_dict:
        input_dict[replacement_key] = input_dict.pop(key_to_replace)


def replace_dict_value(input_dict: dict, key: str, value_to_replace, replacement_value):
    """Replace a value in a dictionary."""
    if key in input_dict and input_dict[key] == value_to_replace:
        input_dict[key] = replacement_value


def convert_tuples_to_lists(input_dict):
    """
    Recursively convert all tuples to lists in a nested dictionary.

    This function traverses through all the elements in the input dictionary and
    converts any tuples it encounters into lists. It also handles nested dictionaries
    and lists by applying the conversion recursively.

    Args:
        input_dict (dict): The input dictionary that may contain tuples.

    Returns:
        dict: A new dictionary with all tuples converted to lists.

    Examples:
        >>> input_dict = {
        ...     'a': (1, 2, 3),
        ...     'b': {
        ...         'c': (4, 5),
        ...         'd': [6, (7, 8)]
        ...     }
        ... }
        >>> convert_tuples_to_lists(input_dict)
        {'a': [1, 2, 3], 'b': {'c': [4, 5], 'd': [6, [7, 8]]}}
    """
    if isinstance(input_dict, dict):
        return {k: convert_tuples_to_lists(v) for k, v in input_dict.items()}
    if isinstance(input_dict, tuple):
        return list(input_dict)
    if isinstance(input_dict, list):
        return [convert_tuples_to_lists(item) for item in input_dict]
    return input_dict


def remove_units_in_dict(input_dict):
    """Remove units from a dimensioned value."""
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
    if isinstance(input_dict, list):
        return [remove_units_in_dict(item) for item in input_dict]
    return input_dict


def has_instance_in_list(obj_list: list, class_type):
    """Check if a list contains an instance of a given type."""
    for obj in obj_list:
        if isinstance(obj, class_type):
            return True
    return False


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


def update_dict_recursively(a, b):
    """
    Recursively updates dictionary 'a' with values from dictionary 'b'.
    If the same key contains dictionaries in both 'a' and 'b', they are merged recursively.
    """
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            # If both a[key] and b[key] are dictionaries, recurse
            update_dict_recursively(a[key], value)
        else:
            # Otherwise, simply update/overwrite the value in 'a' with the value from 'b'
            a[key] = value


def translate_setting_and_apply_to_all_entities(
    obj_list: list,
    class_type,
    translation_func,
    to_list: bool = False,
    entity_injection_func=lambda x: {},
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
                        output[entity.name] = entity_injection_func(entity)
                    # needs to be recursive
                    update_dict_recursively(output[entity.name], translated_setting)
                else:
                    setting = entity_injection_func(entity)
                    setting.update(translated_setting)
                    output.append(setting)
    return output


def merge_unique_item_lists(list1: list[str], list2: list[str]) -> list:
    """Merge two lists and remove duplicates."""
    combined = list1 + list2
    return list(OrderedDict.fromkeys(combined))
