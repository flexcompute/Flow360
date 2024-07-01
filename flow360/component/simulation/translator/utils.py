"""Utilities for the translator module."""

from __future__ import annotations

import functools
import json
from collections import OrderedDict

from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.exceptions import Flow360TranslationError


def preprocess_input(func):
    """Call param preprocess() method before calling the translator."""

    @functools.wraps(func)
    def wrapper(input_params, mesh_unit, *args, **kwargs):
        # pylint: disable=no-member
        if func.__name__ == "get_solver_json":
            preprocess_exclude = ["meshing"]
        elif func.__name__ in ("get_surface_meshing_json", "get_volume_meshing_json"):
            preprocess_exclude = [
                "reference_geometry",
                "operating_condition",
                "models",
                "time_stepping",
                "user_defined_dynamics",
                "outputs",
            ]
        else:
            preprocess_exclude = []
        validated_mesh_unit = LengthType.validate(mesh_unit)
        processed_input = get_simulation_param_dict(
            input_params, validated_mesh_unit, preprocess_exclude
        )
        return func(processed_input, validated_mesh_unit, *args, **kwargs)

    return wrapper


def get_simulation_param_dict(
    input_params: SimulationParams | str | dict,
    validated_mesh_unit: LengthType,
    preprocess_exclude: list[str],
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
        return param.preprocess(validated_mesh_unit, exclude=preprocess_exclude)
    raise ValueError(f"Invalid input <{input_params.__class__.__name__}> for translator. ")


def replace_dict_key(input_dict: dict, key_to_replace: str, replacement_key: str):
    """Replace a key in a dictionary."""
    if key_to_replace in input_dict:
        input_dict[replacement_key] = input_dict.pop(key_to_replace)


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


def is_terminal_instance(obj, cls):
    """Check if an object is an instance of a class and not a subclass."""
    if not isinstance(obj, cls):
        return False
    # Check if there are any subclasses of cls
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        if isinstance(obj, subclass):
            return False
    return True


def has_instance_in_list(obj_list: list, class_type):
    """Check if a list contains an instance of a given type."""
    if obj_list is not None:
        for obj in obj_list:
            if is_terminal_instance(obj, class_type):
                return True
    return False


def _is_last_of_type(lst, obj):
    current_type = type(obj)
    last_index = -1

    for i, item in enumerate(lst):
        if is_terminal_instance(item, current_type):
            last_index = i

    if last_index == -1:
        return False  # The type of obj does not exist in the list

    return lst[last_index] == obj


def get_attribute_from_first_instance(
    obj_list: list, class_type, attribute_name: str, use_empty_entities: bool = False
):
    """In a list loop and find the first instance matching the given type and retrive the attribute"""
    if obj_list is not None:
        for obj in obj_list:
            if (
                is_terminal_instance(obj, class_type)
                and getattr(obj, attribute_name, None) is not None
            ):
                # Route 1: Requested to look into empty-entity instances
                if use_empty_entities and getattr(obj, "entities", None) is not None:
                    # We only look for empty entities instances
                    # Note: This poses requirement that entity list has to be under attribute name 'entities'
                    continue

                # Route 2: Allowed to look into non-empty-entity instances
                default_value = obj.model_fields[attribute_name].default
                field_value = getattr(obj, attribute_name)
                is_last_item = _is_last_of_type(obj_list, obj)

                if (field_value == default_value) and (is_last_item is False):
                    # We skip defaults as much as possible
                    continue
                return getattr(obj, attribute_name)
    return None


def update_dict_recursively(a: dict, b: dict):
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


def _get_key_name(entity: EntityBase):
    if isinstance(entity, Surface):
        # Note: If the entity is a Surface/Boundary, we need to use the full name
        # Note: Ideally this should not happen if ran thorugh the casePipeline
        return entity.full_name

    return entity.name


def translate_setting_and_apply_to_all_entities(
    obj_list: list,
    class_type,
    translation_func,
    to_list: bool = False,
    entity_injection_func=lambda x: {},
    **kwargs,
):
    """Translate settings and apply them to all entities of a given type.

    Args:
        obj_list (list): A list of objects to loop through.
        class_type: The type of objects to match.
        translation_func: The function to use for translation. This function should return a dictionary.
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
        if is_terminal_instance(obj, class_type):
            translated_setting = translation_func(obj, **kwargs)
            if obj.entities is None:
                continue

            list_of_entities = []
            if isinstance(obj.entities, EntityList):
                list_of_entities = obj.entities.stored_entities
            elif isinstance(obj.entities, UniqueItemList):
                list_of_entities = obj.entities.items

            for entity in list_of_entities:
                if not to_list:
                    # Generate a $name:{$value} dict
                    key_name = _get_key_name(entity)
                    if output.get(key_name) is None:
                        output[key_name] = entity_injection_func(entity)
                    update_dict_recursively(output[key_name], translated_setting)
                else:
                    # Generate a list with $name being an item
                    # Note: Surface/Boundary logic should be handeled in the entity_injection_func
                    setting = entity_injection_func(entity)
                    setting.update(translated_setting)
                    output.append(setting)
    return output


def merge_unique_item_lists(list1: list[str], list2: list[str]) -> list:
    """Merge two lists and remove duplicates."""
    combined = list1 + list2
    return list(OrderedDict.fromkeys(combined))


def get_global_setting_from_per_item_setting(
    obj_list: list,
    class_type,
    attribute_name: str,
    allow_first_instance_as_dummy: bool,
    return_none_when_no_global_found: bool = False,
):
    """
    [AI-Generated] Retrieves a global setting from the per-item settings in a list of objects.

    This function searches through a list of objects to find the first instance of a given class type
    with empty entities and retrieves a specified attribute. If no such instance is found and
    `allow_first_instance_as_dummy` is True, it retrieves the attribute from the first instance of the class type
    regardless of whether its entities are empty. If `allow_first_instance_as_dummy` is False and no suitable instance
    is found, it raises a `Flow360TranslationError`.

    Note: This function does not apply to SurfaceOutput situations.

    Args:
        obj_list (list):
            A list of objects to search through.
        class_type (type):
            The class type of objects to match.
        attribute_name (str):
            The name of the attribute to retrieve.
        allow_first_instance_as_dummy (bool, optional):
            Whether to allow retrieving the attribute from any instance of the class
            type if no instance with empty entities is found.

    Returns:
        The value of the specified attribute from the first matching object.

    Raises:
        Flow360TranslationError: If `allow_first_instance_as_dummy` is False and no suitable instance is found.
    """

    # Get from the first instance of `class_type` with empty entities
    global_setting = get_attribute_from_first_instance(
        obj_list,
        class_type,
        attribute_name,
        use_empty_entities=True,
    )

    if global_setting is None:

        if return_none_when_no_global_found is True:
            return None

        if allow_first_instance_as_dummy is True:
            # Assume that no global setting is needed. Just get the first instance of `class_type`
            # This is allowed because simulation will make sure global setting is not used anywhere.
            global_setting = get_attribute_from_first_instance(
                obj_list,
                class_type,
                attribute_name,
                use_empty_entities=False,
            )
        else:
            raise Flow360TranslationError(
                f"Global setting of {attribute_name} is required but not found in `{class_type.__name__}` instances."
            )
    return global_setting
