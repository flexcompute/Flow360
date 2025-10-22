"""Utilities for the translator module."""

from __future__ import annotations

import contextvars
import functools
import json
from collections import OrderedDict
from typing import Union

import unyt as u

from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.primitives import (
    BOUNDARY_FULL_NAME_WHEN_NOT_FOUND,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import Expression
from flow360.component.simulation.utils import is_exact_instance

_unit_system_ctx = contextvars.ContextVar("unit_system", default=None)


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
        processed_input = preprocess_param(input_params, validated_mesh_unit, preprocess_exclude)
        return func(processed_input, validated_mesh_unit, *args, **kwargs)

    return wrapper


def preprocess_param(
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
            if param_dict is None:
                raise ValueError(f"Invalid input <{input_params}> for translator. ")
            param = SimulationParams(file_content=param_dict)
        except json.JSONDecodeError:
            # If input is a file path
            param = SimulationParams(filename=input_params)

    if param is not None:
        # pylint: disable=protected-access
        param._private_set_length_unit(validated_mesh_unit)
        return param._preprocess(validated_mesh_unit, exclude=preprocess_exclude)
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


def evaluate_value_or_expression_object(obj: Union[Expression, u.unyt_quantity, u.unyt_array]):
    """Translate for an ValueOrExpression object"""
    if isinstance(obj, Expression):
        # Only allowing client-time evaluable expressions
        return obj.evaluate(raise_on_non_evaluable=True)
    # Non dimensionalized unyt objects
    return obj


def has_instance_in_list(obj_list: list, class_type):
    """Check if a list contains an instance of a given type."""
    if obj_list is not None:
        for obj in obj_list:
            if is_exact_instance(obj, class_type):
                return True
    return False


def getattr_by_path(obj, path: Union[str, list], *args):
    """Get attribute by path from a list"""
    # If path is a string, return the attribute directly
    if isinstance(path, str):
        return getattr(obj, path, *args)

    # If path is a list, iterate through each attribute name
    for attr in path:
        obj = getattr(obj, attr)

    return obj


def get_global_setting_from_first_instance(
    obj_list: list,
    class_type,
    attribute_name: Union[str, list],
):
    """In a list loop and find the first instance matching the given type and retrive the attribute"""
    if obj_list is not None:
        for obj in obj_list:
            if (
                is_exact_instance(obj, class_type)
                and getattr_by_path(obj, attribute_name, None) is not None
            ):
                # Allowed to look into non-empty-entity instances
                # Then we return the first non-None value.
                # Previously we return the value that is non-default.
                # But this is deemed not intuitive and very hard to implement.
                return getattr_by_path(obj, attribute_name)
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
    if isinstance(entity, (_SurfaceEntityBase, _VolumeEntityBase)):
        # Note: If the entity is a Surface/Boundary, we need to use the full name
        return entity.full_name

    return entity.name


# pylint: disable=too-many-branches, too-many-arguments, too-many-locals, too-many-statements
def translate_setting_and_apply_to_all_entities(
    obj_list: list,
    class_type,
    translation_func,
    to_list: bool = False,
    entity_injection_func=lambda x, **kwargs: {},
    pass_translated_setting_to_entity_injection=False,
    custom_output_dict_entries=False,
    lump_list_of_entities=False,
    use_instance_name_as_key=False,
    use_sub_item_as_key=False,
    **kwargs,
):
    """
    Translate settings and apply them to all entities of a given type.

    This function iterates over a list of objects, applies a translation function to each object if
    it matches the given `class_type`, and then processes its entities based on various customization
    options. The function supports returning either a dictionary or a list of translated settings.

    Parameters
    ----------
    obj_list : list
        A list of objects to process.
    class_type : type
        The type of objects to match. Only objects of this type will be processed.
    translation_func : callable
        A function that translates the settings for each object. It should return a dictionary.
    to_list : bool, optional
        If True, the output is a list. If False (default), the output is a dictionary.
    entity_injection_func : callable, optional
        A function for injecting additional settings into each entity. Defaults to a lambda returning an empty dict.
    pass_translated_setting_to_entity_injection : bool, optional
        If True, passes the translated settings to `entity_injection_func`. Defaults to False.
    custom_output_dict_entries : bool, optional
        If True, allows customization of output dictionary entries. Defaults to False.
    lump_list_of_entities : bool, optional
        If True, lumps all entities into a single list. Defaults to False.
    use_instance_name_as_key : bool, optional
        If True, uses the instance name of the object as the key in the output dictionary.
        Only valid if `lump_list_of_entities` is False. Defaults to False.
    use_sub_item_as_key : bool, optional
        If True, uses subcomponents of entities as keys in the output dictionary. Defaults to False.
    **kwargs : dict, optional
        Additional keyword arguments. Arguments prefixed with `translation_func_` are passed to
        `translation_func`, and those prefixed with `entity_injection_` are passed to `entity_injection_func`.

    Returns
    -------
    dict or list
        A dictionary or list containing the translated settings applied to all entities.
        If `to_list` is False (default), the output is a dictionary, where keys are entity names (or custom keys)
        and values are the translated settings. If `to_list` is True, the output is a list.

    Raises
    ------
    NotImplementedError
        If `lump_list_of_entities` is used with `entity_pairs` or if `use_instance_name_as_key`
        is used when `lump_list_of_entities` is True.

    Notes
    -----
    - The `translation_func` must return a dictionary with the translated settings.
    - The `entity_injection_func` allows additional customizations to be applied to each entity.
    - If `lump_list_of_entities` is True, all entities are treated as a single group, and custom key usage
      (e.g., `use_instance_name_as_key`) may be restricted.

    Examples
    --------
    >>> translate_setting_and_apply_to_all_entities(obj_list, MyClass, my_translation_func)
    {'entity1': {'setting1': 'value1'}, 'entity2': {'setting1': 'value2'}}
    """
    if not to_list:
        output = {}
    else:
        output = []

    translation_func_prefix = "translation_func_"
    translation_func_kwargs = {
        k[len(translation_func_prefix) :]: v
        for k, v in kwargs.items()
        if k.startswith(translation_func_prefix)
    }
    entity_injection_prefix = "entity_injection_"
    entity_injection_kwargs = {
        k[len(entity_injection_prefix) :]: v
        for k, v in kwargs.items()
        if k.startswith(entity_injection_prefix)
    }

    # pylint: disable=too-many-nested-blocks
    for obj in obj_list:
        if class_type and is_exact_instance(obj, class_type):

            list_of_entities = []
            if "entities" in obj.__class__.model_fields:
                if obj.entities is None or (
                    "stored_entities" in obj.entities.__class__.model_fields
                    and obj.entities.stored_entities is None
                ):  # unique item list does not allow None "items" for now.
                    continue
                if isinstance(obj.entities, EntityList):
                    list_of_entities = (
                        obj.entities.stored_entities
                        if lump_list_of_entities is False
                        else [obj.entities]
                    )
                elif isinstance(obj.entities, UniqueItemList):
                    list_of_entities = (
                        obj.entities.items if lump_list_of_entities is False else [obj.entities]
                    )
            elif "entity_pairs" in obj.__class__.model_fields:
                # Note: This is only used in Periodic BC and lump_list_of_entities is not relavant
                if lump_list_of_entities:
                    raise NotImplementedError(
                        "[Internal Error]: lump_list_of_entities cannot be used with entity_pairs"
                    )
                list_of_entities = obj.entity_pairs.items

            translated_setting = translation_func(obj, **translation_func_kwargs)

            if pass_translated_setting_to_entity_injection:
                entity_injection_kwargs["translated_setting"] = translated_setting

            for entity in list_of_entities:
                if not to_list:
                    # Generate a $name:{$value} dict
                    if custom_output_dict_entries:
                        setting = entity_injection_func(entity, **entity_injection_kwargs)
                        if setting is None:
                            continue
                        output.update(setting)
                    else:
                        if use_instance_name_as_key is True and lump_list_of_entities is False:
                            raise NotImplementedError(
                                "[Internal Error]: use_instance_name_as_key cannot be used"
                                " when lump_list_of_entities is True"
                            )
                        if use_sub_item_as_key is True:
                            # pylint: disable=fixme
                            # TODO: Make sure when use_sub_item_as_key is True
                            # TODO: the entity has private_attribute_sub_components
                            key_names = entity.private_attribute_sub_components
                        else:
                            key_names = [
                                (
                                    _get_key_name(entity)
                                    if use_instance_name_as_key is False
                                    else obj.name
                                )
                            ]
                        for key_name in key_names:
                            if key_name == BOUNDARY_FULL_NAME_WHEN_NOT_FOUND:
                                # Skip missing boundary
                                continue
                            if output.get(key_name) is None:
                                setting = entity_injection_func(entity, **entity_injection_kwargs)
                                if setting is None:
                                    continue
                                output[key_name] = setting
                            update_dict_recursively(output[key_name], translated_setting)
                else:
                    # Generate a list with $name being an item
                    # Note: Surface/Boundary logic should be handled in the entity_injection_func
                    setting = entity_injection_func(entity, **entity_injection_kwargs)
                    if setting is None:
                        continue
                    setting.update(translated_setting)
                    output.append(setting)
    return output


def merge_unique_item_lists(list1: list[str], list2: list[str]) -> list:
    """Merge two lists and remove duplicates."""
    combined = list1 + list2
    return list(OrderedDict.fromkeys(combined))


class UnitSystemContext:
    """
    Context manager for setting the current unit system.

    This context manager allows setting a unit system within a specific scope,
    making it accessible to functions that need it without explicitly passing
    it as a parameter.

    Parameters
    ----------
    unit_system : u.UnitSystem
        The unit system to be set in the current context.

    Examples
    --------
    >>> with UnitSystemContext(my_unit_system):
    ...     # Within this block, get_unit_system() will return my_unit_system
    ...     current_system = get_unit_system()
    """

    def __init__(self, unit_system: u.UnitSystem):
        self.unit_system = unit_system
        self.token = None

    def __enter__(self):
        self.token = _unit_system_ctx.set(self.unit_system)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _unit_system_ctx.reset(self.token)


def get_unit_system() -> u.UnitSystem:
    """
    Retrieves the current unit system from the context.

    Returns
    -------
    u.UnitSystem
        The current unit system, or None if no unit system context is active.

    Examples
    --------
    >>> with UnitSystemContext(my_unit_system):
    ...     system = get_unit_system()  # Returns my_unit_system
    """
    return _unit_system_ctx.get()
