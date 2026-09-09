"""Utility functions for the simulation component."""

from typing import Annotated, Union, get_args, get_origin

from flow360_schema.models.simulation.framework.updater_utils import recursive_remove_key


def sanitize_params_dict(model_dict):
    """
    !!!WARNING!!!: This function changes the input dict in place!!!

    Clean the redundant content in the params dict from WebUI
    """
    recursive_remove_key(model_dict, "_id", "private_attribute_image_id")

    model_dict.pop("hash", None)

    return model_dict


def get_combined_subclasses(cls):
    """get subclasses of cls"""
    if isinstance(cls, tuple):
        subclasses = set()
        for single_cls in cls:
            subclasses.update(single_cls.__subclasses__())
        return list(subclasses)
    return cls.__subclasses__()


def is_exact_instance(obj, cls):
    """Check if an object is an instance of a class and not a subclass."""
    if isinstance(cls, tuple):
        return any(is_exact_instance(obj, c) for c in cls)
    if not isinstance(obj, cls):
        return False
    subclasses = get_combined_subclasses(cls)
    return all(not isinstance(obj, subclass) for subclass in subclasses)


def is_instance_of_type_in_union(obj, typ) -> bool:
    """Check whether input `obj` is instance of the types specified in the `Union`(`typ`)"""
    if get_origin(typ) is Annotated:
        typ = get_args(typ)[0]

    if get_origin(typ) is Union:
        types_tuple = get_args(typ)
        return isinstance(obj, types_tuple)

    return isinstance(obj, typ)
