"""Utility functions for the simulation component."""

# pylint: disable=unused-import
from typing import Annotated, Union, get_args, get_origin

# Re-export from schema package for backward compatibility
from flow360_schema.framework.bounding_box import BoundingBox, BoundingBoxType
from flow360_schema.framework.entity.utils import model_attribute_unlock

from flow360.component.simulation.framework.updater_utils import recursive_remove_key


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
    # Check if there are any subclasses of cls
    subclasses = get_combined_subclasses(cls)
    for subclass in subclasses:
        if isinstance(obj, subclass):
            return False
    return True


def is_instance_of_type_in_union(obj, typ) -> bool:
    """Check whether input `obj` is instance of the types specified in the `Union`(`typ`)"""
    # If typ is an Annotated type, extract the underlying type.
    if get_origin(typ) is Annotated:
        typ = get_args(typ)[0]

    # If the underlying type is a Union, extract its arguments (which are types).
    if get_origin(typ) is Union:
        types_tuple = get_args(typ)
        return isinstance(obj, types_tuple)

    # Otherwise, do a normal isinstance check.
    return isinstance(obj, typ)
