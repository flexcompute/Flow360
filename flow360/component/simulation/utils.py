"""Utility functions for the simulation component."""

from contextlib import contextmanager


@contextmanager
def model_attribute_unlock(model, attr: str):
    """
    Helper function to set frozen fields of a pydantic model from internal systems
    """
    try:
        # validate_assignment is set to False to allow for the attribute to be modified
        # Otherwise, the attribute will STILL be frozen and cannot be modified
        model.model_config["validate_assignment"] = False
        model.model_fields[attr].frozen = False
        yield
    finally:
        model.model_config["validate_assignment"] = True
        model.model_fields[attr].frozen = True


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
