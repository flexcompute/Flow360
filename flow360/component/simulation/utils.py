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


def get_unit_system_name_from_simulation_params_dict(params_dict: dict) -> str:
    """Get unit system name from simulation params dict"""
    unit_system_name = None
    for unit_system_key in ["unitSystem", "unit_system"]:
        unit_system_dict = params_dict.get(unit_system_key, None)
        if unit_system_dict is not None:
            unit_system_name = unit_system_dict.get("name", None)
            break

    if unit_system_name is None:
        raise KeyError("Unit system not found in the simulation params. Corrupted file.")

    return unit_system_name


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
