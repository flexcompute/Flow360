import abc
import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.types import TYPE_TAG_STR

# requirements for data models with custom constructors:
# 1. data model can be saved to JSON and read back to pydantic model without problems
# 2. data model can return data provided to custom constructor
# 3. data model can be created from JSON that contains only custom constructor inputs - incomplete JSON
# 4. incomplete JSON is not in a conflict with complete JSON (from point 1), such that there is no need for 2 parsers


@contextmanager
def _model_attribute_unlock(model, attr: str):
    try:
        # validate_assignment is set to False to allow for the attribute to be modified
        # Otherwise, the attribute will STILL be frozen and cannot be modified
        model.model_config["validate_assignment"] = False
        model.model_fields[attr].frozen = False
        yield
    finally:
        model.model_config["validate_assignment"] = True
        model.model_fields[attr].frozen = True


class _MultiConstructorModelBase(Flow360BaseModel, metaclass=abc.ABCMeta):

    type_name: Literal["_MultiConstructorModelBase"] = pd.Field(
        "_MultiConstructorModelBase", frozen=True
    )
    private_attribute_constructor: str = pd.Field("default", frozen=True)
    private_attribute_input_cache: Optional[Any] = pd.Field(None, frozen=True)

    @classmethod
    def model_constructor(cls, func: Callable) -> Callable:
        @classmethod
        @wraps(func)
        def wrapper(cls, **kwargs):
            obj = func(cls, **kwargs)
            sig = inspect.signature(func)
            function_arg_defaults = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            with _model_attribute_unlock(obj, "private_attribute_input_cache"):
                obj.private_attribute_input_cache = obj.private_attribute_input_cache.__class__(
                    # Note: obj.private_attribute_input_cache should not be included here
                    # Note: Because it carries over the previous cached inputs. Whatever the user choose not to specify
                    # Note: should be using default values rather than the previous cached inputs.
                    **{
                        **function_arg_defaults,  # Defaults as the base (needed when load in UI)
                        **kwargs,  # User specified inputs (overwrites defaults)
                    }
                )
            with _model_attribute_unlock(obj, "private_attribute_constructor"):
                obj.private_attribute_constructor = func.__name__
            return obj

        return wrapper


##:: Utility functions for multi-constructor models


def get_class_method(cls, method_name):
    """
    Retrieve a class method by its name.

    Parameters
    ----------
    cls : type
        The class containing the method.
    method_name : str
        The name of the method as a string.

    Returns
    -------
    method : callable
        The class method corresponding to the method name.

    Raises
    ------
    AttributeError
        If the method_name is not a callable method of the class.

    Examples
    --------
    >>> class MyClass:
    ...     @classmethod
    ...     def my_class_method(cls):
    ...         return "Hello from class method!"
    ...
    >>> method = get_class_method(MyClass, "my_class_method")
    >>> method()
    'Hello from class method!'
    """
    method = getattr(cls, method_name)
    if not callable(method):
        raise AttributeError(f"{method_name} is not a callable method of {cls.__name__}")
    return method


def get_class_by_name(class_name, global_vars):
    """
    Retrieve a class by its name from the global scope.

    Parameters
    ----------
    class_name : str
        The name of the class as a string.

    Returns
    -------
    cls : type
        The class corresponding to the class name.

    Raises
    ------
    NameError
        If the class_name is not found in the global scope.
    TypeError
        If the found object is not a class.

    Examples
    --------
    >>> class MyClass:
    ...     pass
    ...
    >>> cls = get_class_by_name("MyClass")
    >>> cls
    <class '__main__.MyClass'>
    """
    cls = global_vars.get(class_name)
    if cls is None:
        raise NameError(f"Class '{class_name}' not found in the global scope.")
    if not isinstance(cls, type):
        raise TypeError(f"'{class_name}' found in global scope, but it is not a class.")
    return cls


def model_custom_constructor_parser(model_as_dict, global_vars):
    constructor_name = model_as_dict.get("private_attribute_constructor", None)
    if constructor_name is not None:
        model_cls = get_class_by_name(model_as_dict.get("type_name"), global_vars)
        input_kwargs = model_as_dict.get("private_attribute_input_cache")
        if constructor_name != "default":
            constructor = get_class_method(model_cls, constructor_name)
            return constructor(**input_kwargs).model_dump(exclude_none=True)
        # else:
        #     return model_cls(**input_kwargs).model_dump(exclude_none=True)
    return model_as_dict


def parse_model_dict(model_as_dict, global_vars) -> dict:
    if isinstance(model_as_dict, dict):
        for key, value in model_as_dict.items():
            model_as_dict[key] = parse_model_dict(value, global_vars)
        model_as_dict = model_custom_constructor_parser(model_as_dict, global_vars)
    elif isinstance(model_as_dict, list):
        model_as_dict = [parse_model_dict(item) for item in model_as_dict]
    return model_as_dict
