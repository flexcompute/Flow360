"""MultiConstructorModelBase class for Pydantic models with multiple constructors."""

# requirements for data models with custom constructors:
# 1. data model can be saved to JSON and read back to pydantic model without problems
# 2. data model can return data provided to custom constructor
# 3. data model can be created from JSON that contains only custom constructor inputs - incomplete JSON
# 4. incomplete JSON is not in a conflict with complete JSON (from point 1), such that there is no need for 2 parsers
import abc
import inspect
from functools import wraps
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.utils import model_attribute_unlock


class MultiConstructorBaseModel(Flow360BaseModel, metaclass=abc.ABCMeta):
    """
    [INTERNAL]

    Base class for models with multiple constructors.

    This class provides a mechanism to create models with multiple constructors, each having its own set
    of parameters and default values. It stores the constructor name and input cache so the class instance
    can be constructed from front end input.
    """

    type_name: Literal["MultiConstructorBaseModel"] = pd.Field(
        "MultiConstructorBaseModel", frozen=True
    )
    private_attribute_constructor: str = pd.Field("default", frozen=True)
    private_attribute_input_cache: Optional[Any] = pd.Field(None, frozen=True)

    @classmethod
    def model_constructor(cls, func: Callable) -> Callable:
        """
        [AI-Generated] Decorator for model constructor functions.

        This method wraps a constructor function to manage default argument values and cache the inputs.

        Args:
            func (Callable): The constructor function to wrap.

        Returns:
            Callable: The wrapped constructor function.
        """

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
            # XXCache should not include private_attribute_id as it is not **User** input
            kwargs.pop("private_attribute_id", None)

            with model_attribute_unlock(obj, "private_attribute_input_cache"):
                obj.private_attribute_input_cache = obj.private_attribute_input_cache.__class__(
                    # Note: obj.private_attribute_input_cache should not be included here
                    # Note: Because it carries over the previous cached inputs. Whatever the user choose not to specify
                    # Note: should be using default values rather than the previous cached inputs.
                    **{
                        **function_arg_defaults,  # Defaults as the base (needed when load in UI)
                        **kwargs,  # User specified inputs (overwrites defaults)
                    }
                )
            with model_attribute_unlock(obj, "private_attribute_constructor"):
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


def _add_parent_location_to_validation_error(
    validation_error: pd.ValidationError, parent_loc=None
) -> pd.ValidationError:
    """Convert the validation error by appending the parent location"""
    if parent_loc is None:
        return validation_error
    updated_errors = []
    for error in validation_error.errors():
        error["loc"] = (parent_loc,) + error["loc"]
        updated_errors.append(error)
    return pd.ValidationError.from_exception_data(
        title=validation_error.title,
        line_errors=updated_errors,
    )


def model_custom_constructor_parser(model_as_dict, global_vars):
    """Parse the dictionary, construct the object and return obj dict."""
    constructor_name = model_as_dict.get("private_attribute_constructor", None)
    if constructor_name is not None and constructor_name != "default":

        def is_optional_argument(annotation) -> bool:
            # Ensure the annotation has been parsed as the typing object
            # Avoid the unnecessary use of from __future__ import annotations
            assert not isinstance(
                arg_obj.annotation, str
            ), "[Internal] Invalid string type annotation. Please check importing future."
            if get_origin(annotation) is Union and type(None) in get_args(annotation):
                return True
            return False

        model_cls = get_class_by_name(model_as_dict.get("type_name"), global_vars)
        input_kwargs = model_as_dict.get("private_attribute_input_cache")

        constructor = get_class_method(model_cls, constructor_name)
        constructor_args = inspect.signature(constructor).parameters
        # Filter the input_kwargs using constructor signatures
        # If the argument is not found in input_kwargs:
        # 1. Error out if the argument is required
        # 2. Use default value if available, else use None if `Optional`
        input_kwargs_filtered = {}
        for arg_name, arg_obj in constructor_args.items():
            if arg_name in input_kwargs.keys():
                input_kwargs_filtered[arg_name] = input_kwargs[arg_name]
            elif (
                is_optional_argument(arg_obj.annotation)
                and arg_obj.default is inspect.Parameter.empty
            ):
                input_kwargs_filtered[arg_name] = None
        try:
            model_dict = constructor(**input_kwargs_filtered).model_dump(
                mode="json", exclude_none=True
            )
            # Make sure we do not generate a new ID.
            if "private_attribute_id" in model_as_dict:
                model_dict["private_attribute_id"] = model_as_dict["private_attribute_id"]
            return model_dict
        except pd.ValidationError as err:
            # pylint:disable = raise-missing-from
            raise _add_parent_location_to_validation_error(
                validation_error=err, parent_loc="private_attribute_input_cache"
            )

    return model_as_dict


def parse_model_dict(model_as_dict, global_vars, parent_loc: Union[str, int] = None) -> dict:
    """Recursively parses the model dictionary and attempts to construct the multi-constructor object."""
    try:
        if isinstance(model_as_dict, dict):
            for key, value in model_as_dict.items():
                model_as_dict[key] = parse_model_dict(value, global_vars, key)

            model_as_dict = model_custom_constructor_parser(model_as_dict, global_vars)
        elif isinstance(model_as_dict, list):
            model_as_dict = [
                parse_model_dict(item, global_vars, idx) for idx, item in enumerate(model_as_dict)
            ]
    except pd.ValidationError as err:
        # pylint:disable = raise-missing-from
        raise _add_parent_location_to_validation_error(validation_error=err, parent_loc=parent_loc)

    return model_as_dict
