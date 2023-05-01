"""
Utility functions
"""
import uuid
from functools import wraps

from ..exceptions import TypeError as FlTypeError
from ..exceptions import ValueError as FlValueError
from ..log import log


# pylint: disable=redefined-builtin
def is_valid_uuid(id, ignore_none=False):
    """
    Checks if id is valid
    """
    if id is None and ignore_none:
        return
    try:
        uuid.UUID(str(id))
    except Exception as exc:
        raise FlValueError(f"{id} is not a valid UUID.") from exc


def beta_feature(feature_name: str):
    """Prints warning message when used on a function which is BETA feature.

    Parameters
    ----------
    feature_name : str
        Name of the feature used in warning message
    """

    def wrapper(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            log.warning(f"{feature_name} is a beta feature.")
            value = func(*args, **kwargs)
            return value

        return wrapper_func

    return wrapper


# pylint: disable=bare-except
def _get_value_or_none(callable):
    try:
        return callable()
    except:
        return None


def validate_type(value, parameter_name: str, expected_type):
    """validate type

    Parameters
    ----------
    value :
        value to be validated
    parameter_name : str
        paremeter name - used for error message
    expected_type : type
        expected type for value

    Raises
    ------
    TypeError
        when value is not expected_type
    """
    if not isinstance(value, expected_type):
        raise FlTypeError(
            f"Expected type={expected_type} for {parameter_name}, but got value={value} (type={type(value)})"
        )
