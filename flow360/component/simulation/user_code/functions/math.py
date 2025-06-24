"""
Math.h for Flow360 Expression system
"""

from numbers import Number
from typing import Any, Union

import numpy as np
from unyt import unyt_array, unyt_quantity

from flow360.component.simulation.user_code.core.types import Expression, Variable


def _handle_expression_list(value: list[Any]):
    is_expression_list = False

    for item in value:
        if isinstance(item, Expression):
            is_expression_list = True

    if is_expression_list:
        value = Expression.model_validate(value)

    return value


VectorInputType = Union[list[float], unyt_array, Expression, Variable]
ScalarInputType = Union[float, unyt_quantity, Expression]


def _get_input_array_length(value):
    try:
        return len(value)
    except Exception as e:
        raise ValueError(
            f"Cannot get length information for {value} but array-like input is expected."
        ) from e


def _check_same_length(left: VectorInputType, right: VectorInputType, operation_name: str):
    """For vector arithmetic operations, we need to check that the vectors have the same length."""
    left_length = _get_input_array_length(left)
    right_length = _get_input_array_length(right)
    if left_length != right_length:
        raise ValueError(
            f"Vectors ({left} | {right}) must have the same length to perform {operation_name}."
        )


def cross(left: VectorInputType, right: VectorInputType):
    """Customized Cross function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return np.cross(left, right)

    _check_same_length(left, right, "cross product")

    if len(left) == 3:
        result = [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    elif len(left) == 2:
        result = left[0] * right[1] - left[1] * right[0]
    else:
        raise ValueError(f"Vector length must be 2 or 3, got {len(left)}.")

    return _handle_expression_list(result)


def dot(left: VectorInputType, right: VectorInputType):
    """Customized Dot function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return np.dot(left, right)

    _check_same_length(left, right, "dot product")

    result = left[0] * right[0]
    for i in range(1, len(left)):
        result += left[i] * right[i]

    return result


########## Scalar functions ##########
def ensure_scalar_input(func):
    """Decorator to check if the input is a scalar and raise an error if so."""

    def wrapper(value):

        def is_scalar(input_value):
            if isinstance(input_value, Number):
                return True
            if isinstance(input_value, unyt_quantity):
                return input_value.shape == ()

            try:
                return len(input_value) == 0
            except Exception:  # pylint: disable=broad-exception-caught
                return False

        if not is_scalar(value):
            raise ValueError(f"Scalar function ({func.__name__}) on {value} not supported.")
        return func(value)

    return wrapper


@ensure_scalar_input
def sqrt(value: ScalarInputType):
    """Customized Sqrt function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.sqrt(value)
    return Expression(expression=f"math.sqrt({value})")
