"""
Math.h for Flow360 Expression system
"""

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


def _check_same_length(left: VectorInputType, right: VectorInputType, operation_name: str):
    """For vector arithmetic operations, we need to check that the vectors have the same length."""
    try:
        len(left)
    except Exception as e:
        raise ValueError(f"Cannot get length information for {left}.") from e

    if len(left) != len(right):
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
