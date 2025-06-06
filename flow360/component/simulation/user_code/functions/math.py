"""
Math.h for Flow360 Expression system
"""

from typing import Any, Union

import numpy as np
from unyt import unyt_array, unyt_quantity

from flow360.component.simulation.user_code.core.types import Expression


def _handle_expression_list(value: list[Any]):
    is_expression_list = False

    for item in value:
        if isinstance(item, Expression):
            is_expression_list = True

    if is_expression_list:
        value = Expression.model_validate(value)

    return value


VectorInputType = Union[list[float], unyt_array, Expression]
ScalarInputType = Union[float, unyt_quantity, Expression]


def cross(left: VectorInputType, right: VectorInputType):
    """Customized Cross function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return np.cross(left, right)

    # Otherwise
    result = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]

    return _handle_expression_list(result)
