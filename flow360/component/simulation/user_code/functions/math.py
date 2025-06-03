import numpy as np
from unyt import unyt_array

from flow360.component.simulation.user_code.core.types import Expression, Variable


def _convert_argument(value):
    """Convert argument for use in builtin expression math functions"""

    # If the argument is a Variable, convert it to an expression
    if isinstance(value, Variable):
        return Expression.model_validate(value)

    return value


def _extract_units(value):
    units = 1  # Neutral element of multiplication

    if isinstance(value, Expression):
        result = value.evaluate(raise_error=False)
        if isinstance(result, unyt_array):
            units = result.units
    elif isinstance(value, unyt_array):
        units = value.units

    return units


def cross(left, right):
    """Customized Cross function to work with the `Expression` and Variables"""

    left = _convert_argument(left)
    right = _convert_argument(right)

    result = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]

    is_expression_type = False

    for item in result:
        if isinstance(item, Expression):
            is_expression_type = True

    if is_expression_type:
        result = Expression.model_validate(result)

    unit = 1

    left_units = _extract_units(left)
    right_units = _extract_units(right)

    if left_units != 1:
        unit = left_units
    if right_units != 1:
        unit = unit * right_units if unit != 1 else right_units

    if unit != 1:
        result *= unit

    return result
