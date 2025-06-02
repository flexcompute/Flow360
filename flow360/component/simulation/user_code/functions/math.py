from enum import Enum

from flow360.component.simulation.user_code.core.types import Expression, Variable


def _convert_argument(value):
    """Convert argument for use in builtin expression math functions"""

    # If the argument is a Variable, convert it to an expression
    if isinstance(value, Variable):
        return Expression.model_validate(value)

    return value


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
        return Expression.model_validate(result)
    else:
        return result
