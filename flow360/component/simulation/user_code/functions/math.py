"""
Math.h for Flow360 Expression system
"""

from typing import Any, Union

from unyt import ucross, unyt_array, unyt_quantity

from flow360.component.simulation.user_code.core.types import Expression, Variable


def _convert_argument(value):
    """Convert argument for use in builtin expression math functions"""

    # If the argument is a Variable, convert it to an expression
    if isinstance(value, Variable):
        return Expression.model_validate(value).evaluate(
            raise_on_non_evaluable=False, force_evaluate=False
        )

    return value


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
    left = _convert_argument(left)
    right = _convert_argument(right)

    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return ucross(left, right)

    # Otherwise
    result = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]

    return _handle_expression_list(result)
