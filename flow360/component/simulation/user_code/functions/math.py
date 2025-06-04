""""""

from unyt import ucross, unyt_array

from flow360.component.simulation.user_code.core.types import Expression, Variable


def _convert_argument(value):
    """Convert argument for use in builtin expression math functions"""

    # If the argument is a Variable, convert it to an expression
    if isinstance(value, Variable):
        return Expression.model_validate(value).evaluate(raise_error=False, force_evaluate=False)

    if isinstance(value, Expression):
        # TODO: Test numerical value?
        return value.evaluate(raise_error=False, force_evaluate=False)
    return value


def cross(left, right):
    """Customized Cross function to work with the `Expression` and Variables"""
    # print("Old left:", left, "  |  ", left.__class__.__name__)
    # print("Old right:", right, "  |  ", right.__class__.__name__)
    left = _convert_argument(left)
    right = _convert_argument(right)

    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return ucross(left, right)

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

    return result
