"""
Math.h for Flow360 Expression system
"""

from numbers import Number
from typing import Any, Union

import numpy as np
from unyt import dimensions, unyt_array, unyt_quantity

from flow360.component.simulation.user_code.core.types import (
    Expression,
    Variable,
    _convert_numeric,
)


def _handle_expression_list(value: list[Any]):
    is_expression_list = False

    for item in value:
        if isinstance(item, Expression):
            is_expression_list = True

    if is_expression_list:
        value = Expression.model_validate(value)

    return value


VectorInputType = Union[list[float], unyt_array, Expression, Variable]
ScalarInputType = Union[float, unyt_quantity, Expression, Variable]


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
            f"Vectors ({left} | {right}) must have the same length to perform {operation_name} operation."
        )


def _get_input_value_dimensionality(value: Union[ScalarInputType, VectorInputType]):
    """Get the dimensionality of the input value"""
    if isinstance(value, list) and len(value) > 0:
        return _get_input_value_dimensionality(value=value[0])
    if isinstance(value, Variable):
        return _get_input_value_dimensionality(value=value.value)
    if isinstance(value, Expression):
        return value.dimensionality
    if isinstance(value, (unyt_array, unyt_quantity)):
        return value.units.dimensions
    if isinstance(value, Number):
        return dimensions.dimensionless
    return None


def _compare_operation_dimensionality(
    value: Union[ScalarInputType, VectorInputType], dimensionality
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector has the specify dimensionality.
    """
    value_dimensionality = _get_input_value_dimensionality(value=value)
    if value_dimensionality:
        return value_dimensionality == dimensionality
    return False


def _check_same_dimensionality(
    value1: Union[ScalarInputType, VectorInputType],
    value2: Union[ScalarInputType, VectorInputType],
    operation_name: str,
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalars/vectors have the same dimensionality.
    """
    value1_dimensionality = _get_input_value_dimensionality(value=value1)
    value2_dimensionality = _get_input_value_dimensionality(value=value2)
    if value1_dimensionality != value2_dimensionality:
        raise ValueError(
            f"Input values ({value1} | {value2}) must have the same dimensinality to perform {operation_name} operation."
        )


def _check_operation_dimensionality(
    value: Union[ScalarInputType, VectorInputType],
    dimensionalities: list,
    operation_name: str,
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector satisfies the specific dimensionality.
    """

    if len(dimensionalities) == 1:
        dimensionalities_err_msg = str(dimensionalities[0])
    else:
        dimensionalities_err_msg = (
            "one of ("
            + ", ".join([str(dimensionality) for dimensionality in dimensionalities])
            + ")"
        )

    if not any(
        _compare_operation_dimensionality(value=value, dimensionality=dimensionality)
        for dimensionality in dimensionalities
    ):
        raise ValueError(
            f"The dimensionality of the input value ({value}) "
            f"must be {dimensionalities_err_msg} to perform {operation_name} operation."
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


def magnitude(value: VectorInputType):
    """Customized Magnitude function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:

    _get_input_array_length(value)

    if isinstance(value, unyt_array):
        return np.linalg.norm(value)

    # For solution variables and expressions, return an Expression
    if isinstance(value, Expression) or (
        isinstance(value, Variable) and hasattr(value, "solver_name")
    ):
        return Expression(expression=f"math.magnitude({value})")

    # For regular lists/arrays and UserVariables, compute the magnitude
    result = value[0] ** 2
    for i in range(1, len(value)):
        result += value[i] ** 2

    return result**0.5


def add(left: VectorInputType, right: VectorInputType):
    """Customized Add function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left + right

    _check_same_length(left, right, "add")
    _check_same_dimensionality(left, right, "add")

    result = [left[i] + right[i] for i in range(len(left))]
    return _handle_expression_list(result)


def subtract(left: VectorInputType, right: VectorInputType):
    """Customized Subtract function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left - right

    _check_same_length(left, right, "subtract")
    _check_same_dimensionality(left, right, "subtract")

    result = [left[i] - right[i] for i in range(len(left))]
    return _handle_expression_list(result)


########## Scalar functions ##########
def ensure_scalar_input(func):
    """Decorator to check if the input is a scalar and raise an error if so."""

    def wrapper(*args, **kwargs):

        def is_scalar(input_value):
            if isinstance(input_value, list):
                return False
            if isinstance(input_value, Number):
                return True
            if isinstance(input_value, unyt_quantity):
                return input_value.shape == ()

            try:
                return len(input_value) == 0
            except Exception:  # pylint: disable=broad-exception-caught
                return False

        for arg in args:
            if not is_scalar(arg):
                raise ValueError(f"Scalar function ({func.__name__}) on {arg} not supported.")
        return func(*args, **kwargs)

    return wrapper


@ensure_scalar_input
def sqrt(value: ScalarInputType):
    """Customized Sqrt function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.sqrt(value)
    return Expression(expression=f"math.sqrt({value})")


@ensure_scalar_input
def log(value: ScalarInputType):
    """Customized Log function to work with the `Expression` and Variables"""
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.dimensionless],
        operation_name="log",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.log(value)
    return Expression(expression=f"math.log({value})")


@ensure_scalar_input
def exp(value: ScalarInputType):
    """Customized Exponential function to work with the `Expression` and Variables"""
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.dimensionless],
        operation_name="exp",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.exp(value)
    return Expression(expression=f"math.exp({value})")


@ensure_scalar_input
def sin(value: ScalarInputType):
    """Customized Sine function to work with the `Expression` and Variables"""
    # TODO: Add check that the value has dimension angle, also does converting to solver unit works for angles?
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.angle, dimensions.dimensionless],
        operation_name="sin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.sin(value)
    return Expression(expression=f"math.sin({value})")


@ensure_scalar_input
def cos(value: ScalarInputType):
    """Customized Cosine function to work with the `Expression` and Variables"""
    # TODO: Add check that the value has dimension angle, also does converting to solver unit works for angles?
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.angle, dimensions.dimensionless],
        operation_name="cos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.cos(value)
    return Expression(expression=f"math.cos({value})")


@ensure_scalar_input
def tan(value: ScalarInputType):
    """Customized Tangent function to work with the `Expression` and Variables"""
    # TODO: Add check that the value has dimension angle, also does converting to solver unit works for angles?
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.angle, dimensions.dimensionless],
        operation_name="tan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.tan(value)
    return Expression(expression=f"math.tan({value})")


@ensure_scalar_input
def asin(value: ScalarInputType):
    """Customized ArcSine function to work with the `Expression` and Variables"""
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.dimensionless],
        operation_name="asin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arcsin(value)
    return Expression(expression=f"math.asin({value})")


@ensure_scalar_input
def acos(value: ScalarInputType):
    """Customized ArcCosine function to work with the `Expression` and Variables"""
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.dimensionless],
        operation_name="acos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arccos(value)
    return Expression(expression=f"math.acos({value})")


@ensure_scalar_input
def atan(value: ScalarInputType):
    """Customized ArcTangent function to work with the `Expression` and Variables"""
    _check_operation_dimensionality(
        value=value,
        dimensionalities=[dimensions.dimensionless],
        operation_name="atan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arctan(value)
    return Expression(expression=f"math.atan({value})")


@ensure_scalar_input
def min(value1: ScalarInputType, value2: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Min function to work with the `Expression` and Variables"""
    _check_same_dimensionality(value1=value1, value2=value2, operation_name="min")
    print(type(value1), type(value2))
    if isinstance(value1, (unyt_quantity, Number)) and isinstance(value2, (unyt_quantity, Number)):
        return np.minimum(value1, value2)
    if isinstance(value1, (Expression, Variable)) and isinstance(value2, unyt_quantity):
        return Expression(expression=f"math.min({value1},{_convert_numeric(value2)})")
    if isinstance(value2, (Expression, Variable)) and isinstance(value1, unyt_quantity):
        return Expression(expression=f"math.min({_convert_numeric(value1)},{value2})")
    return Expression(expression=f"math.min({value1},{value2})")


@ensure_scalar_input
def max(value1: ScalarInputType, value2: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Max function to work with the `Expression` and Variables"""
    _check_same_dimensionality(value1=value1, value2=value2, operation_name="max")
    if isinstance(value1, (unyt_quantity, Number)) and isinstance(value2, (unyt_quantity, Number)):
        return np.maximum(value1, value2)
    if isinstance(value1, (Expression, Variable)) and isinstance(value2, unyt_quantity):
        return Expression(expression=f"math.max({value1},{_convert_numeric(value2)})")
    if isinstance(value2, (Expression, Variable)) and isinstance(value1, unyt_quantity):
        return Expression(expression=f"math.max({_convert_numeric(value1)},{value2})")
    return Expression(expression=f"math.max({value1},{value2})")


@ensure_scalar_input
def abs(value: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Absolute function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.abs(value)
    return Expression(expression=f"math.abs({value})")


@ensure_scalar_input
def ceil(value: ScalarInputType):
    """Customized Ceil function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.ceil(value)
    return Expression(expression=f"math.ceil({value})")


@ensure_scalar_input
def floor(value: ScalarInputType):
    """Customized Floor function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.floor(value)
    return Expression(expression=f"math.floor({value})")


# TODO: Unit tests for translation
