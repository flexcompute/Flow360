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


def magnitude(value: VectorInputType):
    """Customized Magnitude function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(value, unyt_array):
        return np.linalg.norm(value)

    result = value[0] ** 2
    for i in range(1, len(value)):
        result += value[i] ** 2

    return result**0.5


def subtract(left: VectorInputType, right: VectorInputType):
    """Customized Subtract function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left - right

    _check_same_length(left, right, "subtract")

    if len(left) == 3:
        result = [
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2],
        ]
    elif len(left) == 2:
        result = [left[0] - right[0], left[1] - right[1]]
    else:
        raise ValueError(f"Vector length must be 2 or 3, got {len(left)}.")

    return _handle_expression_list(result)


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


# pylint: disable=redefined-builtin
def pow(base: ScalarInputType, exponent: ScalarInputType):
    # pylint: disable=fixme
    # TODO: Needs to ensure the exponent is a float or a unyt_quantity/Expression with units "1"
    """Customized Power function to work with the `Expression` and Variables"""
    if isinstance(base, (unyt_quantity, Number)) and isinstance(exponent, Number):
        return np.power(base, exponent)
    return Expression(expression=f"math.pow({base}, {exponent})")


@ensure_scalar_input
def log(value: ScalarInputType):
    """Customized Log function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.log(value)
    return Expression(expression=f"math.log({value})")


@ensure_scalar_input
def exp(value: ScalarInputType):
    """Customized Exponential function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.exp(value)
    return Expression(expression=f"math.exp({value})")


@ensure_scalar_input
def sin(value: ScalarInputType):
    """Customized Sine function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.sin(value)
    return Expression(expression=f"math.sin({value})")


@ensure_scalar_input
def cos(value: ScalarInputType):
    """Customized Cosine function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.cos(value)
    return Expression(expression=f"math.cos({value})")


@ensure_scalar_input
def tan(value: ScalarInputType):
    """Customized Tangent function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.tan(value)
    return Expression(expression=f"math.tan({value})")


@ensure_scalar_input
def asin(value: ScalarInputType):
    """Customized ArcSine function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.arcsin(value)
    return Expression(expression=f"math.asin({value})")


@ensure_scalar_input
def acos(value: ScalarInputType):
    """Customized ArcCosine function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.arccos(value)
    return Expression(expression=f"math.acos({value})")


@ensure_scalar_input
def atan(value: ScalarInputType):
    """Customized ArcTangent function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.arctan(value)
    return Expression(expression=f"math.atan({value})")


@ensure_scalar_input
def min(value: ScalarInputType):
    """Customized Min function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.min(value)
    return Expression(expression=f"math.min({value})")


@ensure_scalar_input
def max(value: ScalarInputType):
    """Customized Max function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.max(value)
    return Expression(expression=f"math.max({value})")


@ensure_scalar_input
def abs(value: ScalarInputType):
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


def pi():
    """Customized Pi function to work with the `Expression` and Variables"""
    return np.pi
