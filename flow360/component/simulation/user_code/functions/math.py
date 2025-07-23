"""
Math.h for Flow360 Expression system
"""

from numbers import Number
from typing import Any, Literal, Union

import numpy as np
from unyt import dimensions, unyt_array, unyt_quantity

from flow360.component.simulation.user_code.core.types import (
    Expression,
    Variable,
    _check_list_items_are_same_dimensions,
    _convert_numeric,
    get_input_value_dimensions,
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


def _check_same_dimensions(
    value1: Union[ScalarInputType, VectorInputType],
    value2: Union[ScalarInputType, VectorInputType],
    operation_name: str,
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalars/vectors have the same dimensions.
    """

    def _check_list_same_dimensions(value):
        if not isinstance(value, list) or len(value) <= 1:
            return
        try:
            _check_list_items_are_same_dimensions(value=value)
        except ValueError:
            # pylint:disable = raise-missing-from
            raise ValueError(
                f"Each item in the input value ({value}) must have the same dimensions "
                f"to perform {operation_name} operation."
            )

    _check_list_same_dimensions(value=value1)
    _check_list_same_dimensions(value=value2)
    value1_dimensions = get_input_value_dimensions(value=value1)
    value2_dimensions = get_input_value_dimensions(value=value2)
    if value1_dimensions != value2_dimensions:
        raise ValueError(
            f"Input values ({value1} | {value2}) must have the same dimensions to perform {operation_name} operation."
        )


def _compare_operation_dimensions(value: Union[ScalarInputType, VectorInputType], ref_dimensions):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector has the specify dimensions.
    """
    value_dimensions = get_input_value_dimensions(value=value)
    if value_dimensions:
        return value_dimensions == ref_dimensions
    return False


def _check_same_dimensions(
    value1: Union[ScalarInputType, VectorInputType],
    value2: Union[ScalarInputType, VectorInputType],
    operation_name: str,
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalars/vectors have the same dimensions.
    """

    def _check_list_same_dimensions(value):
        if not isinstance(value, list) or len(value) <= 1:
            return
        value_0_dim = get_input_value_dimensions(value=value[0])
        if not all(get_input_value_dimensions(value=item) == value_0_dim for item in value):
            raise ValueError(
                f"Each item in the input value ({value}) must have the same dimensions "
                f"to perform {operation_name} operation."
            )

    _check_list_same_dimensions(value=value1)
    _check_list_same_dimensions(value=value2)
    value1_dimensions = get_input_value_dimensions(value=value1)
    value2_dimensions = get_input_value_dimensions(value=value2)
    if value1_dimensions != value2_dimensions:
        raise ValueError(
            f"Input values ({value1} | {value2}) must have the same dimensions to perform {operation_name} operation."
        )


def _check_value_dimensions(
    value: Union[ScalarInputType, VectorInputType],
    ref_dimensions: list,
    operation_name: str,
):
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector satisfies the specific dimensions.
    """

    if len(ref_dimensions) == 1:
        dimensions_err_msg = str(ref_dimensions[0])
    else:
        dimensions_err_msg = (
            "one of (" + ", ".join([str(dimension) for dimension in ref_dimensions]) + ")"
        )

    if not any(
        _compare_operation_dimensions(value=value, ref_dimensions=dimension)
        for dimension in ref_dimensions
    ):
        raise ValueError(
            f"The dimensions of the input value ({value}) "
            f"must be {dimensions_err_msg} to perform {operation_name} operation."
        )


def _create_min_max_expression(
    value1: ScalarInputType, value2: ScalarInputType, operation_name: Literal["min", "max"]
):
    _check_same_dimensions(value1=value1, value2=value2, operation_name=operation_name)
    if isinstance(value1, (unyt_quantity, Number)) and isinstance(value2, (unyt_quantity, Number)):
        return np.maximum(value1, value2) if operation_name == "max" else np.minimum(value1, value2)
    if isinstance(value1, (Expression, Variable)) and isinstance(value2, unyt_quantity):
        return Expression(expression=f"math.{operation_name}({value1},{_convert_numeric(value2)})")
    if isinstance(value2, (Expression, Variable)) and isinstance(value1, unyt_quantity):
        return Expression(expression=f"math.{operation_name}({_convert_numeric(value1)},{value2})")
    return Expression(expression=f"math.{operation_name}({value1},{value2})")


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
    _check_same_dimensions(left, right, "add")

    result = [left[i] + right[i] for i in range(len(left))]
    return _handle_expression_list(result)


def subtract(left: VectorInputType, right: VectorInputType):
    """Customized Subtract function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left - right

    _check_same_length(left, right, "subtract")
    _check_same_dimensions(left, right, "subtract")

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
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="log",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.log(value)
    return Expression(expression=f"math.log({value})")


@ensure_scalar_input
def exp(value: ScalarInputType):
    """Customized Exponential function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="exp",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.exp(value)
    return Expression(expression=f"math.exp({value})")


@ensure_scalar_input
def sin(value: ScalarInputType):
    """Customized Sine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="sin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.sin(value)
    return Expression(expression=f"math.sin({value})")


@ensure_scalar_input
def cos(value: ScalarInputType):
    """Customized Cosine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="cos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.cos(value)
    return Expression(expression=f"math.cos({value})")


@ensure_scalar_input
def tan(value: ScalarInputType):
    """Customized Tangent function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="tan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.tan(value)
    return Expression(expression=f"math.tan({value})")


@ensure_scalar_input
def asin(value: ScalarInputType):
    """Customized ArcSine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="asin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arcsin(value)
    return Expression(expression=f"math.asin({value})")


@ensure_scalar_input
def acos(value: ScalarInputType):
    """Customized ArcCosine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="acos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arccos(value)
    return Expression(expression=f"math.acos({value})")


@ensure_scalar_input
def atan(value: ScalarInputType):
    """Customized ArcTangent function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="atan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arctan(value)
    return Expression(expression=f"math.atan({value})")


@ensure_scalar_input
def min(value1: ScalarInputType, value2: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Min function to work with the `Expression` and Variables"""
    return _create_min_max_expression(value1, value2, "min")


@ensure_scalar_input
def max(value1: ScalarInputType, value2: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Max function to work with the `Expression` and Variables"""
    return _create_min_max_expression(value1, value2, "max")


@ensure_scalar_input
def abs(value: ScalarInputType):  # pylint: disable=redefined-builtin
    """Customized Absolute function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.abs(value)
    return Expression(expression=f"math.abs({value})")


pi = np.pi
