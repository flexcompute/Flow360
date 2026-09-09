"""
Math.h for Flow360 Expression system
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import Any, Literal, TypeAlias

import numpy as np
from unyt import dimensions, unyt_array, unyt_quantity

from flow360_schema.framework.expression.variable import (
    Expression,
    Variable,
    _convert_numeric,
    get_input_value_dimensions,
)


def _handle_expression_list(value: list[Any]) -> list[Any] | Expression:  # noqa: ANN401
    is_expression_list = False

    for item in value:
        if isinstance(item, Expression):
            is_expression_list = True

    if is_expression_list:
        return Expression.model_validate(value)

    return value


def _make_expression(expression: str) -> Expression:
    """Construct a runtime expression without relying on implicit model-init typing."""
    return Expression.model_validate({"expression": expression})


VectorInputType: TypeAlias = list[float] | unyt_array | Expression | Variable
ScalarInputType: TypeAlias = float | unyt_quantity | Expression | Variable


def _get_input_array_length(value: VectorInputType) -> int:
    try:
        return len(value)
    except Exception as e:
        raise ValueError(f"Cannot get length information for {value} but array-like input is expected.") from e


def _check_same_length(left: VectorInputType, right: VectorInputType, operation_name: str) -> None:
    """For vector arithmetic operations, we need to check that the vectors have the same length."""
    left_length = _get_input_array_length(left)
    right_length = _get_input_array_length(right)
    if left_length != right_length:
        raise ValueError(f"Vectors ({left} | {right}) must have the same length to perform {operation_name} operation.")


def _compare_operation_dimensions(value: ScalarInputType | VectorInputType, ref_dimensions: object) -> bool:
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector has the specify dimensions.
    """
    value_dimensions = get_input_value_dimensions(value=value)
    if value_dimensions:
        return bool(value_dimensions == ref_dimensions)
    return False


def _check_same_dimensions(
    value1: ScalarInputType | VectorInputType,
    value2: ScalarInputType | VectorInputType,
    operation_name: str,
) -> None:
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalars/vectors have the same dimensions.
    """

    def _check_list_same_dimensions(value: Any) -> None:
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
    value: ScalarInputType | VectorInputType,
    ref_dimensions: list[Any],
    operation_name: str,
) -> None:
    """
    For certain scalar/vector arithmetic operations,
    we need to check that the scalar/vector satisfies the specific dimensions.
    """

    if len(ref_dimensions) == 1:
        dimensions_err_msg = str(ref_dimensions[0])
    else:
        dimensions_err_msg = "one of (" + ", ".join([str(dimension) for dimension in ref_dimensions]) + ")"

    if not any(_compare_operation_dimensions(value=value, ref_dimensions=dimension) for dimension in ref_dimensions):
        raise ValueError(
            f"The dimensions of the input value ({value}) "
            f"must be {dimensions_err_msg} to perform {operation_name} operation."
        )


def _create_min_max_expression(
    value1: ScalarInputType, value2: ScalarInputType, operation_name: Literal["min", "max"]
) -> float | unyt_quantity | Expression:
    _check_same_dimensions(value1=value1, value2=value2, operation_name=operation_name)
    if isinstance(value1, (unyt_quantity, Number)) and isinstance(value2, (unyt_quantity, Number)):
        if operation_name == "max":
            return np.maximum(value1, value2)  # type: ignore[arg-type]
        return np.minimum(value1, value2)  # type: ignore[arg-type]
    if isinstance(value1, (Expression, Variable)) and isinstance(value2, unyt_quantity):
        return _make_expression(f"math.{operation_name}({value1},{_convert_numeric(value2)})")
    if isinstance(value2, (Expression, Variable)) and isinstance(value1, unyt_quantity):
        return _make_expression(f"math.{operation_name}({_convert_numeric(value1)},{value2})")
    return _make_expression(f"math.{operation_name}({value1},{value2})")


def cross(left: VectorInputType, right: VectorInputType) -> unyt_array | Expression:
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
        return _handle_expression_list(result)
    if len(left) == 2:
        return left[0] * right[1] - left[1] * right[0]
    raise ValueError(f"Vector length must be 2 or 3, got {len(left)}.")


def dot(left: VectorInputType, right: VectorInputType) -> unyt_quantity | float | Expression:
    """Customized Dot function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return np.dot(left, right)

    _check_same_length(left, right, "dot product")

    result = left[0] * right[0]
    for i in range(1, len(left)):
        result += left[i] * right[i]

    return result


def magnitude(value: VectorInputType) -> unyt_quantity | float | Expression:
    """Customized Magnitude function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:

    _get_input_array_length(value)

    if isinstance(value, unyt_array):
        return np.linalg.norm(value)

    # For solution variables and expressions, return an Expression
    if isinstance(value, Expression) or (isinstance(value, Variable) and hasattr(value, "solver_name")):
        return _make_expression(f"math.magnitude({value})")

    # For regular lists/arrays and UserVariables, compute the magnitude
    result = value[0] ** 2
    for i in range(1, len(value)):
        result += value[i] ** 2

    return result**0.5


def add(left: VectorInputType, right: VectorInputType) -> unyt_array | Expression:
    """Customized Add function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left + right

    _check_same_length(left, right, "add")
    _check_same_dimensions(left, right, "add")

    result = [left[i] + right[i] for i in range(len(left))]
    return _handle_expression_list(result)


def subtract(left: VectorInputType, right: VectorInputType) -> unyt_array | Expression:
    """Customized Subtract function to work with the `Expression` and Variables"""
    # Taking advantage of unyt as much as possible:
    if isinstance(left, unyt_array) and isinstance(right, unyt_array):
        return left - right

    _check_same_length(left, right, "subtract")
    _check_same_dimensions(left, right, "subtract")

    result = [left[i] - right[i] for i in range(len(left))]
    return _handle_expression_list(result)


########## Scalar functions ##########
def ensure_scalar_input(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to check if the input is a scalar and raise an error if so."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        def is_scalar(input_value: Any) -> bool:
            if isinstance(input_value, list):
                return False
            if isinstance(input_value, Number):
                return True
            if isinstance(input_value, unyt_quantity):
                return bool(input_value.shape == ())

            try:
                return len(input_value) == 0
            except Exception:
                return False

        for arg in args:
            if not is_scalar(arg):
                raise ValueError(f"Scalar function ({func.__name__}) on {arg} not supported.")
        return func(*args, **kwargs)

    return wrapper


@ensure_scalar_input
def sqrt(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Sqrt function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.sqrt(value)  # type: ignore[arg-type]
    return _make_expression(f"math.sqrt({value})")


@ensure_scalar_input
def log(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Log function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="log",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.log(value)  # type: ignore[arg-type]
    return _make_expression(f"math.log({value})")


@ensure_scalar_input
def exp(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Exponential function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="exp",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.exp(value)  # type: ignore[arg-type]
    return _make_expression(f"math.exp({value})")


@ensure_scalar_input
def sin(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Sine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="sin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.sin(value)  # type: ignore[arg-type]
    return _make_expression(f"math.sin({value})")


@ensure_scalar_input
def cos(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Cosine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="cos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.cos(value)  # type: ignore[arg-type]
    return _make_expression(f"math.cos({value})")


@ensure_scalar_input
def tan(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Tangent function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.angle, dimensions.dimensionless],
        operation_name="tan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.tan(value)  # type: ignore[arg-type]
    return _make_expression(f"math.tan({value})")


@ensure_scalar_input
def asin(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized ArcSine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="asin",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arcsin(value)  # type: ignore[arg-type]
    return _make_expression(f"math.asin({value})")


@ensure_scalar_input
def acos(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized ArcCosine function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="acos",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arccos(value)  # type: ignore[arg-type]
    return _make_expression(f"math.acos({value})")


@ensure_scalar_input
def atan(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized ArcTangent function to work with the `Expression` and Variables"""
    _check_value_dimensions(
        value=value,
        ref_dimensions=[dimensions.dimensionless],
        operation_name="atan",
    )
    if isinstance(value, (unyt_quantity, Number)):
        return np.arctan(value)  # type: ignore[arg-type]
    return _make_expression(f"math.atan({value})")


@ensure_scalar_input
def min(value1: ScalarInputType, value2: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Min function to work with the `Expression` and Variables"""
    return _create_min_max_expression(value1, value2, "min")


@ensure_scalar_input
def max(value1: ScalarInputType, value2: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Max function to work with the `Expression` and Variables"""
    return _create_min_max_expression(value1, value2, "max")


@ensure_scalar_input
def abs(value: ScalarInputType) -> float | unyt_quantity | Expression:
    """Customized Absolute function to work with the `Expression` and Variables"""
    if isinstance(value, (unyt_quantity, Number)):
        return np.abs(value)  # type: ignore[arg-type]
    return _make_expression(f"math.abs({value})")


pi = np.pi
