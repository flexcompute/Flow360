import re

import numpy as np
import pydantic as pd
import pytest
import unyt as u

import flow360.component.simulation.user_code.core.context as context
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.material import Water
from flow360.component.simulation.operating_condition.operating_condition import (
    LiquidOperatingCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system, VelocityType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    ValueOrExpression,
)
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution


@pytest.fixture(autouse=True)
def reset_context():
    context.default_context.clear()


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def scaling_provider():
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=50 * u.m / u.s,
                reference_velocity_magnitude=100 * u.m / u.s,
                material=Water(name="water"),
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=10 * u.m),
        )
    return params


# ---------------------------#
# Cross product
# ---------------------------#
def test_cross_product():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType.Vector] = pd.Field()

    x = UserVariable(name="x", value=[1, 2, 3])

    model = TestModel(field=math.cross(x, [3, 2, 1]) * u.m / u.s)
    assert (
        str(model.field)
        == "[x[1] * 1 - x[2] * 2, x[2] * 3 - x[0] * 1, x[0] * 2 - x[1] * 3] * u.m / u.s"
    )

    assert (model.field.evaluate() == [-4, 8, -4] * u.m / u.s).all()

    model = TestModel(field="math.cross(x, [3, 2, 1]) * u.m / u.s")
    assert str(model.field) == "math.cross(x, [3, 2, 1]) * u.m / u.s"

    result = model.field.evaluate()
    assert (result == [-4, 8, -4] * u.m / u.s).all()

    some_var = UserVariable(name="some_var", value=[1, 2] * u.m)
    a = UserVariable(name="a", value=math.cross(some_var, [3, 1]) * u.m / u.s)
    assert str(a.value) == "(some_var[0] * 1 - some_var[1] * 3) * u.m / u.s"
    result = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
    assert result == -5 * u.m * u.m / u.s

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vectors ([1, 2] | [3 1 2 3] m) must have the same length to perform cross product operation."
        ),
    ):
        a.value = math.cross(
            [
                1,
                2,
            ],
            [3, 1, 2, 3] * u.m,
        )

    with pytest.raises(ValueError, match="Vector length must be 2 or 3, got 4."):
        a.value = math.cross([1, 2, 2, 3], [3, 1, 2, 3] * u.m)


def test_cross_function_use_case(scaling_provider):

    print("\n1 Python mode\n")
    a = UserVariable(name="a", value=math.cross([3, 2, 1] * u.m, solution.coordinate))
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1], "
        "1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2], "
        "3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({(((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])), (((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])), (((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0]))})"
    )

    print("\n1.1 Python mode but arg swapped\n")
    a.value = math.cross(solution.coordinate, [3, 2, 1] * u.m)
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[solution.coordinate[1] * 1 * u.m - solution.coordinate[2] * 2 * u.m, "
        "solution.coordinate[2] * 3 * u.m - solution.coordinate[0] * 1 * u.m, "
        "solution.coordinate[0] * 2 * u.m - solution.coordinate[1] * 3 * u.m]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({(((coordinate[1] * 1) * 0.1) - ((coordinate[2] * 2) * 0.1)), (((coordinate[2] * 3) * 0.1) - ((coordinate[0] * 1) * 0.1)), (((coordinate[0] * 2) * 0.1) - ((coordinate[1] * 3) * 0.1))})"
    )

    print("\n2 Taking advantage of unyt as much as possible\n")
    a.value = math.cross([3, 2, 1] * u.m, [2, 2, 1] * u.m)
    assert all(a.value == [0, -1, 2] * u.m * u.m)

    print("\n3 (Units defined in components)\n")
    a.value = math.cross([3 * u.m, 2 * u.m, 1 * u.m], [2 * u.m, 2 * u.m, 1 * u.m])
    assert all(a.value == [0, -1, 2] * u.m * u.m)

    print("\n4 Serialized version\n")
    a.value = "math.cross([3, 2, 1] * u.m, solution.coordinate)"
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1], "
        "1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2], "
        "3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({(((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])), (((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])), (((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0]))})"
    )

    print("\n5 Recursive cross in Python mode\n")
    a.value = math.cross(math.cross([3, 2, 1] * u.m, solution.coordinate), [3, 2, 1] * u.m)
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[(1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 1 * u.m - (3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 2 * u.m, "
        "(3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 3 * u.m - (2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 1 * u.m, "
        "(2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 2 * u.m - (1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 3 * u.m]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({((((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 1) * 0.1) - (((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 2) * 0.1)), ((((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 3) * 0.1) - (((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 1) * 0.1)), ((((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 2) * 0.1) - (((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 3) * 0.1))})"
    )

    print("\n6 Recursive cross in String mode\n")
    a.value = "math.cross(math.cross([3, 2, 1] * u.m, solution.coordinate), [3, 2, 1] * u.m)"
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert (
        str(res)
        == "[(1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 1 * u.m - (3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 2 * u.m, "
        "(3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 3 * u.m - (2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 1 * u.m, "
        "(2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 2 * u.m - (1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 3 * u.m]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({((((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 1) * 0.1) - (((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 2) * 0.1)), ((((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 3) * 0.1) - (((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 1) * 0.1)), ((((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 2) * 0.1) - (((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 3) * 0.1))})"
    )

    print("\n7 Using other variables in Python mode\n")
    b = UserVariable(name="b", value=math.cross([3, 2, 1] * u.m, solution.coordinate))
    a.value = math.cross(b, [3, 2, 1] * u.m)
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[(1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 1 * u.m - (3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 2 * u.m, "
        "(3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]) * 3 * u.m - (2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 1 * u.m, "
        "(2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1]) * 2 * u.m - (1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2]) * 3 * u.m]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({((((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 1) * 0.1) - (((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 2) * 0.1)), ((((((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0])) * 3) * 0.1) - (((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 1) * 0.1)), ((((((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])) * 2) * 0.1) - (((((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])) * 3) * 0.1))})"
    )

    print("\n8 Using other constant variables in Python mode\n")
    b.value = [3, 2, 1] * u.m
    a.value = math.cross(b, solution.coordinate)
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1], "
        "1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2], "
        "3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({(((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])), (((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])), (((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0]))})"
    )

    print("\n9 Using non-unyt_array\n")
    b.value = [3 * u.m, 2 * u.m, 1 * u.m]
    a.value = math.cross(b, solution.coordinate)
    res = a.value.evaluate(raise_on_non_evaluable=False, force_evaluate=False)
    assert str(res) == (
        "[2 * u.m * solution.coordinate[2] - 1 * u.m * solution.coordinate[1], "
        "1 * u.m * solution.coordinate[0] - 3 * u.m * solution.coordinate[2], "
        "3 * u.m * solution.coordinate[1] - 2 * u.m * solution.coordinate[0]]"
    )
    assert (
        a.value.to_solver_code(scaling_provider)
        == "std::vector<float>({(((2 * 0.1) * coordinate[2]) - ((1 * 0.1) * coordinate[1])), (((1 * 0.1) * coordinate[0]) - ((3 * 0.1) * coordinate[2])), (((3 * 0.1) * coordinate[1]) - ((2 * 0.1) * coordinate[0]))})"
    )


# ---------------------------#
# Dot product
# ---------------------------#
def test_dot_product():
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    y = UserVariable(name="y", value=[1, 2, 3] * u.K)

    # Python mode
    assert math.dot(x, y).evaluate() == 14 * u.m * u.K
    assert math.dot(x, [1, 2, 3] * u.m).evaluate() == 14 * u.m * u.m
    assert math.dot([1, 2, 3] * u.m, [1, 2, 3] * u.m) == 14 * u.m * u.m
    assert math.dot([1, 2] * u.m, [1, 2] * u.m) == 5 * u.m * u.m
    assert math.dot([1 * u.m, 2 * u.m], [1 * u.s, 3 * u.s]) == 7 * u.m * u.s
    assert (
        str(math.dot(solution.coordinate, solution.velocity))
        == "solution.coordinate[0] * solution.velocity[0] + "
        "solution.coordinate[1] * solution.velocity[1] + "
        "solution.coordinate[2] * solution.velocity[2]"
    )
    assert (
        str(math.dot(solution.coordinate, x))
        == "solution.coordinate[0] * x[0] + solution.coordinate[1] * x[1] + solution.coordinate[2] * x[2]"
    )

    # String mode
    assert Expression(expression="math.dot(x, y)").evaluate() == 14 * u.m * u.K
    assert Expression(expression="math.dot(x, [1, 2, 3] * u.m)").evaluate() == 14 * u.m * u.m
    assert (
        Expression(expression="math.dot([1, 2, 3] * u.m, [1, 2, 3] * u.m)").evaluate()
        == 14 * u.m * u.m
    )
    assert Expression(expression="math.dot([1, 2] * u.m, [1, 2] * u.m)").evaluate() == 5 * u.m * u.m
    assert (
        Expression(expression="math.dot([1* u.m, 2* u.m] , [1* u.s, 3* u.s])").evaluate()
        == 7 * u.m * u.s
    )  # We do not probably want to user to know this works though....

    # Error handling
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vectors ([1 2] m | y) must have the same length to perform dot product operation."
        ),
    ):
        math.dot([1, 2] * u.m, y)


# ---------------------------#
# Add and Subtract
# ---------------------------#
def test_add_vector():
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    y = UserVariable(name="y", value=[4, 5, 6] * u.m)

    # Python mode
    assert all(math.add(x, y).evaluate() == [5, 7, 9] * u.m)
    assert all(math.add(x, [1, 2, 3] * u.m).evaluate() == [2, 4, 6] * u.m)
    assert all(math.add([1, 2, 3] * u.m, [1, 2, 3] * u.m) == [2, 4, 6] * u.m)
    assert all(math.add([1, 2] * u.m, [1, 2] * u.m) == [2, 4] * u.m)
    assert all(math.add([1 * u.s, 2 * u.s], [1 * u.s, 3 * u.s]) == [2, 5] * u.s)
    assert (
        str(math.add(solution.grad_u, solution.grad_w))
        == "[solution.grad_u[0] + solution.grad_w[0], "
        "solution.grad_u[1] + solution.grad_w[1], "
        "solution.grad_u[2] + solution.grad_w[2]]"
    )
    assert (
        str(math.add(solution.coordinate, x))
        == "[solution.coordinate[0] + x[0], solution.coordinate[1] + x[1], solution.coordinate[2] + x[2]]"
    )

    # String mode
    assert all(Expression(expression="math.add(x, y)").evaluate() == [5, 7, 9] * u.m)
    assert all(Expression(expression="math.add(x, [1, 2, 3] * u.m)").evaluate() == [2, 4, 6] * u.m)
    assert all(
        Expression(expression="math.add([1, 2, 3] * u.m, [1, 2, 3] * u.m)").evaluate()
        == [2, 4, 6] * u.m
    )
    assert all(
        Expression(expression="math.add([1, 2] * u.m, [1, 2] * u.m)").evaluate() == [2, 4] * u.m
    )
    assert all(
        Expression(expression="math.add([1 * u.s, 2 * u.s] , [1 * u.s, 3 * u.s])").evaluate()
        == [2, 5] * u.s
    )

    # Error handling
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vectors ([1 2] m | y) must have the same length to perform add operation."
        ),
    ):
        math.add([1, 2] * u.m, y)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values ([1 2 3] s | y) must have the same dimensinality to perform add operation."
        ),
    ):
        math.add([1, 2, 3] * u.s, y)


def test_subtract_vector():
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    y = UserVariable(name="y", value=[-4, -5, -6] * u.m)

    # Python mode
    assert all(math.subtract(x, y).evaluate() == [5, 7, 9] * u.m)
    assert all(math.subtract(x, [-1, -2, -3] * u.m).evaluate() == [2, 4, 6] * u.m)
    assert all(math.subtract([-1, -2, -3] * u.m, [1, 2, 3] * u.m) == [-2, -4, -6] * u.m)
    assert all(math.subtract([1, 2] * u.m, [-1, -2] * u.m) == [2, 4] * u.m)
    assert all(math.subtract([1 * u.s, -2 * u.s], [-1 * u.s, 3 * u.s]) == [2, -5] * u.s)
    assert (
        str(math.subtract(solution.grad_u, solution.grad_w))
        == "[solution.grad_u[0] - solution.grad_w[0], "
        "solution.grad_u[1] - solution.grad_w[1], "
        "solution.grad_u[2] - solution.grad_w[2]]"
    )
    assert (
        str(math.subtract(solution.coordinate, x))
        == "[solution.coordinate[0] - x[0], solution.coordinate[1] - x[1], solution.coordinate[2] - x[2]]"
    )

    # String mode
    assert all(Expression(expression="math.subtract(x, y)").evaluate() == [5, 7, 9] * u.m)
    assert all(
        Expression(expression="math.subtract(x, [-1, -2, -3] * u.m)").evaluate() == [2, 4, 6] * u.m
    )
    assert all(
        Expression(expression="math.subtract([-1, -2, -3] * u.m, [1, 2, 3] * u.m)").evaluate()
        == [-2, -4, -6] * u.m
    )
    assert all(
        Expression(expression="math.subtract([1, 2] * u.m, [-1, -2] * u.m)").evaluate()
        == [2, 4] * u.m
    )
    assert all(
        Expression(expression="math.subtract([1 * u.s, -2 * u.s] , [-1 * u.s, 3 * u.s])").evaluate()
        == [2, -5] * u.s
    )

    # Error handling
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vectors ([1 2] m | y) must have the same length to perform subtract operation."
        ),
    ):
        math.subtract([1, 2] * u.m, y)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values ([1 2 3] s | y) must have the same dimensinality to perform subtract operation."
        ),
    ):
        math.subtract([1, 2, 3] * u.s, y)


# ---------------------------#
# Scalar functions with ensure_scalar_input wrapper
# ---------------------------#
def test_sqrt_scalar_input():
    """Test sqrt function with valid scalar inputs."""

    # Test with regular numbers
    assert math.sqrt(4) == 2.0
    assert math.sqrt(0) == 0.0
    assert math.sqrt(2) == pytest.approx(1.4142135623730951)

    with pytest.raises(RuntimeWarning, match="invalid value encountered in sqrt"):
        math.sqrt(-1)

    # Test with unyt quantities
    assert math.sqrt(4 * u.m * u.m) == 2 * u.m
    assert math.sqrt(9 * u.K * u.K) == 3 * u.K
    assert math.sqrt(16 * u.m * u.m / u.s / u.s) == 4 * u.m / u.s

    # Test with expressions
    x = UserVariable(name="x", value=4 * u.m * u.m)
    result = math.sqrt(x)
    assert str(result) == "math.sqrt(x)"
    assert result.evaluate() == 2 * u.m

    x.value = 9 * u.m
    result = math.sqrt(x)
    assert result.evaluate() == 3 * u.Unit("sqrt(m)")

    # Test with string expressions
    expr = Expression(expression="math.sqrt(16)")
    assert expr.evaluate() == 4.0


def test_sqrt_non_scalar_input_errors():
    """Test sqrt function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on [1, 2, 3] not supported.")
    ):
        math.sqrt([1, 2, 3])

    with pytest.raises(
        ValueError,
        match=re.escape("Scalar function (sqrt) on [unyt_quantity(1, 'm'), unyt_quantity(2, 'm')]"),
    ):
        math.sqrt([1 * u.m, 2 * u.m])

    # Test with unyt arrays
    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on [1 2 3] m not supported.")
    ):
        math.sqrt(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match=re.escape("Scalar function (sqrt) on x not supported.")):
        math.sqrt(x)

    # Test with solution variables (which are arrays)
    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on solution.coordinate not supported.")
    ):
        math.sqrt(solution.coordinate)

    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on solution.velocity not supported.")
    ):
        math.sqrt(solution.velocity)


def test_sqrt_edge_cases():
    """Test sqrt function with edge cases."""

    math.sqrt([])

    # Test with None (should raise TypeError from numpy)
    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on None not supported.")
    ):
        math.sqrt(None)

    # Test with string (should raise ValueError from numpy)
    with pytest.raises(
        ValueError, match=re.escape("Scalar function (sqrt) on not a number not supported.")
    ):
        math.sqrt("not a number")


def test_sqrt_with_expressions(scaling_provider):
    """Test sqrt function with various expression types."""

    # Test with UserVariable containing scalar
    x = UserVariable(name="x", value=9 * u.m * u.m)
    result = math.sqrt(x)
    assert str(result) == "math.sqrt(x)"
    assert result.evaluate() == 3 * u.m

    # Test with Expression containing scalar
    expr = Expression(expression="math.sqrt(25)")
    assert expr.evaluate() == 5.0

    # Test with nested expressions
    y = UserVariable(name="y", value=4 * u.m)
    z = UserVariable(name="z", value=16 * u.m * u.m)
    result = math.sqrt(y * math.sqrt(z))
    assert str(result) == "math.sqrt(y * math.sqrt(z))"
    assert result.evaluate() == 4 * u.m

    # Test with solution variables
    result = math.sqrt(solution.Cp * math.sqrt(solution.mut))
    assert str(result) == "math.sqrt(solution.Cp * math.sqrt(solution.mut))"
    assert result.to_solver_code(scaling_provider) == "sqrt((___Cp * sqrt(mut)))"


# ---------------------------#
# Magnitude function
# ---------------------------#
def test_magnitude():
    """Test magnitude function with various inputs."""

    # Test with regular lists
    assert math.magnitude([3, 4]) == 5.0
    assert math.magnitude([1, 1, 1]) == pytest.approx(1.7320508075688772)
    assert math.magnitude([0, 0, 0]) == 0.0

    # Test with unyt quantities
    assert math.magnitude([3, 4] * u.m) == 5 * u.m
    assert math.magnitude([1, 1, 1] * u.K).value == pytest.approx(1.7320508075688772)

    # Test with variables
    x = UserVariable(name="x", value=[3, 4] * u.m)
    result = math.magnitude(x)
    assert str(result) == "(x[0] ** 2 + x[1] ** 2) ** 0.5"
    assert result.evaluate() == 5 * u.m

    # Test with solution variables
    result = math.magnitude(solution.coordinate)
    assert str(result) == "math.magnitude(solution.coordinate)"

    # Test with string expressions
    expr = Expression(expression="math.magnitude([3, 4])")
    assert expr.evaluate() == 5.0


def test_magnitude_errors():
    """Test magnitude function error handling."""

    # Test with scalar input
    with pytest.raises(ValueError, match="Cannot get length information"):
        math.magnitude(5)

    with pytest.raises(ValueError, match="Cannot get length information"):
        math.magnitude(5 * u.m)


# ---------------------------#
# Subtract function
# ---------------------------#
def test_subtract():
    """Test subtract function with various inputs."""

    # Test with regular lists
    assert math.subtract([5, 3], [2, 1]) == [3, 2]
    assert math.subtract([1, 2, 3], [0, 1, 2]) == [1, 1, 1]

    # Test with unyt quantities
    assert all(math.subtract([5, 3] * u.m, [2, 1] * u.m) == [3, 2] * u.m)
    assert all(math.subtract([1, 2, 3] * u.K, [0, 1, 2] * u.K) == [1, 1, 1] * u.K)

    # Test with expressions
    x = UserVariable(name="x", value=[5, 3] * u.m)
    y = UserVariable(name="y", value=[2, 1] * u.m)
    result = math.subtract(x, y)
    assert str(result) == "[x[0] - y[0], x[1] - y[1]]"
    assert all(result.evaluate() == [3, 2] * u.m)

    # Test with solution variables
    result = math.subtract(solution.coordinate, [1, 0, 0] * u.m)
    assert (
        str(result)
        == "[solution.coordinate[0] - 1 * u.m, solution.coordinate[1] - 0 * u.m, solution.coordinate[2] - 0 * u.m]"
    )

    # Test with string expressions
    expr = Expression(expression="math.subtract([5, 3], [2, 1])")
    assert expr.evaluate() == [3, 2]


def test_subtract_errors():
    """Test subtract function error handling."""

    # Test with different lengths
    with pytest.raises(ValueError, match="must have the same length"):
        math.subtract([1, 2], [1, 2, 3])

    # Test with scalar inputs
    with pytest.raises(ValueError, match="Cannot get length information"):
        math.subtract(5, 3)


# ---------------------------#
# Power operator (**)
# ---------------------------#
def test_power_operator_scalar_input():
    """Test ** operator with valid scalar inputs."""

    # Test with regular numbers
    assert 2**3 == 8.0
    assert 4**0.5 == 2.0
    assert 2**0 == 1.0
    assert 0**2 == 0.0

    # Test with unyt quantities
    assert (2 * u.m) ** 2 == 4 * u.m * u.m
    assert (4 * u.K) ** 0.5 == 2 * u.Unit("sqrt(K)")

    # Test with expressions
    x = UserVariable(name="x", value=2 * u.m)
    result = x**2
    assert str(result) == "x ** 2"
    assert result.evaluate() == 4 * u.m * u.m

    # Test with string expressions
    expr = Expression(expression="2 ** 3")
    assert expr.evaluate() == 8.0


def test_power_operator_non_scalar_input_errors():
    """Test ** operator raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for \\*\\* or pow\\(\\)"):
        [1, 2, 3] ** 2

    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for \\*\\* or pow\\(\\)"):
        2 ** [1, 2, 3]

    # Test with unyt arrays
    with pytest.raises(ValueError, match="is not well defined."):
        2 ** ([1, 2, 3] * u.m)

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    assert all((x**2).evaluate() == [1, 4, 9] * u.m * u.m)

    # Test with expressions
    expr = Expression(expression="2 ** [1, 2, 3]")
    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for \\*\\* or pow\\(\\)"):
        expr.evaluate()


def test_power_operator_with_expressions(scaling_provider):
    """Test ** operator with various expression types."""

    # Test with UserVariable containing scalar
    x = UserVariable(name="x", value=3 * u.m)
    result = x**2
    assert str(result) == "x ** 2"
    assert result.evaluate() == 9 * u.m * u.m

    # Test with Expression containing scalar
    expr = Expression(expression="4 ** 2")
    assert expr.evaluate() == 16.0

    # Test with nested expressions
    y = UserVariable(name="y", value=2 * u.m)
    z = UserVariable(name="z", value=3)
    result = y**z
    assert str(result) == "y ** z"
    assert result.evaluate() == 8 * u.m * u.m * u.m


def test_power_operator_exponent_validation():
    """Test ** operator exponent validation - exponents must be dimensionless scalars."""

    # Test with unyt_quantity exponent (should raise error)
    with pytest.raises(ValueError, match="is not well defined."):
        (2 * u.m) ** (3 * u.m)

    with pytest.raises(ValueError, match="is not well defined."):
        (2 * u.m) ** (3 * u.K)

    with pytest.raises(ValueError, match="is not well defined."):
        2 ** (3 * u.m)

    # Test with Variable containing unyt_quantity exponent (should raise error)
    x = UserVariable(name="x", value=3 * u.m)
    with pytest.raises(ValueError, match="is not well defined."):
        ((2 * u.m) ** x).evaluate()

    # Test with Variable containing dimensionless scalar (should work)
    y = UserVariable(name="y", value=3)  # dimensionless scalar
    result = (2 * u.m) ** y
    assert str(result) == "(2 * u.m) ** y"
    assert result.evaluate() == 8 * u.m * u.m * u.m

    # Test with Variable containing dimensionless unyt_quantity (should work)
    z = UserVariable(name="z", value=3 * u.dimensionless)
    result = (2 * u.m) ** z
    assert str(result) == "(2 * u.m) ** z"
    assert result.evaluate() == 8 * u.m * u.m * u.m


def test_power_operator_exponent_validation_edge_cases():
    """Test ** operator exponent validation with edge cases."""

    # Test with zero exponent (should work)
    result = (2 * u.m) ** 0
    assert result == 1.0

    # Test with negative exponent (should work)
    result = (2.0 * u.m) ** (-1)
    assert result == 0.5 / u.m

    # Test with fractional exponent (should work)
    result = (4 * u.m * u.m) ** 0.5
    assert result == 2 * u.m

    # Test with Variable containing zero (should work)
    zero_var = UserVariable(name="zero_var", value=0)
    result = (2 * u.m) ** zero_var
    assert str(result) == "(2 * u.m) ** zero_var"
    assert result.evaluate() == 1.0

    # Test with Variable containing negative number (should work)
    neg_var = UserVariable(name="neg_var", value=-1)
    result = (2 * u.m) ** neg_var
    assert str(result) == "(2 * u.m) ** neg_var"
    assert result.evaluate() == 0.5 / u.m

    # Test with Expression containing zero (should work)
    zero_expr = Expression(expression="0")
    result = (2 * u.m) ** zero_expr
    assert str(result) == "(2 * u.m) ** 0"
    assert result.evaluate() == 1.0


def test_power_operator_exponent_validation_complex_expressions():
    """Test ** operator exponent validation with complex expressions."""

    # # Test with Variable containing complex dimensionless expression
    # x = UserVariable(name="x", value=2)
    # y = UserVariable(name="y", value=1)
    # complex_var = UserVariable(name="complex_var", value=x + y)  # 2 + 1 = 3 (dimensionless)
    # result = (2 * u.m) ** complex_var
    # assert str(result) == "(2 * u.m) ** complex_var"
    # assert result.evaluate() == 8 * u.m * u.m * u.m

    # Test with Expression containing complex dimensionless expression
    complex_expr = Expression(expression="2 + 1")  # 3 (dimensionless)
    result = (2 * u.m) ** complex_expr
    assert str(result) == "(2 * u.m) ** (2 + 1)"
    assert result.evaluate() == 8 * u.m * u.m * u.m

    # Test with Variable containing expression that has units (should raise error)
    x_with_units = UserVariable(name="x_with_units", value=2 * u.m)
    y_with_units = UserVariable(name="y_with_units", value=1 * u.m)
    complex_var_with_units = UserVariable(
        name="complex_var_with_units", value=x_with_units + y_with_units
    )
    with pytest.raises(ValueError, match="not well defined."):
        ((2 * u.m) ** complex_var_with_units).evaluate()

        # Test with Expression containing expression that has units (should raise error)
    complex_expr_with_units = Expression(expression="2 * u.m + 1 * u.m")
    with pytest.raises(ValueError, match="not well defined."):
        ((2 * u.m) ** complex_var_with_units).evaluate()


# ---------------------------#
# Logarithm function
# ---------------------------#
def test_log_scalar_input():
    """Test log function with valid scalar inputs."""

    # Test with regular numbers
    assert math.log(1) == 0.0
    assert math.log(math.exp(1)) == pytest.approx(1.0)
    assert math.log(10) == pytest.approx(2.302585092994046)

    # Test with unyt quantities
    assert math.log(1 * u.m / u.m) == 0.0
    assert math.log(math.exp(1) * u.K / u.K) == pytest.approx(1.0)

    # Test with expressions
    x = UserVariable(name="x", value=10 * u.m / u.m)
    result = math.log(x)
    assert str(result) == "math.log(x)"
    assert result.evaluate() == pytest.approx(2.302585092994046)

    # Test with string expressions
    expr = Expression(expression="math.log(10)")
    assert expr.evaluate() == pytest.approx(2.302585092994046)


def test_log_non_scalar_input_errors():
    """Test log function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.log([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.log(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.log(x)


def test_log_edge_cases():
    """Test log function with edge cases."""

    # Test with zero (should raise error)
    with pytest.raises(RuntimeWarning, match="divide by zero"):
        math.log(0)

    # Test with negative number (should raise error)
    with pytest.raises(RuntimeWarning, match="invalid value"):
        math.log(-1)


def test_log_with_expressions(scaling_provider):
    """Test log function with various expression types."""

    # Test with UserVariable containing scalar
    x = UserVariable(name="x", value=10 * u.m / u.m)
    result = math.log(x)
    assert str(result) == "math.log(x)"
    assert result.evaluate() == pytest.approx(2.302585092994046)

    # Test with Expression containing scalar
    expr = Expression(expression="math.log(10)")
    assert expr.evaluate() == pytest.approx(2.302585092994046)

    # Test with nested expressions
    y = UserVariable(name="y", value=2 * u.m / u.m)
    result = math.log(y * math.log(math.exp(10)))
    assert str(result) == "math.log(y * 10.0)"


# ---------------------------#
# Exponential function
# ---------------------------#
def test_exp_scalar_input():
    """Test exp function with valid scalar inputs."""

    # Test with regular numbers
    assert math.exp(0) == 1.0
    assert math.exp(1) == pytest.approx(2.718281828459045)
    assert math.exp(-1) == pytest.approx(0.36787944117144233)

    # Test with unyt quantities
    assert math.exp(0 * u.m / u.m) == 1.0
    assert math.exp(1 * u.K / u.K) == pytest.approx(2.718281828459045)

    # Test with expressions
    x = UserVariable(name="x", value=1 * u.m / u.m)
    result = math.exp(x)
    assert str(result) == "math.exp(x)"
    assert result.evaluate() == pytest.approx(2.718281828459045)

    # Test with string expressions
    expr = Expression(expression="math.exp(1)")
    assert expr.evaluate() == pytest.approx(2.718281828459045)


def test_exp_non_scalar_input_errors():
    """Test exp function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.exp([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.exp(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.exp(x)


def test_exp_with_expressions(scaling_provider):
    """Test exp function with various expression types."""

    # Test with UserVariable containing scalar
    x = UserVariable(name="x", value=1 * u.m / u.m)
    result = math.exp(x)
    assert str(result) == "math.exp(x)"
    assert result.evaluate() == pytest.approx(2.718281828459045)

    # Test with Expression containing scalar
    expr = Expression(expression="math.exp(1)")
    assert expr.evaluate() == pytest.approx(2.718281828459045)

    # Test with nested expressions
    y = UserVariable(name="y", value=2 * u.m / u.m)
    result = math.exp(y * math.log(math.exp(2)))
    assert str(result) == "math.exp(y * 2.0)"


# ---------------------------#
# Trigonometric functions
# ---------------------------#
def test_sin_scalar_input():
    """Test sin function with valid scalar inputs."""

    # Test with regular numbers
    assert math.sin(0) == 0.0
    assert math.sin(np.pi / 2) == pytest.approx(1.0)
    assert math.sin(np.pi) == pytest.approx(0.0)

    # Test with unyt quantities
    assert math.sin(0 * u.rad) == 0.0
    assert math.sin(np.pi / 2 * u.rad) == pytest.approx(1.0)

    # Test with expressions
    x = UserVariable(name="x", value=np.pi / 2 * u.rad)
    result = math.sin(x)
    assert str(result) == "math.sin(x)"
    assert result.evaluate() == pytest.approx(1.0)

    # Test with string expressions
    expr = Expression(expression="math.sin(0)")
    assert expr.evaluate() == 0.0


def test_sin_non_scalar_input_errors():
    """Test sin function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.sin([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.sin(u.unyt_array([1, 2, 3], u.rad))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.rad)
    with pytest.raises(ValueError, match="Scalar function"):
        math.sin(x)


def test_sin_dimensionality_errors():
    """Test sin function raise errors for incorrect dimensionality."""

    # Test sin with incorrect dimensionality
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (5 m) must be one of ((angle), 1) to perform sin operation."
        ),
    ):
        math.sin(5 * u.m)
    x_length = UserVariable(name="x_length", value=6 * u.m)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_length) must be one of ((angle), 1) to perform sin operation."
        ),
    ):
        math.sin(x_length)
    expr_length = Expression(expression="7 * u.m")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (7 * u.m) must be one of ((angle), 1) to perform sin operation."
        ),
    ):
        math.sin(expr_length)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (solution.Cp * math.sqrt(solution.mut)) must be one of ((angle), 1) to perform sin operation."
        ),
    ):
        math.sin(solution.Cp * math.sqrt(solution.mut))


def test_cos_scalar_input():
    """Test cos function with valid scalar inputs."""

    # Test with regular numbers
    assert math.cos(0) == 1.0
    assert math.cos(np.pi / 2) == pytest.approx(0.0)
    assert math.cos(np.pi) == pytest.approx(-1.0)

    # Test with unyt quantities
    assert math.cos(0 * u.rad) == 1.0
    assert math.cos(np.pi / 2 * u.rad) == pytest.approx(0.0)

    # Test with expressions
    x = UserVariable(name="x", value=0 * u.rad)
    result = math.cos(x)
    assert str(result) == "math.cos(x)"
    assert result.evaluate() == 1.0

    # Test with string expressions
    expr = Expression(expression="math.cos(0)")
    assert expr.evaluate() == 1.0


def test_cos_non_scalar_input_errors():
    """Test cos function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.cos([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.cos(u.unyt_array([1, 2, 3], u.rad))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.rad)
    with pytest.raises(ValueError, match="Scalar function"):
        math.cos(x)


def test_cos_dimensionality_errors():
    """Test cos function raise errors for incorrect dimensionality."""

    # Test cos with incorrect dimensionality
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (10 s) must be one of ((angle), 1) to perform cos operation."
        ),
    ):
        math.cos(10 * u.s)
    x_time = UserVariable(name="x_time", value=10 * u.s)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_time) must be one of ((angle), 1) to perform cos operation."
        ),
    ):
        math.cos(x_time)
    expr_time = Expression(expression="10 * u.s")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (10 * u.s) must be one of ((angle), 1) to perform cos operation."
        ),
    ):
        math.cos(expr_time)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (solution.Cf * math.sqrt(solution.velocity[0])) must be one of ((angle), 1) to perform cos operation."
        ),
    ):
        math.cos(solution.Cf * math.sqrt(solution.velocity[0]))


def test_tan_scalar_input():
    """Test tan function with valid scalar inputs."""

    # Test with regular numbers
    assert math.tan(0) == 0.0
    assert math.tan(np.pi / 4) == pytest.approx(1.0)

    # Test with unyt quantities
    assert math.tan(0 * u.rad) == 0.0
    assert math.tan(np.pi / 4 * u.rad) == pytest.approx(1.0)

    # Test with expressions
    x = UserVariable(name="x", value=0 * u.rad)
    result = math.tan(x)
    assert str(result) == "math.tan(x)"
    assert result.evaluate() == 0.0

    # Test with string expressions
    expr = Expression(expression="math.tan(0)")
    assert expr.evaluate() == 0.0


def test_tan_non_scalar_input_errors():
    """Test tan function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.tan([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.tan(u.unyt_array([1, 2, 3], u.rad))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.rad)
    with pytest.raises(ValueError, match="Scalar function"):
        math.tan(x)


def test_tan_dimensionality_errors():
    """Test tan function raise errors for incorrect dimensionality."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 kg) must be one of ((angle), 1) to perform tan operation."
        ),
    ):
        math.tan(2 * u.kg)
    x_mass = UserVariable(name="x_mass", value=2 * u.kg)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_mass) must be one of ((angle), 1) to perform tan operation."
        ),
    ):
        math.tan(x_mass)
    expr_mass = Expression(expression="2 * u.kg")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 * u.kg) must be one of ((angle), 1) to perform tan operation."
        ),
    ):
        math.tan(expr_mass)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (solution.Cf * math.sqrt(solution.entropy)) must be one of ((angle), 1) to perform cos operation."
        ),
    ):
        math.cos(solution.Cf * math.sqrt(solution.entropy))


# ---------------------------#
# Inverse trigonometric functions
# ---------------------------#
def test_asin_scalar_input():
    """Test asin function with valid scalar inputs."""

    # Test with regular numbers
    assert math.asin(0) == 0.0
    assert math.asin(1) == pytest.approx(np.pi / 2)
    assert math.asin(-1) == pytest.approx(-np.pi / 2)

    # Test with unyt quantities
    assert math.asin(0 * u.dimensionless) == 0.0
    assert math.asin(1 * u.dimensionless) == pytest.approx(np.pi / 2)

    # Test with expressions
    x = UserVariable(name="x", value=0 * u.dimensionless)
    result = math.asin(x)
    assert str(result) == "math.asin(x)"
    assert result.evaluate() == 0.0

    # Test with string expressions
    expr = Expression(expression="math.asin(0)")
    assert expr.evaluate() == 0.0


def test_asin_non_scalar_input_errors():
    """Test asin function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.asin([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.asin(u.unyt_array([1, 2, 3], u.dimensionless))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.dimensionless)
    with pytest.raises(ValueError, match="Scalar function"):
        math.asin(x)


def test_asin_dimensionality_errors():
    """Test asin function raise errors for incorrect dimensionality."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 K) must be 1 to perform asin operation."
        ),
    ):
        math.asin(2 * u.K)
    x_temperature = UserVariable(name="x_temperature", value=2 * u.K)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_temperature) must be 1 to perform asin operation."
        ),
    ):
        math.asin(x_temperature)
    expr_temperature = Expression(expression="2 * u.K")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 * u.K) must be 1 to perform asin operation."
        ),
    ):
        math.asin(expr_temperature)


def test_acos_scalar_input():
    """Test acos function with valid scalar inputs."""

    # Test with regular numbers
    assert math.acos(1) == 0.0
    assert math.acos(0) == pytest.approx(np.pi / 2)
    assert math.acos(-1) == pytest.approx(np.pi)

    # Test with unyt quantities
    assert math.acos(1 * u.dimensionless) == 0.0
    assert math.acos(0 * u.dimensionless) == pytest.approx(np.pi / 2)

    # Test with expressions
    x = UserVariable(name="x", value=1 * u.dimensionless)
    result = math.acos(x)
    assert str(result) == "math.acos(x)"
    assert result.evaluate() == 0.0

    # Test with string expressions
    expr = Expression(expression="math.acos(1)")
    assert expr.evaluate() == 0.0


def test_acos_non_scalar_input_errors():
    """Test acos function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.acos([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.acos(u.unyt_array([1, 2, 3], u.dimensionless))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.dimensionless)
    with pytest.raises(ValueError, match="Scalar function"):
        math.acos(x)


def test_acos_dimensionality_errors():
    """Test tan function raise errors for incorrect dimensionality."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 K) must be 1 to perform acos operation."
        ),
    ):
        math.acos(2 * u.K)
    x_temperature = UserVariable(name="x_temperature", value=2 * u.K)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_temperature) must be 1 to perform acos operation."
        ),
    ):
        math.acos(x_temperature)
    expr_temperature = Expression(expression="2 * u.K")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 * u.K) must be 1 to perform acos operation."
        ),
    ):
        math.acos(expr_temperature)


def test_atan_scalar_input():
    """Test atan function with valid scalar inputs."""

    # Test with regular numbers
    assert math.atan(0) == 0.0
    assert math.atan(1) == pytest.approx(np.pi / 4)
    assert math.atan(-1) == pytest.approx(-np.pi / 4)

    # Test with unyt quantities
    assert math.atan(0 * u.dimensionless) == 0.0
    assert math.atan(1 * u.dimensionless) == pytest.approx(np.pi / 4)

    # Test with expressions
    x = UserVariable(name="x", value=0 * u.dimensionless)
    result = math.atan(x)
    assert str(result) == "math.atan(x)"
    assert result.evaluate() == 0.0

    # Test with string expressions
    expr = Expression(expression="math.atan(0)")
    assert expr.evaluate() == 0.0


def test_atan_non_scalar_input_errors():
    """Test atan function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.atan([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.atan(u.unyt_array([1, 2, 3], u.dimensionless))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.dimensionless)
    with pytest.raises(ValueError, match="Scalar function"):
        math.atan(x)


def test_atan_dimensionality_errors():
    """Test tan function raise errors for incorrect dimensionality."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 K) must be 1 to perform atan operation."
        ),
    ):
        math.atan(2 * u.K)
    x_temperature = UserVariable(name="x_temperature", value=2 * u.K)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (x_temperature) must be 1 to perform atan operation."
        ),
    ):
        math.atan(x_temperature)
    expr_temperature = Expression(expression="2 * u.K")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The dimensionality of the input value (2 * u.K) must be 1 to perform atan operation."
        ),
    ):
        math.atan(expr_temperature)


# ---------------------------#
# Min/Max functions
# ---------------------------#
def test_min_scalar_input():
    """Test min function with valid scalar inputs."""

    # Test with regular numbers
    assert math.min(5, 20) == 5.0
    assert math.min(-3, 5) == -3.0
    assert math.min(0, 19) == 0.0

    # Test with unyt quantities
    assert math.min(5 * u.m, 20 * u.m) == 5 * u.m
    assert math.min(-3 * u.K, 5 * u.K) == -3 * u.K

    # Test with expressions
    x = UserVariable(name="x", value=5 * u.m)
    y = UserVariable(name="y", value=20 * u.m)
    result = math.min(x, y)
    assert str(result) == "math.min(x, y)"
    assert result.evaluate() == 5 * u.m

    # Test with string expressions
    expr = Expression(expression="math.min(5,20)")
    assert expr.evaluate() == 5.0


def test_min_non_scalar_input_errors():
    """Test min function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.min([1, 2, 3], [4, 5, 6])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.min(u.unyt_array([1, 2, 3], u.m), u.unyt_array([4, 5, 6], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    y = UserVariable(name="y", value=[4, 5, 6] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.min(x, y)


def test_min_different_dimensionality_errors():
    """Test min function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (1 | 4 s) must have the same dimensinality to perform min operation."
        ),
    ):
        math.min(1, 4 * u.s)

    # Test with unyt quantity
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (1 m | 4 K) must have the same dimensinality to perform min operation."
        ),
    ):
        math.min(u.unyt_quantity(1, u.m), u.unyt_quantity(4, u.K))

    # Test with variables
    x = UserVariable(name="x", value=solution.mut)
    y = UserVariable(name="y", value=solution.velocity[0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (x | y) must have the same dimensinality to perform min operation."
        ),
    ):
        math.min(x, y)


def test_max_scalar_input():
    """Test max function with valid scalar inputs."""

    # Test with regular numbers
    assert math.max(5, -20) == 5.0
    assert math.max(-3, -5) == -3.0
    assert math.max(0, -19) == 0.0

    # Test with unyt quantities
    assert math.max(5 * u.m, -20 * u.m) == 5 * u.m
    assert math.max(-3 * u.K, -5 * u.K) == -3 * u.K

    # Test with expressions
    x = UserVariable(name="x", value=5 * u.m)
    y = UserVariable(name="y", value=-20 * u.m)
    result = math.max(x, y)
    assert str(result) == "math.max(x, y)"
    assert result.evaluate() == 5 * u.m

    # Test with string expressions
    expr = Expression(expression="math.max(5,-20)")
    assert expr.evaluate() == 5.0


def test_max_non_scalar_input_errors():
    """Test max function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.max([1, 2, 3], [4, 5, 6])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.max(u.unyt_array([1, 2, 3], u.m), u.unyt_array([4, 5, 6], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    y = UserVariable(name="y", value=[4, 5, 6] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.max(x, y)


def test_max_different_dimensionality_errors():
    """Test max function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (1 | 4 s) must have the same dimensinality to perform max operation."
        ),
    ):
        math.max(1, 4 * u.s)

    # Test with unyt quantity
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (1 m | 4 K) must have the same dimensinality to perform max operation."
        ),
    ):
        math.max(u.unyt_quantity(1, u.m), u.unyt_quantity(4, u.K))

    # Test with variables
    x = UserVariable(name="x", value=solution.mut)
    y = UserVariable(name="y", value=solution.velocity[0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input values (x | y) must have the same dimensinality to perform max operation."
        ),
    ):
        math.max(x, y)


# ---------------------------#
# Absolute value function
# ---------------------------#
def test_abs_scalar_input():
    """Test abs function with valid scalar inputs."""

    # Test with regular numbers
    assert math.abs(5) == 5.0
    assert math.abs(-3) == 3.0
    assert math.abs(0) == 0.0

    # Test with unyt quantities
    assert math.abs(5 * u.m) == 5 * u.m
    assert math.abs(-3 * u.K) == 3 * u.K

    # Test with expressions
    x = UserVariable(name="x", value=-3 * u.m)
    result = math.abs(x)
    assert str(result) == "math.abs(x)"
    assert result.evaluate() == 3 * u.m

    # Test with string expressions
    expr = Expression(expression="math.abs(-3)")
    assert expr.evaluate() == 3.0


def test_abs_non_scalar_input_errors():
    """Test abs function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.abs([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.abs(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.abs(x)


# ---------------------------#
# Ceiling and Floor functions
# ---------------------------#
def test_ceil_scalar_input():
    """Test ceil function with valid scalar inputs."""

    # Test with regular numbers
    assert math.ceil(3.2) == 4.0
    assert math.ceil(3.9) == 4.0
    assert math.ceil(3.0) == 3.0
    assert math.ceil(-3.2) == -3.0

    # Test with unyt quantities
    assert math.ceil(3.2 * u.m) == 4 * u.m
    assert math.ceil(-3.2 * u.K) == -3 * u.K

    # Test with expressions
    x = UserVariable(name="x", value=3.2 * u.m)
    result = math.ceil(x)
    assert str(result) == "math.ceil(x)"
    assert result.evaluate() == 4 * u.m

    # Test with string expressions
    expr = Expression(expression="math.ceil(3.2)")
    assert expr.evaluate() == 4.0


def test_ceil_non_scalar_input_errors():
    """Test ceil function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.ceil([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.ceil(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.ceil(x)


def test_floor_scalar_input():
    """Test floor function with valid scalar inputs."""

    # Test with regular numbers
    assert math.floor(3.2) == 3.0
    assert math.floor(3.9) == 3.0
    assert math.floor(3.0) == 3.0
    assert math.floor(-3.2) == -4.0

    # Test with unyt quantities
    assert math.floor(3.2 * u.m) == 3 * u.m
    assert math.floor(-3.2 * u.K) == -4 * u.K

    # Test with expressions
    x = UserVariable(name="x", value=3.2 * u.m)
    result = math.floor(x)
    assert str(result) == "math.floor(x)"
    assert result.evaluate() == 3 * u.m

    # Test with string expressions
    expr = Expression(expression="math.floor(3.2)")
    assert expr.evaluate() == 3.0


def test_floor_non_scalar_input_errors():
    """Test floor function raises errors for non-scalar inputs."""

    # Test with lists/arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.floor([1, 2, 3])

    # Test with unyt arrays
    with pytest.raises(ValueError, match="Scalar function"):
        math.floor(u.unyt_array([1, 2, 3], u.m))

    # Test with variables containing arrays
    x = UserVariable(name="x", value=[1, 2, 3] * u.m)
    with pytest.raises(ValueError, match="Scalar function"):
        math.floor(x)
