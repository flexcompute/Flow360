import json
import re

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
            "Vectors ([1, 2] | [3 1 2 3] m) must have the same length to perform cross product."
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
        match=re.escape("Vectors ([1 2] m | y) must have the same length to perform dot product."),
    ):
        math.dot([1, 2] * u.m, y)


# ---------------------------#
# Scalar functions with ensure_scalar_input wrapper
# ---------------------------#
def test_sqrt_scalar_input():
    """Test sqrt function with valid scalar inputs."""

    # Test with regular numbers
    assert math.sqrt(4) == 2.0
    assert math.sqrt(0) == 0.0
    assert math.sqrt(1) == 1.0
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
