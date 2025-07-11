import json
import re
from typing import Annotated, Optional

import numpy as np
import pydantic as pd
import pytest

import flow360.component.simulation.user_code.core.context as context
from flow360 import (
    AerospaceCondition,
    HeatEquationInitialCondition,
    LiquidOperatingCondition,
    SimulationParams,
    Solid,
    Unsteady,
    math,
    u,
)
from flow360.component.simulation.blueprint.core.dependency_graph import DependencyGraph
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_lists
from flow360.component.simulation.models.material import Water, aluminum
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import (
    ProbeOutput,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    GenericVolume,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.translator.solver_translator import (
    user_variable_to_udf,
)
from flow360.component.simulation.unit_system import (
    AbsoluteTemperatureType,
    AngleType,
    AngularVelocityType,
    AreaType,
    DensityType,
    ForceType,
    FrequencyType,
    HeatFluxType,
    HeatSourceType,
    InverseAreaType,
    InverseLengthType,
    LengthType,
    MassFlowRateType,
    MassType,
    MomentType,
    PowerType,
    PressureType,
    SI_unit_system,
    SpecificEnergyType,
    SpecificHeatCapacityType,
    ThermalConductivityType,
    TimeType,
    VelocityType,
    ViscosityType,
)
from flow360.component.simulation.user_code.core.context import WHITELISTED_CALLABLES
from flow360.component.simulation.user_code.core.types import (
    Expression,
    SolverVariable,
    UserVariable,
    ValueOrExpression,
    get_user_variable,
    remove_user_variable,
    save_user_variables,
)
from flow360.component.simulation.user_code.variables import control, solution
from tests.utils import to_file_from_file_test


@pytest.fixture(autouse=True)
def reset_context():
    """Clear user variables from the context."""
    clear_context()


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def constant_variable():
    return UserVariable(name="constant_variable", value=10)


@pytest.fixture()
def constant_array():
    return UserVariable(name="constant_array", value=[10, 20])


@pytest.fixture()
def constant_unyt_quantity():
    return UserVariable(name="constant_unyt_quantity", value=10 * u.m)


@pytest.fixture()
def constant_unyt_array():
    return UserVariable(name="constant_unyt_array", value=[10, 20] * u.m)


@pytest.fixture()
def solution_variable():
    return UserVariable(name="solution_variable", value=solution.velocity)


def test_variable_init():
    # Variables can be initialized with a...

    # Value
    a = UserVariable(name="a", value=1)

    # Dimensioned value
    b = UserVariable(name="b", value=1 * u.m)

    # Expression (possibly with other variable)
    c = UserVariable(name="c", value=b + 1 * u.m)

    with pytest.raises(ValueError, match="Empty list is not allowed."):
        UserVariable(name="d", value=[])


def test_expression_init():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    # Declare a variable
    x = UserVariable(name="x", value=1)

    # Initialize with value
    model_1 = TestModel(field=1)
    assert isinstance(model_1.field, float)
    assert model_1.field == 1
    assert str(model_1.field) == "1.0"

    # Initialize with variable
    model_2 = TestModel(field=x)
    assert isinstance(model_2.field, Expression)
    assert model_2.field.evaluate() == 1
    assert str(model_2.field) == "x"

    # Initialize with variable and value
    model_3 = TestModel(field=x + 1)
    assert isinstance(model_3.field, Expression)
    assert model_3.field.evaluate() == 2
    assert str(model_3.field) == "x + 1"

    # Initialize with another expression
    model_4 = TestModel(field=model_3.field + 1)
    assert isinstance(model_4.field, Expression)
    assert model_4.field.evaluate() == 3
    assert str(model_4.field) == "x + 1 + 1"


def test_variable_reassignment():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    # Declare a variable
    x = UserVariable(name="x", value=1)

    model = TestModel(field=x)
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "x"

    # Change variable value
    x.value = 2

    assert model.field.evaluate() == 2


def test_expression_operators():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    # Declare two variables
    x = UserVariable(name="x", value=3)
    y = UserVariable(name="y", value=2)

    model = TestModel(field=x + y)

    # Addition
    model.field = x + y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 5
    assert str(model.field) == "x + y"

    # Subtraction
    model.field = x - y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "x - y"

    # Multiplication
    model.field = x * y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 6
    assert str(model.field) == "x * y"

    # Division
    model.field = x / y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1.5
    assert str(model.field) == "x / y"

    # Exponentiation
    model.field = x**y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 9
    assert str(model.field) == "x ** y"

    # Modulus
    model.field = x % y
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "x % y"

    # Negation
    model.field = -x
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == -3
    assert str(model.field) == "-x"

    # Identity
    model.field = +x
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 3
    assert str(model.field) == "+x"

    # Complex statement
    model.field = ((x - 2 * x) + (x + y) / 2 - 2**x) % 4
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 3.5
    assert str(model.field) == "(x - 2 * x + (x + y) / 2 - 2 ** x) % 4"


def test_dimensioned_expressions():
    class TestModel(Flow360BaseModel):
        length: ValueOrExpression[LengthType] = pd.Field()
        angle: ValueOrExpression[AngleType] = pd.Field()
        mass: ValueOrExpression[MassType] = pd.Field()
        time: ValueOrExpression[TimeType] = pd.Field()
        absolute_temp: ValueOrExpression[AbsoluteTemperatureType] = pd.Field()
        velocity: ValueOrExpression[VelocityType] = pd.Field()
        area: ValueOrExpression[AreaType] = pd.Field()
        force: ValueOrExpression[ForceType] = pd.Field()
        pressure: ValueOrExpression[PressureType] = pd.Field()
        density: ValueOrExpression[DensityType] = pd.Field()
        viscosity: ValueOrExpression[ViscosityType] = pd.Field()
        power: ValueOrExpression[PowerType] = pd.Field()
        moment: ValueOrExpression[MomentType] = pd.Field()
        angular_velocity: ValueOrExpression[AngularVelocityType] = pd.Field()
        heat_flux: ValueOrExpression[HeatFluxType] = pd.Field()
        heat_source: ValueOrExpression[HeatSourceType] = pd.Field()
        specific_heat_capacity: ValueOrExpression[SpecificHeatCapacityType] = pd.Field()
        thermal_conductivity: ValueOrExpression[ThermalConductivityType] = pd.Field()
        inverse_area: ValueOrExpression[InverseAreaType] = pd.Field()
        inverse_length: ValueOrExpression[InverseLengthType] = pd.Field()
        mass_flow_rate: ValueOrExpression[MassFlowRateType] = pd.Field()
        specific_energy: ValueOrExpression[SpecificEnergyType] = pd.Field()
        frequency: ValueOrExpression[FrequencyType] = pd.Field()

    model_legacy = TestModel(
        length=1 * u.m,
        angle=1 * u.rad,
        mass=1 * u.kg,
        time=1 * u.s,
        absolute_temp=1 * u.K,
        velocity=1 * u.m / u.s,
        area=1 * u.m**2,
        force=1 * u.N,
        pressure=1 * u.Pa,
        density=1 * u.kg / u.m**3,
        viscosity=1 * u.Pa * u.s,
        power=1 * u.W,
        moment=1 * u.N * u.m,
        angular_velocity=1 * u.rad / u.s,
        heat_flux=1 * u.W / u.m**2,
        heat_source=1 * u.kg / u.m / u.s**3,
        specific_heat_capacity=1 * u.J / u.kg / u.K,
        thermal_conductivity=1 * u.W / u.m / u.K,
        inverse_area=1 / u.m**2,
        inverse_length=1 / u.m,
        mass_flow_rate=1 * u.kg / u.s,
        specific_energy=1 * u.J / u.kg,
        frequency=1 * u.Hz,
    )

    assert model_legacy

    x = UserVariable(name="x", value=1)

    model_expression = TestModel(
        length=x * u.m,
        angle=x * u.rad,
        mass=x * u.kg,
        time=x * u.s,
        absolute_temp=x * u.K,
        velocity=x * u.m / u.s,
        area=x * u.m**2,
        force=x * u.N,
        pressure=x * u.Pa,
        density=x * u.kg / u.m**3,
        viscosity=x * u.Pa * u.s,
        power=x * u.W,
        moment=x * u.N * u.m,
        angular_velocity=x * u.rad / u.s,
        heat_flux=x * u.W / u.m**2,
        heat_source=x * u.kg / u.m / u.s**3,
        specific_heat_capacity=x * u.J / u.kg / u.K,
        thermal_conductivity=x * u.W / u.m / u.K,
        inverse_area=x / u.m**2,
        inverse_length=x / u.m,
        mass_flow_rate=x * u.kg / u.s,
        specific_energy=x * u.J / u.kg,
        frequency=x * u.Hz,
    )

    assert model_expression


def test_constrained_scalar_type():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[Annotated[float, pd.Field(strict=True, ge=0)]] = pd.Field()

    x = UserVariable(name="x", value=1)

    model = TestModel(field=x)

    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "x"

    with pytest.raises(pd.ValidationError):
        model.field = -x


def test_disallow_run_time_expressions():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    with pytest.raises(ValueError, match="Run-time expression is not allowed in this field."):
        TestModel(field=solution.Cp)


def test_constrained_dimensioned_type():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[LengthType.Positive] = pd.Field()

    x = UserVariable(name="x", value=1)

    model = TestModel(field=x * u.m)

    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1 * u.m
    assert str(model.field) == "x * u.m"

    with pytest.raises(pd.ValidationError):
        model.field = -x * u.m


def test_vector_types():
    class TestModel(Flow360BaseModel):
        vector: ValueOrExpression[LengthType.Vector] = pd.Field()
        axis: ValueOrExpression[LengthType.Axis] = pd.Field()
        array: ValueOrExpression[LengthType.Array] = pd.Field()
        direction: ValueOrExpression[LengthType.Direction] = pd.Field()
        moment: ValueOrExpression[LengthType.Moment] = pd.Field()

    x = UserVariable(name="x", value=[1, 0, 0] * u.m)
    y = UserVariable(name="y", value=[0, 0, 0] * u.m)
    z = UserVariable(name="z", value=[1, 0, 0, 0] * u.m)
    w = UserVariable(name="w", value=[1, 1, 1] * u.m)

    model = TestModel(vector=y, axis=x, array=z, direction=x, moment=w)

    assert isinstance(model.vector, Expression)
    assert (model.vector.evaluate() == [0, 0, 0] * u.m).all()
    assert str(model.vector) == "y"

    assert isinstance(model.axis, Expression)
    assert (model.axis.evaluate() == [1, 0, 0] * u.m).all()
    assert str(model.axis) == "x"

    assert isinstance(model.array, Expression)
    assert (model.array.evaluate() == [1, 0, 0, 0] * u.m).all()
    assert str(model.array) == "z"

    assert isinstance(model.direction, Expression)
    assert (model.direction.evaluate() == [1, 0, 0] * u.m).all()
    assert str(model.direction) == "x"

    assert isinstance(model.moment, Expression)
    assert (model.moment.evaluate() == [1, 1, 1] * u.m).all()
    assert str(model.moment) == "w"

    with pytest.raises(pd.ValidationError):
        model.vector = z

    with pytest.raises(pd.ValidationError):
        model.axis = y

    with pytest.raises(pd.ValidationError):
        model.direction = y

    with pytest.raises(pd.ValidationError):
        model.moment = x


def test_serializer(
    constant_variable,
    constant_array,
    constant_unyt_quantity,
    constant_unyt_array,
    solution_variable,
):
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()
        non_dim_field: Optional[
            ValueOrExpression.configure(allow_run_time_expression=True)[float]  # type: ignore
        ] = pd.Field(default=None)

    x = UserVariable(name="x", value=4)
    cp = UserVariable(name="my_cp", value=solution.Cp)

    model = TestModel(field=x * u.m / u.s + 4 * x**2 * u.m / u.s, non_dim_field=cp)

    assert str(model.field) == "x * u.m / u.s + 4 * x ** 2 * u.m / u.s"

    serialized = model.model_dump()

    assert serialized["field"]["type_name"] == "expression"
    assert serialized["field"]["expression"] == "x * u.m / u.s + 4 * x ** 2 * u.m / u.s"
    assert serialized["non_dim_field"]["expression"] == "my_cp"

    model = TestModel(field=4 * u.m / u.s)

    serialized = model.model_dump(exclude_none=True)

    assert serialized["field"]["type_name"] == "number"
    assert serialized["field"]["value"] == 4
    assert serialized["field"]["units"] == "m/s"

    assert constant_variable.model_dump() == {
        "name": "constant_variable",
        "type_name": "UserVariable",
    }

    assert constant_array.model_dump() == {
        "name": "constant_array",
        "type_name": "UserVariable",
    }
    assert constant_unyt_quantity.model_dump() == {
        "name": "constant_unyt_quantity",
        "type_name": "UserVariable",
    }

    assert constant_unyt_array.model_dump() == {
        "name": "constant_unyt_array",
        "type_name": "UserVariable",
    }

    assert solution_variable.model_dump() == {
        "name": "solution_variable",
        "type_name": "UserVariable",
    }


def test_deserializer(
    constant_unyt_quantity,
    constant_unyt_array,
    constant_variable,
    constant_array,
    solution_variable,
):
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    model = {
        "type_name": "expression",
        "expression": "x * u.m / u.s + 4 * x ** 2 * u.m / u.s",
    }

    deserialized = TestModel(field=model)

    assert str(deserialized.field) == "x * u.m / u.s + 4 * x ** 2 * u.m / u.s"

    model = {"type_name": "number", "value": 4.0, "units": "m/s"}

    deserialized = TestModel(field=model)

    assert str(deserialized.field) == "4.0 m/s"

    # Constant unyt quantity
    model = {
        "name": "constant_unyt_quantity",
        "value": {
            "expression": None,
            "output_units": None,
            "type_name": "number",
            "units": "m",
            "value": 10.0,
        },
    }
    deserialized = UserVariable.model_validate(model)
    assert deserialized == constant_unyt_quantity

    # Constant unyt array
    model = {
        "name": "constant_unyt_array",
        "value": {
            "expression": None,
            "output_units": None,
            "type_name": "number",
            "units": "m",
            "value": [10, 20],
        },
    }
    deserialized = UserVariable.model_validate(model)
    assert deserialized == constant_unyt_array

    # Constant quantity
    model = {
        "name": "constant_variable",
        "value": {
            "expression": None,
            "output_units": None,
            "type_name": "number",
            "units": None,
            "value": 10.0,
        },
    }
    deserialized = UserVariable.model_validate(model)
    assert deserialized == constant_variable

    # Constant array
    model = {
        "name": "constant_array",
        "value": {
            "expression": None,
            "output_units": None,
            "type_name": "number",
            "units": None,
            "value": [10, 20],
        },
    }
    deserialized = UserVariable.model_validate(model)
    assert deserialized == constant_array

    # Solver variable (NaN-None handling)
    model = {
        "name": "solution_variable",
        "value": {
            "expression": "solution.velocity",
            "output_units": None,
            "type_name": "expression",
            "units": None,
            "value": None,
        },
    }
    deserialized = UserVariable.model_validate(model)
    assert deserialized == solution_variable
    assert all(
        np.isnan(item)
        for item in deserialized.value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
    )


def test_subscript_access():
    class ScalarModel(Flow360BaseModel):
        scalar: ValueOrExpression[float] = pd.Field()

    x = UserVariable(name="x", value=[2, 3, 4])

    model = ScalarModel(scalar=x[0] + x[1] + x[2] + 1)

    assert str(model.scalar) == "x[0] + x[1] + x[2] + 1"

    assert model.scalar.evaluate() == 10

    model = ScalarModel(scalar="x[0] + x[1] + x[2] + 1")

    assert str(model.scalar) == "x[0] + x[1] + x[2] + 1"

    assert model.scalar.evaluate() == 10


def test_error_message():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    try:
        TestModel(field="1 + nonexisting * 1")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "Name 'nonexisting' is not defined" in validation_errors[0]["msg"]

    try:
        TestModel(field="1 + x * 1")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "does not match (length)/(time) dimension" in validation_errors[0]["msg"]

    try:
        TestModel(field="1 * 1 +")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "invalid syntax" in validation_errors[0]["msg"]
        assert "1 * 1 +" in validation_errors[0]["msg"]
        assert "line" in validation_errors[0]["ctx"]
        assert "column" in validation_errors[0]["ctx"]
        assert validation_errors[0]["ctx"]["column"] == 8

    try:
        TestModel(field="1 * 1 +* 2")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "invalid syntax" in validation_errors[0]["msg"]
        assert "1 * 1 +* 2" in validation_errors[0]["msg"]
        assert "line" in validation_errors[0]["ctx"]
        assert "column" in validation_errors[0]["ctx"]
        assert validation_errors[0]["ctx"]["column"] == 8

    try:
        TestModel(field="1 * 1 + (2")
    except pd.ValidationError as err:
        validation_errors = err.errors()

    assert len(validation_errors) == 1
    assert validation_errors[0]["type"] == "value_error"
    assert "line" in validation_errors[0]["ctx"]
    assert "column" in validation_errors[0]["ctx"]
    assert validation_errors[0]["ctx"]["column"] in (
        9,
        11,
    )  # Python 3.9 report error on col 11, error message is also different

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vector operation (__add__ between solution.velocity and [1 2 3] cm/ms) not supported for variables. Please write expression for each component."
        ),
    ):
        UserVariable(name="x", value=solution.velocity + [1, 2, 3] * u.cm / u.ms)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Vector operation (__add__ between solution.velocity and [1 2 3] cm/ms) not supported for variables. Please write expression for each component."
        ),
    ):
        UserVariable(name="xx", value="solution.velocity + [1, 2, 3] * u.cm / u.ms")


def test_temperature_units_usage():
    with pytest.raises(
        ValueError,
        match="Relative temperature scale usage is not allowed. Please use u.R or u.K instead.",
    ):
        Expression(expression="[1,2,3] * u.degF")

    with pytest.raises(
        ValueError,
        match="Relative temperature scale usage is not allowed in output units. Please use u.R or u.K instead.",
    ):
        UserVariable(name="x", value=solution.temperature + 123 * u.K).in_units(new_unit="u.degF")

    with pytest.raises(
        ValueError,
        match="Relative temperature scale usage is not allowed in output units. Please use u.R or u.K instead.",
    ):
        solution.temperature.in_units(new_name="my_temperature", new_unit="u.degF")


def test_solver_translation():
    timestepping_unsteady = Unsteady(steps=12, step_size=0.1 * u.s)
    solid_model = Solid(
        volumes=[GenericVolume(name="CHTSolid")],
        material=aluminum,
        volumetric_heat_source="0",
        initial_condition=HeatEquationInitialCondition(temperature="10"),
    )
    surface_output_with_residual_heat_solver = SurfaceOutput(
        name="surface",
        surfaces=[Surface(name="noSlipWall")],
        write_single_file=True,
        output_fields=["residualHeatSolver"],
    )
    water = Water(
        name="h2o", density=1000 * u.kg / u.m**3, dynamic_viscosity=0.001 * u.kg / u.m / u.s
    )
    liquid_operating_condition = LiquidOperatingCondition(
        velocity_magnitude=50 * u.m / u.s,
        reference_velocity_magnitude=100 * u.m / u.s,
        material=water,
    )

    # Valid simulation params
    with SI_unit_system:
        params = SimulationParams(
            models=[solid_model],
            operating_condition=liquid_operating_condition,
            time_stepping=timestepping_unsteady,
            outputs=[surface_output_with_residual_heat_solver],
            private_attribute_asset_cache=AssetCache(project_length_unit=2 * u.m),
        )

        x = UserVariable(name="x", value=4)
        y = UserVariable(name="y", value=x + 1)

        # Showcased features:
        expression = Expression.model_validate(x * u.m**2)

        # 1. Units are converted to flow360 unit system using the provided params (1m**2 -> 0.25 because of length unit)
        # 2. User variables are inlined (for numeric value types)
        assert expression.to_solver_code(params) == "(4.0 * pow(0.5, 2))"

        # 3. User variables are inlined (for expression value types)
        expression = Expression.model_validate(y * u.m**2)
        assert expression.to_solver_code(params) == "(5.0 * pow(0.5, 2))"

        # 4. For solver variables, the units are stripped (assumed to be in solver units so factor == 1.0)
        expression = Expression.model_validate(y * u.m / u.s + control.MachRef)
        assert expression.to_solver_code(params) == "(((5.0 * 0.5) / 500.0) + machRef)"


def test_cyclic_dependencies():
    x = UserVariable(name="x", value=4)
    y = UserVariable(name="y", value=x)

    # If we try to create a cyclic dependency we throw a validation error
    # The error contains info about the cyclic dependency, so here its x -> y -> x
    with pytest.raises(
        pd.ValidationError, match=re.escape("Circular dependency detected among: ['x', 'y']")
    ):
        x.value = y

    z = UserVariable(name="z", value=4)

    with pytest.raises(pd.ValidationError):
        z.value = z


def test_auto_alias():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    unaliased = {
        "type_name": "expression",
        "expression": "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)",
    }

    aliased = {
        "typeName": "expression",
        "expression": "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)",
        "evaluatedValue": 68.0,
        "evaluatedUnits": "m/s",
    }

    model_1 = TestModel(field=unaliased)
    model_2 = TestModel(field=aliased)

    assert str(model_1.field) == "x * u.m / u.s + 4 * x ** 2 * u.m / u.s"
    assert str(model_2.field) == "x * u.m / u.s + 4 * x ** 2 * u.m / u.s"


def test_variable_space_init():
    # Simulating loading a SimulationParams object from file - ensure that the variable space is loaded correctly
    with open("data/simulation.json", "r+") as fh:
        data = json.load(fh)

    params, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
    )

    assert errors is None
    evaluated = params.reference_geometry.area.evaluate()

    assert evaluated == 1.0 * u.m**2


def test_expression_indexing():
    a = UserVariable(name="a", value=1)
    b = UserVariable(name="b", value=[1, 2, 3])
    c = UserVariable(name="c", value=[3, 2, 1])

    # Cannot simplify without non-statically evaluable index object (expression for example)
    cross_result = math.cross(b, c)
    expr = Expression.model_validate(cross_result[a])

    assert (
        str(expr)
        == "[b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]][a]"
    )
    assert expr.evaluate() == 8

    # Cannot simplify without non-statically evaluable index object (expression for example)
    expr = Expression.model_validate(cross_result[1])

    assert str(expr) == "b[2] * c[0] - b[0] * c[2]"
    assert expr.evaluate() == 8


def test_to_file_from_file_expression(
    constant_variable, constant_array, constant_unyt_quantity, constant_unyt_array
):
    with SI_unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=10 * u.m**2,
            ),
            outputs=[
                VolumeOutput(
                    output_fields=[
                        solution.mut.in_units(new_name="mut_in_SI", new_unit="g/cm/min"),
                        constant_variable,
                        constant_array,
                        constant_unyt_quantity,
                        constant_unyt_array,
                    ]
                )
            ],
        )

    to_file_from_file_test(params)
    params.display_output_units()  # Just to make sure not exception.


def assert_ignore_space(expected: str, actual: str):
    """For expression comparison, ignore spaces"""
    assert expected.replace(" ", "") == actual.replace(" ", "")


def test_udf_generator():
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=5 * u.m / u.s,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=10 * u.m),
        )
    # Scalar output
    result = user_variable_to_udf(
        solution.mut.in_units(new_name="mut_in_km", new_unit="kg/km/s"), input_params=params
    )
    # velocity scale = 100 m/s, length scale = 10m, density scale = 1000 kg/m**3
    # mut_scale = Rho*L*V -> 1000*10*100 * kg/m/s == 1000*10*100*1000 * kg/km/s
    assert result.expression == "mut_in_km = (mut * 1000000000.0);"

    vel_cross_vec = UserVariable(
        name="vel_cross_vec", value=math.cross(solution.velocity, [1, 2, 3] * u.cm)
    ).in_units(new_unit="CGS_unit_system")
    assert vel_cross_vec.value.get_output_units(input_params=params) == u.cm**2 / u.s

    # We disabled degC and degF on the interface and therefore the inferred units should be K or R.
    my_temp = UserVariable(name="my_temperature", value=solution.temperature).in_units(
        new_unit="Imperial_unit_system"
    )
    assert my_temp.value.get_output_units(input_params=params) == u.R

    # Test __pow__ on SolverVariable:
    vel_sq = UserVariable(name="vel_sq", value=solution.velocity**2)
    result = user_variable_to_udf(vel_sq, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;vel_sq[0] = (pow(___velocity[0], 2) * 10000.0); vel_sq[1] = (pow(___velocity[1], 2) * 10000.0); vel_sq[2] = (pow(___velocity[2], 2) * 10000.0);"
    )

    # Test __neg__ on SolverVariable:
    neg_vel = UserVariable(name="neg_vel", value=-solution.velocity)
    result = user_variable_to_udf(neg_vel, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;neg_vel[0] = (-___velocity[0] * 100.0); neg_vel[1] = (-___velocity[1] * 100.0); neg_vel[2] = (-___velocity[2] * 100.0);"
    )

    # Test __pos__ on SolverVariable:
    pos_vel = UserVariable(name="pos_vel", value=+solution.velocity)
    result = user_variable_to_udf(pos_vel, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;pos_vel[0] = (+___velocity[0] * 100.0); pos_vel[1] = (+___velocity[1] * 100.0); pos_vel[2] = (+___velocity[2] * 100.0);"
    )


def test_project_variables_serialization():
    ccc = UserVariable(name="ccc", value=12 * u.m / u.s)
    aaa = UserVariable(
        name="aaa", value=[solution.velocity[0] + ccc, solution.velocity[1], solution.velocity[2]]
    )
    bbb = UserVariable(name="bbb", value=[aaa[0] + 14 * u.m / u.s, aaa[1], aaa[2]]).in_units(
        new_unit="km/ms"
    )

    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(
                velocity_magnitude=Expression(expression="10 * u.m / u.s"),
                reference_velocity_magnitude=10 * u.m / u.s,
            ),
            outputs=[
                VolumeOutput(
                    output_fields=[
                        bbb,
                    ]
                ),
                ProbeOutput(
                    probe_points=[
                        Point(name="pt1", location=(1, 2, 3), private_attribute_id="111")
                    ],
                    output_fields=[bbb],
                ),
            ],
        )

    params = save_user_variables(params)

    with open("ref/simulation_with_project_variables.json", "r+") as fh:
        ref_data = fh.read()

    assert ref_data == params.model_dump_json(indent=4, exclude_none=True)


def test_project_variables_deserialization():
    with open("ref/simulation_with_project_variables.json", "r+") as fh:
        data = json.load(fh)

    # Assert no variables registered yet
    with pytest.raises(NameError):
        context.default_context.get("aaa")
    with pytest.raises(NameError):
        context.default_context.get("bbb")
    with pytest.raises(NameError):
        context.default_context.get("ccc")

    params, _, _ = validate_model(
        params_as_dict=data,
        root_item_type=None,
        validated_by=ValidationCalledBy.LOCAL,
    )
    assert params
    assert (
        params.outputs[0].output_fields.items[0].value.expression
        == "[aaa[0] + 14 * u.m / u.s, aaa[1], aaa[2]]"
    )

    assert params.outputs[0].output_fields.items[0].value.output_units == "km/ms"

    assert (
        params.outputs[0]
        .output_fields.items[0]
        .value.evaluate(force_evaluate=False, raise_on_non_evaluable=False)
        .expression
        == "[solution.velocity[0] + 12 * u.m / u.s + 14 * u.m / u.s, solution.velocity[1], solution.velocity[2]]"
    )  # Fully resolvable


def test_overwriting_project_variables():
    a = UserVariable(name="a", value=1)

    with pytest.raises(
        ValueError,
        match="Redeclaring user variable 'a' with new value: 2.0. Previous value: 1.0",
    ):
        UserVariable(name="a", value=2)

    a.value = 2
    assert a.value == 2


def test_unique_dimensions():
    with pytest.raises(ValueError, match="All items in the list must have the same dimensions."):
        UserVariable(name="a", value=[1 * u.m, 1 * u.s])

    with pytest.raises(
        ValueError, match="List must contain only all unyt_quantities or all numbers."
    ):
        UserVariable(name="a", value=[1.0 * u.m, 1.0])

    a = UserVariable(name="a", value=[1.0 * u.m, 1.0 * u.mm])
    assert all(a.value == [1.0, 0.001] * u.m)


@pytest.mark.parametrize(
    "bad_name, expected_msg",
    [
        ("", "Identifier cannot be empty."),
        ("1stPlace", "Identifier must start with a letter (A-Z/a-z) or underscore (_)."),
        ("bad-name", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        ("has space", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        (" leading", "Identifier must start with a letter (A-Z/a-z) or underscore (_)."),
        ("trailing ", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        ("tab\tname", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        ("new\nline", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        ("name$", "Identifier can only contain letters, digits (0-9), or underscore (_)."),
        ("class", "'class' is a reserved keyword."),
        ("namespace", "'namespace' is a reserved keyword."),
        ("template", "'template' is a reserved keyword."),
        ("temperature", "'temperature' is a reserved solver side variable name."),
        ("area", "'area' is a reserved solver side variable name."),
        ("velocity", "'velocity' is a reserved (legacy) output field name."),
        ("rho", "'rho' is a reserved (legacy) output field name."),
        ("pressure", "'pressure' is a reserved (legacy) output field name."),
    ],
)
def test_invalid_names_raise(bad_name, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        UserVariable(name=bad_name, value=0)


def test_output_units_dimensions():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Output units 'ms' have different dimensions (time) than the expression (length)."
        ),
    ):
        a = UserVariable(name="a", value="1 * u.m")
        a.in_units(new_unit="ms")


def test_whitelisted_callables():
    def get_user_variable_names(module):
        return [attr for attr in dir(module) if isinstance(getattr(module, attr), SolverVariable)]

    solution_vars = get_user_variable_names(solution)
    control_vars = get_user_variable_names(control)

    assert compare_lists(solution_vars, WHITELISTED_CALLABLES["flow360.solution"]["callables"])
    assert compare_lists(control_vars, WHITELISTED_CALLABLES["flow360.control"]["callables"])


def test_deserialization_with_wrong_syntax():
    with open("data/simulation_with_wrong_expr_syntax.json", "r+") as fh:
        data = json.load(fh)

    _, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
    )

    assert len(errors) == 1
    assert (
        r"Value error, expression evaluation failed: Name 'alphaAngle' is not defined"
        in errors[0]["msg"]
    )
    assert errors[0]["loc"] == ("private_attribute_asset_cache", "variable_context", 0)


# DependencyGraph Tests
class TestDependencyGraph:
    """Test suite for DependencyGraph class."""

    def test_init(self):
        """Test DependencyGraph initialization."""
        graph = DependencyGraph()
        assert graph._graph == {}
        assert graph._deps == {}

    def test_extract_deps_simple(self):
        """Test dependency extraction from simple expressions."""
        graph = DependencyGraph()
        all_names = {"x", "y", "z"}

        # Simple variable reference
        deps = graph._extract_deps("x", all_names)
        assert deps == {"x"}

        # Expression with multiple variables
        deps = graph._extract_deps("x + y * z", all_names)
        assert deps == {"x", "y", "z"}

        # Expression with unknown variables (should be filtered out)
        deps = graph._extract_deps("x + unknown_var", all_names)
        assert deps == {"x"}

    def test_extract_deps_complex_expressions(self):
        """Test dependency extraction from complex expressions."""
        graph = DependencyGraph()
        all_names = {"a", "b", "c", "d", "e"}

        # Complex mathematical expression
        deps = graph._extract_deps("(a + b) * c / (d - e)", all_names)
        assert deps == {"a", "b", "c", "d", "e"}

        # Function calls
        deps = graph._extract_deps("abs(a) + max(b, c)", all_names)
        assert deps == {"a", "b", "c"}

        # Nested expressions
        deps = graph._extract_deps("a + (b * (c + d))", all_names)
        assert deps == {"a", "b", "c", "d"}

    def test_extract_deps_syntax_error(self):
        """Test dependency extraction with syntax errors."""
        graph = DependencyGraph()
        all_names = {"x", "y"}

        # Invalid syntax should return empty set
        deps = graph._extract_deps("x +", all_names)
        assert deps == set()

        deps = graph._extract_deps("x + * y", all_names)
        assert deps == set()

    def test_load_from_list_simple(self):
        """Test loading variables from a simple list."""
        graph = DependencyGraph()
        vars_list = [
            {"name": "x", "value": "1"},
            {"name": "y", "value": "x + 1"},
        ]

        graph.load_from_list(vars_list)

        # Check dependencies
        assert "x" in graph._graph
        assert "y" in graph._graph
        assert "y" in graph._graph["x"]  # y depends on x
        assert "x" in graph._deps["y"]  # y depends on x

    def test_load_from_list_complex_dependencies(self):
        """Test loading variables with complex dependency relationships."""
        graph = DependencyGraph()
        vars_list = [
            {"name": "a", "value": "1"},
            {"name": "b", "value": "a + 1"},
            {"name": "c", "value": "b * 2"},
            {"name": "d", "value": "a + c"},
        ]

        graph.load_from_list(vars_list)

        # Check dependency relationships
        assert "b" in graph._graph["a"]  # b depends on a
        assert "c" in graph._graph["b"]  # c depends on b
        assert "d" in graph._graph["a"]  # d depends on a
        assert "d" in graph._graph["c"]  # d depends on c

        assert "a" in graph._deps["b"]  # b depends on a
        assert "b" in graph._deps["c"]  # c depends on b
        assert "a" in graph._deps["d"]  # d depends on a
        assert "c" in graph._deps["d"]  # d depends on c

    def test_load_from_list_unknown_variable_reference(self):
        """Test loading with reference to unknown variable."""
        graph = DependencyGraph()
        vars_list = [
            {"name": "x", "value": "y + 1"},
        ]

        # The DependencyGraph only creates dependencies for variables that exist in the graph
        # Since 'y' is not in the vars_list, no dependency is created
        graph.load_from_list(vars_list)

        # Check that x exists but has no dependencies
        assert "x" in graph._graph
        assert "x" in graph._deps
        assert graph._graph["x"] == set()
        assert graph._deps["x"] == set()

    def test_load_from_list_clear_existing(self):
        """Test that loading clears existing graph."""
        graph = DependencyGraph()

        # Add some initial data
        graph.add_variable("old_var", "1")

        # Load new data
        vars_list = [
            {"name": "new_var", "value": "1"},
        ]
        graph.load_from_list(vars_list)

        # Check old data is gone
        assert "old_var" not in graph._graph
        assert "new_var" in graph._graph

    def test_add_variable_simple(self):
        """Test adding a simple variable."""
        graph = DependencyGraph()
        graph.add_variable("x")

        assert "x" in graph._graph
        assert "x" in graph._deps
        assert graph._graph["x"] == set()
        assert graph._deps["x"] == set()

    def test_add_variable_with_expression(self):
        """Test adding a variable with an expression."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        assert "y" in graph._graph["x"]  # y depends on x
        assert "x" in graph._deps["y"]  # y depends on x

    def test_add_variable_overwrite(self):
        """Test overwriting an existing variable."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # Overwrite y with new expression
        graph.add_variable("y", "x * 2")

        assert "y" in graph._graph["x"]  # y still depends on x
        assert "x" in graph._deps["y"]  # y depends on x

        # Overwrite y with no expression
        graph.add_variable("y")

        assert "y" not in graph._graph["x"]  # y no longer depends on x
        assert "x" not in graph._deps["y"]  # y no longer depends on x

    def test_add_variable_unknown_dependency(self):
        """Test adding variable with unknown dependency."""
        graph = DependencyGraph()

        # The DependencyGraph only creates dependencies for variables that exist in the graph
        # Since 'x' is not in the graph, no dependency is created
        graph.add_variable("y", "x + 1")

        # Check that y exists but has no dependencies
        assert "y" in graph._graph
        assert "y" in graph._deps
        assert graph._graph["y"] == set()
        assert graph._deps["y"] == set()

    def test_remove_variable(self):
        """Test removing a variable."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")
        graph.add_variable("z", "y + 1")

        # Remove y
        graph.remove_variable("y")

        assert "y" not in graph._graph
        assert "y" not in graph._deps
        assert "y" not in graph._graph["x"]  # y's dependency on x is removed
        assert "z" not in graph._graph["y"]  # z's dependency on y is removed

    def test_remove_variable_nonexistent(self):
        """Test removing a nonexistent variable."""
        graph = DependencyGraph()

        with pytest.raises(KeyError, match="Variable 'nonexistent' does not exist"):
            graph.remove_variable("nonexistent")

    def test_update_expression(self):
        """Test updating an existing variable's expression."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # Update y's expression
        graph.update_expression("y", "x * 2")

        assert "y" in graph._graph["x"]  # y still depends on x
        assert "x" in graph._deps["y"]  # y depends on x

        # Remove expression
        graph.update_expression("y", None)

        assert "y" not in graph._graph["x"]  # y no longer depends on x
        assert "x" not in graph._deps["y"]  # y no longer depends on x

    def test_update_expression_nonexistent(self):
        """Test updating expression for nonexistent variable."""
        graph = DependencyGraph()

        with pytest.raises(KeyError, match="Variable 'nonexistent' does not exist"):
            graph.update_expression("nonexistent", "1 + 1")

    def test_update_expression_unknown_dependency(self):
        """Test updating expression with unknown dependency."""
        graph = DependencyGraph()
        graph.add_variable("x")

        # The DependencyGraph only creates dependencies for variables that exist in the graph
        # Since 'y' is not in the graph, no dependency is created
        graph.update_expression("x", "y + 1")

        # Check that x exists but has no dependencies
        assert "x" in graph._graph
        assert "x" in graph._deps
        assert graph._graph["x"] == set()
        assert graph._deps["x"] == set()

    def test_check_for_cycle_simple(self):
        """Test cycle detection with simple cycle."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # This should not raise an error (no cycle)
        graph._check_for_cycle()

        # Create a cycle manually
        graph._graph["y"].add("x")
        graph._deps["x"].add("y")

        with pytest.raises(
            pd.ValidationError, match="Circular dependency detected among: \\['x', 'y'\\]"
        ):
            graph._check_for_cycle()

    def test_check_for_cycle_complex(self):
        """Test cycle detection with complex cycle."""
        graph = DependencyGraph()
        graph.add_variable("a")
        graph.add_variable("b", "a + 1")
        graph.add_variable("c", "b + 1")
        graph.add_variable("d", "c + 1")

        # Create a cycle: a -> b -> c -> d -> a
        graph._graph["d"].add("a")
        graph._deps["a"].add("d")

        with pytest.raises(
            pd.ValidationError, match="Circular dependency detected among: \\['a', 'b', 'c', 'd'\\]"
        ):
            graph._check_for_cycle()

    def test_topology_sort_simple(self):
        """Test topological sorting with simple dependencies."""
        graph = DependencyGraph()
        graph.add_variable("a")
        graph.add_variable("b", "a + 1")
        graph.add_variable("c", "b + 1")

        order = graph.topology_sort()

        # Check that dependencies come before dependents
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_topology_sort_complex(self):
        """Test topological sorting with complex dependencies."""
        graph = DependencyGraph()
        graph.add_variable("a")
        graph.add_variable("b", "a + 1")
        graph.add_variable("c", "a + 1")
        graph.add_variable("d", "b + c")
        graph.add_variable("e", "d + 1")

        order = graph.topology_sort()

        # Check dependency order
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")
        assert order.index("d") < order.index("e")

    def test_topology_sort_with_cycle(self):
        """Test topological sorting with cycle detection."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # Create a cycle
        graph._graph["y"].add("x")
        graph._deps["x"].add("y")

        with pytest.raises(
            pd.ValidationError, match="Circular dependency detected among: \\['x', 'y'\\]"
        ):
            graph.topology_sort()

    def test_load_from_list_restore_on_error(self):
        """Test that graph state is restored on error during load."""
        graph = DependencyGraph()

        # Add initial state
        graph.add_variable("initial", "1")

        # Try to load data that would create a cycle
        vars_list = [
            {"name": "a", "value": "1"},
            {"name": "b", "value": "a + 1"},
            {"name": "c", "value": "b + 1"},
            {
                "name": "a",
                "value": "c + 1",
            },  # This creates a cycle
        ]

        with pytest.raises(pd.ValidationError, match="Circular dependency detected"):
            graph.load_from_list(vars_list)

        # Check that initial state is preserved
        assert "initial" in graph._graph
        assert "a" not in graph._graph
        assert "b" not in graph._graph
        assert "c" not in graph._graph

    def test_add_variable_restore_on_error(self):
        """Test that graph state is restored on error during add."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # Try to add variable that creates a cycle
        with pytest.raises(pd.ValidationError, match="Circular dependency detected"):
            graph.add_variable("x", "y + 1")  # This creates x -> y -> x cycle

        # Check that graph state is unchanged
        assert "y" in graph._graph["x"]  # y still depends on x
        assert "x" in graph._deps["y"]  # y depends on x

    def test_update_expression_restore_on_error(self):
        """Test that graph state is restored on error during update."""
        graph = DependencyGraph()
        graph.add_variable("x")
        graph.add_variable("y", "x + 1")

        # Try to update with expression that creates a cycle
        with pytest.raises(pd.ValidationError, match="Circular dependency detected"):
            graph.update_expression("x", "y + 1")  # This creates x -> y -> x cycle

        # Check that original dependency is preserved
        assert "y" in graph._graph["x"]  # y still depends on x
        assert "x" in graph._deps["y"]  # y depends on x

    def test_self_reference_cycle(self):
        """Test cycle detection with self-reference."""
        graph = DependencyGraph()
        graph.add_variable("x")

        # Try to create self-reference
        with pytest.raises(
            pd.ValidationError, match="Circular dependency detected among: \\['x'\\]"
        ):
            graph.update_expression("x", "x + 1")

    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        graph = DependencyGraph()

        # Topological sort of empty graph
        order = graph.topology_sort()
        assert order == []

        # Add variable to empty graph
        graph.add_variable("x", "1")
        assert "x" in graph._graph
        assert "x" in graph._deps

    def test_isolated_variables(self):
        """Test handling of variables with no dependencies."""
        graph = DependencyGraph()
        graph.add_variable("a")
        graph.add_variable("b")
        graph.add_variable("c", "a + b")

        order = graph.topology_sort()

        # a and b can come in any order, but c must come after both
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_multiple_dependencies(self):
        """Test variables with multiple dependencies."""
        graph = DependencyGraph()
        graph.add_variable("a")
        graph.add_variable("b")
        graph.add_variable("c")
        graph.add_variable("d", "a + b + c")

        order = graph.topology_sort()

        # d must come after all its dependencies
        assert order.index("a") < order.index("d")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_dependency_extraction_edge_cases(self):
        """Test dependency extraction with edge cases."""
        graph = DependencyGraph()
        all_names = {"x", "y", "z"}

        # Empty expression
        deps = graph._extract_deps("", all_names)
        assert deps == set()

        # Expression with no variables
        deps = graph._extract_deps("1 + 2 * 3", all_names)
        assert deps == set()

        # Expression with only unknown variables
        deps = graph._extract_deps("unknown1 + unknown2", all_names)
        assert deps == set()

        # Expression with mixed known and unknown variables
        deps = graph._extract_deps("x + unknown + y", all_names)
        assert deps == {"x", "y"}

    def test_trailing_semicolon(self):
        graph = DependencyGraph()
        input_variable_list = [
            {"name": "contains_unit_sudu", "value": "[math.ceil(math.sqrt(81)) + math.floor(5.5)]"},
            {"name": "post_processing_variables", "value": "1+1"},
            {"name": "gamma", "value": "1.4"},
            {"name": "pow1", "value": "gamma/(gamma-1)"},
            {"name": "pow2", "value": "(gamma-1) / 2"},
            {"name": "primitiveVar", "value": "[1,2,3,4,5]"},
            {"name": "v", "value": "90"},
            {
                "name": "TotalPressureCoeff",
                "value": "(gamma*primitiveVar[4]*(1+pow2*solution.Mach*solution.Mach)^pow1-(1+pow2*MachRefSq)^pow1)/(gamma/2*MachRefSq);",
            },
            {"name": "MachRefSq", "value": "control.MachRef*control.MachRef"},
        ]
        graph.load_from_list(input_variable_list)
        order = graph.topology_sort()
        assert order.index("gamma") < order.index("pow1")
        assert order.index("gamma") < order.index("pow2")
        assert order.index("gamma") < order.index("TotalPressureCoeff")
        assert order.index("primitiveVar") < order.index("TotalPressureCoeff")
        assert order.index("pow2") < order.index("TotalPressureCoeff")
        assert order.index("pow1") < order.index("TotalPressureCoeff")
        assert order.index("MachRefSq") < order.index("TotalPressureCoeff")


def test_remove_variable_with_yes_confirmation(monkeypatch, capsys):
    # Simulate user typing 'yes'
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    var_a = UserVariable(name="var_a", value=solution.Cpt)
    var_b = UserVariable(name="var_b", value=var_a**2)
    var_c = UserVariable(name="var_c", value=var_a**3)
    var_d = UserVariable(name="var_d", value=var_b**2 + var_a + var_c)

    remove_user_variable(name="var_b")

    captured = capsys.readouterr()
    assert "--- Confirmation Required ---" in captured.out
    assert "The following variables will be removed:" in captured.out
    assert "- var_b" in captured.out
    assert "- var_d" in captured.out
    assert "--- Proceeding with removal ---" in captured.out
    assert "Removed 'var_b' from values." in captured.out
    assert "Removed 'var_d' from values." in captured.out

    assert var_a == get_user_variable("var_a")
    with pytest.raises(NameError, match="Name 'var_b' is not defined"):
        get_user_variable("var_b")
    assert var_c == get_user_variable("var_c")
    with pytest.raises(NameError, match="Name 'var_d' is not defined"):
        get_user_variable("var_d")

    remove_user_variable(name="var_a")
    captured = capsys.readouterr()
    assert "--- Confirmation Required ---" in captured.out
    assert "The following variables will be removed:" in captured.out
    assert "- var_a" in captured.out
    assert "- var_c" in captured.out
    assert "--- Proceeding with removal ---" in captured.out
    assert "Removed 'var_a' from values." in captured.out
    assert "Removed 'var_c' from values." in captured.out

    with pytest.raises(NameError, match="Name 'var_a' is not defined"):
        get_user_variable("var_a")
    with pytest.raises(NameError, match="Name 'var_c' is not defined"):
        get_user_variable("var_c")

    var_d = UserVariable(name="var_d", value=solution.Cp)
    remove_user_variable(name="var_d")
    with pytest.raises(NameError, match="Name 'var_d' is not defined"):
        get_user_variable("var_d")


def test_remove_variable_with_no_confirmation(monkeypatch, capsys):
    # Simulate user typing 'no'
    monkeypatch.setattr("builtins.input", lambda _: "no")

    var_a = UserVariable(name="var_a", value=solution.Cpt)
    var_b = UserVariable(name="var_b", value=var_a**2)
    var_c = UserVariable(name="var_c", value=var_a**3)
    var_d = UserVariable(name="var_d", value=var_b**2 + var_a + var_c)

    remove_user_variable(name="var_a")

    # Optionally, check the output messages (stdout)
    captured = capsys.readouterr()
    assert "--- Confirmation Required ---" in captured.out
    assert ("The following variables will be removed:") in captured.out
    assert "- var_a" in captured.out
    assert "- var_b" in captured.out
    assert "- var_c" in captured.out
    assert "- var_d" in captured.out
    assert "Operation cancelled. No variables were removed." in captured.out

    assert var_a == get_user_variable("var_a")
    assert var_b == get_user_variable("var_b")
    assert var_c == get_user_variable("var_c")
    assert var_d == get_user_variable("var_d")


def test_remove_non_existent_variable():
    with pytest.raises(NameError, match="There is no variable named 'non_existent_var'."):
        remove_user_variable("non_existent_var")


def test_sanitize_expression_validator():
    """Test the sanitize_expression validator that removes whitespace and trailing characters."""

    # Test basic whitespace removal
    assert Expression.sanitize_expression("  x + y  ") == "x + y"
    assert Expression.sanitize_expression("\t x + y \t") == "x + y"
    assert Expression.sanitize_expression("\n x + y \n") == "x + y"

    # Test trailing semicolon removal
    assert Expression.sanitize_expression("x + y;") == "x + y"
    assert Expression.sanitize_expression("x + y ;") == "x + y"
    assert Expression.sanitize_expression("x + y;\t") == "x + y"
    assert Expression.sanitize_expression("x + y;\n") == "x + y"

    # Test multiple trailing characters
    assert Expression.sanitize_expression("x + y; \n\t") == "x + y"
    assert Expression.sanitize_expression("x + y \t\n;") == "x + y"

    # Test no changes needed
    assert Expression.sanitize_expression("x + y") == "x + y"
    assert Expression.sanitize_expression("") == ""

    # Test with complex expressions
    assert Expression.sanitize_expression("  (a + b) * c / d;  ") == "(a + b) * c / d"
    assert (
        Expression.sanitize_expression("\t math.sqrt(x**2 + y**2); \n") == "math.sqrt(x**2 + y**2)"
    )

    # Test with units
    assert Expression.sanitize_expression("  velocity * u.m / u.s;  ") == "velocity * u.m / u.s"

    # Test with mixed whitespace characters
    assert Expression.sanitize_expression("  \t\n x + y \n\t ;  ") == "x + y"


def test_disable_confusing_operators_validator():
    """Test the disable_confusing_operators validator that prevents use of ^ and & operators."""

    # Test valid expressions (should pass through unchanged)
    assert Expression.disable_confusing_operators("x + y") == "x + y"
    assert Expression.disable_confusing_operators("x ** y") == "x ** y"  # Valid power operator
    assert Expression.disable_confusing_operators("x and y") == "x and y"  # Valid logical AND
    assert Expression.disable_confusing_operators("math.sqrt(x)") == "math.sqrt(x)"
    assert Expression.disable_confusing_operators("") == ""

    # Test ^ operator (should raise ValueError)
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("x ^ y")

    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("2 ^ 3")

    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("x + y ^ z")

    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("(x + y) ^ 2")

    # Test & operator (should raise ValueError)
    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators("x & y")

    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators("a & b & c")

    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators("x + y & z")

    # Test both operators in same expression (should catch first one)
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("x ^ y & z")

    # Test operators in complex expressions
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("math.sqrt(x ^ 2 + y ^ 2)")

    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators("(a + b) & (c + d)")

    # Test operators in string literals (should still be caught)
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators("'x ^ y'")

    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators("'x & y'")


def test_expression_validators_integration():
    """Test that the validators work together when processing expression strings."""

    # Test sanitization and operator validation work together
    # This should pass: sanitize removes whitespace and semicolon, no invalid operators
    sanitized = Expression.sanitize_expression("  x + y;  ")
    assert sanitized == "x + y"
    # Should pass operator validation
    assert Expression.disable_confusing_operators(sanitized) == "x + y"

    # This should pass: sanitize removes whitespace, ** is valid power operator
    sanitized = Expression.sanitize_expression("  x ** y;  ")
    assert sanitized == "x ** y"
    # Should pass operator validation
    assert Expression.disable_confusing_operators(sanitized) == "x ** y"

    # This should fail: sanitize removes whitespace but ^ is still invalid
    sanitized = Expression.sanitize_expression("  x ^ y;  ")
    assert sanitized == "x ^ y"
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators(sanitized)

    # This should fail: sanitize removes whitespace but & is still invalid
    sanitized = Expression.sanitize_expression("  x & y;  ")
    assert sanitized == "x & y"
    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators(sanitized)

    # Test with complex expressions
    sanitized = Expression.sanitize_expression("  (a + b) ** 2;  ")
    assert sanitized == "(a + b) ** 2"
    assert Expression.disable_confusing_operators(sanitized) == "(a + b) ** 2"

    # Test with units
    sanitized = Expression.sanitize_expression("  velocity ** 2 * u.m / u.s;  ")
    assert sanitized == "velocity ** 2 * u.m / u.s"
    assert Expression.disable_confusing_operators(sanitized) == "velocity ** 2 * u.m / u.s"

    # Test edge cases
    sanitized = Expression.sanitize_expression("  ;  ")  # Just whitespace and semicolon
    assert sanitized == ""
    assert Expression.disable_confusing_operators(sanitized) == ""

    # Test with math functions
    sanitized = Expression.sanitize_expression("  math.sqrt(x ** 2 + y ** 2);  ")
    assert sanitized == "math.sqrt(x ** 2 + y ** 2)"
    assert Expression.disable_confusing_operators(sanitized) == "math.sqrt(x ** 2 + y ** 2)"


def test_expression_validators_with_user_variables():
    """Test validators work correctly with UserVariable expressions."""

    # Create a user variable
    x = UserVariable(name="x", value=5)

    # Test valid expressions
    sanitized = Expression.sanitize_expression("  x + 1;  ")
    assert sanitized == "x + 1"
    assert Expression.disable_confusing_operators(sanitized) == "x + 1"

    sanitized = Expression.sanitize_expression("  x ** 2;  ")
    assert sanitized == "x ** 2"
    assert Expression.disable_confusing_operators(sanitized) == "x ** 2"

    # Test invalid expressions
    sanitized = Expression.sanitize_expression("  x ^ 2;  ")
    assert sanitized == "x ^ 2"
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators(sanitized)

    sanitized = Expression.sanitize_expression("  x & 1;  ")
    assert sanitized == "x & 1"
    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators(sanitized)

    # Test with complex user variable expressions
    y = UserVariable(name="y", value=x + 1)
    sanitized = Expression.sanitize_expression("  y ** 2 + x;  ")
    assert sanitized == "y ** 2 + x"
    assert Expression.disable_confusing_operators(sanitized) == "y ** 2 + x"


def test_expression_validators_edge_cases():
    """Test validators with edge cases and boundary conditions."""

    # Test empty string
    sanitized = Expression.sanitize_expression("")
    assert sanitized == ""
    assert Expression.disable_confusing_operators(sanitized) == ""

    # Test string with only whitespace and special characters
    sanitized = Expression.sanitize_expression("  \t\n;  ")
    assert sanitized == ""
    assert Expression.disable_confusing_operators(sanitized) == ""

    # Test string with only valid operators
    sanitized = Expression.sanitize_expression("  + - * / ** // %  ")
    assert sanitized == "+ - * / ** // %"
    assert Expression.disable_confusing_operators(sanitized) == "+ - * / ** // %"

    # Test string with mixed valid and invalid operators
    sanitized = Expression.sanitize_expression("x + y ^ z")
    assert sanitized == "x + y ^ z"
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators(sanitized)

    sanitized = Expression.sanitize_expression("x + y & z")
    assert sanitized == "x + y & z"
    with pytest.raises(ValueError, match="& operator is not allowed in expressions."):
        Expression.disable_confusing_operators(sanitized)

    # Test with very long expressions
    long_expr = "  " + "x + " * 100 + "y;" + "  "
    expected = "x + " * 100 + "y"
    sanitized = Expression.sanitize_expression(long_expr)
    assert sanitized == expected
    assert Expression.disable_confusing_operators(sanitized) == expected

    # Test with unicode characters (should still work)
    sanitized = Expression.sanitize_expression("  α + β;  ")
    assert sanitized == "α + β"
    assert Expression.disable_confusing_operators(sanitized) == "α + β"

    # Test with numbers and operators
    sanitized = Expression.sanitize_expression("  123 ** 456;  ")
    assert sanitized == "123 ** 456"
    assert Expression.disable_confusing_operators(sanitized) == "123 ** 456"

    sanitized = Expression.sanitize_expression("  123 ^ 456;  ")
    assert sanitized == "123 ^ 456"
    with pytest.raises(
        ValueError,
        match="\\^ operator is not allowed in expressions. For power operator, please use \\*\\* instead.",
    ):
        Expression.disable_confusing_operators(sanitized)
