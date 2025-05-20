from typing import List

import numpy as np
import pydantic as pd
import pytest

from flow360 import control, solution, u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
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
    SpecificEnergyType,
    SpecificHeatCapacityType,
    ThermalConductivityType,
    TimeType,
    VelocityType,
    ViscosityType,
)
from flow360.component.simulation.user_code import (
    Expression,
    UserVariable,
    ValueOrExpression,
)


def test_variable_init():
    class TestModel(Flow360BaseModel):
        field: List[UserVariable] = pd.Field()

    # Variables can be initialized with a...

    # Value
    a = UserVariable(name="a", value=1)

    # Dimensioned value
    b = UserVariable(name="b", value=1 * u.m)

    # Expression (possibly with other variable)
    c = UserVariable(name="c", value=b + 1 * u.m)

    # A dictionary (can contain extra values - important for frontend)
    d = UserVariable.model_validate({"name": "d", "value": 1, "extra": "foo"})


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

    # Absolute value
    model.field = abs(x)
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 3
    assert str(model.field) == "abs(x)"

    # Complex statement
    model.field = ((abs(x) - 2 * x) + (x + y) / 2 - 2**x) % 4
    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 3.5
    assert str(model.field) == "(abs(x) - (2 * x) + ((x + y) / 2) - (2 ** x)) % 4"


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
        field: ValueOrExpression[pd.confloat(ge=0)] = pd.Field()

    x = UserVariable(name="x", value=1)

    model = TestModel(field=x)

    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "x"

    with pytest.raises(pd.ValidationError):
        model.field = -x


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

    x = UserVariable(name="x", value=np.array([1, 0, 0]))
    y = UserVariable(name="y", value=np.array([0, 0, 0]))
    z = UserVariable(name="z", value=np.array([1, 0, 0, 0]))
    w = UserVariable(name="w", value=np.array([1, 1, 1]))

    model = TestModel(
        vector=y * u.m, axis=x * u.m, array=z * u.m, direction=x * u.m, moment=w * u.m
    )

    assert isinstance(model.vector, Expression)
    assert (model.vector.evaluate() == [0, 0, 0] * u.m).all()
    assert str(model.vector) == "y * u.m"

    assert isinstance(model.axis, Expression)
    assert (model.axis.evaluate() == [1, 0, 0] * u.m).all()
    assert str(model.axis) == "x * u.m"

    assert isinstance(model.array, Expression)
    assert (model.array.evaluate() == [1, 0, 0, 0] * u.m).all()
    assert str(model.array) == "z * u.m"

    assert isinstance(model.direction, Expression)
    assert (model.direction.evaluate() == [1, 0, 0] * u.m).all()
    assert str(model.direction) == "x * u.m"

    assert isinstance(model.moment, Expression)
    assert (model.moment.evaluate() == [1, 1, 1] * u.m).all()
    assert str(model.moment) == "w * u.m"

    with pytest.raises(pd.ValidationError):
        model.vector = z * u.m

    with pytest.raises(pd.ValidationError):
        model.axis = y * u.m

    with pytest.raises(pd.ValidationError):
        model.direction = y * u.m

    with pytest.raises(pd.ValidationError):
        model.moment = x * u.m


def test_solver_builtin():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    x = UserVariable(name="x", value=4)

    model = TestModel(field=x * u.m + solution.kOmega * u.cm)

    assert str(model.field) == "x * u.m + (solution.kOmega * u.cm)"

    # Raises when trying to evaluate with a message about this variable being blacklisted
    with pytest.raises(ValueError):
        model.field.evaluate()


def test_serializer():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    model = TestModel(field=x * u.m / u.s + 4 * x**2 * u.m / u.s)

    assert str(model.field) == "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)"

    serialized = model.model_dump(exclude_none=True)

    assert serialized["field"]["type_name"] == "expression"
    assert serialized["field"]["expression"] == "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)"

    model = TestModel(field=4 * u.m / u.s)

    serialized = model.model_dump(exclude_none=True)

    assert serialized["field"]["type_name"] == "number"
    assert serialized["field"]["value"] == 4
    assert serialized["field"]["units"] == "m/s"


def test_deserializer():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    model = {
        "type_name": "expression",
        "expression": "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)",
        "evaluated_value": 68.0,
        "evaluated_units": "m/s",
    }

    deserialized = TestModel(field=model)

    assert str(deserialized.field) == "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)"

    model = {"type_name": "number", "value": 4.0, "units": "m/s"}

    deserialized = TestModel(field=model)

    assert str(deserialized.field) == "4.0 m/s"


def test_numpy_interop_scalars():
    class ScalarModel(Flow360BaseModel):
        scalar: ValueOrExpression[float] = pd.Field()

    x = UserVariable(name="x", value=1)

    # Since numpy arrays already give us the behavior we want there is no point
    # to building our own vector arithmetic. We just add the symbols to the whitelist

    # Using expression types inside numpy arrays works OK
    a = np.array([x + 1, 0, x**2])
    b = np.array([0, x / 2, 3])

    c = np.linalg.norm(a + b)  # This yields an expression containing the inlined dot product...

    # Sadly it seems like we cannot stop numpy from inlining some functions by
    # implementing a specific method (like with trigonometic functions for example)

    d = np.sin(c)  # This yields an expression
    e = np.cos(c)  # This also yields an expression

    model = ScalarModel(
        scalar=np.arctan(d + e + 1)
    )  # So we can later compose those into expressions further...

    assert str(model.scalar) == (
        "np.arctan(np.sin(np.sqrt((x + 1 + 0) * (x + 1 + 0) + "
        "((0 + x / 2) * (0 + x / 2)) + ((x ** 2 + 3) * (x ** 2 + 3)))) + "
        "(np.cos(np.sqrt((x + 1 + 0) * (x + 1 + 0) + ((0 + x / 2) * "
        "(0 + x / 2)) + ((x ** 2 + 3) * (x ** 2 + 3))))) + 1)"
    )

    result = model.scalar.evaluate()

    assert result == -0.1861456975646416

    # Notice that when we inline some operations (like cross/dot product or norm, for example)
    # we make the underlying generated string of the expression ugly...
    #
    # Luckily this is transparent to the user. When the user is defining expressions in python he does
    # not have to worry about the internal string representation or the CUDA-generated code
    # (for CUDA code inlining might actually probably be our best bet to reduce function calls...)
    #
    # Conversely, when we are dealing with frontend strings we can deal with non-inlined numpy functions
    # because they are whitelisted by the blueprint parser:

    # Let's simulate a frontend use case by parsing raw string input:

    # The user defines some variables for convenience

    a = UserVariable(name="a", value="np.array([x + 1, 0, x ** 2])")
    b = UserVariable(name="b", value="np.array([0, x / 2, 3])")

    c = UserVariable(name="c", value="np.linalg.norm(a + b)")

    d = UserVariable(name="d", value="np.sin(c)")
    e = UserVariable(name="e", value="np.cos(c)")

    # Then he inputs the actual expression somewhere within
    # simulation.json using the helper variables defined before

    model = ScalarModel(scalar="np.arctan(d + e + 1)")

    assert str(model.scalar) == "np.arctan(d + e + 1)"

    result = model.scalar.evaluate()

    assert result == -0.1861456975646416


def test_numpy_interop_vectors():
    Vec3 = tuple[float, float, float]

    class VectorModel(Flow360BaseModel):
        vector: ValueOrExpression[Vec3] = pd.Field()

    x = UserVariable(name="x", value=np.array([2, 3, 4]))
    y = UserVariable(name="y", value=2 * x)

    model = VectorModel(vector=x**2 + y + np.array([1, 0, 0]))

    assert str(model.vector) == "x ** 2 + y + np.array([1,0,0])"

    result = model.vector.evaluate()

    assert np.array_equal(result, np.array([9, 15, 24]))


def test_subscript_access():
    class ScalarModel(Flow360BaseModel):
        scalar: ValueOrExpression[float] = pd.Field()

    x = UserVariable(name="x", value=np.array([2, 3, 4]))

    model = ScalarModel(scalar=x[0] + x[1] + x[2] + 1)

    assert str(model.scalar) == "x[0] + (x[1]) + (x[2]) + 1"

    assert model.scalar.evaluate() == 10

    model = ScalarModel(scalar="x[0] + x[1] + x[2] + 1")

    assert str(model.scalar) == "x[0] + x[1] + x[2] + 1"

    assert model.scalar.evaluate() == 10


def test_error_message():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    try:
        model = TestModel(field="1 + nonexisting * 1")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "Name 'nonexisting' is not defined" in validation_errors[0]["msg"]

    try:
        model = TestModel(field="1 + x * 1")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "does not match (length)/(time) dimension" in validation_errors[0]["msg"]

    try:
        model = TestModel(field="1 * 1 +")
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
        model = TestModel(field="1 * 1 +* 2")
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
        model = TestModel(field="1 * 1 + (2")
    except pd.ValidationError as err:
        validation_errors = err.errors()

        assert len(validation_errors) >= 1
        assert validation_errors[0]["type"] == "value_error"
        assert "unexpected EOF" in validation_errors[0]["msg"]
        assert "1 * 1 + (2" in validation_errors[0]["msg"]
        assert "line" in validation_errors[0]["ctx"]
        assert "column" in validation_errors[0]["ctx"]
        assert validation_errors[0]["ctx"]["column"] == 11


def test_solver_translation():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[float] = pd.Field()

    x = UserVariable(name="x", value=4)

    model = TestModel(field=(x // 3) ** 2)

    assert isinstance(model.field, Expression)
    assert model.field.evaluate() == 1
    assert str(model.field) == "(x // 3) ** 2"

    solver_code = model.field.to_solver_code()

    assert solver_code == "pow(floor(x / 3), 2)"


def test_cyclic_dependencies():
    x = UserVariable(name="x", value=4)
    y = UserVariable(name="y", value=x)

    # If we try to create a cyclic dependency we throw a validation error
    # The error contains info about the cyclic dependency, so here its x -> y -> x
    with pytest.raises(pd.ValidationError):
        x.value = y

    x = UserVariable(name="x", value=4)

    with pytest.raises(pd.ValidationError):
        x.value = x


def test_auto_alias():
    class TestModel(Flow360BaseModel):
        field: ValueOrExpression[VelocityType] = pd.Field()

    x = UserVariable(name="x", value=4)

    unaliased = {
        "type_name": "expression",
        "expression": "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)",
        "evaluated_value": 68.0,
        "evaluated_units": "m/s",
    }

    aliased = {
        "typeName": "expression",
        "expression": "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)",
        "evaluatedValue": 68.0,
        "evaluatedUnits": "m/s",
    }

    model_1 = TestModel(field=unaliased)
    model_2 = TestModel(field=aliased)

    assert str(model_1.field) == "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)"
    assert str(model_2.field) == "(x * u.m) / u.s + (((4 * (x ** 2)) * u.m) / u.s)"
