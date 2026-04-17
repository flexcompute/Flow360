# NOTE: Schema-pure expression tests (DependencyGraph, Variable, Expression operators/validators,
# ValueOrExpression, registry, utils, etc.) have been migrated to:
#   flex/share/flow360-schema/tests/framework/expression/
# The tests below depend on SimulationParams/validate_model/translator and cannot be migrated
# until those components are also migrated to flow360-schema.

import json

import flow360_schema.framework.expression.registry as context
import pytest
from flow360_schema.framework.expression import Expression, UserVariable
from flow360_schema.models.variables import control, solution

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
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.param_utils import AssetCache
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
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import (
    ValueOrExpression,
    save_user_variables,
)
from tests.simulation.conftest import to_file_from_file_test_approx


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
        output_format="tecplot",
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
        assert expression.to_solver_code(params.flow360_unit_system) == "(4.0 * pow(0.5, 2))"

        # 3. User variables are inlined (for expression value types)
        expression = Expression.model_validate(y * u.m**2)
        assert expression.to_solver_code(params.flow360_unit_system) == "(5.0 * pow(0.5, 2))"

        # 4. For solver variables, the units are stripped (assumed to be in solver units so factor == 1.0)
        expression = Expression.model_validate(y * u.m / u.s + control.MachRef)
        assert (
            expression.to_solver_code(params.flow360_unit_system)
            == "(((5.0 * 0.5) / 500.0) + machRef)"
        )


def test_variable_space_init():
    # Simulating loading a SimulationParams object from file - ensure that the variable space is loaded correctly
    with open("data/simulation.json", "r") as fh:
        data = json.load(fh)

    params, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
    )

    assert errors is None
    evaluated = params.reference_geometry.area.evaluate()

    assert evaluated == 1.0 * u.m**2


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
    # velocity scale = 5 m/s, length scale = 10m, density scale = 1000 kg/m**3
    # mut_scale = Rho*L*V -> 1000*10*5 * kg/m/s == 1000*10*5*1000 * kg/km/s
    assert (
        result.expression
        == "double ___mut; ___mut = mut * velocityScale;mut_in_km = (___mut * 50000000.0);"
    )

    vel_cross_vec = UserVariable(
        name="vel_cross_vec", value=math.cross(solution.velocity, [1, 2, 3] * u.cm)
    ).in_units(new_unit="CGS_unit_system")
    assert (
        vel_cross_vec.value.get_output_units(unit_system_name=params.unit_system.name)
        == u.cm**2 / u.s
    )

    # We disabled degC and degF on the interface and therefore the inferred units should be K or R.
    my_temp = UserVariable(name="my_temperature", value=solution.temperature).in_units(
        new_unit="Imperial_unit_system"
    )
    assert my_temp.value.get_output_units(unit_system_name=params.unit_system.name) == u.R

    # Test __pow__ on SolverVariable:
    vel_sq = UserVariable(name="vel_sq", value=solution.velocity**2)
    result = user_variable_to_udf(vel_sq, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;vel_sq[0] = (pow(___velocity[0], 2) * 25.0); vel_sq[1] = (pow(___velocity[1], 2) * 25.0); vel_sq[2] = (pow(___velocity[2], 2) * 25.0);"
    )

    # Test __neg__ on SolverVariable:
    neg_vel = UserVariable(name="neg_vel", value=-solution.velocity)
    result = user_variable_to_udf(neg_vel, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;neg_vel[0] = (-___velocity[0] * 5.0); neg_vel[1] = (-___velocity[1] * 5.0); neg_vel[2] = (-___velocity[2] * 5.0);"
    )

    # Test __pos__ on SolverVariable:
    pos_vel = UserVariable(name="pos_vel", value=+solution.velocity)
    result = user_variable_to_udf(pos_vel, input_params=params)
    assert (
        result.expression
        == "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;pos_vel[0] = (+___velocity[0] * 5.0); pos_vel[1] = (+___velocity[1] * 5.0); pos_vel[2] = (+___velocity[2] * 5.0);"
    )

    density_kg_per_m3 = UserVariable(name="density_kg_per_m3", value=solution.density).in_units(
        new_unit="kg /m**3"
    )
    velocity_metric = UserVariable(name="velocity_metric", value=solution.velocity).in_units(
        new_unit="m/s"
    )
    mass_flow_rate_kg_per_s_per_m2 = UserVariable(
        name="mass_flow_rate_kg_per_s",
        value=math.dot(velocity_metric, solution.node_unit_normal) * density_kg_per_m3,
    ).in_units(new_unit="kg/s/m**2")

    assert user_variable_to_udf(mass_flow_rate_kg_per_s_per_m2, input_params=params).expression == (
        "double ___density;"
        "___density = usingLiquidAsMaterial ? 1.0 : primitiveVars[0];"
        "double ___node_unit_normal[3];"
        "double ___normalMag = magnitude(nodeNormals);"
        "for (int i = 0; i < 3; i++)"
        "{"
        "___node_unit_normal[i] = nodeNormals[i] / ___normalMag;"
        "}"
        "double ___velocity[3];"
        "___velocity[0] = primitiveVars[1] * velocityScale;"
        "___velocity[1] = primitiveVars[2] * velocityScale;"
        "___velocity[2] = primitiveVars[3] * velocityScale;"
        "mass_flow_rate_kg_per_s = ("
        "((("
        "(___velocity[0] * ___node_unit_normal[0]) + "
        "(___velocity[1] * ___node_unit_normal[1])"
        ") + "
        "(___velocity[2] * ___node_unit_normal[2])"
        ") * ___density) * 5000.0);"
    )


def test_project_variables_serialization():
    ccc = UserVariable(name="ccc", value=12 * u.m / u.s, description="ccc description")
    aaa = UserVariable(
        name="aaa", value=[solution.velocity[0] + ccc, solution.velocity[1], solution.velocity[2]]
    )
    bbb = UserVariable(name="bbb", value=[aaa[0] + 14 * u.m / u.s, aaa[1], aaa[2]]).in_units(
        new_unit="km/ms",
    )
    ddd = UserVariable(name="ddd", value=Expression(expression="12 * u.ft / u.s"))
    eee = UserVariable(name="eee", value=1.0)

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
                        ddd,
                        eee,
                    ],
                    private_attribute_id="000",
                ),
                ProbeOutput(
                    probe_points=[
                        Point(name="pt1", location=(1, 2, 3), private_attribute_id="111")
                    ],
                    output_fields=[bbb],
                    private_attribute_id="222",
                ),
            ],
        )

    params = save_user_variables(params)
    params_dict = params.model_dump(mode="json", exclude_none=True)
    variable_context = params_dict["private_attribute_asset_cache"]["variable_context"]
    output_units_by_name = {
        item["name"]: item["value"].get("output_units") for item in variable_context
    }
    assert output_units_by_name["bbb"] == "km/ms"
    assert output_units_by_name["ddd"] == "m/s"
    assert output_units_by_name["eee"] == "dimensionless"

    paramsJson = json.dumps(
        json.loads(params.model_dump_json(exclude_none=True)), indent=4, sort_keys=True
    )
    with open("ref/simulation_with_project_variables.json", "w") as f:
        f.write(paramsJson)

    with open("ref/simulation_with_project_variables.json", "r") as fh:
        ref_data = fh.read()

    assert ref_data == json.dumps(
        json.loads(params.model_dump_json(exclude_none=True)), indent=4, sort_keys=True
    )


def test_project_variables_deserialization():
    with open("ref/simulation_with_project_variables.json", "r") as fh:
        data = json.load(fh)

    # Assert no variables registered yet
    with pytest.raises(NameError):
        context.default_context.get("aaa")
    with pytest.raises(NameError):
        context.default_context.get("bbb")
    with pytest.raises(NameError):
        context.default_context.get("ccc")

    params, errors, _ = validate_model(
        params_as_dict=data,
        root_item_type=None,
        validated_by=ValidationCalledBy.LOCAL,
    )
    assert errors is None, errors
    assert params
    assert (
        params.outputs[0].output_fields.items[0].value.expression
        == "[aaa[0] + 14 * u.m / u.s, aaa[1], aaa[2]]"
    )
    assert context.default_context.get_metadata("ccc", "description") == "ccc description"

    assert params.outputs[0].output_fields.items[0].value.output_units == "km/ms"

    assert (
        params.outputs[0]
        .output_fields.items[0]
        .value.evaluate(force_evaluate=False, raise_on_non_evaluable=False)
        .expression
        == "[solution.velocity[0] + 12 * u.m / u.s + 14 * u.m / u.s, solution.velocity[1], solution.velocity[2]]"
    )  # Fully resolvable


def test_project_variables_metadata():
    # Testing of loading and saving metadata for project variable
    with open("ref/simulation_with_project_variables.json", "r") as fh:
        data = json.load(fh)
    mock_metadata = {
        "radom_key": "some_value",
        "additional_dict": {"key>>>": ["value1", "value2"]},
    }
    data["private_attribute_asset_cache"]["variable_context"][0]["metadata"] = mock_metadata
    params, _, _ = validate_model(
        params_as_dict=data,
        root_item_type=None,
        validated_by=ValidationCalledBy.LOCAL,
    )

    a_var = UserVariable(name="a_var", value=1111)
    params.outputs[0].output_fields.items.append(a_var)

    params = save_user_variables(params)

    param_dumped = params.model_dump(mode="json", exclude_none=True)

    assert param_dumped["private_attribute_asset_cache"]["variable_context"] == [
        {
            "name": "ccc",
            "value": {"type_name": "expression", "expression": "12 * u.m / u.s"},
            "post_processing": False,
            "description": "ccc description",
            "metadata": {
                "radom_key": "some_value",
                "additional_dict": {"key>>>": ["value1", "value2"]},
            },
        },
        {
            "name": "aaa",
            "value": {
                "type_name": "expression",
                "expression": "[solution.velocity[0] + ccc, solution.velocity[1], solution.velocity[2]]",
            },
            "post_processing": False,
        },
        {
            "name": "bbb",
            "value": {
                "type_name": "expression",
                "expression": "[aaa[0] + 14 * u.m / u.s, aaa[1], aaa[2]]",
                "output_units": "km/ms",
            },
            "post_processing": True,
        },
        {
            "name": "ddd",
            "value": {
                "type_name": "expression",
                "expression": "12 * u.ft / u.s",
                "output_units": "m/s",
            },
            "post_processing": True,
        },
        {
            "name": "eee",
            "value": {
                "type_name": "expression",
                "expression": "1.0",
                "output_units": "dimensionless",
            },
            "post_processing": True,
        },
        {
            "name": "a_var",
            "value": {
                "type_name": "expression",
                "expression": "1111.0",
                "output_units": "dimensionless",
            },
            "post_processing": True,
        },
    ]
    assert param_dumped["outputs"][0]["output_fields"]["items"] == [
        {"name": "bbb", "type_name": "UserVariable"},
        {"name": "ddd", "type_name": "UserVariable"},
        {"name": "eee", "type_name": "UserVariable"},
        {
            "name": "a_var",
            "type_name": "UserVariable",
        },
    ]


def test_deserialization_with_wrong_syntax():
    with open("data/simulation_with_wrong_expr_syntax.json", "r") as fh:
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


def test_correct_expression_error_location():

    with open("data/simulation.json", "r") as fh:
        data = json.load(fh)
    data["private_attribute_asset_cache"]["variable_context"][1]["value"][
        "expression"
    ] = "math.sqrt(z) + 12*u.m"

    _, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="Geometry"
    )
    assert len(errors) == 1
    assert errors[0]["loc"] == ("private_attribute_asset_cache", "variable_context", 1)
    assert (
        "operator for unyt_arrays with units 'dimensionless' (dimensions '1') and 'm' (dimensions '(length)') is not well defined."
        in errors[0]["msg"]
    )
