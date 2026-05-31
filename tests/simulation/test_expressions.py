# NOTE: Schema-pure expression tests (DependencyGraph, Variable, Expression operators/validators,
# ValueOrExpression, registry, utils, etc.) have been migrated to:
#   flex/share/flow360-schema/tests/framework/expression/
# The tests below still cover client-side serialization and validation behavior.

import json

import flow360_schema.framework.expression.registry as context
import pytest
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.expression import (
    Expression,
    UserVariable,
    ValueOrExpression,
)
from flow360_schema.framework.param_utils import AssetCache
from flow360_schema.models.entities.output_entities import Point
from flow360_schema.models.entities.surface_entities import Surface
from flow360_schema.models.entities.volume_entities import GenericVolume
from flow360_schema.models.reference_geometry import ReferenceGeometry
from flow360_schema.models.simulation.models.material import Water, aluminum
from flow360_schema.models.simulation.outputs.outputs import (
    ProbeOutput,
    SurfaceOutput,
    VolumeOutput,
)
from flow360_schema.models.variables import solution

from flow360 import (
    AerospaceCondition,
    HeatEquationInitialCondition,
    SimulationParams,
    Solid,
    Unsteady,
    u,
)
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import save_user_variables
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
