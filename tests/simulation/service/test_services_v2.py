import copy
import json
from typing import get_args

import pytest
from flow360_schema import __version__ as _SCHEMA_VERSION
from flow360_schema.framework.expression import UserVariable
from unyt import Unit

import flow360.component.simulation.units as u
from flow360.component.simulation import services
from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.services_report import get_default_report_config
from flow360.component.simulation.unit_system import DimensionedTypes


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_forward_compatibility_error():
    # Mock a future simulation.json
    with open("data/updater_should_pass.json", "r") as fp:
        future_dict = json.load(fp)
    future_dict["version"] = "99.99.99"
    _, errors, _ = services.validate_model(
        params_as_dict=future_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )

    assert errors[0] == {
        "type": f"99.99.99 > {_SCHEMA_VERSION}",
        "loc": [],
        "msg": f"The cloud `SimulationParam` (version: 99.99.99) is too new for your local schema package (version: {_SCHEMA_VERSION}). "
        "Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }

    _, errors, _ = services.validate_model(
        params_as_dict=future_dict,
        validated_by=services.ValidationCalledBy.PIPELINE,
        root_item_type="Geometry",
    )

    assert errors[0] == {
        "type": f"99.99.99 > {_SCHEMA_VERSION}",
        "loc": [],
        "msg": f"[Internal] Your `SimulationParams` (version: 99.99.99) is too new for the solver (version: {_SCHEMA_VERSION}). Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }


def test_unit_conversion_front_end_compatibility():

    ##### 1. Ensure that the units are valid in `supported_units_by_front_end`
    def _get_all_units(value):
        if isinstance(value, dict):
            return [item for item in value.values()]
        else:
            assert isinstance(value, list)
            return value

    for dimension, value in supported_units_by_front_end.items():
        for unit in _get_all_units(value=value):
            if str(Unit(unit).dimensions) == dimension:
                continue
            elif (
                dimension == "(temperature_difference)"
                and str(Unit(unit).dimensions) == "(temperature)"
            ):
                continue
            else:
                raise ValueError(f"Unit {unit} is not valid for dimension {dimension}")

    ##### 2.  Ensure that all units supported have set their front-end approved units
    for dim_type in get_args(DimensionedTypes):
        inner_type = get_args(dim_type)[0]  # unwrap Annotated
        unit_system_dimension_string = str(inner_type.dim)
        dim_name = inner_type.dim_name
        if unit_system_dimension_string not in supported_units_by_front_end.keys():
            raise ValueError(
                f"Unit {unit_system_dimension_string} (A.K.A {dim_name}) is not supported by the front-end.",
                "Please ensure front end team is aware of this new unit and add its support.",
            )


def test_get_default_report_config_json():
    report_config_dict = get_default_report_config()
    with open("ref/default_report_config.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(report_config_dict, ref_dict, ignore_keys=["formatter"])


@pytest.mark.parametrize("unit_system_name", ["SI", "Imperial", "CGS"])
def test_validate_model_preserves_unit_system(unit_system_name):
    """validate_model must not mutate the unit_system entry in the input dict."""
    with open("data/simulation.json", "r") as fp:
        params_data = json.load(fp)

    # Override the declared unit system. The serialized values are always SI on
    # the wire (display_unit metadata is what the WebUI honors for rendering),
    # so the name change is all that's needed for this test's purpose.
    params_data["unit_system"]["name"] = unit_system_name
    unit_system_before = copy.deepcopy(params_data["unit_system"])

    validated_param, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    assert params_data["unit_system"] == unit_system_before
    if validated_param is not None:
        assert validated_param.unit_system.name == unit_system_name
