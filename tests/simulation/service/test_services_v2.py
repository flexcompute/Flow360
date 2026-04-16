import copy
import json
import re
from typing import get_args

import pytest
from flow360_schema.framework.expression import UserVariable
from unyt import Unit

import flow360.component.simulation.units as u
from flow360.component.simulation import services
from flow360.component.simulation.exposed_units import supported_units_by_front_end
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.services_report import get_default_report_config
from flow360.component.simulation.unit_system import DimensionedTypes
from flow360.component.simulation.validation.validation_context import (
    CASE,
)
from flow360.version import __version__
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_validate_error_from_initialize_variable_space():
    with open("../translator/data/simulation_isosurface.json", "r") as fp:
        param_dict = json.load(fp)

    UserVariable(name="my_time_stepping_var", value=0.6 * u.s)
    _, errors, _ = services.validate_model(
        params_as_dict=param_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": ["unknown"],
            "msg": "Loading user variable 'my_time_stepping_var' from simulation.json "
            "which is already defined in local context. Please change your local user variable definition.",
        }
    ]
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["msg"] == exp_err["msg"]

    services.clear_context()
    _ = UserVariable(name="my_time_stepping_var", value=0.6 * u.s)
    _, errors, _ = services.validate_model(
        params_as_dict=param_dict,
        validated_by=services.ValidationCalledBy.SERVICE,
        root_item_type="VolumeMesh",
    )
    assert errors is None


def test_front_end_JSON_with_multi_constructor():
    params_data = {
        "meshing": {
            "defaults": {
                "boundary_layer_first_layer_thickness": {"value": 1, "units": "m"},
                "surface_max_edge_length": {"value": 1, "units": "m"},
            },
            "refinement_factor": 1.45,
            "refinements": [
                {
                    "entities": {
                        "stored_entities": [
                            {
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_entity_type_name": "Box",
                                "private_attribute_id": "hardcoded_id-1",
                                "name": "my_box_default",
                                "private_attribute_zone_boundary_names": {"items": []},
                                "type_name": "Box",
                                "private_attribute_constructor": "default",
                                "private_attribute_input_cache": {},
                                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                                "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                                "axis_of_rotation": [1.0, 0.0, 0.0],
                                "angle_of_rotation": {"value": 20.0, "units": "degree"},
                            },
                            {
                                "type_name": "Box",
                                "private_attribute_id": "hardcoded_id-2",
                                "private_attribute_constructor": "from_principal_axes",
                                "private_attribute_input_cache": {
                                    "axes": [[0.6, 0.8, 0.0], [0.8, -0.6, 0.0]],
                                    "center": {"value": [7.0, 1.0, 2.0], "units": "m"},
                                    "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                                    "name": "my_box_from",
                                },
                            },
                            {
                                "private_attribute_registry_bucket_name": "VolumetricEntityType",
                                "private_attribute_entity_type_name": "Cylinder",
                                "private_attribute_id": "hardcoded_id-3",
                                "name": "my_cylinder_default",
                                "private_attribute_zone_boundary_names": {"items": []},
                                "axis": [0.0, 1.0, 0.0],
                                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                                "height": {"value": 3.0, "units": "m"},
                                "outer_radius": {"value": 2.0, "units": "m"},
                            },
                        ]
                    },
                    "refinement_type": "UniformRefinement",
                    "spacing": {"units": "cm", "value": 7.5},
                }
            ],
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "private_attribute_id": "hardcoded_id-4",
                        "name": "automated_farfield_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.2.0",
        "private_attribute_asset_cache": {
            "project_length_unit": 1.0,
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x_1", "face_x_2", "face_x_3"],
                "face_group_tag": "some_tag",
                "face_attribute_names": ["some_tag"],
                "grouped_faces": [
                    [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                ],
            },
        },
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                },
                "use_wall_function": False,
                "private_attribute_id": "wall1",
            }
        ],
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "private_attribute_constructor": "from_mach",
            "private_attribute_input_cache": {
                "alpha": {"value": 5.0, "units": "degree"},
                "beta": {"value": 0.0, "units": "degree"},
                "thermal_state": {
                    "type_name": "ThermalState",
                    "private_attribute_constructor": "from_standard_atmosphere",
                    "private_attribute_input_cache": {
                        "altitude": {"value": 1000.0, "units": "m"},
                        "temperature_offset": {"value": 0.0, "units": "K"},
                    },
                },
                "mach": 0.8,
            },
        },
    }

    simulation_param, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    assert errors is None
    with open("../../ref/simulation/simulation_json_with_multi_constructor_used.json", "r") as f:
        ref_data = json.load(f)
        ref_param, err, _ = services.validate_model(
            params_as_dict=ref_data,
            root_item_type="Geometry",
            validated_by=services.ValidationCalledBy.LOCAL,
        )
        assert err is None

    param_dict = simulation_param.model_dump(exclude_none=True)
    ref_param_dict = ref_param.model_dump(exclude_none=True)
    assert compare_values(ref_param_dict, param_dict)


def test_generate_process_json():
    params_data = {
        "meshing": {
            "defaults": {},
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfield_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "private_attribute_constructor": "default",
            "private_attribute_input_cache": {},
            "alpha": {"value": 5.0, "units": "degree"},
            "beta": {"value": 0.0, "units": "degree"},
            # "velocity_magnitude": {"value": 0.8, "units": "km/s"},
        },
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                },
                "use_wall_function": False,
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": 1.0,
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x_1", "face_x_2", "face_x_3"],
                "face_group_tag": "some_tag",
                "face_attribute_names": ["some_tag"],
                "grouped_faces": [
                    [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                ],
            },
        },
    }

    params_data["meshing"]["defaults"]["surface_max_edge_length"] = {"value": 1, "units": "m"}
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="SurfaceMesh"
    )

    assert res1 is not None
    assert res2 is None
    assert res3 is None

    params_data["meshing"]["defaults"]["boundary_layer_first_layer_thickness"] = {
        "value": 1,
        "units": "m",
    }
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="VolumeMesh"
    )

    assert res1 is not None
    assert res2 is not None
    assert res3 is None

    params_data["operating_condition"]["velocity_magnitude"] = {"value": 0.8, "units": "km/s"}
    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="Case"
    )

    assert res1 is not None
    assert res2 is not None
    assert res3 is not None


def test_generate_process_json_skips_case_validation_for_meshing():
    """velocity_magnitude=0 without reference_velocity should not fail when only generating mesh JSON."""
    params_data = {
        "meshing": {
            "defaults": {"surface_max_edge_length": 1.0},
            "volume_zones": [
                {
                    "method": "auto",
                    "type": "AutomatedFarfield",
                    "private_attribute_entity": {
                        "private_attribute_registry_bucket_name": "VolumetricEntityType",
                        "private_attribute_entity_type_name": "GenericVolume",
                        "name": "automated_farfield_entity",
                        "private_attribute_zone_boundary_names": {"items": []},
                    },
                }
            ],
        },
        "unit_system": {"name": "SI"},
        "version": "24.11.5",
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "private_attribute_constructor": "default",
            "private_attribute_input_cache": {},
            "alpha": {"value": 0, "units": "degree"},
            "beta": {"value": 0, "units": "degree"},
            "velocity_magnitude": {"value": 0, "units": "m/s"},
        },
        "models": [
            {
                "type": "Wall",
                "entities": {
                    "stored_entities": [
                        {
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_is_interface": False,
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                },
                "use_wall_function": False,
            }
        ],
        "private_attribute_asset_cache": {
            "project_length_unit": 1.0,
            "project_entity_info": {
                "type_name": "GeometryEntityInfo",
                "face_ids": ["face_x_1", "face_x_2", "face_x_3"],
                "face_group_tag": "some_tag",
                "face_attribute_names": ["some_tag"],
                "grouped_faces": [
                    [
                        {
                            "private_attribute_entity_type_name": "Surface",
                            "name": "surface_x",
                            "private_attribute_tag_key": "some_tag",
                            "private_attribute_sub_components": [
                                "face_x_1",
                                "face_x_2",
                                "face_x_3",
                            ],
                        }
                    ]
                ],
            },
        },
    }

    res1, res2, res3 = services.generate_process_json(
        simulation_json=json.dumps(params_data), root_item_type="Geometry", up_to="SurfaceMesh"
    )

    assert res1 is not None
    assert res2 is None
    assert res3 is None

def test_forward_compatibility_error():

    from flow360.version import __version__

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
        "type": f"99.99.99 > {__version__}",
        "loc": [],
        "msg": f"The cloud `SimulationParam` (version: 99.99.99) is too new for your local Python client (version: {__version__}). "
        "Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }

    _, errors, _ = services.validate_model(
        params_as_dict=future_dict,
        validated_by=services.ValidationCalledBy.PIPELINE,
        root_item_type="Geometry",
    )

    assert errors[0] == {
        "type": f"99.99.99 > {__version__}",
        "loc": [],
        "msg": f"[Internal] Your `SimulationParams` (version: 99.99.99) is too new for the solver (version: {__version__}). Errors may occur since forward compatibility is limited.",
        "ctx": {},
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Your `SimulationParams` (version: 99.99.99) is too new for the solver (version: {__version__}). Errors may occur since forward compatibility is limited."
        ),
    ):
        _, _, _ = services.generate_process_json(
            simulation_json=json.dumps(future_dict),
            up_to=CASE,
            root_item_type="Geometry",
        )


def validate_proper_unit(obj, allowed_units_string):
    def is_expected_unit(unit_str, allowed_units_string):
        tokens = re.findall(r"[A-Za-z_]+", unit_str)
        return all(token in allowed_units_string for token in tokens)

    if isinstance(obj, dict):
        if "value" in obj and "units" in obj:
            assert is_expected_unit(obj["units"], allowed_units_string)

        for key, val in obj.items():
            if key == "project_length_unit":
                continue
            validate_proper_unit(val, allowed_units_string)

    elif isinstance(obj, list):
        for item in obj:
            validate_proper_unit(item, allowed_units_string)


def test_imperial_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, target_unit_system="Imperial")
    imperial_units = {"ft", "lbf", "lb", "s", "degF", "delta_degF", "rad", "degree"}
    unit_system_names = {
        "SI_unit_system",
        "Imperial_unit_system",
        "CGS_unit_system",
    }

    validate_proper_unit(dict_to_convert, (imperial_units | unit_system_names))
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "degF"
    assert abs(temperature_tester["value"] - 302) / 302 < 1e-10

    # Assert stop criterion tolerance unit is correct
    assert (
        dict_to_convert["run_control"]["stopping_criteria"][1]["tolerance"]["units"]
        == "SI_unit_system"
    )

    # General comparison\
    with open("./ref/unit_system_converted_imperial.json", "r") as fp:
        ref_dict = json.load(fp)

    assert compare_values(dict_to_convert, ref_dict)


def test_CGS_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, target_unit_system="CGS")
    CGS_units = {"dyn", "cm", "g", "s", "K", "rad", "degree"}
    unit_system_names = {
        "SI_unit_system",
        "Imperial_unit_system",
        "CGS_unit_system",
    }

    validate_proper_unit(dict_to_convert, (CGS_units | unit_system_names))
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "K"
    assert abs(temperature_tester["value"] - 423.15) / 423.15 < 1e-10

    # Assert stop criterion tolerance unit is correct
    assert (
        dict_to_convert["run_control"]["stopping_criteria"][1]["tolerance"]["units"]
        == "SI_unit_system"
    )

    # General comparison
    with open("./ref/unit_system_converted_CGS.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(dict_to_convert, ref_dict, rtol=1e-7)  # Default tol fail for Windows


def test_SI_unit_system_conversion():
    with open("data/simulation_param.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, target_unit_system="SI")
    SI_units = {"m", "kg", "s", "K", "rad", "degree", "Pa"}
    unit_system_names = {
        "SI_unit_system",
        "Imperial_unit_system",
        "CGS_unit_system",
    }

    validate_proper_unit(dict_to_convert, (SI_units | unit_system_names))
    # Check that the angles are not changed
    assert dict_to_convert["meshing"]["refinements"][0]["entities"]["stored_entities"][0][
        "angle_of_rotation"
    ] == {"units": "degree", "value": 20.0}

    # Assert no change in angle unit
    assert dict_to_convert["operating_condition"]["alpha"] == {"units": "rad", "value": 5.0}

    # Assert temperature unit name is correct
    temperature_tester = dict_to_convert["operating_condition"]["thermal_state"]["material"][
        "dynamic_viscosity"
    ]["effective_temperature"]
    assert temperature_tester["units"] == "K"
    assert abs(temperature_tester["value"] - 423.15) / 423.15 < 1e-10

    # Assert stop criterion tolerance unit is correct
    assert (
        dict_to_convert["run_control"]["stopping_criteria"][1]["tolerance"]["units"]
        == "SI_unit_system"
    )

    # General comparison
    with open("./ref/unit_system_converted_SI.json", "r") as fp:
        ref_dict = json.load(fp)
    assert compare_values(dict_to_convert, ref_dict, rtol=1e-7)  # Default tol fail for Windows


def test_unchanged_BETDisk_length_unit():
    with open("data/simulation_bet_disk.json", "r") as fp:
        dict_to_convert = json.load(fp)
    services.change_unit_system(data=dict_to_convert, target_unit_system="CGS")
    assert dict_to_convert["unit_system"]["name"] == "CGS"
    assert dict_to_convert["models"][4]["private_attribute_input_cache"]["length_unit"] == {
        "value": 1,
        "units": "m",
    }
    assert dict_to_convert["models"][6]["private_attribute_input_cache"]["length_unit"] == {
        "value": 1,
        "units": "ft",
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


def test_sanitize_stack_trace():
    """Test that _sanitize_stack_trace properly sanitizes file paths and removes traceback prefix."""
    from flow360.component.simulation.services import _sanitize_stack_trace

    # Test case 1: Full stack trace with traceback prefix and absolute paths
    input_stack = """Traceback (most recent call last):
  File "/disk2/ben/Flow360-R2/flow360/component/simulation/services.py", line 553, in validate_model
    validation_info = ParamsValidationInfo(
  File "/disk2/ben/Flow360-R2/flow360/component/simulation/validation/validation_context.py", line 437, in __init__
    self.farfield_method = self._get_farfield_method_(param_as_dict=param_as_dict)
  File "/disk2/ben/Flow360-R2/flow360/component/simulation/validation/validation_context.py", line 162, in _get_farfield_method_
    if meshing["type_name"] == "MeshingParams":
KeyError: 'type_name'"""

    expected_output = """File "flow360/component/simulation/services.py", line 553, in validate_model
    validation_info = ParamsValidationInfo(
  File "flow360/component/simulation/validation/validation_context.py", line 437, in __init__
    self.farfield_method = self._get_farfield_method_(param_as_dict=param_as_dict)
  File "flow360/component/simulation/validation/validation_context.py", line 162, in _get_farfield_method_
    if meshing["type_name"] == "MeshingParams":
KeyError: 'type_name'"""

    result = _sanitize_stack_trace(input_stack)
    assert result == expected_output

    # Test case 2: Stack trace without traceback prefix (already sanitized prefix)
    input_stack_no_prefix = """File "/home/user/projects/flow360/component/simulation/services.py", line 100, in some_function
    some_code()"""

    expected_no_prefix = """File "flow360/component/simulation/services.py", line 100, in some_function
    some_code()"""

    result_no_prefix = _sanitize_stack_trace(input_stack_no_prefix)
    assert result_no_prefix == expected_no_prefix

    # Test case 3: Stack trace with non-flow360 paths should remain unchanged for those paths
    input_mixed = """Traceback (most recent call last):
  File "/usr/lib/python3.10/site-packages/pydantic/main.py", line 100, in validate
    return cls.model_validate(obj)
  File "/disk2/ben/Flow360-R2/flow360/component/simulation/services.py", line 50, in my_func
    do_something()"""

    expected_mixed = """File "/usr/lib/python3.10/site-packages/pydantic/main.py", line 100, in validate
    return cls.model_validate(obj)
  File "flow360/component/simulation/services.py", line 50, in my_func
    do_something()"""

    result_mixed = _sanitize_stack_trace(input_mixed)
    assert result_mixed == expected_mixed

    # Test case 4: Empty string should return empty string
    assert _sanitize_stack_trace("") == ""

    # Test case 5: String with no file paths should remain unchanged (except traceback prefix)
    input_no_paths = "Some error message without file paths"
    assert _sanitize_stack_trace(input_no_paths) == input_no_paths

    # Test case 6: Windows-style paths
    input_windows = """Traceback (most recent call last):
  File "C:\\Users\\dev\\Flow360-R2\\flow360\\component\\simulation\\services.py", line 100, in func
    code()"""

    expected_windows = """File "flow360\\component\\simulation\\services.py", line 100, in func
    code()"""

    result_windows = _sanitize_stack_trace(input_windows)
    assert result_windows == expected_windows

@pytest.mark.parametrize("unit_system_name", ["SI", "Imperial", "CGS"])
def test_validate_model_preserves_unit_system(unit_system_name):
    """validate_model must not mutate the unit_system entry in the input dict."""
    with open("data/simulation.json", "r") as fp:
        params_data = json.load(fp)

    # Convert to the target unit system so all values carry matching units
    services.change_unit_system(data=params_data, target_unit_system=unit_system_name)
    unit_system_before = copy.deepcopy(params_data["unit_system"])

    validated_param, errors, _ = services.validate_model(
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    assert params_data["unit_system"] == unit_system_before
    if validated_param is not None:
        assert validated_param.unit_system.name == unit_system_name
