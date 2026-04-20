import copy
import json
import re
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
from flow360.component.simulation.validation.validation_context import CASE
from flow360.version import __version__


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


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


@pytest.mark.parametrize("unit_system_name", ["SI", "Imperial", "CGS"])
def test_validate_model_preserves_unit_system(unit_system_name):
    """validate_model must not mutate the unit_system entry in the input dict."""
    with open("data/simulation.json", "r") as fp:
        params_data = json.load(fp)

    # Convert to the target unit system so all values carry matching units
    services.change_unit_system(data=params_data, target_unit_system=unit_system_name)
    unit_system_before = copy.deepcopy(params_data["unit_system"])

<<<<<<< HEAD
    validated_param, errors, _ = services.validate_model(
=======
    def check_setting_preserved(
        result_entity_info: GeometryEntityInfo,
        reference_entity_infos: list[GeometryEntityInfo],
        entity_type: str,
        setting_name: str,
    ):
        """
        Check that a specific setting is preserved from reference entity infos.

        Args:
            result_entity_info: The merged entity info to verify
            reference_entity_infos: List of reference entity infos to check against
            entity_type: Either "body" or "face"
            setting_name: The setting to check (e.g., "mesh_exterior", "name")
        """
        if entity_type == "body":
            group_tag = result_entity_info.body_group_tag
            attribute_names = result_entity_info.body_attribute_names
            grouped_entities = result_entity_info.grouped_bodies
        elif entity_type == "face":
            group_tag = result_entity_info.face_group_tag
            attribute_names = result_entity_info.face_attribute_names
            grouped_entities = result_entity_info.grouped_faces
        else:
            raise ValueError(f"Invalid entity_type: {entity_type}")

        group_index_result = attribute_names.index(group_tag)

        for entity in grouped_entities[group_index_result]:
            found = False
            for reference_info in reference_entity_infos:
                if entity_type == "body":
                    ref_attribute_names = reference_info.body_attribute_names
                    ref_grouped_entities = reference_info.grouped_bodies
                else:  # face
                    ref_attribute_names = reference_info.face_attribute_names
                    ref_grouped_entities = reference_info.grouped_faces

                group_index_ref = ref_attribute_names.index(group_tag)

                for ref_entity in ref_grouped_entities[group_index_ref]:
                    if entity.private_attribute_id == ref_entity.private_attribute_id:
                        result_value = getattr(entity, setting_name)
                        ref_value = getattr(ref_entity, setting_name)
                        assert result_value == ref_value, (
                            f"{setting_name} mismatch for {entity_type} "
                            f"'{entity.name}' (id: {entity.private_attribute_id}): "
                            f"expected {ref_value}, got {result_value}"
                        )
                        found = True
                        break
                if found:
                    break
            assert found, (
                f"{entity_type.capitalize()} '{entity.name}' (id: {entity.private_attribute_id}) "
                f"not found in any reference entity info"
            )

    # Load test data
    with open("data/root_geometry_cube_simulation.json", "r") as f:
        root_cube_simulation_dict = json.load(f)
        root_cube_entity_info = GeometryEntityInfo.model_validate(
            root_cube_simulation_dict["private_attribute_asset_cache"]["project_entity_info"]
        )
    with open("data/dependency_geometry_sphere1_simulation.json", "r") as f:
        dependency_sphere1_simulation_dict = json.load(f)
        dependency_sphere1_entity_info = GeometryEntityInfo.model_validate(
            dependency_sphere1_simulation_dict["private_attribute_asset_cache"][
                "project_entity_info"
            ]
        )
    with open("data/dependency_geometry_sphere2_simulation.json", "r") as f:
        dependency_sphere2_simulation_dict = json.load(f)
        dependency_sphere2_entity_info = GeometryEntityInfo.model_validate(
            dependency_sphere2_simulation_dict["private_attribute_asset_cache"][
                "project_entity_info"
            ]
        )

    # Test 1: Merge root geometry and one dependency geometry
    # Should preserve mesh_exterior and name settings of geometryBodyGroup in root geometry
    result_entity_info_dict1 = services.merge_geometry_entity_info(
        draft_param_as_dict=root_cube_simulation_dict,
        geometry_dependencies_param_as_dict=[
            root_cube_simulation_dict,
            dependency_sphere1_simulation_dict,
        ],
    )
    result_entity_info1 = GeometryEntityInfo.model_validate(result_entity_info_dict1)

    # Load expected result for test 1
    with open("data/result_merged_geometry_entity_info1.json", "r") as f:
        expected_result1 = json.load(f)

    # Compare results
    assert compare_values(
        result_entity_info_dict1, expected_result1
    ), "Test 1 failed: Merged entity info does not match expected result"

    # Verify key properties are preserved using helper function
    check_setting_preserved(
        result_entity_info1,
        [root_cube_entity_info, dependency_sphere1_entity_info],
        entity_type="body",
        setting_name="mesh_exterior",
    )
    check_setting_preserved(
        result_entity_info1,
        [root_cube_entity_info, dependency_sphere1_entity_info],
        entity_type="body",
        setting_name="name",
    )
    check_setting_preserved(
        result_entity_info1,
        [root_cube_entity_info, dependency_sphere1_entity_info],
        entity_type="face",
        setting_name="name",
    )

    # Test 2: Start from result of (1), replace the dependency with another dependency geometry
    # Should check:
    #  a. the preservation of mesh_exterior and name settings of geometryBodyGroup in new_draft_param_as_dict
    #  b. the new dependency geometry should have replaced the old dependency geometry
    new_draft_param_as_dict = copy.deepcopy(root_cube_simulation_dict)
    new_draft_param_as_dict["private_attribute_asset_cache"]["project_entity_info"] = copy.deepcopy(
        result_entity_info_dict1
    )

    result_entity_info_dict2 = services.merge_geometry_entity_info(
        draft_param_as_dict=new_draft_param_as_dict,
        geometry_dependencies_param_as_dict=[
            root_cube_simulation_dict,
            dependency_sphere2_simulation_dict,
        ],
    )

    # Load expected result for test 2
    with open("data/result_merged_geometry_entity_info2.json", "r") as f:
        expected_result2 = json.load(f)

    # Compare results
    assert compare_values(
        result_entity_info_dict2, expected_result2
    ), "Test 2 failed: Merged entity info with replaced dependency does not match expected result"

    result_entity_info2 = GeometryEntityInfo.model_validate(result_entity_info_dict2)

    # Verify key properties are preserved using helper function
    check_setting_preserved(
        result_entity_info2,
        [root_cube_entity_info, dependency_sphere2_entity_info],
        entity_type="body",
        setting_name="mesh_exterior",
    )
    check_setting_preserved(
        result_entity_info2,
        [root_cube_entity_info, dependency_sphere2_entity_info],
        entity_type="body",
        setting_name="name",
    )
    check_setting_preserved(
        result_entity_info2,
        [root_cube_entity_info, dependency_sphere2_entity_info],
        entity_type="face",
        setting_name="name",
    )


def test_merge_geometry_entity_info_runs_updater_on_old_version():
    """
    When an old-version simulation params dict is passed to merge_geometry_entity_info,
    SimulationParams._update_param_dict must be invoked on the draft dict AND on each
    dependency dict before the entity info is deserialized. Without this, a schema change
    between the stored version and the current version would surface as a validation
    error in GeometryEntityInfo.model_validate.
    """
    import copy
    from unittest.mock import patch

    from flow360.component.simulation.simulation_params import SimulationParams

    with open("data/root_geometry_cube_simulation.json", "r") as f:
        root_dict = json.load(f)
    with open("data/dependency_geometry_sphere1_simulation.json", "r") as f:
        dep_dict1 = json.load(f)
    with open("data/dependency_geometry_sphere2_simulation.json", "r") as f:
        dep_dict2 = json.load(f)

    # Pre-milestone version so the updater walks every registered migration.
    old_version = "24.11.0"
    old_draft = copy.deepcopy(root_dict)
    old_draft["version"] = old_version
    old_deps = [copy.deepcopy(dep_dict1), copy.deepcopy(dep_dict2)]
    for d in old_deps:
        d["version"] = old_version

    # Part A: spy on _update_param_dict while still executing the real updater.
    # Version is captured at call time because the updater mutates the dict in place.
    real_updater = SimulationParams._update_param_dict
    observed_versions: list = []

    def capturing_updater(model_dict, *args, **kwargs):
        observed_versions.append(model_dict.get("version"))
        return real_updater(model_dict, *args, **kwargs)

    with patch.object(
        SimulationParams,
        "_update_param_dict",
        side_effect=capturing_updater,
    ) as updater_spy:
        result_from_old = services.merge_geometry_entity_info(
            draft_param_as_dict=old_draft,
            geometry_dependencies_param_as_dict=old_deps,
        )

        # One call for the draft + one per dependency.
        assert updater_spy.call_count == 1 + len(
            old_deps
        ), f"Expected {1 + len(old_deps)} updater calls, got {updater_spy.call_count}"
        # Every call entered the updater with the old (pre-migration) version.
        assert observed_versions == [old_version] * (1 + len(old_deps)), observed_versions

    # Part B: migrated input must yield the same merged GeometryEntityInfo as
    # running the function against the current-version fixtures directly.
    result_from_current = services.merge_geometry_entity_info(
        draft_param_as_dict=copy.deepcopy(root_dict),
        geometry_dependencies_param_as_dict=[
            copy.deepcopy(dep_dict1),
            copy.deepcopy(dep_dict2),
        ],
    )

    assert result_from_old["type_name"] == "GeometryEntityInfo"
    assert compare_values(
        result_from_old, result_from_current
    ), "Merged entity info from old-version inputs diverged from current-version baseline"


def test_merge_geometry_entity_info_on_real_old_version_json():
    """
    Regression test using an actual v25.7.7 geometry JSON that carries a
    'transformation' key inside grouped_bodies. That key was removed by the
    `_to_25_8_1` updater migration (see framework/updater.py) and is rejected
    by the current GeometryEntityInfo schema ('Extra inputs are not permitted').

    Without the updater call in merge_geometry_entity_info, deserializing this
    fixture raises pydantic.ValidationError. This test therefore fails loudly
    if the updater step is ever removed or reordered after deserialization.
    """
    import copy

    import pydantic as pd

    with open("data/old_version_geometry_with_transformation.json", "r") as f:
        old_version_dict = json.load(f)

    # Sanity: fixture genuinely exercises an old-schema field that current schema rejects.
    assert old_version_dict["version"] == "25.7.7"
    pre_update_bodies = old_version_dict["private_attribute_asset_cache"]["project_entity_info"][
        "grouped_bodies"
    ]
    assert any(
        "transformation" in body for group in pre_update_bodies for body in group
    ), "Fixture no longer contains the obsolete 'transformation' key"

    # Confirm current schema rejects the pre-migration payload directly.
    with pytest.raises(pd.ValidationError, match="transformation"):
        GeometryEntityInfo.model_validate(
            old_version_dict["private_attribute_asset_cache"]["project_entity_info"]
        )

    # merge_geometry_entity_info must run the updater before deserialization,
    # so the same fixture now round-trips successfully.
    merged = services.merge_geometry_entity_info(
        draft_param_as_dict=copy.deepcopy(old_version_dict),
        geometry_dependencies_param_as_dict=[copy.deepcopy(old_version_dict)],
    )
    assert merged["type_name"] == "GeometryEntityInfo"
    assert not any(
        "transformation" in body for group in merged["grouped_bodies"] for body in group
    ), "Obsolete 'transformation' field leaked into merged output"


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


def test_validate_error_location_with_selector():
    """
    Test that validation error locations are correctly preserved when errors occur
    within EntitySelector's children field.

    This test verifies the fix for the bug where error locations were incorrectly
    reduced to just ("children",) instead of the full path like
    ("models", 0, "entities", "selectors", 0, "children", 0, "value").

    The bug was in _traverse_error_location which used `current.get(field)` instead
    of `field in current`, causing fields with falsy values (empty list, 0, etc.)
    to be incorrectly filtered out.
    """
    params_data = {
        "models": [
            {
                "type": "Wall",
                "name": "Wall with selector",
                "entities": {
                    "stored_entities": [],
                    "selectors": ["test-selector-id"],
                },
                "use_wall_function": False,
            }
        ],
        "unit_system": {"name": "SI"},
        "version": __version__,
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "VolumeMeshEntityInfo",
                "draft_entities": [],
                "zones": [],
                "boundaries": [],
            },
            "used_selectors": [
                {
                    "name": "test_selector",
                    "target_class": "Surface",
                    "logic": "AND",
                    "selector_id": "test-selector-id",
                }
            ],
        },
    }

    _, errors, _ = services.validate_model(
>>>>>>> b0f0eccf ([Hotfix 25.9]: [FXC-8307][25.8] Run updater on inputs in merge_geometry_entity_info (#2005))
        params_as_dict=params_data,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )

    assert params_data["unit_system"] == unit_system_before
    if validated_param is not None:
        assert validated_param.unit_system.name == unit_system_name
