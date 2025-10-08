import copy
import json
import os
import re
from enum import Enum

import pytest
import toml

from flow360.component.simulation.framework.updater import (
    VERSION_MILESTONES,
    _find_update_path,
    updater,
)
from flow360.component.simulation.framework.updater_utils import Flow360Version
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.validation.validation_context import ALL
from flow360.version import __version__


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_version_consistency():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    pyproject_path = os.path.join(project_root, "pyproject.toml")

    # Load the pyproject.toml file
    with open(pyproject_path, "r") as f:
        config = toml.load(f)

    # Extract the version value from the pyproject.toml under [tool.poetry]
    pyproject_version = config["project"]["version"]

    # Assert the version in pyproject.toml matches the internal __version__
    assert pyproject_version == "v" + __version__, (
        f"Version mismatch: pyproject.toml version is {pyproject_version}, "
        f"but __version__ is {__version__}"
    )


def test_version_greater_than_highest_updater_version():
    current_python_version = Flow360Version(__version__)
    assert (
        current_python_version >= VERSION_MILESTONES[-1][0]
    ), "Highest version updater can handle is higher than Python client version. This is not allowed."


def test_milestone_ordering():
    for index in range(len(VERSION_MILESTONES) - 1):
        assert VERSION_MILESTONES[index][0] < VERSION_MILESTONES[index + 1][0]


def test_updater_completeness():
    class DummyUpdaters(Enum):
        to_1 = "to_1"
        to_3 = "to_3"
        to_5 = "to_5"

    version_milestones = [
        (Flow360Version("99.11.1"), DummyUpdaters.to_1),
        (Flow360Version("99.11.3"), DummyUpdaters.to_3),
    ]

    with pytest.raises(
        ValueError,
        match=r"Trying to update `SimulationParams` to a version lower than any known version.",
    ):
        # 1) from <99.11.1, to <99.11.1 => ValueError
        _find_update_path(
            version_from=Flow360Version("99.10.9"),
            version_to=Flow360Version("99.10.10"),
            version_milestones=version_milestones,
        )

    # 2) from <99.11.1, to ==99.11.1 => crosses milestone => [to_1]
    res = _find_update_path(
        version_from=Flow360Version("99.10.9"),
        version_to=Flow360Version("99.11.1"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_1], "Case 2: crosses only 99.11.1 => [to_1]"

    # 3) from <99.11.1, to in-between (e.g., 99.11.2) => crosses 99.11.1 => [to_1]
    res = _find_update_path(
        version_from=Flow360Version("99.10.9"),
        version_to=Flow360Version("99.11.2"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_1], "Case 3: crosses only 99.11.1 => [to_1]"

    # 4) from <99.11.1, to ==99.11.3 => crosses 99.11.1 and 99.11.3 => [to_1, to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.10.9"),
        version_to=Flow360Version("99.11.3"),
        version_milestones=version_milestones,
    )
    assert res == [
        DummyUpdaters.to_1,
        DummyUpdaters.to_3,
    ], "Case 4: crosses 99.11.1, 99.11.3 => [to_1, to_3]"

    # 5) from <99.11.1, to >99.11.3 => crosses 99.11.1 and 99.11.3 => [to_1, to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.10.9"),
        version_to=Flow360Version("99.12.0"),
        version_milestones=version_milestones,
    )
    assert res == [
        DummyUpdaters.to_1,
        DummyUpdaters.to_3,
    ], "Case 5: crosses 99.11.1, 99.11.3 => [to_1, to_3]"

    # 6) from ==99.11.1, to ==99.11.1 => no updates
    res = _find_update_path(
        version_from=Flow360Version("99.11.1"),
        version_to=Flow360Version("99.11.1"),
        version_milestones=version_milestones,
    )
    assert res == [], "Case 6: same version => no updates"

    # 7) from ==99.11.1, to in-between (99.11.2) => no milestone crossed => []
    res = _find_update_path(
        version_from=Flow360Version("99.11.1"),
        version_to=Flow360Version("99.11.2"),
        version_milestones=version_milestones,
    )
    assert res == [], "Case 7: crosses nothing => no updates"

    # 8) from ==99.11.1, to ==99.11.3 => crosses milestone => [to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.11.1"),
        version_to=Flow360Version("99.11.3"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_3], "Case 8: crosses 99.11.3 => [to_3]"

    # 8.1) from ==99.11.1, to >99.11.3 => crosses milestone => [to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.11.1"),
        version_to=Flow360Version("99.11.4"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_3], "Case 8.1: crosses 99.11.3 => [to_3]"

    # 9) from in-between (99.11.2), to ==99.11.3 => crosses milestone => [to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.11.2"),
        version_to=Flow360Version("99.11.3"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_3], "Case 9: crosses 99.11.3 => [to_3]"

    # 10) from in-between (99.11.2), to >99.11.3 => crosses milestone => [to_3]
    res = _find_update_path(
        version_from=Flow360Version("99.11.2"),
        version_to=Flow360Version("99.11.4"),
        version_milestones=version_milestones,
    )
    assert res == [DummyUpdaters.to_3], "Case 10: crosses 99.11.3 => [to_3]"

    # 11) from ==99.11.3, to >99.11.3 => crosses nothing => []
    res = _find_update_path(
        version_from=Flow360Version("99.11.3"),
        version_to=Flow360Version("99.11.4"),
        version_milestones=version_milestones,
    )
    assert res == [], "Case 11: crosses nothing => []"

    # 12) from >99.11.3, to >99.11.3 => forward compatability mode
    res = _find_update_path(
        version_from=Flow360Version("99.11.4"),
        version_to=Flow360Version("99.11.5"),
        version_milestones=version_milestones,
    )
    assert res == []

    # 13) to < from => forward compatability mode

    res = _find_update_path(
        version_from=Flow360Version("99.11.3"),
        version_to=Flow360Version("99.11.2"),
        version_milestones=version_milestones,
    )
    assert res == []

    # 14) [more than 2 versions] to > max version
    version_milestones = [
        (Flow360Version("99.11.1"), DummyUpdaters.to_1),
        (Flow360Version("99.11.3"), DummyUpdaters.to_3),
        (Flow360Version("99.11.5"), DummyUpdaters.to_5),
    ]

    res = _find_update_path(
        version_from=Flow360Version("99.11.0"),
        version_to=Flow360Version("99.11.8"),
        version_milestones=version_milestones,
    )
    assert res == [
        DummyUpdaters.to_1,
        DummyUpdaters.to_3,
        DummyUpdaters.to_5,
    ], "Case 14: crosses all 3"

    # 15) [more than 2 versions] to == second max version
    res = _find_update_path(
        version_from=Flow360Version("99.11.0"),
        version_to=Flow360Version("99.11.3"),
        version_milestones=version_milestones,
    )
    assert res == [
        DummyUpdaters.to_1,
        DummyUpdaters.to_3,
    ], "Case 15: crosses first 2"


def test_updater_to_24_11_1():

    with open("../data/simulation/simulation_pre_24_11_1.json", "r") as fp:
        params_pre_24_11_1 = json.load(fp)

    params_24_11_1 = updater(
        version_from="24.11.0", version_to="24.11.1", params_as_dict=params_pre_24_11_1
    )

    assert params_24_11_1.get("meshing") is None

    for model in params_24_11_1["models"]:
        if model["type"] == "Wall":
            assert model["heat_spec"] == {
                "type_name": "HeatFlux",
                "value": {"value": 0, "units": "W / m**2"},
            }

    assert params_24_11_1["time_stepping"].get("order_of_accuracy") is None

    with open("../data/simulation/simulation_pre_24_11_1_symmetry.json", "r") as fp:
        params_pre_24_11_1_symmetry = json.load(fp)

    params_pre_24_11_1_symmetry = updater(
        version_from="24.11.0", version_to="24.11.1", params_as_dict=params_pre_24_11_1_symmetry
    )

    updated_surface_1 = params_pre_24_11_1_symmetry["private_attribute_asset_cache"][
        "project_entity_info"
    ]["ghost_entities"][1]
    updated_surface_2 = params_pre_24_11_1_symmetry["private_attribute_asset_cache"][
        "project_entity_info"
    ]["ghost_entities"][2]

    assert updated_surface_1["name"] == "symmetric-1"
    assert updated_surface_2["name"] == "symmetric-2"


def test_updater_to_24_11_7():

    with open("../data/simulation/simulation_pre_24_11_7.json", "r") as fp:
        params_pre_24_11_7 = json.load(fp)

    params_24_11_7 = updater(
        version_from="24.11.6", version_to="24.11.7", params_as_dict=params_pre_24_11_7
    )

    assert params_24_11_7["outputs"][0]["entities"]["stored_entities"][0]["private_attribute_id"]
    assert params_24_11_7["outputs"][1]["entities"]["stored_entities"][0]["private_attribute_id"]
    assert params_24_11_7["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][
        0
    ]["private_attribute_id"]
    assert params_24_11_7["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][
        1
    ]["private_attribute_id"]

    assert (
        params_24_11_7["outputs"][1]["entities"]["stored_entities"][0]["private_attribute_id"]
        == params_24_11_7["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][
            0
        ]["private_attribute_id"]
    )
    assert (
        params_24_11_7["outputs"][0]["entities"]["stored_entities"][0]["private_attribute_id"]
        == params_24_11_7["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][
            1
        ]["private_attribute_id"]
    )

    with open("../data/simulation/simulation_pre_24_11_1_symmetry.json", "r") as fp:
        params_pre_24_11_1_symmetry = json.load(fp)

    params_pre_24_11_1_symmetry = updater(
        version_from="24.11.6", version_to="24.11.7", params_as_dict=params_pre_24_11_1_symmetry
    )

    updated_surface_1 = params_pre_24_11_1_symmetry["private_attribute_asset_cache"][
        "project_entity_info"
    ]["ghost_entities"][1]
    updated_surface_2 = params_pre_24_11_1_symmetry["private_attribute_asset_cache"][
        "project_entity_info"
    ]["ghost_entities"][2]

    assert updated_surface_1["name"] == "symmetric-1"
    assert updated_surface_2["name"] == "symmetric-2"


def test_updater_to_25_2_0():
    with open("../data/simulation/simulation_pre_25_2_0.json", "r") as fp:
        params = json.load(fp)

    params_pre_25_2_0 = copy.deepcopy(params)
    params_new = updater(
        version_from=f"24.11.8",
        version_to=f"25.2.0",
        params_as_dict=params,
    )

    for idx, model in enumerate(params_pre_25_2_0["models"]):
        if model["type"] == "Fluid":
            assert params_new["models"][idx]["turbulence_model_solver"]["hybrid_model"] == {
                "shielding_function": "DDES",
                "grid_size_for_LES": model["turbulence_model_solver"]["grid_size_for_LES"],
            }
        if model["type"] == "Inflow":
            assert (
                params_new["models"][idx]["spec"]["velocity_direction"]
                == params_pre_25_2_0["models"][idx]["velocity_direction"]
            )
        if model["type"] == "Outflow":
            assert params_new["models"][idx]["spec"]["ramp_steps"] == None

    for idx_output, output in enumerate(params_pre_25_2_0["outputs"]):
        if output["output_type"] == "VolumeOutput":
            for idx_field, field in enumerate(output["output_fields"]):
                if field == "SpalartAllmaras_DDES":
                    assert (
                        params_new["outputs"][idx_output]["output_fields"][idx_field]
                        == "SpalartAllmaras_hybridModel"
                    )
                if field == "kOmegaSST_DDES":
                    assert (
                        params_new["outputs"][idx_output]["output_fields"][idx_field]
                        == "kOmegaSST_hybridModel"
                    )

        if output["output_type"] == "AeroAcousticOutput":
            for idx_observer, observer in enumerate(output["observers"]):
                assert (
                    params_new["outputs"][idx_output]["observers"][idx_observer]["position"]
                    == observer
                )
                assert (
                    params_new["outputs"][idx_output]["observers"][idx_observer]["group_name"]
                    == "0"
                )
                assert (
                    params_new["outputs"][idx_output]["observers"][idx_observer][
                        "private_attribute_expand"
                    ]
                    is None
                )


def test_updater_to_24_11_10():
    with open("../data/simulation/simulation_24_11_9.json", "r") as fp:
        params = json.load(fp)

    params_new = updater(
        version_from=f"24.11.9",
        version_to=f"24.11.10",
        params_as_dict=params,
    )
    updated_ghost_sphere = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ][0]
    assert updated_ghost_sphere["private_attribute_entity_type_name"] == "GhostSphere"
    assert "type_name" not in updated_ghost_sphere
    assert updated_ghost_sphere["center"] == [5.0007498695, 0, 0]
    assert updated_ghost_sphere["max_radius"] == 504.16453591327473


def test_updater_to_25_2_1():
    with open("../data/simulation/simulation_pre_25_2_1.json", "r") as fp:
        params = json.load(fp)

    params_new = updater(
        version_from=f"25.2.0",
        version_to=f"25.2.1",
        params_as_dict=params,
    )
    updated_ghost_sphere = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ][0]
    assert updated_ghost_sphere["private_attribute_entity_type_name"] == "GhostSphere"
    assert "type_name" not in updated_ghost_sphere
    assert updated_ghost_sphere["center"] == [0, 0, 0]
    assert updated_ghost_sphere["max_radius"] == 5.000000000000003


def test_updater_to_25_2_3():
    with open("../data/simulation/simulation_pre_25_2_3_geo.json", "r") as fp:
        params = json.load(fp)

    params_new = updater(
        version_from=f"25.2.2",
        version_to=f"25.2.3",
        params_as_dict=params,
    )
    updated_edge = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "grouped_edges"
    ][0][0]
    updated_face = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "grouped_faces"
    ][0][0]
    updated_ghost_entity = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ][1]
    updated_draft_entity = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "draft_entities"
    ][0]

    updated_output_surface_entity = params_new["outputs"][1]["entities"]["stored_entities"][1]
    updated_model_freestream_entity = params["models"][1]["entities"]["stored_entities"][0]

    assert updated_edge["private_attribute_id"] == updated_edge["name"] == "wingtrailingEdge"
    assert updated_face["private_attribute_id"] == updated_face["name"] == "wingTrailing"
    assert (
        updated_ghost_entity["private_attribute_id"]
        == updated_ghost_entity["name"]
        == "symmetric-1"
    )
    assert (
        updated_output_surface_entity["private_attribute_id"]
        == updated_output_surface_entity["name"]
        == "wing"
    )
    assert (
        updated_model_freestream_entity["private_attribute_id"]
        == updated_model_freestream_entity["name"]
        == "farfield"
    )
    assert updated_draft_entity["private_attribute_id"] != updated_draft_entity["name"]

    with open("../data/simulation/simulation_pre_25_2_3_volume_zones.json", "r") as fp:
        params = json.load(fp)

    params_new = updater(
        version_from=f"25.2.1",
        version_to=f"25.2.3",
        params_as_dict=params,
    )

    updated_zone = params_new["private_attribute_asset_cache"]["project_entity_info"]["zones"][-1]
    updated_boundary = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "boundaries"
    ][0]
    updated_model_rotation_entity = params["models"][1]["entities"]["stored_entities"][0]

    assert updated_zone["private_attribute_id"] == updated_zone["name"] == "stationaryField"
    assert (
        updated_boundary["private_attribute_id"]
        == updated_boundary["name"]
        == "rotationField/blade"
    )
    assert (
        updated_model_rotation_entity["private_attribute_id"]
        == updated_model_rotation_entity["name"]
        == "rotationField"
    )


def test_updater_to_25_4_1():
    with open("../data/simulation/simulation_pre_25_4_1.json", "r") as fp:
        params = json.load(fp)

    geometry_relative_accuracy = params["meshing"]["defaults"]["geometry_relative_accuracy"]

    params_new = updater(
        version_from=f"25.4.0",
        version_to=f"25.4.1",
        params_as_dict=params,
    )

    assert (
        params_new["meshing"]["defaults"]["geometry_accuracy"]["value"]
        == geometry_relative_accuracy
    )
    assert params_new["meshing"]["defaults"]["geometry_accuracy"]["units"] == "m"


def test_updater_to_25_6_2():
    with open("../data/simulation/simulation_pre_25_6_0.json", "r") as fp:
        params = json.load(fp)

    def _update_to_25_6_2(pre_update_param_as_dict, version_from):
        params_new = updater(
            version_from=version_from,
            version_to=f"25.6.2",
            params_as_dict=pre_update_param_as_dict,
        )
        return params_new

    def _ensure_validity(params):
        params_new, _, _ = validate_model(
            params_as_dict=copy.deepcopy(params),
            validated_by=ValidationCalledBy.LOCAL,
            root_item_type="VolumeMesh",
        )
        assert params_new

    pre_update_param_as_dict = copy.deepcopy(params)
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    assert params_new["models"][2]["velocity_direction"] == [0, -1, 0]
    assert "velocity_direction" not in params_new["models"][2]["spec"]
    _ensure_validity(params_new)
    assert params_new["outputs"] == pre_update_param_as_dict["outputs"]

    pre_update_param_as_dict = copy.deepcopy(params)
    pre_update_param_as_dict["models"][2]["spec"]["velocity_direction"] = None
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    assert "velocity_direction" not in params_new["models"][2]
    assert "velocity_direction" not in params_new["models"][2]["spec"]
    _ensure_validity(params_new)

    pre_update_param_as_dict = copy.deepcopy(params)
    pre_update_param_as_dict["models"][2]["spec"].pop("velocity_direction")
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    assert "velocity_direction" not in params_new["models"][2]
    assert "velocity_direction" not in params_new["models"][2]["spec"]
    _ensure_validity(params_new)

    pre_update_param_as_dict = copy.deepcopy(params)
    pre_update_param_as_dict["models"][2]["spec"].pop("velocity_direction")
    pre_update_param_as_dict["models"][2]["velocity_direction"] = [0, -1, 0]
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    assert params_new["models"][2]["velocity_direction"] == [0, -1, 0]
    assert "velocity_direction" not in params_new["models"][2]["spec"]
    _ensure_validity(params_new)

    pre_update_param_as_dict = copy.deepcopy(params)
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    reynolds = params["operating_condition"]["private_attribute_input_cache"]["reynolds"]
    assert "reynolds" not in params_new["operating_condition"]["private_attribute_input_cache"]
    assert (
        "reynolds_mesh_unit" in params_new["operating_condition"]["private_attribute_input_cache"]
    )
    assert (
        params_new["operating_condition"]["private_attribute_input_cache"]["reynolds_mesh_unit"]
        == reynolds
    )
    _ensure_validity(params_new)

    # Ensure the updater can handle reynolds with None value correctly
    pre_update_param_as_dict = copy.deepcopy(params)
    pre_update_param_as_dict["operating_condition"]["private_attribute_input_cache"][
        "reynolds"
    ] = None
    params_new = _update_to_25_6_2(pre_update_param_as_dict, version_from="25.5.1")
    assert (
        "reynolds_mesh_unit"
        not in params_new["operating_condition"]["private_attribute_input_cache"].keys()
    )

    with open("../data/simulation/simulation_pre_25_6_0-2.json", "r") as fp:
        params = json.load(fp)
    params_new = _update_to_25_6_2(params, version_from="25.5.0")
    _ensure_validity(params_new)

    assert len(params_new["outputs"]) == 5
    assert params_new["outputs"] == [
        {
            "frequency": 16,
            "frequency_offset": 0,
            "name": "Volume output",
            "output_fields": {
                "items": [
                    "vorticity",
                    "primitiveVars",
                    "residualNavierStokes",
                    "Mach",
                    "qcriterion",
                    "T",
                    "Cp",
                    "mut",
                ]
            },
            "output_format": "paraview",
            "output_type": "VolumeOutput",
        },
        {
            "entities": {
                "stored_entities": [
                    {
                        "name": "middle/middle_bottom",
                        "private_attribute_color": None,
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_full_name": "middle/middle_bottom",
                        "private_attribute_id": "middle/middle_bottom",
                        "private_attribute_is_interface": False,
                        "private_attribute_potential_issues": [],
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_sub_components": [],
                        "private_attribute_tag_key": None,
                    },
                    {
                        "name": "static/static_bottom",
                        "private_attribute_color": None,
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_full_name": "static/static_bottom",
                        "private_attribute_id": "static/static_bottom",
                        "private_attribute_is_interface": False,
                        "private_attribute_potential_issues": [],
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_sub_components": [],
                        "private_attribute_tag_key": None,
                    },
                ]
            },
            "frequency": -1,
            "frequency_offset": 0,
            "name": "Surface output",
            "output_fields": {"items": ["Cp"]},
            "output_format": "paraview",
            "output_type": "SurfaceOutput",
            "write_single_file": False,
        },
        {
            "frequency": -1,
            "frequency_offset": 0,
            "name": "Surface output",
            "output_fields": {"items": ["Cf", "Cp", "primitiveVars"]},
            "output_format": "paraview",
            "output_type": "SurfaceOutput",
            "write_single_file": False,
            "entities": {
                "stored_entities": [
                    {
                        "name": "middle/middle_top",
                        "private_attribute_color": None,
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_full_name": "middle/middle_top",
                        "private_attribute_id": "middle/middle_top",
                        "private_attribute_is_interface": False,
                        "private_attribute_potential_issues": [],
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_sub_components": [],
                        "private_attribute_tag_key": None,
                    }
                ]
            },
        },
        {
            "frequency": -1,
            "frequency_offset": 0,
            "name": "Surface output",
            "output_fields": {"items": ["Cf", "Cp", "primitiveVars"]},
            "output_format": "paraview",
            "output_type": "SurfaceOutput",
            "write_single_file": False,
            "entities": {
                "stored_entities": [
                    {
                        "name": "static/static_top",
                        "private_attribute_color": None,
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_full_name": "static/static_top",
                        "private_attribute_id": "static/static_top",
                        "private_attribute_is_interface": False,
                        "private_attribute_potential_issues": [],
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_sub_components": [],
                        "private_attribute_tag_key": None,
                    }
                ]
            },
        },
        {
            "frequency": -1,
            "frequency_offset": 0,
            "name": "Surface output",
            "output_fields": {"items": ["Cf", "Cp", "primitiveVars"]},
            "output_format": "paraview",
            "output_type": "SurfaceOutput",
            "write_single_file": False,
            "entities": {
                "stored_entities": [
                    {
                        "name": "inner/cylinder",
                        "private_attribute_color": None,
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_full_name": "inner/cylinder",
                        "private_attribute_id": "inner/cylinder",
                        "private_attribute_is_interface": False,
                        "private_attribute_potential_issues": [],
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_sub_components": [],
                        "private_attribute_tag_key": None,
                    }
                ]
            },
        },
    ]


def test_deserialization_with_updater():
    # From 24.11.0 to 25.2.0
    with open("../data/simulation/simulation_24_11_0.json", "r") as fp:
        params = json.load(fp)
    validate_model(
        params_as_dict=params,
        root_item_type="VolumeMesh",
        validated_by=ValidationCalledBy.LOCAL,
        validation_level=ALL,
    )


def test_updater_to_25_6_4():
    with open("../data/simulation/simulation_pre_25_4_1.json", "r") as fp:
        params_as_dict = json.load(fp)

    params_new = updater(
        version_from="25.4.0b1",
        version_to=f"25.6.4",
        params_as_dict=params_as_dict,
    )
    assert params_new["meshing"]["defaults"]["planar_face_tolerance"] == 1e-6
    params_new, _, _ = validate_model(
        params_as_dict=params_new,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    assert params_new


def test_updater_to_25_6_5():
    with open("../data/simulation/simulation_pre_25_4_1.json", "r") as fp:
        params_as_dict = json.load(fp)

    params_new = updater(
        version_from="25.4.0b1",
        version_to=f"25.6.5",
        params_as_dict=params_as_dict,
    )
    assert params_new["meshing"]["defaults"]["planar_face_tolerance"] == 1e-6
    params_new, _, _ = validate_model(
        params_as_dict=params_new,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
    )
    assert params_new


def test_updater_to_25_7_2():
    with open("../data/simulation/simulation_pre_25_7_2.json", "r") as fp:
        params_as_dict = json.load(fp)

    params_new = updater(
        version_from="25.7.1",
        version_to=f"25.7.2",
        params_as_dict=params_as_dict,
    )
    assert (
        params_new["private_attribute_asset_cache"]["variable_context"][0]["post_processing"]
        == True
    )
    assert (
        params_new["private_attribute_asset_cache"]["variable_context"][1]["post_processing"]
        == True
    )
    assert (
        params_new["private_attribute_asset_cache"]["variable_context"][2]["post_processing"]
        == False
    )
    assert (
        params_new["private_attribute_asset_cache"]["variable_context"][3]["post_processing"]
        == False
    )
