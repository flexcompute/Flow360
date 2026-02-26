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
    _to_25_9_0,
    _to_25_9_1,
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
    pyproject_version = config["tool"]["poetry"]["version"]

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


def test_updater_to_25_7_6_remove_entity_bucket_field():
    # Construct minimal params containing entity dicts with the legacy bucket field
    params_as_dict = {
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "output_fields": {"items": ["Cp"]},
                "entities": {
                    "stored_entities": [
                        {
                            "name": "wing",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "wing",
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        },
                        {
                            "name": "tail",
                            "private_attribute_entity_type_name": "Surface",
                            "private_attribute_id": "tail",
                            "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        },
                    ]
                },
            }
        ],
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "ghost_entities": [
                    {
                        "name": "symmetric-1",
                        "private_attribute_entity_type_name": "GhostCircularPlane",
                        "private_attribute_registry_bucket_name": "GhostEntityType",
                    }
                ],
                "draft_entities": [
                    {
                        "name": "point-array-1",
                        "private_attribute_entity_type_name": "PointArray",
                        "private_attribute_registry_bucket_name": "PointArrayEntityType",
                    }
                ],
            }
        },
        # Non-entity dict should keep the field
        "misc": {"private_attribute_registry_bucket_name": "keep_me"},
    }

    params_new = updater(
        version_from="25.7.4",
        version_to="25.7.6",
        params_as_dict=params_as_dict,
    )

    # Verify removal from all entity dicts
    stored_entities = params_new["outputs"][0]["entities"]["stored_entities"]
    assert all("private_attribute_registry_bucket_name" not in entity for entity in stored_entities)

    ghost_entities = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ]
    assert all("private_attribute_registry_bucket_name" not in entity for entity in ghost_entities)

    draft_entities = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "draft_entities"
    ]
    assert all("private_attribute_registry_bucket_name" not in entity for entity in draft_entities)

    # Non-entity dict remains unchanged
    assert params_new["misc"]["private_attribute_registry_bucket_name"] == "keep_me"


def test_updater_to_25_7_6_rename_rotation_cylinder():
    # Minimal input containing a RotationCylinder in meshing.volume_zones
    params_as_dict = {
        "meshing": {
            "volume_zones": [
                {
                    "type": "RotationCylinder",
                    "name": "rot_zone",
                    "entities": {
                        "stored_entities": [
                            {
                                "private_attribute_entity_type_name": "Cylinder",
                                "name": "c1",
                            }
                        ]
                    },
                    "spacing_axial": {"value": 1.0, "units": "m"},
                    "spacing_radial": {"value": 1.0, "units": "m"},
                    "spacing_circumferential": {"value": 1.0, "units": "m"},
                }
            ]
        },
        "unit_system": {"name": "SI"},
        "version": "25.7.2",
    }

    params_new = updater(version_from="25.7.2", version_to="25.7.6", params_as_dict=params_as_dict)

    assert params_new["version"] == "25.7.6"
    assert params_new["meshing"]["volume_zones"][0]["type"] == "RotationVolume"


def test_updater_to_25_7_7():
    """Test updater for version 25.7.7 which handles:
    1. Resetting frequency/frequency_offset to defaults for steady simulations
    2. Removing transition-specific output fields when transition model is None
    """

    # Construct test params with:
    # - Steady simulation with non-default frequency settings
    # - Transition model set to None with transition-specific output fields
    params_as_dict = {
        "version": "25.7.6",
        "time_stepping": {
            "type_name": "Steady",
            "max_steps": 1000,
        },
        "models": [
            {
                "type": "Fluid",
                "transition_model_solver": {
                    "type_name": "None",
                },
                "turbulence_model_solver": {
                    "type_name": "SpalartAllmaras",
                },
            }
        ],
        "outputs": [
            {
                "output_type": "VolumeOutput",
                "name": "Volume output",
                "frequency": 10,
                "frequency_offset": 5,
                "output_format": "paraview",
                "output_fields": {
                    "items": [
                        "primitiveVars",
                        "residualNavierStokes",
                        "residualTransition",
                        "Mach",
                    ]
                },
            },
            {
                "output_type": "SurfaceOutput",
                "name": "Surface output",
                "frequency": 20,
                "frequency_offset": 10,
                "output_format": "paraview",
                "entities": {"stored_entities": []},
                "output_fields": {
                    "items": [
                        "Cp",
                        "Cf",
                        "solutionTransition",
                    ]
                },
            },
            {
                "output_type": "SliceOutput",
                "name": "Slice output",
                "frequency": 15,
                "frequency_offset": 2,
                "output_format": "paraview",
                "entities": {"stored_entities": []},
                "output_fields": {
                    "items": [
                        "vorticity",
                        "linearResidualTransition",
                    ]
                },
            },
            {
                "output_type": "ProbeOutput",
                "name": "Probe output",
                "entities": {"stored_entities": []},
                "output_fields": {
                    "items": [
                        "primitiveVars",
                        "residualTransition",
                    ]
                },
            },
            {
                "output_type": "AeroAcousticOutput",
                "name": "Aeroacoustic output",
                "observers": [],
            },
        ],
    }

    params_new = updater(
        version_from="25.7.6",
        version_to="25.7.7",
        params_as_dict=params_as_dict,
    )

    # Test 1: Verify frequency settings were reset to defaults for steady simulation
    assert (
        params_new["outputs"][0]["frequency"] == -1
    ), "VolumeOutput frequency should be reset to -1"
    assert (
        params_new["outputs"][0]["frequency_offset"] == 0
    ), "VolumeOutput frequency_offset should be reset to 0"

    assert (
        params_new["outputs"][1]["frequency"] == -1
    ), "SurfaceOutput frequency should be reset to -1"
    assert (
        params_new["outputs"][1]["frequency_offset"] == 0
    ), "SurfaceOutput frequency_offset should be reset to 0"

    assert (
        params_new["outputs"][2]["frequency"] == -1
    ), "SliceOutput frequency should be reset to -1"
    assert (
        params_new["outputs"][2]["frequency_offset"] == 0
    ), "SliceOutput frequency_offset should be reset to 0"

    # Test 2: Verify transition-specific output fields were removed
    volume_output_fields = params_new["outputs"][0]["output_fields"]["items"]
    assert "residualTransition" not in volume_output_fields, "residualTransition should be removed"
    assert "primitiveVars" in volume_output_fields, "primitiveVars should remain"
    assert "residualNavierStokes" in volume_output_fields, "residualNavierStokes should remain"
    assert "Mach" in volume_output_fields, "Mach should remain"

    surface_output_fields = params_new["outputs"][1]["output_fields"]["items"]
    assert "solutionTransition" not in surface_output_fields, "solutionTransition should be removed"
    assert "Cp" in surface_output_fields, "Cp should remain"
    assert "Cf" in surface_output_fields, "Cf should remain"

    slice_output_fields = params_new["outputs"][2]["output_fields"]["items"]
    assert (
        "linearResidualTransition" not in slice_output_fields
    ), "linearResidualTransition should be removed"
    assert "vorticity" in slice_output_fields, "vorticity should remain"

    probe_output_fields = params_new["outputs"][3]["output_fields"]["items"]
    assert (
        "residualTransition" not in probe_output_fields
    ), "residualTransition should be removed from ProbeOutput"
    assert "primitiveVars" in probe_output_fields, "primitiveVars should remain"

    # Test 3: Verify version was updated
    assert params_new["version"] == "25.7.7"


def test_updater_to_25_7_7_unsteady_no_frequency_change():
    """Test that frequency settings are NOT changed for unsteady simulations"""

    params_as_dict = {
        "version": "25.7.6",
        "time_stepping": {
            "type_name": "Unsteady",
            "max_steps": 1000,
        },
        "models": [
            {
                "type": "Fluid",
                "transition_model_solver": {
                    "type_name": "None",
                },
            }
        ],
        "outputs": [
            {
                "output_type": "VolumeOutput",
                "name": "Volume output",
                "frequency": 10,
                "frequency_offset": 5,
                "output_format": "paraview",
                "output_fields": {"items": ["primitiveVars", "Mach"]},
            },
        ],
    }

    params_new = updater(
        version_from="25.7.6",
        version_to="25.7.7",
        params_as_dict=params_as_dict,
    )

    assert params_new["outputs"][0]["frequency"] == 10, "Unsteady frequency should not be changed"
    assert (
        params_new["outputs"][0]["frequency_offset"] == 5
    ), "Unsteady frequency_offset should not be changed"


def test_updater_to_25_7_7_with_transition_model():
    """Test that transition output fields are NOT removed when transition model is enabled"""

    params_as_dict = {
        "version": "25.7.6",
        "time_stepping": {
            "type_name": "Steady",
            "max_steps": 1000,
        },
        "models": [
            {
                "type": "Fluid",
                "transition_model_solver": {
                    "type_name": "AmplificationFactorTransport",
                },
            }
        ],
        "outputs": [
            {
                "output_type": "VolumeOutput",
                "name": "Volume output",
                "frequency": 10,
                "frequency_offset": 5,
                "output_format": "paraview",
                "output_fields": {
                    "items": [
                        "primitiveVars",
                        "residualTransition",
                        "solutionTransition",
                    ]
                },
            },
        ],
    }

    params_new = updater(
        version_from="25.7.6",
        version_to="25.7.7",
        params_as_dict=params_as_dict,
    )

    # Frequency settings should still be reset for steady simulation
    assert params_new["outputs"][0]["frequency"] == -1
    assert params_new["outputs"][0]["frequency_offset"] == 0

    # Transition output fields should NOT be removed when transition model is enabled
    volume_output_fields = params_new["outputs"][0]["output_fields"]["items"]
    assert (
        "residualTransition" in volume_output_fields
    ), "residualTransition should remain with AmplificationFactorTransport"
    assert (
        "solutionTransition" in volume_output_fields
    ), "solutionTransition should remain with AmplificationFactorTransport"
    assert "primitiveVars" in volume_output_fields, "primitiveVars should remain"


def test_updater_to_25_8_0_add_meshing_type_name():
    params_as_dict = {
        "meshing": {
            "refinement_factor": 1,
            "gap_treatment_strength": 0,
            "defaults": {
                "surface_edge_growth_rate": 1.2,
                "surface_max_edge_length": {"value": 0.1, "units": "m"},
                "curvature_resolution_angle": {"value": 14, "units": "degree"},
                "boundary_layer_growth_rate": 1.2,
                "boundary_layer_first_layer_thickness": {"value": 0.05, "units": "m"},
                "planar_face_tolerance": 0.01,
            },
            "volume_zones": [
                {
                    "type": "AutomatedFarfield",
                    "name": "Farfield",
                    "method": "auto",
                    "_id": "0kd7rt12-7c82-0fma-js93-bf7jx7216532",
                }
            ],
            "refinements": [],
        }
    }

    params_new = updater(
        version_from="25.7.6",
        version_to="25.8.0",
        params_as_dict=params_as_dict,
    )

    assert "type_name" in params_new["meshing"]
    assert params_new["meshing"]["type_name"] == "MeshingParams"


def test_updater_to_25_8_1_remove_transformation_key():
    params_as_dict = {
        "version": "25.8.0",
        "unit_system": {"name": "SI"},
        # Top-level key to be removed
        "transformation": {"should_be_removed": True},
        # Nested key to be removed inside a likely real subtree
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "draft_entities": [
                    {
                        "name": "cs-1",
                        "private_attribute_entity_type_name": "CoordinateSystem",
                        "transformation": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                        # Ensure other unrelated keys are not affected
                        "other_key": {"keep_me": True},
                    }
                ]
            }
        },
        # List nesting
        "outputs": [
            {
                "output_type": "VolumeOutput",
                "output_fields": {"items": ["Mach"]},
                "metadata": [{"transformation": 123}],
            }
        ],
    }

    params_new = updater(
        version_from="25.8.0",
        version_to="25.8.1",
        params_as_dict=params_as_dict,
    )

    assert params_new["version"] == "25.8.1"
    assert "transformation" not in params_new
    assert (
        "transformation"
        not in params_new["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][
            0
        ]
    )
    assert (
        params_new["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"][0][
            "other_key"
        ]["keep_me"]
        is True
    )
    assert "transformation" not in params_new["outputs"][0]["metadata"][0]


def test_updater_to_25_8_3_rename_origin_to_reference_point():
    """Test updater for version 25.8.3 which renames 'origin' to 'reference_point' in CoordinateSystem"""

    params_as_dict = {
        "version": "25.8.2",
        "unit_system": {"name": "SI"},
        "private_attribute_asset_cache": {
            "coordinate_system_status": {
                "coordinate_systems": [
                    {
                        "name": "frame1",
                        "type_name": "CoordinateSystem",
                        "origin": {"value": [1.0, 2.0, 3.0], "units": "m"},
                        "axis_of_rotation": [0, 0, 1],
                        "angle_of_rotation": {"value": 90, "units": "degree"},
                        "scale": [1, 1, 1],
                        "translation": {"value": [0, 0, 0], "units": "m"},
                        "private_attribute_id": "cs-1",
                    },
                    {
                        "name": "frame2",
                        "type_name": "CoordinateSystem",
                        "origin": {"value": [5.0, 6.0, 7.0], "units": "m"},
                        "axis_of_rotation": [1, 0, 0],
                        "angle_of_rotation": {"value": 45, "units": "degree"},
                        "scale": [2, 2, 2],
                        "translation": {"value": [1, 1, 1], "units": "m"},
                        "private_attribute_id": "cs-2",
                    },
                ]
            }
        },
    }

    params_new = updater(
        version_from="25.8.2",
        version_to="25.8.3",
        params_as_dict=params_as_dict,
    )

    assert params_new["version"] == "25.8.3"

    # Verify 'origin' is renamed to 'reference_point' in all coordinate systems
    coord_systems = params_new["private_attribute_asset_cache"]["coordinate_system_status"][
        "coordinate_systems"
    ]

    assert len(coord_systems) == 2

    # Check first coordinate system
    assert "origin" not in coord_systems[0], "'origin' should be removed from frame1"
    assert "reference_point" in coord_systems[0], "'reference_point' should exist in frame1"
    assert coord_systems[0]["reference_point"] == {
        "value": [1.0, 2.0, 3.0],
        "units": "m",
    }, "reference_point should have the old origin value"

    # Check second coordinate system
    assert "origin" not in coord_systems[1], "'origin' should be removed from frame2"
    assert "reference_point" in coord_systems[1], "'reference_point' should exist in frame2"
    assert coord_systems[1]["reference_point"] == {
        "value": [5.0, 6.0, 7.0],
        "units": "m",
    }, "reference_point should have the old origin value"

    # Verify other fields remain unchanged
    assert coord_systems[0]["name"] == "frame1"
    assert coord_systems[0]["axis_of_rotation"] == [0, 0, 1]
    assert coord_systems[1]["name"] == "frame2"
    assert coord_systems[1]["scale"] == [2, 2, 2]


def test_updater_to_25_8_3_no_coordinate_systems():
    """Test updater handles cases where coordinate_system_status is missing or empty"""

    # Case 1: No asset_cache
    params_as_dict_1 = {
        "version": "25.8.2",
        "unit_system": {"name": "SI"},
    }

    params_new_1 = updater(
        version_from="25.8.2",
        version_to="25.8.3",
        params_as_dict=params_as_dict_1,
    )

    assert params_new_1["version"] == "25.8.3"
    assert "private_attribute_asset_cache" not in params_new_1

    # Case 2: No coordinate_system_status
    params_as_dict_2 = {
        "version": "25.8.2",
        "unit_system": {"name": "SI"},
        "private_attribute_asset_cache": {},
    }

    params_new_2 = updater(
        version_from="25.8.2",
        version_to="25.8.3",
        params_as_dict=params_as_dict_2,
    )

    assert params_new_2["version"] == "25.8.3"
    assert params_new_2["private_attribute_asset_cache"] == {}

    # Case 3: Empty coordinate_systems list
    params_as_dict_3 = {
        "version": "25.8.2",
        "unit_system": {"name": "SI"},
        "private_attribute_asset_cache": {"coordinate_system_status": {"coordinate_systems": []}},
    }

    params_new_3 = updater(
        version_from="25.8.2",
        version_to="25.8.3",
        params_as_dict=params_as_dict_3,
    )

    assert params_new_3["version"] == "25.8.3"
    assert (
        params_new_3["private_attribute_asset_cache"]["coordinate_system_status"][
            "coordinate_systems"
        ]
        == []
    )


def test_updater_to_25_8_4_add_wind_tunnel_ghost_surfaces():
    """Ensures ghost_entities is populated with wind tunnel ghost surfaces"""

    # from translator/data/simulation_with_auto_area.json
    params_as_dict = {
        "version": "25.6.6",
        "unit_system": {"name": "CGS"},
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "ghost_entities": [
                    {
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_entity_type_name": "GhostSphere",
                        "private_attribute_id": "farfield",
                        "name": "farfield",
                        "private_attribute_full_name": None,
                        "center": [11, 6, 5],
                        "max_radius": 1100.0000000000005,
                    },
                    {
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_entity_type_name": "GhostCircularPlane",
                        "private_attribute_id": "symmetric-1",
                        "name": "symmetric-1",
                        "private_attribute_full_name": None,
                        "center": [11, 0, 5],
                        "max_radius": 22.00000000000001,
                        "normal_axis": [0, 1, 0],
                    },
                    {
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_entity_type_name": "GhostCircularPlane",
                        "private_attribute_id": "symmetric-2",
                        "name": "symmetric-2",
                        "private_attribute_full_name": None,
                        "center": [11, 12, 5],
                        "max_radius": 22.00000000000001,
                        "normal_axis": [0, 1, 0],
                    },
                    {
                        "private_attribute_registry_bucket_name": "SurfaceEntityType",
                        "private_attribute_entity_type_name": "GhostCircularPlane",
                        "private_attribute_id": "symmetric",
                        "name": "symmetric",
                        "private_attribute_full_name": None,
                        "center": [11, 0, 5],
                        "max_radius": 22.00000000000001,
                        "normal_axis": [0, 1, 0],
                    },
                ],
            }
        },
    }

    # Verify no WindTunnelGhostSurface currently exists
    ghost_entities_before = params_as_dict["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ]
    assert not any(
        e.get("private_attribute_entity_type_name") == "WindTunnelGhostSurface"
        for e in ghost_entities_before
    )

    # Update
    params_new = updater(
        version_from="25.6.6",
        version_to="25.8.4",
        params_as_dict=params_as_dict,
    )
    assert params_new["version"] == "25.8.4"

    ghost_entities = params_new["private_attribute_asset_cache"]["project_entity_info"][
        "ghost_entities"
    ]

    # Should still have original ghost entities (GhostSphere, GhostCircularPlane)
    assert any(e["name"] == "farfield" for e in ghost_entities)
    assert any(e["name"] == "symmetric" for e in ghost_entities)

    # Should now have all 10 wind tunnel ghost surfaces
    wind_tunnel_names = [
        "windTunnelInlet",
        "windTunnelOutlet",
        "windTunnelCeiling",
        "windTunnelFloor",
        "windTunnelLeft",
        "windTunnelRight",
        "windTunnelFrictionPatch",
        "windTunnelCentralBelt",
        "windTunnelFrontWheelBelt",
        "windTunnelRearWheelBelt",
    ]
    for name in wind_tunnel_names:
        assert any(
            e.get("private_attribute_entity_type_name") == "WindTunnelGhostSurface"
            and e["name"] == name
            for e in ghost_entities
        ), f"Missing wind tunnel ghost surface: {name}"


def test_updater_to_25_8_4_fix_write_single_file_paraview():
    """Test updater for version 25.8.4 which fixes write_single_file incompatibility with Paraview format"""

    # Construct test params with write_single_file=True and various output formats
    params_as_dict = {
        "version": "25.8.3",
        "unit_system": {"name": "SI"},
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "Surface output paraview",
                "write_single_file": True,
                "output_format": "paraview",
                "output_fields": {"items": ["Cp"]},
                "entities": {"stored_entities": []},
            },
            {
                "output_type": "TimeAverageSurfaceOutput",
                "name": "Time average surface output paraview",
                "write_single_file": True,
                "output_format": "paraview",
                "output_fields": {"items": ["Cf"]},
                "entities": {"stored_entities": []},
            },
            {
                "output_type": "SurfaceOutput",
                "name": "Surface output both",
                "write_single_file": True,
                "output_format": "both",
                "output_fields": {"items": ["Cp"]},
                "entities": {"stored_entities": []},
            },
            {
                "output_type": "SurfaceOutput",
                "name": "Surface output tecplot",
                "write_single_file": True,
                "output_format": "tecplot",
                "output_fields": {"items": ["Cp"]},
                "entities": {"stored_entities": []},
            },
            {
                "output_type": "VolumeOutput",
                "name": "Volume output",
                "output_format": "paraview",
                "output_fields": {"items": ["Mach"]},
            },
        ],
    }

    params_new = updater(
        version_from="25.8.3",
        version_to="25.8.4",
        params_as_dict=params_as_dict,
    )

    assert params_new["version"] == "25.8.4"

    # Test 1: write_single_file should be reset to False for paraview format
    assert (
        params_new["outputs"][0]["write_single_file"] is False
    ), "SurfaceOutput with paraview should have write_single_file=False"
    assert (
        params_new["outputs"][1]["write_single_file"] is False
    ), "TimeAverageSurfaceOutput with paraview should have write_single_file=False"

    # Test 2: write_single_file should NOT be changed for "both" format (only warning, not error)
    assert (
        params_new["outputs"][2]["write_single_file"] is True
    ), "SurfaceOutput with 'both' format should keep write_single_file=True"

    # Test 3: write_single_file should NOT be changed for tecplot format (valid)
    assert (
        params_new["outputs"][3]["write_single_file"] is True
    ), "SurfaceOutput with tecplot should keep write_single_file=True"

    # Test 4: Non-SurfaceOutput types should not be affected
    assert (
        "write_single_file" not in params_new["outputs"][4]
    ), "VolumeOutput should not be affected"


def test_updater_to_25_8_4_no_outputs():
    """Test updater handles cases where outputs is missing or empty"""

    # Case 1: No outputs
    params_as_dict_1 = {
        "version": "25.8.3",
        "unit_system": {"name": "SI"},
    }

    params_new_1 = updater(
        version_from="25.8.3",
        version_to="25.8.4",
        params_as_dict=params_as_dict_1,
    )

    assert params_new_1["version"] == "25.8.4"
    assert "outputs" not in params_new_1

    # Case 2: Empty outputs list
    params_as_dict_2 = {
        "version": "25.8.3",
        "unit_system": {"name": "SI"},
        "outputs": [],
    }

    params_new_2 = updater(
        version_from="25.8.3",
        version_to="25.8.4",
        params_as_dict=params_as_dict_2,
    )

    assert params_new_2["version"] == "25.8.4"
    assert params_new_2["outputs"] == []


def test_updater_to_25_8_4_write_single_file_false():
    """Test updater doesn't change outputs that already have write_single_file=False"""

    params_as_dict = {
        "version": "25.8.3",
        "unit_system": {"name": "SI"},
        "outputs": [
            {
                "output_type": "SurfaceOutput",
                "name": "Surface output",
                "write_single_file": False,
                "output_format": "paraview",
                "output_fields": {"items": ["Cp"]},
                "entities": {"stored_entities": []},
            },
        ],
    }

    params_new = updater(
        version_from="25.8.3",
        version_to="25.8.4",
        params_as_dict=params_as_dict,
    )

    assert params_new["version"] == "25.8.4"
    assert (
        params_new["outputs"][0]["write_single_file"] is False
    ), "write_single_file should remain False"


def test_updater_to_25_9_0_remove_deprecated_remove_non_manifold_faces():
    """Test 25.9.0 updater step removes deprecated meshing.defaults.remove_non_manifold_faces key."""

    params_as_dict = {
        "version": "25.8.3",
        "unit_system": {"name": "SI"},
        "meshing": {
            "defaults": {
                "surface_max_edge_length": {"value": 0.2, "units": "m"},
                "remove_non_manifold_faces": False,
            }
        },
    }

    params_new = _to_25_9_0(params_as_dict)
    defaults = params_new["meshing"]["defaults"]
    assert "remove_non_manifold_faces" not in defaults


def test_updater_to_25_9_0_convert_use_wall_function_bool():
    """Test 25.9.0 updater converts use_wall_function bool to WallFunction dict or removes it."""

    params_as_dict = {
        "version": "25.8.4",
        "unit_system": {"name": "SI"},
        "models": [
            {"type": "Wall", "use_wall_function": True, "name": "Wall"},
            {"type": "Wall", "use_wall_function": False, "name": "NoSlipWall"},
            {"type": "Wall", "name": "DefaultWall"},
            {"type": "Freestream"},
        ],
    }

    params_new = _to_25_9_0(params_as_dict)
    models = params_new["models"]

    assert models[0]["use_wall_function"] == {"type_name": "BoundaryLayer"}
    assert "use_wall_function" not in models[1]
    assert "use_wall_function" not in models[2]
    assert models[3].get("type") == "Freestream"


def test_updater_to_25_9_1_add_linear_solver_type_name():
    """Test 25.9.1 updater adds type_name to linear_solver inside navier_stokes_solver."""

    params_as_dict = {
        "version": "25.9.0",
        "unit_system": {"name": "SI"},
        "models": [
            {
                "type": "Fluid",
                "navier_stokes_solver": {
                    "absolute_tolerance": 1e-10,
                    "linear_solver": {
                        "max_iterations": 30,
                    },
                },
            },
            {
                "type": "Fluid",
                "navier_stokes_solver": {
                    "absolute_tolerance": 1e-10,
                    "linear_solver": {
                        "type_name": "LinearSolver",
                        "max_iterations": 50,
                    },
                },
            },
            {
                "type": "Fluid",
                "navier_stokes_solver": {
                    "absolute_tolerance": 1e-10,
                },
            },
            {
                "type": "Wall",
                "name": "wall-1",
            },
        ],
    }

    params_new = _to_25_9_1(params_as_dict)
    models = params_new["models"]

    # linear_solver without type_name should get it added
    assert models[0]["navier_stokes_solver"]["linear_solver"]["type_name"] == "LinearSolver"
    assert models[0]["navier_stokes_solver"]["linear_solver"]["max_iterations"] == 30

    # linear_solver that already has type_name should be unchanged
    assert models[1]["navier_stokes_solver"]["linear_solver"]["type_name"] == "LinearSolver"
    assert models[1]["navier_stokes_solver"]["linear_solver"]["max_iterations"] == 50

    # navier_stokes_solver without linear_solver should be unaffected
    assert "linear_solver" not in models[2]["navier_stokes_solver"]

    # Non-Fluid model without navier_stokes_solver should be unaffected
    assert "navier_stokes_solver" not in models[3]


def test_updater_to_25_9_1_via_updater():
    """Test the full updater path from 25.9.0 to 25.9.1 adds linear_solver type_name."""

    params_as_dict = {
        "version": "25.9.0",
        "unit_system": {"name": "SI"},
        "models": [
            {
                "type": "Fluid",
                "navier_stokes_solver": {
                    "linear_solver": {
                        "max_iterations": 25,
                    },
                },
            },
        ],
    }

    params_new = updater(
        version_from="25.9.0",
        version_to="25.9.1",
        params_as_dict=params_as_dict,
    )

    assert params_new["version"] == "25.9.1"
    assert (
        params_new["models"][0]["navier_stokes_solver"]["linear_solver"]["type_name"]
        == "LinearSolver"
    )


def test_updater_to_25_9_1_no_models():
    """Test 25.9.1 updater handles missing or empty models gracefully."""

    # No models key at all
    params_no_models = {"version": "25.9.0"}
    params_new = _to_25_9_1(params_no_models)
    assert "models" not in params_new

    # Empty models list
    params_empty_models = {"version": "25.9.0", "models": []}
    params_new = _to_25_9_1(params_empty_models)
    assert params_new["models"] == []
