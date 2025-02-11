import copy
import json
from enum import Enum

import pytest

import flow360 as fl
from flow360.component.simulation.framework.updater import (
    VERSION_MILESTONES,
    _find_update_path,
    updater,
)
from flow360.component.simulation.framework.updater_utils import Flow360Version
from flow360.component.simulation.services import validate_model
from flow360.component.simulation.validation.validation_context import ALL


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


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

    # 12) from >99.11.3, to >99.11.3 => ValueError => []
    with pytest.raises(
        ValueError,
        match=r"Input `SimulationParams` have higher version than all known versions and thus cannot be handled.",
    ):
        _find_update_path(
            version_from=Flow360Version("99.11.4"),
            version_to=Flow360Version("99.11.5"),
            version_milestones=version_milestones,
        )

    # 13) to < from => ValueError
    with pytest.raises(
        ValueError,
        match=r"Input `SimulationParams` have higher version than the target version and thus cannot be handled.",
    ):
        _find_update_path(
            version_from=Flow360Version("99.11.3"),
            version_to=Flow360Version("99.11.2"),
            version_milestones=version_milestones,
        )

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


def test_deserialization_with_updater():
    # From 24.11.0 to 25.2.0
    with open("../data/simulation/simulation_24_11_0.json", "r") as fp:
        params = json.load(fp)
    validate_model(params_as_dict=params, root_item_type="VolumeMesh", validation_level=ALL)
