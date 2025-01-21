import json

import pytest

import flow360 as fl
from flow360.component.simulation.framework.updater import updater
from flow360.component.simulation.framework.updater_utils import compare_dicts


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_updater_to_24_11_1():
    files = ["simulation_24_11_0.json"]

    for file in files:
        params = fl.SimulationParams(f"../data/simulation/{file}")
        assert params


def test_updater_from_24_11_1_5_to_24_11_6():
    with open("../data/simulation/simulation_no_updater.json", "r") as fp:
        params = json.load(fp)

    for idx_from in range(1, 6):
        for idx_to in range(idx_from + 1, 7):
            params_new = updater(
                version_from=f"24.11.{idx_from}",
                version_to=f"24.11.{idx_to}",
                params_as_dict=params,
            )
            assert compare_dicts(params, params_new)


def test_updater_from_24_11_0_6_to_24_11_7():

    with open("../data/simulation/simulation_24_11_6.json", "r") as fp:
        params_24_11_6 = json.load(fp)

    params_24_11_7 = updater(
        version_from="24.11.6", version_to="24.11.7", params_as_dict=params_24_11_6
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


def test_updater_from_24_11_8_to_24_11_9():
    with open("../data/simulation/simulation_24_11_8.json", "r") as fp:
        params = json.load(fp)

    for idx_from in range(1, 9):
        params_new = updater(
            version_from=f"24.11.{idx_from}",
            version_to=f"25.2.0",
            params_as_dict=params,
        )
        assert compare_dicts(params, params_new)
