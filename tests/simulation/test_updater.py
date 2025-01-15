import json

import pytest

import flow360 as fl
from flow360.component.simulation.framework.updater import updater


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_updater_to_24_11_7():

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
