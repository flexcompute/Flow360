import importlib
import json

import pytest
from click.testing import CliRunner

from flow360.cli import flow360

PROJECT_ID = "prj-41d2333b-85fd-4bed-ae13-15dcb6da519e"
CASE_ID = "case-69b8c249-fce5-412a-9927-6a79049deebb"
FOLDER_ID = "folder-3834758b-3d39-4a4a-ad85-710b7652267c"


@pytest.fixture
def recorded_webapi_calls(monkeypatch, mock_response):
    mock_server = importlib.import_module("tests.mock_server")
    original_mock_webapi = mock_server.mock_webapi
    calls = []

    def recording_mock_webapi(request_type, url, params):
        calls.append({"type": request_type, "url": url, "params": params})
        return original_mock_webapi(request_type, url, params)

    monkeypatch.setattr(mock_server, "mock_webapi", recording_mock_webapi)
    return calls


def _load_json_output(output):
    json_start = output.rfind("\n{")
    if json_start == -1:
        json_start = output.find("{")
    else:
        json_start += 1
    return json.loads(output[json_start:])


def test_project_info_uses_project_info_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "info", PROJECT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == PROJECT_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
    }


def test_project_get_alias_uses_project_info_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "get", PROJECT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == PROJECT_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
    }


def test_project_tree_uses_tree_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "tree", PROJECT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["root"]["id"] == "geo-2877e124-96ff-473d-864b-11eec8648d42"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}/tree",
        "params": None,
    }


def test_project_list_uses_limit_search_and_folder_filters(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        [
            "project",
            "list",
            "--search",
            "wing",
            "--limit",
            "10",
            "--folder-id",
            FOLDER_ID,
            "--exclude-subfolders",
        ],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["returned"] >= 1
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": "/v2/projects",
        "params": {
            "page": "0",
            "size": 10,
            "filterKeywords": "wing",
            "filterTags": None,
            "sortFields": ["createdAt"],
            "sortDirections": ["desc"],
            "filterFolderIds": [FOLDER_ID],
            "filterExcludeSubfolders": True,
        },
    }


def test_project_path_uses_path_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        [
            "project",
            "path",
            PROJECT_ID,
            "--item-id",
            CASE_ID,
            "--item-type",
            "Case",
        ],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["items"][0]["type"] == "Geometry"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}/path",
        "params": {"itemId": CASE_ID, "itemType": "Case"},
    }


def test_folder_get_uses_folder_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["folder", "get", FOLDER_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == FOLDER_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/folders/{FOLDER_ID}",
        "params": None,
    }


def test_folder_tree_uses_folder_list_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["folder", "tree"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["root"]["id"] == "ROOT.FLOW360"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": "/v2/folders",
        "params": {"includeSubfolders": True, "page": 0, "size": 1000},
    }
