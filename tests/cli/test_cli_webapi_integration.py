import importlib
import json

import pytest
from click.testing import CliRunner

from flow360.cli import flow360

PROJECT_ID = "prj-41d2333b-85fd-4bed-ae13-15dcb6da519e"
GEOMETRY_ID = "geo-2877e124-96ff-473d-864b-11eec8648d42"
SURFACE_MESH_ID = "sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a"
VOLUME_MESH_ID = "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
CASE_ID = "case-69b8c249-fce5-412a-9927-6a79049deebb"
DRAFT_ID = "dft-84b20880-937d-4ef2-983b-7f75089f6dd6"
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


def test_project_tree_uses_tree_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "tree", PROJECT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["root"]["id"] == GEOMETRY_ID
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


@pytest.mark.parametrize(
    ("command", "resource_id", "resource_type", "endpoint"),
    [
        ("geometry", GEOMETRY_ID, "Geometry", "geometries"),
        ("surface-mesh", SURFACE_MESH_ID, "SurfaceMesh", "surface-meshes"),
        ("volume-mesh", VOLUME_MESH_ID, "VolumeMesh", "volume-meshes"),
    ],
)
def test_asset_info_uses_v2_endpoint(
    command, resource_id, resource_type, endpoint, recorded_webapi_calls
):
    runner = CliRunner()

    result = runner.invoke(flow360, [command, "info", resource_id])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == resource_id
    assert payload["type"] == resource_type
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/{endpoint}/{resource_id}",
        "params": None,
    }


@pytest.mark.parametrize(
    ("command", "resource_id", "resource_type", "endpoint"),
    [
        ("geometry", GEOMETRY_ID, "Geometry", "geometries"),
        ("surface-mesh", SURFACE_MESH_ID, "SurfaceMesh", "surface-meshes"),
        ("volume-mesh", VOLUME_MESH_ID, "VolumeMesh", "volume-meshes"),
    ],
)
def test_asset_state_uses_v2_endpoint(
    command, resource_id, resource_type, endpoint, recorded_webapi_calls
):
    runner = CliRunner()

    result = runner.invoke(flow360, [command, "state", resource_id])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == resource_id
    assert payload["type"] == resource_type
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/{endpoint}/{resource_id}",
        "params": None,
    }


@pytest.mark.parametrize(
    ("command", "resource_id", "endpoint"),
    [
        ("geometry", GEOMETRY_ID, "geometries"),
        ("surface-mesh", SURFACE_MESH_ID, "surface-meshes"),
        ("volume-mesh", VOLUME_MESH_ID, "volume-meshes"),
        ("case", CASE_ID, "cases"),
    ],
)
def test_asset_simulation_get_uses_simulation_endpoint(
    command, resource_id, endpoint, recorded_webapi_calls
):
    runner = CliRunner()

    result = runner.invoke(flow360, [command, "simulation", "get", resource_id])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert isinstance(payload["simulation"], dict)
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/{endpoint}/{resource_id}/simulation/file",
        "params": {"type": "simulation"},
    }


@pytest.mark.parametrize(
    ("command", "resource_id", "endpoint"),
    [
        ("geometry", GEOMETRY_ID, "geometries"),
        ("surface-mesh", SURFACE_MESH_ID, "surface-meshes"),
        ("volume-mesh", VOLUME_MESH_ID, "volume-meshes"),
        ("case", CASE_ID, "cases"),
    ],
)
def test_asset_summary_uses_simulation_endpoint(
    command, resource_id, endpoint, monkeypatch, recorded_webapi_calls
):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_summarize_simulation_json",
        lambda simulation_json: {"models": {"surface": []}},
    )

    result = runner.invoke(flow360, [command, "summary", resource_id])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == resource_id
    assert "summary" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/{endpoint}/{resource_id}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_case_info_uses_case_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "info", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == CASE_ID
    assert payload["mesh_id"] == VOLUME_MESH_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}",
        "params": None,
    }


def test_case_state_uses_case_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "state", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == CASE_ID
    assert payload["status"] == "completed"
    assert payload["mesh_id"] == VOLUME_MESH_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}",
        "params": None,
    }


def test_wait_uses_resource_state_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["wait", VOLUME_MESH_ID, "--timeout", "0.1"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert payload["status"] == "completed"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
        "params": None,
    }


def test_draft_list_uses_project_scoped_list_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "list", "--project-id", PROJECT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["records"][0]["id"] == DRAFT_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": "/v2/drafts",
        "params": {"projectId": PROJECT_ID},
    }


def test_draft_info_uses_draft_info_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "info", DRAFT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == DRAFT_ID
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/drafts/{DRAFT_ID}",
        "params": None,
    }


def test_draft_state_uses_draft_info_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "state", DRAFT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == DRAFT_ID
    assert payload["status"] == "queued"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/drafts/{DRAFT_ID}",
        "params": None,
    }


def test_draft_simulation_get_uses_simulation_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "simulation", "get", DRAFT_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/drafts/{DRAFT_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_folder_get_uses_folder_info_endpoint(recorded_webapi_calls):
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
        "params": {
            "includeSubfolders": True,
            "page": 0,
            "size": 1000,
        },
    }
