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


def test_project_ls_uses_limit_search_and_folder_filters(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        [
            "project",
            "ls",
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


def test_geometry_info_uses_geometry_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "info", GEOMETRY_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == GEOMETRY_ID
    assert payload["type"] == "Geometry"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }


def test_geometry_get_alias_uses_geometry_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "get", GEOMETRY_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == GEOMETRY_ID
    assert payload["type"] == "Geometry"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }


def test_geometry_state_uses_geometry_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "state", GEOMETRY_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == GEOMETRY_ID
    assert payload["status"] == "processed"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }


def test_geometry_simulation_get_uses_geometry_simulation_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "simulation", "get", GEOMETRY_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert isinstance(payload["simulation"], dict)
    assert "models" in payload["simulation"]
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_geometry_summary_uses_geometry_simulation_endpoint(monkeypatch, recorded_webapi_calls):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_summarize_simulation_json",
        lambda simulation_json: {"models": {"surface": []}},
    )

    result = runner.invoke(flow360, ["geometry", "summary", GEOMETRY_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == GEOMETRY_ID
    assert "summary" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_case_info_uses_case_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "info", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == CASE_ID
    assert payload["mesh_id"] == "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
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
    assert payload["mesh_id"] == "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}",
        "params": None,
    }


def test_surface_mesh_info_uses_surface_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "info", SURFACE_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == SURFACE_MESH_ID
    assert payload["type"] == "SurfaceMesh"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}",
        "params": None,
    }


def test_surface_mesh_get_alias_uses_surface_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "get", SURFACE_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == SURFACE_MESH_ID
    assert payload["type"] == "SurfaceMesh"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}",
        "params": None,
    }


def test_surface_mesh_state_uses_surface_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "state", SURFACE_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == SURFACE_MESH_ID
    assert payload["status"] == "processed"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}",
        "params": None,
    }


def test_surface_mesh_simulation_get_uses_surface_mesh_simulation_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "simulation", "get", SURFACE_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert isinstance(payload["simulation"], dict)
    assert "models" in payload["simulation"]
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_surface_mesh_summary_uses_surface_mesh_simulation_endpoint(
    monkeypatch, recorded_webapi_calls
):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_summarize_simulation_json",
        lambda simulation_json: {"models": {"surface": []}},
    )

    result = runner.invoke(flow360, ["surface-mesh", "summary", SURFACE_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == SURFACE_MESH_ID
    assert "summary" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_volume_mesh_info_uses_volume_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "info", VOLUME_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert payload["type"] == "VolumeMesh"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
        "params": None,
    }


def test_volume_mesh_get_alias_uses_volume_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "get", VOLUME_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert payload["type"] == "VolumeMesh"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
        "params": None,
    }


def test_volume_mesh_state_uses_volume_mesh_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "state", VOLUME_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert payload["status"] == "completed"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
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


def test_volume_mesh_simulation_get_uses_volume_mesh_simulation_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "simulation", "get", VOLUME_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert isinstance(payload["simulation"], dict)
    assert "models" in payload["simulation"]
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_volume_mesh_summary_uses_volume_mesh_simulation_endpoint(
    monkeypatch, recorded_webapi_calls
):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_summarize_simulation_json",
        lambda simulation_json: {"models": {"surface": []}},
    )

    result = runner.invoke(flow360, ["volume-mesh", "summary", VOLUME_MESH_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert "summary" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_case_get_alias_uses_case_v2_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "get", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == CASE_ID
    assert payload["mesh_id"] == "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}",
        "params": None,
    }


def test_case_simulation_get_uses_case_simulation_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "simulation", "get", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert "simulation" in payload
    assert isinstance(payload["simulation"], dict)
    assert "models" in payload["simulation"]
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_case_summary_uses_case_simulation_endpoint(monkeypatch, recorded_webapi_calls):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_summarize_simulation_json",
        lambda simulation_json: {"models": {"fluid": []}},
    )

    result = runner.invoke(flow360, ["case", "summary", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == CASE_ID
    assert "summary" in payload
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/v2/cases/{CASE_ID}/simulation/file",
        "params": {"type": "simulation"},
    }


def test_case_results_ls_uses_legacy_case_files_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "results", "list", CASE_ID])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["records"]
    assert all(record["path"].startswith("results/") for record in payload["records"])
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/cases/{CASE_ID}/files",
        "params": None,
    }


def test_case_results_get_uses_legacy_case_files_endpoint(monkeypatch, recorded_webapi_calls):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_download_case_result",
        lambda case_id, result_path, to_path=None, overwrite=False: "/tmp/total_forces_v2.csv",
    )

    result = runner.invoke(
        flow360,
        ["case", "results", "get", CASE_ID, "force_output_wing_all_planes_forces_v2.csv"],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["saved_to"] == "/tmp/total_forces_v2.csv"
    assert payload["result"]["path"] == "results/force_output_wing_all_planes_forces_v2.csv"
    assert recorded_webapi_calls[-1] == {
        "type": "get",
        "url": f"/cases/{CASE_ID}/files",
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


def test_draft_get_alias_uses_draft_info_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "get", DRAFT_ID])

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


def test_draft_simulation_set_uses_simulation_endpoint(recorded_webapi_calls, tmp_path):
    runner = CliRunner()
    file_path = tmp_path / "params.json"
    file_path.write_text('{"version":"24.11.0","unit_system":{"name":"SI"}}')

    result = runner.invoke(
        flow360,
        ["draft", "simulation", "set", DRAFT_ID, str(file_path)],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": DRAFT_ID, "updated": True}
    assert recorded_webapi_calls[-1] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/simulation/file",
        "params": {
            "data": '{"version": "24.11.0", "unit_system": {"name": "SI"}}',
            "type": "simulation",
            "version": "",
        },
    }


def test_draft_create_from_project_resolves_root_and_posts_to_drafts(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "create", PROJECT_ID, "--name", "Draft A"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == DRAFT_ID
    assert payload["source_item_id"] == GEOMETRY_ID
    calls = recorded_webapi_calls[-3:]
    assert calls[0] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
    }
    assert calls[1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }
    assert calls[2]["type"] == "post"
    assert calls[2]["url"] == "/v2/drafts"
    assert calls[2]["params"]["name"] == "Draft A"
    assert calls[2]["params"]["projectId"] == PROJECT_ID
    assert calls[2]["params"]["sourceItemId"] == GEOMETRY_ID
    assert calls[2]["params"]["sourceItemType"] == "Geometry"
    assert calls[2]["params"]["solverVersion"] == "release-24.11"
    assert calls[2]["params"]["forkCase"] is False


def test_draft_run_uses_run_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "run", DRAFT_ID, "--up-to", "volume-mesh"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == VOLUME_MESH_ID
    assert payload["type"] == "VolumeMesh"
    assert recorded_webapi_calls[-1] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/run",
        "params": {
            "upTo": "VolumeMesh",
            "useInHouse": False,
            "useGai": False,
        },
    }


def test_draft_run_wait_polls_result_state_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "run", DRAFT_ID, "--up-to", "volume-mesh", "--wait"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["result"]["id"] == VOLUME_MESH_ID
    assert payload["state"]["id"] == VOLUME_MESH_ID
    calls = recorded_webapi_calls[-2:]
    assert calls[0] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/run",
        "params": {
            "upTo": "VolumeMesh",
            "useInHouse": False,
            "useGai": False,
        },
    }
    assert calls[1] == {
        "type": "get",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
        "params": None,
    }


def test_draft_run_from_project_creates_sets_and_runs(recorded_webapi_calls, tmp_path):
    runner = CliRunner()
    simulation_path = tmp_path / "simulation.json"
    simulation_path.write_text('{"version":"24.11.0","unit_system":{"name":"SI"}}')

    result = runner.invoke(
        flow360,
        [
            "draft",
            "run",
            PROJECT_ID,
            str(simulation_path),
            "--name",
            "Alpha -18",
            "--up-to",
            "volume-mesh",
        ],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["draft"]["id"] == DRAFT_ID
    assert payload["result"]["id"] == VOLUME_MESH_ID
    calls = recorded_webapi_calls[-5:]
    assert calls[0] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
    }
    assert calls[1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }
    assert calls[2]["type"] == "post"
    assert calls[2]["url"] == "/v2/drafts"
    assert calls[2]["params"]["projectId"] == PROJECT_ID
    assert calls[2]["params"]["sourceItemId"] == GEOMETRY_ID
    assert calls[2]["params"]["sourceItemType"] == "Geometry"
    assert calls[2]["params"]["solverVersion"] == "release-24.11"
    assert calls[2]["params"]["forkCase"] is False
    assert calls[2]["params"]["name"] == "Alpha -18"
    assert calls[3] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/simulation/file",
        "params": {
            "data": '{"version": "24.11.0", "unit_system": {"name": "SI"}}',
            "type": "simulation",
            "version": "",
        },
    }
    assert calls[4] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/run",
        "params": {
            "upTo": "VolumeMesh",
            "useInHouse": False,
            "useGai": False,
        },
    }


def test_draft_run_from_project_patch_fetches_merges_sets_and_runs(recorded_webapi_calls, tmp_path):
    runner = CliRunner()
    patch_path = tmp_path / "patch.json"
    patch_path.write_text('{"meshing":{"refinement_factor":2.5}}')

    result = runner.invoke(
        flow360,
        ["draft", "run", PROJECT_ID, "--patch", str(patch_path), "--up-to", "volume-mesh"],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["draft"]["id"] == DRAFT_ID
    assert payload["result"]["id"] == VOLUME_MESH_ID
    calls = recorded_webapi_calls[-6:]
    assert calls[0] == {
        "type": "get",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
    }
    assert calls[1] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": None,
    }
    assert calls[2] == {
        "type": "get",
        "url": f"/v2/geometries/{GEOMETRY_ID}/simulation/file",
        "params": {"type": "simulation"},
    }
    assert calls[3] == {
        "type": "post",
        "url": "/v2/drafts",
        "params": {
            **calls[3]["params"],
        },
    }
    assert calls[3]["params"]["projectId"] == PROJECT_ID
    assert calls[3]["params"]["sourceItemId"] == GEOMETRY_ID
    assert calls[3]["params"]["sourceItemType"] == "Geometry"
    assert calls[3]["params"]["solverVersion"] == "release-24.11"
    assert calls[3]["params"]["forkCase"] is False
    assert isinstance(calls[3]["params"]["name"], str)
    assert calls[3]["params"]["name"]
    assert calls[4]["type"] == "post"
    assert calls[4]["url"] == f"/v2/drafts/{DRAFT_ID}/simulation/file"
    assert calls[4]["params"]["type"] == "simulation"
    assert calls[4]["params"]["version"] == ""
    assert calls[5] == {
        "type": "post",
        "url": f"/v2/drafts/{DRAFT_ID}/run",
        "params": {
            "upTo": "VolumeMesh",
            "useInHouse": False,
            "useGai": False,
        },
    }

    merged_payload = json.loads(calls[4]["params"]["data"])
    assert merged_payload["meshing"]["refinement_factor"] == 2.5
    assert "defaults" in merged_payload["meshing"]


def test_project_rename_uses_project_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "rename", PROJECT_ID, "--name", "Renamed Project"])

    assert result.exit_code == 0
    assert f"Renamed project {PROJECT_ID} to Renamed Project." in result.output
    assert recorded_webapi_calls[-1]["type"] == "patch"
    assert recorded_webapi_calls[-1]["url"] == f"/v2/projects/{PROJECT_ID}"
    assert recorded_webapi_calls[-1]["params"]["name"] == "Renamed Project"


def test_case_rename_uses_case_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "rename", CASE_ID, "--name", "Alpha -18"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": CASE_ID, "name": "Alpha -18"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/cases/{CASE_ID}",
        "params": {"name": "Alpha -18"},
    }


def test_geometry_rename_uses_geometry_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360, ["geometry", "rename", GEOMETRY_ID, "--name", "Renamed Geometry"]
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": GEOMETRY_ID, "name": "Renamed Geometry"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/geometries/{GEOMETRY_ID}",
        "params": {"name": "Renamed Geometry"},
    }


def test_surface_mesh_rename_uses_surface_mesh_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        ["surface-mesh", "rename", SURFACE_MESH_ID, "--name", "Renamed Surface Mesh"],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": SURFACE_MESH_ID, "name": "Renamed Surface Mesh"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/surface-meshes/{SURFACE_MESH_ID}",
        "params": {"name": "Renamed Surface Mesh"},
    }


def test_volume_mesh_rename_uses_volume_mesh_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        ["volume-mesh", "rename", VOLUME_MESH_ID, "--name", "Renamed Volume Mesh"],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": VOLUME_MESH_ID, "name": "Renamed Volume Mesh"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/volume-meshes/{VOLUME_MESH_ID}",
        "params": {"name": "Renamed Volume Mesh"},
    }


def test_draft_rename_uses_draft_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "rename", DRAFT_ID, "--name", "Renamed Draft"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": DRAFT_ID, "name": "Renamed Draft"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/drafts/{DRAFT_ID}",
        "params": {"name": "Renamed Draft"},
    }


def test_project_delete_uses_project_delete_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "delete", PROJECT_ID, "--yes"])

    assert result.exit_code == 0
    assert f"Deleted project {PROJECT_ID}." in result.output
    assert recorded_webapi_calls[-1] == {
        "type": "delete",
        "url": f"/v2/projects/{PROJECT_ID}",
        "params": None,
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


def test_folder_create_uses_folder_create_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        [
            "folder",
            "create",
            "--name",
            "Folder A",
            "--parent-folder-id",
            "ROOT.FLOW360",
            "--tag",
            "demo",
        ],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload["id"] == "folder-3834758b-3d39-4a4a-ad85-710b7652267c"
    assert recorded_webapi_calls[-1] == {
        "type": "post",
        "url": "/folders",
        "params": {
            "name": "Folder A",
            "tags": ["demo"],
            "parentFolderId": "ROOT.FLOW360",
            "type": "folder",
        },
    }


def test_folder_rename_uses_folder_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(flow360, ["folder", "rename", FOLDER_ID, "--name", "Renamed Folder"])

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {"id": FOLDER_ID, "name": "Renamed Folder"}
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/folders/{FOLDER_ID}",
        "params": {"name": "Renamed Folder"},
    }


def test_folder_move_uses_folder_patch_endpoint(recorded_webapi_calls):
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        [
            "folder",
            "move",
            FOLDER_ID,
            "--parent-folder-id",
            "folder-4da3cdd0-c5b6-4130-9ca1-196237322ab9",
        ],
    )

    assert result.exit_code == 0
    payload = _load_json_output(result.output)
    assert payload == {
        "id": FOLDER_ID,
        "parent_id": "folder-4da3cdd0-c5b6-4130-9ca1-196237322ab9",
    }
    assert recorded_webapi_calls[-1] == {
        "type": "patch",
        "url": f"/v2/folders/{FOLDER_ID}",
        "params": {"parentFolderId": "folder-4da3cdd0-c5b6-4130-9ca1-196237322ab9"},
    }
