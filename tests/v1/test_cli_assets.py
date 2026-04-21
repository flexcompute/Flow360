import json

from click.testing import CliRunner

from flow360.cli import flow360


def test_flow360_help_shows_asset_groups():
    runner = CliRunner()

    result = runner.invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "geometry" in result.output
    assert "surface-mesh" in result.output
    assert "volume-mesh" in result.output
    assert "case" in result.output
    assert "folder" in result.output


def test_case_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["case", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "simulation" in result.output
    assert "results" in result.output
    assert "get" not in result.output


def test_geometry_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_surface_mesh_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_volume_mesh_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_geometry_info_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "geo-123",
        "name": "Wing",
        "projectId": "prj-123",
        "parentId": None,
        "solverVersion": "release-25.2",
        "status": "processed",
        "tags": ["demo"],
        "type": "Geometry",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["geometry", "info", "geo-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "geo-123"
    assert payload["project_id"] == "prj-123"
    assert payload["type"] == "Geometry"


def test_geometry_get_alias_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "geo-123",
        "name": "Wing",
        "projectId": "prj-123",
        "parentId": None,
        "solverVersion": "release-25.2",
        "status": "processed",
        "tags": ["demo"],
        "type": "Geometry",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["geometry", "get", "geo-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "geo-123"
    assert payload["project_id"] == "prj-123"
    assert payload["type"] == "Geometry"


def test_geometry_rename_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        assets_cli,
        "_rename_asset",
        lambda webapi_cls, asset_id, new_name: calls.update(
            {"webapi_cls": webapi_cls, "asset_id": asset_id, "new_name": new_name}
        ),
    )

    result = runner.invoke(flow360, ["geometry", "rename", "geo-123", "--name", "Wing Renamed"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "geo-123", "name": "Wing Renamed"}
    assert calls["asset_id"] == "geo-123"
    assert calls["new_name"] == "Wing Renamed"


def test_geometry_state_outputs_lifecycle_projection(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "get_resource_state_for_type",
        lambda resource_type, resource_id: {
            "id": resource_id,
            "type": "Geometry",
            "status": "processed",
            "is_terminal": True,
            "is_success": True,
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["geometry", "state", "geo-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "geo-123",
        "type": "Geometry",
        "status": "processed",
        "is_terminal": True,
        "is_success": True,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_geometry_simulation_get_outputs_json(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_json",
        lambda webapi_cls, asset_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["geometry", "simulation", "get", "geo-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation"]["version"] == "24.11.0"
    assert payload["simulation"]["unit_system"]["name"] == "SI"


def test_surface_mesh_info_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "sm-123",
        "name": "Surface Mesh",
        "projectId": "prj-123",
        "parentId": "geo-123",
        "solverVersion": "release-25.2",
        "status": "processed",
        "tags": [],
        "type": "SurfaceMesh",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["surface-mesh", "info", "sm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "sm-123"
    assert payload["parent_id"] == "geo-123"
    assert payload["type"] == "SurfaceMesh"


def test_surface_mesh_get_alias_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "sm-123",
        "name": "Surface Mesh",
        "projectId": "prj-123",
        "parentId": "geo-123",
        "solverVersion": "release-25.2",
        "status": "processed",
        "tags": [],
        "type": "SurfaceMesh",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["surface-mesh", "get", "sm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "sm-123"
    assert payload["parent_id"] == "geo-123"
    assert payload["type"] == "SurfaceMesh"


def test_surface_mesh_rename_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        assets_cli,
        "_rename_asset",
        lambda webapi_cls, asset_id, new_name: calls.update(
            {"webapi_cls": webapi_cls, "asset_id": asset_id, "new_name": new_name}
        ),
    )

    result = runner.invoke(
        flow360,
        ["surface-mesh", "rename", "sm-123", "--name", "Surface Mesh Renamed"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "sm-123", "name": "Surface Mesh Renamed"}
    assert calls["asset_id"] == "sm-123"
    assert calls["new_name"] == "Surface Mesh Renamed"


def test_surface_mesh_state_outputs_lifecycle_projection(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "get_resource_state_for_type",
        lambda resource_type, resource_id: {
            "id": resource_id,
            "type": "SurfaceMesh",
            "status": "queued",
            "is_terminal": False,
            "is_success": False,
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["surface-mesh", "state", "sm-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "sm-123",
        "type": "SurfaceMesh",
        "status": "queued",
        "is_terminal": False,
        "is_success": False,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_surface_mesh_simulation_get_outputs_json(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_json",
        lambda webapi_cls, asset_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["surface-mesh", "simulation", "get", "sm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation"]["version"] == "24.11.0"
    assert payload["simulation"]["unit_system"]["name"] == "SI"


def test_volume_mesh_info_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "vm-123",
        "name": "Volume Mesh",
        "projectId": "prj-123",
        "parentId": "sm-123",
        "solverVersion": "release-25.2",
        "status": "completed",
        "tags": ["demo"],
        "type": "VolumeMesh",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["volume-mesh", "info", "vm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "vm-123"
    assert payload["type"] == "VolumeMesh"


def test_volume_mesh_get_alias_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "vm-123",
        "name": "Volume Mesh",
        "projectId": "prj-123",
        "parentId": "sm-123",
        "solverVersion": "release-25.2",
        "status": "completed",
        "tags": ["demo"],
        "type": "VolumeMesh",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["volume-mesh", "get", "vm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "vm-123"
    assert payload["type"] == "VolumeMesh"


def test_volume_mesh_rename_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        assets_cli,
        "_rename_asset",
        lambda webapi_cls, asset_id, new_name: calls.update(
            {"webapi_cls": webapi_cls, "asset_id": asset_id, "new_name": new_name}
        ),
    )

    result = runner.invoke(
        flow360,
        ["volume-mesh", "rename", "vm-123", "--name", "Volume Mesh Renamed"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "vm-123", "name": "Volume Mesh Renamed"}
    assert calls["asset_id"] == "vm-123"
    assert calls["new_name"] == "Volume Mesh Renamed"


def test_volume_mesh_state_outputs_lifecycle_projection(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "get_resource_state_for_type",
        lambda resource_type, resource_id: {
            "id": resource_id,
            "type": "VolumeMesh",
            "status": "failed",
            "is_terminal": True,
            "is_success": False,
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["volume-mesh", "state", "vm-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "vm-123",
        "type": "VolumeMesh",
        "status": "failed",
        "is_terminal": True,
        "is_success": False,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_volume_mesh_simulation_get_outputs_json(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_json",
        lambda webapi_cls, asset_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["volume-mesh", "simulation", "get", "vm-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation"]["version"] == "24.11.0"
    assert payload["simulation"]["unit_system"]["name"] == "SI"


def test_case_info_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "case-123",
        "name": "Case 1",
        "projectId": "prj-123",
        "caseMeshId": "vm-123",
        "solverVersion": "release-25.2",
        "status": "completed",
        "tags": ["demo"],
        "type": "Case",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["case", "info", "case-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "case-123"
    assert payload["mesh_id"] == "vm-123"
    assert payload["type"] == "Case"


def test_case_get_alias_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    info = {
        "id": "case-123",
        "name": "Case 1",
        "projectId": "prj-123",
        "caseMeshId": "vm-123",
        "solverVersion": "release-25.2",
        "status": "completed",
        "tags": ["demo"],
        "type": "Case",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
    }
    monkeypatch.setattr(assets_cli, "_get_asset_info", lambda webapi_cls, asset_id: info)

    result = runner.invoke(flow360, ["case", "get", "case-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "case-123"
    assert payload["mesh_id"] == "vm-123"
    assert payload["type"] == "Case"


def test_case_rename_outputs_metadata(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        assets_cli,
        "_rename_asset",
        lambda webapi_cls, asset_id, new_name: calls.update(
            {"webapi_cls": webapi_cls, "asset_id": asset_id, "new_name": new_name}
        ),
    )

    result = runner.invoke(flow360, ["case", "rename", "case-123", "--name", "Alpha -18"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "case-123", "name": "Alpha -18"}
    assert calls["asset_id"] == "case-123"
    assert calls["new_name"] == "Alpha -18"


def test_case_state_outputs_lifecycle_projection(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "get_resource_state_for_type",
        lambda resource_type, resource_id: {
            "id": resource_id,
            "type": "Case",
            "status": "completed",
            "is_terminal": True,
            "is_success": True,
            "mesh_id": "vm-123",
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["case", "state", "case-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "case-123",
        "type": "Case",
        "status": "completed",
        "is_terminal": True,
        "is_success": True,
        "mesh_id": "vm-123",
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_case_simulation_get_outputs_json(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_json",
        lambda webapi_cls, asset_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["case", "simulation", "get", "case-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation"]["version"] == "24.11.0"
    assert payload["simulation"]["unit_system"]["name"] == "SI"


def test_case_results_list_outputs_only_result_artifacts(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_list_case_results",
        lambda case_id: [
            {
                "fileName": "results/total_forces_v2.csv",
                "filePath": "results/total_forces_v2.csv",
                "fileType": ".csv",
                "length": 1024,
                "updatedAt": "2025-01-01T01:00:00Z",
            },
            {
                "fileName": "results/nonlinear_residual_v2.csv",
                "filePath": "results/nonlinear_residual_v2.csv",
                "fileType": ".csv",
                "length": 2048,
                "updatedAt": None,
            },
        ],
    )

    result = runner.invoke(flow360, ["case", "results", "list", "case-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "records": [
            {
                "name": "total_forces_v2.csv",
                "path": "results/total_forces_v2.csv",
                "file_type": ".csv",
                "size_bytes": 1024,
                "updated_at": "2025-01-01T01:00:00Z",
            },
            {
                "name": "nonlinear_residual_v2.csv",
                "path": "results/nonlinear_residual_v2.csv",
                "file_type": ".csv",
                "size_bytes": 2048,
                "updated_at": None,
            },
        ]
    }


def test_case_results_ls_alias_outputs_only_result_artifacts(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_list_case_results",
        lambda case_id: [
            {
                "fileName": "results/total_forces_v2.csv",
                "filePath": "results/total_forces_v2.csv",
                "fileType": ".csv",
                "length": 1024,
                "updatedAt": "2025-01-01T01:00:00Z",
            }
        ],
    )

    result = runner.invoke(flow360, ["case", "results", "ls", "case-123"])

    assert result.exit_code == 0
    assert json.loads(result.output)["records"][0]["path"] == "results/total_forces_v2.csv"


def test_case_result_serialization_normalizes_full_storage_path():
    from flow360.cli import assets as assets_cli

    record = {
        "fileName": "results/total_forces_v2.csv",
        "filePath": "users/AID/case-123/results/total_forces_v2.csv",
        "fileType": ".csv",
        "length": 1024,
        "updatedAt": "2025-01-01T01:00:00Z",
    }

    assert assets_cli._serialize_case_result(record) == {
        "name": "total_forces_v2.csv",
        "path": "results/total_forces_v2.csv",
        "file_type": ".csv",
        "size_bytes": 1024,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_case_results_get_downloads_selected_artifact(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_resolve_case_result",
        lambda case_id, result_ref: {
            "fileName": "results/total_forces_v2.csv",
            "filePath": "results/total_forces_v2.csv",
            "fileType": ".csv",
            "length": 1024,
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )
    monkeypatch.setattr(
        assets_cli,
        "_download_case_result",
        lambda case_id, result_path, to_path=None, overwrite=False: "/tmp/total_forces_v2.csv",
    )

    result = runner.invoke(flow360, ["case", "results", "get", "case-123", "total_forces_v2.csv"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "case_id": "case-123",
        "result": {
            "name": "total_forces_v2.csv",
            "path": "results/total_forces_v2.csv",
            "file_type": ".csv",
            "size_bytes": 1024,
            "updated_at": "2025-01-01T01:00:00Z",
        },
        "saved_to": "/tmp/total_forces_v2.csv",
    }


def test_case_results_get_normalizes_storage_path_before_download(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    captured = {}
    monkeypatch.setattr(
        assets_cli,
        "_resolve_case_result",
        lambda case_id, result_ref: {
            "fileName": "results/total_forces_v2.csv",
            "filePath": "users/AID/case-123/results/total_forces_v2.csv",
            "fileType": ".csv",
            "length": 1024,
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )

    def _fake_download(case_id, result_path, to_path=None, overwrite=False):
        captured["case_id"] = case_id
        captured["result_path"] = result_path
        return "/tmp/total_forces_v2.csv"

    monkeypatch.setattr(assets_cli, "_download_case_result", _fake_download)

    result = runner.invoke(flow360, ["case", "results", "get", "case-123", "total_forces_v2.csv"])

    assert result.exit_code == 0
    assert captured == {
        "case_id": "case-123",
        "result_path": "results/total_forces_v2.csv",
    }
    assert json.loads(result.output) == {
        "case_id": "case-123",
        "result": {
            "name": "total_forces_v2.csv",
            "path": "results/total_forces_v2.csv",
            "file_type": ".csv",
            "size_bytes": 1024,
            "updated_at": "2025-01-01T01:00:00Z",
        },
        "saved_to": "/tmp/total_forces_v2.csv",
    }
