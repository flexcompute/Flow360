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
    assert "summary" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_geometry_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["geometry", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "summary" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_surface_mesh_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["surface-mesh", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "summary" in result.output
    assert "simulation" in result.output
    assert "get" not in result.output


def test_volume_mesh_group_help_shows_info_and_simulation():
    runner = CliRunner()

    result = runner.invoke(flow360, ["volume-mesh", "--help"])

    assert result.exit_code == 0
    assert "info" in result.output
    assert "state" in result.output
    assert "summary" in result.output
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
