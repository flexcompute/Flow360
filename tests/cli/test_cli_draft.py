import json

from click.testing import CliRunner

from flow360.cli import flow360


def test_draft_group_help_shows_read_commands():
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "info" in result.output
    assert "state" in result.output
    assert "create" not in result.output
    assert "run" not in result.output
    assert "get" not in result.output
    assert "simulation-params" in result.output


def test_draft_list_outputs_records(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_list_drafts",
        lambda project_id: [
            {
                "id": "dft-123",
                "name": "Draft 1",
                "projectId": project_id,
                "solverVersion": "release-25.2",
                "type": "Draft",
            }
        ],
    )

    result = runner.invoke(flow360, ["draft", "list", "--project-id", "prj-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["records"] == [
        {
            "fork_case": None,
            "id": "dft-123",
            "name": "Draft 1",
            "project_id": "prj-123",
            "solver_version": "release-25.2",
            "source_item_id": None,
            "source_item_type": None,
            "type": "Draft",
        }
    ]


def test_draft_hidden_aliases_are_not_registered():
    runner = CliRunner()

    list_result = runner.invoke(flow360, ["draft", "ls", "--project-id", "prj-123"])
    info_result = runner.invoke(flow360, ["draft", "get", "dft-123"])

    assert list_result.exit_code != 0
    assert "No such command" in list_result.output
    assert info_result.exit_code != 0
    assert "No such command" in info_result.output


def test_draft_info_outputs_metadata(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    info = {
        "id": "dft-123",
        "name": "Draft 1",
        "projectId": "prj-123",
        "solverVersion": "release-25.2",
        "type": "Draft",
    }
    monkeypatch.setattr(draft_cli, "_get_draft_info", lambda draft_id: info)

    result = runner.invoke(flow360, ["draft", "info", "dft-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "dft-123"
    assert payload["project_id"] == "prj-123"
    assert payload["type"] == "Draft"


def test_draft_state_outputs_lifecycle_projection(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "get_resource_state_for_type",
        lambda resource_type, resource_id: {
            "id": resource_id,
            "type": "Draft",
            "status": "queued",
            "is_terminal": False,
            "is_success": False,
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["draft", "state", "dft-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "dft-123",
        "type": "Draft",
        "status": "queued",
        "is_terminal": False,
        "is_success": False,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_draft_simulation_params_get_outputs_json(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_get_draft_simulation_params",
        lambda draft_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["draft", "simulation-params", "get", "dft-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation_params"]["version"] == "24.11.0"
    assert payload["simulation_params"]["unit_system"]["name"] == "SI"
