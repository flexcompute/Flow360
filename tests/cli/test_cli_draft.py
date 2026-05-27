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
    assert "create" in result.output
    assert "run" in result.output
    assert "rename" in result.output
    assert "delete" in result.output
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


def test_draft_create_outputs_new_draft(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_resolve_draft_source",
        lambda source_id: {
            "project_id": "prj-123",
            "source_item_id": "geo-123",
            "source_item_type": "Geometry",
            "solver_version": "release-25.2",
            "fork_case": False,
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_create_draft_from_source",
        lambda source, name=None: {
            "id": "dft-123",
            "name": name,
            "projectId": source["project_id"],
            "sourceItemId": source["source_item_id"],
            "sourceItemType": source["source_item_type"],
            "solverVersion": source["solver_version"],
            "forkCase": source["fork_case"],
            "type": "Draft",
        },
    )

    result = runner.invoke(flow360, ["draft", "create", "geo-123", "--name", "Alpha -4"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "dft-123"
    assert payload["name"] == "Alpha -4"
    assert payload["source_item_id"] == "geo-123"


def test_draft_rename_and_delete(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = []
    monkeypatch.setattr(
        draft_cli, "_rename_draft", lambda draft_id, name: calls.append((draft_id, name))
    )
    monkeypatch.setattr(
        draft_cli, "_delete_draft", lambda draft_id: calls.append((draft_id, "delete"))
    )

    rename_result = runner.invoke(flow360, ["draft", "rename", "dft-123", "--name", "Updated"])
    delete_result = runner.invoke(flow360, ["draft", "delete", "dft-123", "--yes"])

    assert rename_result.exit_code == 0
    assert delete_result.exit_code == 0
    assert calls == [("dft-123", "Updated"), ("dft-123", "delete")]
    assert json.loads(rename_result.output) == {"id": "dft-123", "name": "Updated"}
    assert json.loads(delete_result.output) == {"id": "dft-123", "deleted": True}


def test_draft_delete_requires_confirmation(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_delete_draft",
        lambda draft_id: (_ for _ in ()).throw(AssertionError("should not delete")),
    )

    result = runner.invoke(flow360, ["draft", "delete", "dft-123"])

    assert result.exit_code != 0
    assert "Pass --yes" in result.output


def test_draft_simulation_params_set_reads_json_file(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = []
    params_file = tmp_path / "simulation.json"
    params_file.write_text('{"version": "24.11.0"}', encoding="utf-8")
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_params",
        lambda draft_id, params: calls.append((draft_id, params)),
    )

    result = runner.invoke(
        flow360,
        ["draft", "simulation-params", "set", "dft-123", str(params_file)],
    )

    assert result.exit_code == 0
    assert calls == [("dft-123", {"version": "24.11.0"})]
    assert json.loads(result.output) == {"id": "dft-123", "updated": True}


def test_draft_simulation_params_patch_merges_json(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = []
    patch_file = tmp_path / "alpha.patch.json"
    patch_file.write_text(
        '{"operating_condition": {"alpha": {"value": -10, "units": "degree"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        draft_cli,
        "_get_draft_simulation_params",
        lambda draft_id: {
            "version": "24.11.0",
            "operating_condition": {
                "alpha": {"value": 0, "units": "degree"},
                "beta": {"value": 0, "units": "degree"},
            },
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_params",
        lambda draft_id, params: calls.append((draft_id, params)),
    )

    result = runner.invoke(
        flow360,
        ["draft", "simulation-params", "patch", "dft-123", str(patch_file)],
    )

    assert result.exit_code == 0
    assert calls[0][0] == "dft-123"
    assert calls[0][1]["operating_condition"]["alpha"]["value"] == -10
    assert calls[0][1]["operating_condition"]["beta"]["value"] == 0


def test_draft_run_existing_draft_sets_params_and_waits(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = []
    params_file = tmp_path / "simulation.json"
    params_file.write_text('{"version": "24.11.0"}', encoding="utf-8")
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_params",
        lambda draft_id, params: calls.append(("set", draft_id, params)),
    )
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: calls.append(("run", draft_id, up_to))
        or {"id": "case-123", "type": "Case", "status": "queued"},
    )
    monkeypatch.setattr(
        draft_cli,
        "_wait_for_resource_state",
        lambda resource_id, timeout, poll_interval: calls.append(
            ("wait", resource_id, timeout, poll_interval)
        )
        or {"id": resource_id, "status": "completed"},
    )

    result = runner.invoke(
        flow360,
        [
            "draft",
            "run",
            "dft-123",
            str(params_file),
            "--up-to",
            "case",
            "--wait",
            "--timeout",
            "2",
            "--poll-interval",
            "0.1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["result"]["id"] == "case-123"
    assert payload["state"]["status"] == "completed"
    assert calls == [
        ("set", "dft-123", {"version": "24.11.0"}),
        ("run", "dft-123", "Case"),
        ("wait", "case-123", 2.0, 0.1),
    ]


def test_draft_run_from_asset_creates_draft_and_applies_patch(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = []
    patch_file = tmp_path / "alpha.patch.json"
    patch_file.write_text('{"operating_condition": {"alpha": {"value": 4}}}', encoding="utf-8")
    source = {
        "project_id": "prj-123",
        "source_item_id": "geo-123",
        "source_item_type": "Geometry",
        "solver_version": "release-25.2",
        "fork_case": False,
    }
    monkeypatch.setattr(draft_cli, "_resolve_draft_source", lambda source_id: source)
    monkeypatch.setattr(
        draft_cli,
        "_create_draft_from_source",
        lambda source, name=None: calls.append(("create", source, name))
        or {
            "id": "dft-123",
            "name": name,
            "projectId": source["project_id"],
            "sourceItemId": source["source_item_id"],
            "sourceItemType": source["source_item_type"],
            "solverVersion": source["solver_version"],
            "forkCase": source["fork_case"],
            "type": "Draft",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_get_asset_simulation_params_for_type",
        lambda resource_type, resource_id: {
            "version": "24.11.0",
            "operating_condition": {"alpha": {"value": 0}},
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_params",
        lambda draft_id, params: calls.append(("set", draft_id, params)),
    )
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: calls.append(("run", draft_id, up_to))
        or {"id": "vm-123", "type": "VolumeMesh", "status": "queued"},
    )

    result = runner.invoke(
        flow360,
        [
            "draft",
            "run",
            "geo-123",
            "--name",
            "Alpha 4",
            "--patch",
            str(patch_file),
            "--up-to",
            "volume-mesh",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["draft"]["id"] == "dft-123"
    assert payload["result"]["id"] == "vm-123"
    assert calls[0] == ("create", source, "Alpha 4")
    assert calls[1][0] == "set"
    assert calls[1][2]["operating_condition"]["alpha"]["value"] == 4
    assert calls[2] == ("run", "dft-123", "VolumeMesh")
