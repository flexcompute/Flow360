import json

from click.testing import CliRunner

from flow360.cli import flow360


def test_draft_group_help_shows_read_commands():
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "create" in result.output
    assert "info" in result.output
    assert "state" in result.output
    assert "run" in result.output
    assert "get" not in result.output
    assert "simulation" in result.output


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


def test_draft_ls_alias_outputs_records(monkeypatch):
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

    result = runner.invoke(flow360, ["draft", "ls", "--project-id", "prj-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["records"][0]["id"] == "dft-123"


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


def test_draft_get_alias_outputs_metadata(monkeypatch):
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

    result = runner.invoke(flow360, ["draft", "get", "dft-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "dft-123"
    assert payload["project_id"] == "prj-123"
    assert payload["type"] == "Draft"


def test_draft_rename_outputs_metadata(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        draft_cli,
        "_rename_draft",
        lambda draft_id, new_name: calls.update({"draft_id": draft_id, "new_name": new_name}),
    )

    result = runner.invoke(flow360, ["draft", "rename", "dft-123", "--name", "Alpha Sweep 1"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "dft-123", "name": "Alpha Sweep 1"}
    assert calls == {"draft_id": "dft-123", "new_name": "Alpha Sweep 1"}


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


def test_draft_simulation_get_outputs_json(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_get_draft_simulation_json",
        lambda draft_id: {"version": "24.11.0", "unit_system": {"name": "SI"}},
    )

    result = runner.invoke(flow360, ["draft", "simulation", "get", "dft-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["simulation"]["version"] == "24.11.0"
    assert payload["simulation"]["unit_system"]["name"] == "SI"


def test_draft_simulation_set_reads_json_and_updates(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    calls = {}
    file_path = tmp_path / "params.json"
    file_path.write_text('{"version":"24.11.0","unit_system":{"name":"SI"}}')

    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_json",
        lambda draft_id, simulation_json: calls.update(
            {"draft_id": draft_id, "simulation_json": simulation_json}
        ),
    )

    result = runner.invoke(
        flow360,
        ["draft", "simulation", "set", "dft-123", str(file_path)],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "dft-123", "updated": True}
    assert calls == {
        "draft_id": "dft-123",
        "simulation_json": {"version": "24.11.0", "unit_system": {"name": "SI"}},
    }


def test_draft_simulation_set_rejects_invalid_json(tmp_path):
    runner = CliRunner()
    file_path = tmp_path / "params.json"
    file_path.write_text("{not-json}")

    result = runner.invoke(
        flow360,
        ["draft", "simulation", "set", "dft-123", str(file_path)],
    )

    assert result.exit_code != 0
    assert f"Invalid JSON in {file_path}" in result.output


def test_draft_run_outputs_result_metadata(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["draft", "run", "dft-123", "--up-to", "volume-mesh"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "created_at": "2025-01-01T00:00:00Z",
        "id": "vm-123",
        "name": "Volume Mesh",
        "parent_id": "sm-123",
        "project_id": "prj-123",
        "solver_version": "release-25.2",
        "status": "queued",
        "tags": ["demo"],
        "type": "VolumeMesh",
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_wait_for_resource_state_polls_until_terminal(monkeypatch):
    from flow360.cli import resource_state

    states = iter(
        [
            {
                "id": "vm-123",
                "type": "VolumeMesh",
                "status": "queued",
                "is_terminal": False,
                "is_success": False,
                "updated_at": "2025-01-01T00:00:00Z",
            },
            {
                "id": "vm-123",
                "type": "VolumeMesh",
                "status": "completed",
                "is_terminal": True,
                "is_success": True,
                "updated_at": "2025-01-01T00:00:02Z",
            },
        ]
    )
    sleeps = []
    monotonic_values = iter([0.0, 0.5])

    monkeypatch.setattr(resource_state, "get_resource_state", lambda resource_id: next(states))
    monkeypatch.setattr(resource_state.time, "sleep", lambda interval: sleeps.append(interval))
    monkeypatch.setattr(resource_state.time, "monotonic", lambda: next(monotonic_values))

    state = resource_state.wait_for_resource_state("vm-123", timeout=10.0, poll_interval=2.0)

    assert state["status"] == "completed"
    assert sleeps == [2.0]


def test_draft_run_wait_outputs_result_and_state(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_wait_for_resource_state",
        lambda resource_id, timeout, poll_interval: {
            "id": resource_id,
            "type": "VolumeMesh",
            "status": "completed",
            "is_terminal": True,
            "is_success": True,
            "updated_at": "2025-01-01T01:00:02Z",
        },
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "dft-123", "--up-to", "volume-mesh", "--wait"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "result": {
            "created_at": "2025-01-01T00:00:00Z",
            "id": "vm-123",
            "name": "Volume Mesh",
            "parent_id": "sm-123",
            "project_id": "prj-123",
            "solver_version": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "updated_at": "2025-01-01T01:00:00Z",
        },
        "state": {
            "id": "vm-123",
            "type": "VolumeMesh",
            "status": "completed",
            "is_terminal": True,
            "is_success": True,
            "updated_at": "2025-01-01T01:00:02Z",
        },
    }


def test_draft_run_wait_failed_state_exits_nonzero(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": [],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_wait_for_resource_state",
        lambda resource_id, timeout, poll_interval: {
            "id": resource_id,
            "type": "VolumeMesh",
            "status": "failed",
            "is_terminal": True,
            "is_success": False,
            "updated_at": "2025-01-01T01:00:02Z",
        },
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "dft-123", "--up-to", "volume-mesh", "--wait"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["state"]["status"] == "failed"
    assert payload["state"]["is_success"] is False


def test_draft_run_wait_timeout_exits_124(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": [],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_wait_for_resource_state",
        lambda resource_id, timeout, poll_interval: (_ for _ in ()).throw(
            draft_cli.WaitTimeoutError(
                {
                    "id": resource_id,
                    "type": "VolumeMesh",
                    "status": "running",
                    "is_terminal": False,
                    "is_success": False,
                    "updated_at": "2025-01-01T01:00:02Z",
                }
            )
        ),
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "dft-123", "--up-to", "volume-mesh", "--wait"],
    )

    assert result.exit_code == 124
    payload = json.loads(result.output)
    assert payload["timed_out"] is True
    assert payload["state"]["status"] == "running"


def test_draft_run_rejects_non_draft_id():
    runner = CliRunner()

    result = runner.invoke(flow360, ["draft", "run", "prj-123", "--up-to", "volume-mesh"])

    assert result.exit_code != 0
    assert "Simulation JSON path or --patch is required when running from a non-draft ref." in result.output


def test_draft_create_outputs_metadata(monkeypatch):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    monkeypatch.setattr(
        draft_cli,
        "_create_draft_from_ref",
        lambda ref_id, name=None: {
            "id": "dft-123",
            "name": name,
            "projectId": "prj-123",
            "solverVersion": "release-25.2",
            "sourceItemId": "geo-123",
            "sourceItemType": "Geometry",
            "forkCase": False,
            "type": "Draft",
        },
    )

    result = runner.invoke(flow360, ["draft", "create", "prj-123", "--name", "Draft 1"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "fork_case": False,
        "id": "dft-123",
        "name": "Draft 1",
        "project_id": "prj-123",
        "solver_version": "release-25.2",
        "source_item_id": "geo-123",
        "source_item_type": "Geometry",
        "type": "Draft",
    }


def test_draft_run_from_project_creates_sets_and_runs(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    simulation_path = tmp_path / "simulation.json"
    simulation_path.write_text('{"version":"24.11.0","unit_system":{"name":"SI"}}')
    calls = {}

    monkeypatch.setattr(
        draft_cli,
        "_resolve_draft_source",
        lambda ref_id: {
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
            "name": name or "Draft 1",
            "projectId": source["project_id"],
            "solverVersion": source["solver_version"],
            "sourceItemId": source["source_item_id"],
            "sourceItemType": source["source_item_type"],
            "forkCase": source["fork_case"],
            "type": "Draft",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_json",
        lambda draft_id, simulation_json: calls.update(
            {"draft_id": draft_id, "simulation_json": simulation_json}
        ),
    )
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "prj-123", str(simulation_path), "--name", "Alpha -18", "--up-to", "volume-mesh"],
    )

    assert result.exit_code == 0
    assert calls == {
        "draft_id": "dft-123",
        "simulation_json": {"version": "24.11.0", "unit_system": {"name": "SI"}},
    }
    assert json.loads(result.output) == {
        "draft": {
            "fork_case": False,
            "id": "dft-123",
            "name": "Alpha -18",
            "project_id": "prj-123",
            "solver_version": "release-25.2",
            "source_item_id": "geo-123",
            "source_item_type": "Geometry",
            "type": "Draft",
        },
        "result": {
            "created_at": "2025-01-01T00:00:00Z",
            "id": "vm-123",
            "name": "Volume Mesh",
            "parent_id": "sm-123",
            "project_id": "prj-123",
            "solver_version": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "updated_at": "2025-01-01T01:00:00Z",
        },
    }


def test_draft_run_from_project_with_wait_outputs_draft_result_and_state(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    simulation_path = tmp_path / "simulation.json"
    simulation_path.write_text('{"version":"24.11.0","unit_system":{"name":"SI"}}')

    monkeypatch.setattr(
        draft_cli,
        "_resolve_draft_source",
        lambda ref_id: {
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
            "name": "Draft 1",
            "projectId": source["project_id"],
            "solverVersion": source["solver_version"],
            "sourceItemId": source["source_item_id"],
            "sourceItemType": source["source_item_type"],
            "forkCase": source["fork_case"],
            "type": "Draft",
        },
    )
    monkeypatch.setattr(draft_cli, "_set_draft_simulation_json", lambda draft_id, simulation_json: None)
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_wait_for_resource_state",
        lambda resource_id, timeout, poll_interval: {
            "id": resource_id,
            "type": "VolumeMesh",
            "status": "completed",
            "is_terminal": True,
            "is_success": True,
            "updated_at": "2025-01-01T01:00:02Z",
        },
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "prj-123", str(simulation_path), "--up-to", "volume-mesh", "--wait"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["draft"]["id"] == "dft-123"
    assert payload["result"]["id"] == "vm-123"
    assert payload["state"]["status"] == "completed"


def test_draft_run_existing_draft_rejects_simulation_path(tmp_path):
    runner = CliRunner()
    simulation_path = tmp_path / "simulation.json"
    simulation_path.write_text('{"version":"24.11.0"}')

    result = runner.invoke(
        flow360,
        ["draft", "run", "dft-123", str(simulation_path), "--up-to", "volume-mesh"],
    )

    assert result.exit_code != 0
    assert (
        "Simulation JSON, patch, or name cannot be passed when running an existing draft."
        in result.output
    )


def test_draft_run_existing_draft_rejects_name():
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        ["draft", "run", "dft-123", "--name", "Alpha -18", "--up-to", "volume-mesh"],
    )

    assert result.exit_code != 0
    assert (
        "Simulation JSON, patch, or name cannot be passed when running an existing draft."
        in result.output
    )


def test_draft_run_from_project_patch_creates_sets_and_runs(monkeypatch, tmp_path):
    from flow360.cli import draft as draft_cli

    runner = CliRunner()
    patch_path = tmp_path / "patch.json"
    patch_path.write_text('{"meshing":{"refinement_factor":2.5}}')
    calls = {}

    monkeypatch.setattr(
        draft_cli,
        "_resolve_draft_source",
        lambda ref_id: {
            "project_id": "prj-123",
            "source_item_id": "geo-123",
            "source_item_type": "Geometry",
            "solver_version": "release-25.2",
            "fork_case": False,
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_apply_patch_to_source_simulation",
        lambda source, patch_json: {
            "version": "24.11.0",
            "meshing": {
                "defaults": {"first_layer_thickness": 0.1},
                "refinement_factor": patch_json["meshing"]["refinement_factor"],
            },
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_create_draft_from_source",
        lambda source, name=None: {
            "id": "dft-123",
            "name": "Draft 1",
            "projectId": source["project_id"],
            "solverVersion": source["solver_version"],
            "sourceItemId": source["source_item_id"],
            "sourceItemType": source["source_item_type"],
            "forkCase": source["fork_case"],
            "type": "Draft",
        },
    )
    monkeypatch.setattr(
        draft_cli,
        "_set_draft_simulation_json",
        lambda draft_id, simulation_json: calls.update(
            {"draft_id": draft_id, "simulation_json": simulation_json}
        ),
    )
    monkeypatch.setattr(
        draft_cli,
        "_run_draft",
        lambda draft_id, up_to: {
            "id": "vm-123",
            "name": "Volume Mesh",
            "projectId": "prj-123",
            "parentId": "sm-123",
            "solverVersion": "release-25.2",
            "status": "queued",
            "tags": ["demo"],
            "type": "VolumeMesh",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(
        flow360,
        ["draft", "run", "prj-123", "--patch", str(patch_path), "--up-to", "volume-mesh"],
    )

    assert result.exit_code == 0
    assert calls == {
        "draft_id": "dft-123",
        "simulation_json": {
            "version": "24.11.0",
            "meshing": {
                "defaults": {"first_layer_thickness": 0.1},
                "refinement_factor": 2.5,
            },
        },
    }


def test_draft_run_rejects_simulation_and_patch_together(tmp_path):
    runner = CliRunner()
    simulation_path = tmp_path / "simulation.json"
    patch_path = tmp_path / "patch.json"
    simulation_path.write_text('{"version":"24.11.0"}')
    patch_path.write_text('{"meshing":{"refinement_factor":2.5}}')

    result = runner.invoke(
        flow360,
        [
            "draft",
            "run",
            "prj-123",
            str(simulation_path),
            "--patch",
            str(patch_path),
            "--up-to",
            "volume-mesh",
        ],
    )

    assert result.exit_code != 0
    assert "Provide either a full simulation JSON path or --patch, not both." in result.output
