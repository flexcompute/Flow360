import json

from click.testing import CliRunner

from flow360.cli import flow360


def test_root_help_shows_wait():
    runner = CliRunner()

    result = runner.invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "wait" in result.output


def test_wait_help_shows_polling_options():
    runner = CliRunner()

    result = runner.invoke(flow360, ["wait", "--help"])

    assert result.exit_code == 0
    assert "--timeout" in result.output
    assert "--poll-interval" in result.output


def test_wait_outputs_terminal_success_state(monkeypatch):
    from flow360.cli import wait as wait_cli

    runner = CliRunner()
    monkeypatch.setattr(
        wait_cli,
        "_wait_for_resource_state",
        lambda ref_id, timeout, poll_interval: {
            "id": ref_id,
            "type": "VolumeMesh",
            "status": "completed",
            "is_terminal": True,
            "is_success": True,
            "updated_at": "2025-01-01T01:00:00Z",
        },
    )

    result = runner.invoke(flow360, ["wait", "vm-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "vm-123",
        "type": "VolumeMesh",
        "status": "completed",
        "is_terminal": True,
        "is_success": True,
        "updated_at": "2025-01-01T01:00:00Z",
    }


def test_wait_failed_terminal_state_exits_nonzero(monkeypatch):
    from flow360.cli import wait as wait_cli

    runner = CliRunner()
    monkeypatch.setattr(
        wait_cli,
        "_wait_for_resource_state",
        lambda ref_id, timeout, poll_interval: {
            "id": ref_id,
            "type": "Case",
            "status": "failed",
            "is_terminal": True,
            "is_success": False,
            "updated_at": "2025-01-01T01:00:00Z",
            "mesh_id": "vm-123",
        },
    )

    result = runner.invoke(flow360, ["wait", "case-123"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["mesh_id"] == "vm-123"


def test_wait_timeout_exits_124(monkeypatch):
    from flow360.cli import wait as wait_cli

    runner = CliRunner()
    monkeypatch.setattr(
        wait_cli,
        "_wait_for_resource_state",
        lambda ref_id, timeout, poll_interval: (_ for _ in ()).throw(
            wait_cli.WaitTimeoutError(
                {
                    "id": ref_id,
                    "type": "Draft",
                    "status": "queued",
                    "is_terminal": False,
                    "is_success": False,
                    "updated_at": "2025-01-01T01:00:00Z",
                }
            )
        ),
    )

    result = runner.invoke(flow360, ["wait", "dft-123"])

    assert result.exit_code == 124
    payload = json.loads(result.output)
    assert payload["timed_out"] is True
    assert payload["status"] == "queued"
