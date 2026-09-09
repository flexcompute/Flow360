import json
import sys

from click.testing import CliRunner

from flow360.cli import flow360


def test_folder_group_help_shows_read_commands():
    runner = CliRunner()

    result = runner.invoke(flow360, ["folder", "--help"])

    assert result.exit_code == 0
    assert "get" in result.output
    assert "tree" in result.output
    assert "create" in result.output
    assert "rename" in result.output
    assert "move" in result.output
    assert "delete" in result.output


def test_folder_help_does_not_import_folder_webapi():
    runner = CliRunner()
    sys.modules.pop("flow360.cli.folder", None)
    sys.modules.pop("flow360.component.simulation.web.folder_webapi", None)
    flow360.commands.pop("folder", None)

    result = runner.invoke(flow360, ["folder", "--help"])

    assert result.exit_code == 0
    assert "flow360.component.simulation.web.folder_webapi" not in sys.modules


def test_folder_get_outputs_metadata(monkeypatch):
    from flow360.cli import folder as folder_cli

    runner = CliRunner()
    monkeypatch.setattr(
        folder_cli,
        "_get_folder_info",
        lambda folder_id: {
            "id": folder_id,
            "name": "Folder A",
            "parentFolderId": "ROOT.FLOW360",
            "type": "folder",
            "tags": ["demo"],
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T01:00:00Z",
            "parentFolders": [{"id": "ROOT.FLOW360", "name": "My workspace"}],
        },
    )

    result = runner.invoke(flow360, ["folder", "get", "folder-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "folder-123"
    assert payload["parent_id"] == "ROOT.FLOW360"
    assert payload["type"] == "folder"


def test_folder_tree_outputs_nested_tree(monkeypatch):
    from flow360.cli import folder as folder_cli

    runner = CliRunner()
    monkeypatch.setattr(
        folder_cli,
        "_get_folder_tree",
        lambda folder_id: {
            "id": folder_id,
            "name": "My workspace",
            "subfolders": [
                {
                    "id": "folder-123",
                    "name": "Folder A",
                    "subfolders": [],
                }
            ],
        },
    )

    result = runner.invoke(flow360, ["folder", "tree"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["root"]["id"] == "ROOT.FLOW360"
    assert payload["root"]["subfolders"][0]["id"] == "folder-123"


def test_folder_create_outputs_metadata(monkeypatch):
    from flow360.cli import folder as folder_cli

    runner = CliRunner()
    calls = {}
    monkeypatch.setattr(
        folder_cli,
        "_create_folder",
        lambda name, parent_folder_id, tags: calls.update(
            {"name": name, "parent_folder_id": parent_folder_id, "tags": tags}
        )
        or {
            "id": "folder-123",
            "name": name,
            "parentFolderId": parent_folder_id,
            "type": "folder",
            "tags": list(tags),
        },
    )

    result = runner.invoke(
        flow360,
        [
            "folder",
            "create",
            "--name",
            "Folder A",
            "--parent-folder-id",
            "folder-parent",
            "--tag",
            "demo",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "folder-123"
    assert payload["name"] == "Folder A"
    assert calls == {
        "name": "Folder A",
        "parent_folder_id": "folder-parent",
        "tags": ("demo",),
    }


def test_folder_rename_move_and_delete(monkeypatch):
    from flow360.cli import folder as folder_cli

    runner = CliRunner()
    calls = []
    monkeypatch.setattr(
        folder_cli,
        "_rename_folder",
        lambda folder_id, name: calls.append(("rename", folder_id, name)),
    )
    monkeypatch.setattr(
        folder_cli,
        "_move_folder",
        lambda folder_id, parent_folder_id: calls.append(("move", folder_id, parent_folder_id)),
    )
    monkeypatch.setattr(
        folder_cli, "_delete_folder", lambda folder_id: calls.append(("delete", folder_id))
    )

    rename_result = runner.invoke(flow360, ["folder", "rename", "folder-123", "--name", "Updated"])
    move_result = runner.invoke(
        flow360, ["folder", "move", "folder-123", "--parent-folder-id", "folder-parent"]
    )
    delete_result = runner.invoke(flow360, ["folder", "delete", "folder-123", "--yes"])

    assert rename_result.exit_code == 0
    assert move_result.exit_code == 0
    assert delete_result.exit_code == 0
    assert calls == [
        ("rename", "folder-123", "Updated"),
        ("move", "folder-123", "folder-parent"),
        ("delete", "folder-123"),
    ]
    assert json.loads(rename_result.output) == {"id": "folder-123", "name": "Updated"}
    assert json.loads(move_result.output) == {"id": "folder-123", "parent_id": "folder-parent"}
    assert json.loads(delete_result.output) == {"id": "folder-123", "deleted": True}


def test_folder_delete_requires_confirmation(monkeypatch):
    from flow360.cli import folder as folder_cli

    runner = CliRunner()
    monkeypatch.setattr(
        folder_cli,
        "_delete_folder",
        lambda folder_id: (_ for _ in ()).throw(AssertionError("should not delete")),
    )

    result = runner.invoke(flow360, ["folder", "delete", "folder-123"])

    assert result.exit_code != 0
    assert "Pass --yes" in result.output
