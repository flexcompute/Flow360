import json

from click.testing import CliRunner

from flow360.cli import flow360


def test_folder_group_help_shows_read_commands():
    runner = CliRunner()

    result = runner.invoke(flow360, ["folder", "--help"])

    assert result.exit_code == 0
    assert "get" in result.output
    assert "tree" in result.output


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

