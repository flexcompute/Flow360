import json
from types import SimpleNamespace

from click.testing import CliRunner

from flow360.cli import flow360


def test_project_group_help_shows_read_commands():
    runner = CliRunner()

    result = runner.invoke(flow360, ["project", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "info" in result.output
    assert "tree" in result.output
    assert "path" in result.output


def test_project_list_supports_search_limit_and_folder_filters(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    calls = {}
    record = SimpleNamespace(
        name="Wing Study",
        project_id="prj-123",
        tags=["demo"],
        description="test project",
        solver_version="release-25.2",
        created_at="2025-01-01T00:00:00Z",
        root_item_type="Geometry",
    )

    monkeypatch.setattr(
        project_cli,
        "_get_project_records",
        lambda search=None, limit=25, folder_ids=None, exclude_subfolders=False: (
            calls.update(
                {
                    "search": search,
                    "limit": limit,
                    "folder_ids": folder_ids,
                    "exclude_subfolders": exclude_subfolders,
                }
            )
            or ([record], 1)
        ),
    )

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
            "folder-123",
            "--exclude-subfolders",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["records"][0]["id"] == "prj-123"
    assert payload["returned"] == 1
    assert calls == {
        "search": "wing",
        "limit": 10,
        "folder_ids": ("folder-123",),
        "exclude_subfolders": True,
    }


def test_get_project_records_accepts_none_folder_ids(monkeypatch):
    from flow360.cli import project as project_cli
    from flow360.component.simulation.web import project_records

    calls = {}

    def fake_get_project_records(**kwargs):
        calls.update(kwargs)
        return [], 0

    monkeypatch.setattr(project_records, "get_project_records", fake_get_project_records)

    records, total = project_cli._get_project_records(folder_ids=None)

    assert records == []
    assert total == 0
    assert calls["folder_ids"] is None


def test_global_dev_and_profile_apply_to_project_commands(monkeypatch):
    from flow360.cli import project as project_cli
    from flow360.environment import Env
    from flow360.user_config import UserConfig

    runner = CliRunner()
    seen = {}

    monkeypatch.setattr(
        project_cli,
        "_get_project_info",
        lambda project_id: seen.update({"env": Env.current.name, "profile": UserConfig.profile})
        or {
            "id": project_id,
            "name": "Wing Study",
            "solverVersion": "release-25.2",
            "tags": [],
            "rootItemId": "geo-123",
            "rootItemType": "Geometry",
        },
    )

    result = runner.invoke(
        flow360,
        [
            "--dev",
            "--profile",
            "alt",
            "project",
            "info",
            "prj-12345678-1234-1234-1234-123456789abc",
        ],
    )

    assert result.exit_code == 0
    assert seen["env"] == "dev"
    assert seen["profile"] == "alt"


def test_project_list_outputs_records(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    record = SimpleNamespace(
        name="Wing Study",
        project_id="prj-123",
        tags=["demo"],
        description="test project",
        solver_version="release-25.2",
        created_at="2025-01-01T00:00:00Z",
        root_item_type="Geometry",
    )

    monkeypatch.setattr(
        project_cli,
        "_get_project_records",
        lambda search=None, limit=25, folder_ids=None, exclude_subfolders=False: ([record], 1),
    )

    result = runner.invoke(flow360, ["project", "list", "--keyword", "wing"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["records"][0]["id"] == "prj-123"
    assert payload["records"][0]["name"] == "Wing Study"
    assert payload["returned"] == 1
    assert payload["total"] == 1


def test_project_list_can_output_legacy_style_text(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    record = SimpleNamespace(
        name="Wing Study",
        project_id="prj-123",
        tags=["demo"],
        description="test project",
        solver_version="release-25.2",
        created_at="2025-01-01T00:00:00Z",
        root_item_type="Geometry",
        statistics=SimpleNamespace(
            geometry=SimpleNamespace(
                count=1,
                successCount=1,
                runningCount=0,
                divergedCount=0,
                errorCount=0,
            ),
            surface_mesh=None,
            volume_mesh=None,
            case=SimpleNamespace(
                count=3,
                successCount=2,
                runningCount=0,
                divergedCount=1,
                errorCount=0,
            ),
        ),
    )

    monkeypatch.setattr(
        project_cli,
        "_get_project_records",
        lambda search=None, limit=25, folder_ids=None, exclude_subfolders=False: ([record], 7),
    )
    monkeypatch.setattr(
        project_cli,
        "_project_browser_url",
        lambda project_id: f"https://example.test/workbench/{project_id}",
    )

    result = runner.invoke(flow360, ["project", "list", "--keyword", "wing", "--format", "text"])

    assert result.exit_code == 0
    assert ">>> Projects sorted by creation time:" in result.output
    assert "Name:         Wing Study" in result.output
    assert "Created with: Geometry" in result.output
    assert "Solver:       release-25.2" in result.output
    assert "Link:         https://example.test/workbench/prj-123" in result.output
    assert "Geometry count:     1" in result.output
    assert "Case count:         3" in result.output
    assert "Showing 1 of 7 matching projects." in result.output


def test_show_projects_uses_project_list_formatter(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    calls = {}
    record = SimpleNamespace(
        name="Wing Study",
        project_id="prj-123",
        tags=[],
        description=None,
        solver_version="release-25.2",
        created_at="2025-01-01T00:00:00Z",
        root_item_type="Geometry",
    )

    monkeypatch.setattr(
        project_cli,
        "_get_project_records",
        lambda search=None, limit=25, folder_ids=None, exclude_subfolders=False: (
            calls.update({"search": search, "limit": limit}) or ([record], 1)
        ),
    )
    monkeypatch.setattr(
        project_cli,
        "_project_browser_url",
        lambda project_id: f"https://example.test/workbench/{project_id}",
    )

    result = runner.invoke(flow360, ["show_projects", "-k", "wing"])

    assert result.exit_code == 0
    assert calls == {"search": "wing", "limit": 200}
    assert "Name:         Wing Study" in result.output
    assert "Link:         https://example.test/workbench/prj-123" in result.output


def test_project_info_outputs_metadata(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    info = {
        "id": "prj-123",
        "name": "Wing Study",
        "solverVersion": "release-25.2",
        "tags": ["demo"],
        "rootItemId": "geo-123",
        "rootItemType": "Geometry",
    }

    monkeypatch.setattr(
        project_cli,
        "_get_project_info",
        lambda project_id: info,
    )

    result = runner.invoke(flow360, ["project", "info", "prj-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "prj-123"
    assert payload["name"] == "Wing Study"
    assert payload["root_item"]["id"] == "geo-123"
    assert payload["root_item"]["type"] == "Geometry"


def test_project_tree_outputs_nested_tree(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    leaf = SimpleNamespace(
        asset_id="case-123",
        asset_name="Case 1",
        asset_type="Case",
        children=[],
    )
    root = SimpleNamespace(
        asset_id="geo-123",
        asset_name="Wing",
        asset_type="Geometry",
        children=[leaf],
    )
    monkeypatch.setattr(
        project_cli,
        "_get_project_tree",
        lambda project_id: SimpleNamespace(root=root),
    )

    result = runner.invoke(flow360, ["project", "tree", "prj-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["root"]["id"] == "geo-123"
    assert payload["root"]["children"][0]["id"] == "case-123"


def test_project_items_outputs_flat_items(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    monkeypatch.setattr(
        project_cli,
        "_get_project_tree_records",
        lambda project_id: [
            {
                "id": "geo-123",
                "name": "Wing",
                "type": "Geometry",
                "parentId": None,
                "parentCaseId": None,
            },
            {
                "id": "case-123",
                "name": "Case 1",
                "type": "Case",
                "parentId": None,
                "parentCaseId": "geo-123",
            },
        ],
    )

    result = runner.invoke(flow360, ["project", "items", "prj-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["items"] == [
        {
            "id": "geo-123",
            "name": "Wing",
            "parent_id": None,
            "type": "Geometry",
        },
        {
            "id": "case-123",
            "name": "Case 1",
            "parent_id": "geo-123",
            "type": "Case",
        },
    ]


def test_project_path_outputs_flat_branch(monkeypatch):
    from flow360.cli import project as project_cli

    runner = CliRunner()
    monkeypatch.setattr(
        project_cli,
        "_get_project_path",
        lambda project_id, item_id, item_type: [
            {
                "id": "geo-123",
                "name": "Wing",
                "type": "Geometry",
                "parentId": None,
                "status": "processed",
                "updatedAt": "2025-01-01T00:00:00Z",
            },
            {
                "id": "case-123",
                "name": "Case 1",
                "type": "Case",
                "parentId": "geo-123",
                "status": "completed",
                "updatedAt": "2025-01-01T01:00:00Z",
            },
        ],
    )

    result = runner.invoke(
        flow360,
        [
            "project",
            "path",
            "prj-123",
            "--item-id",
            "case-123",
            "--item-type",
            "Case",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["items"] == [
        {
            "id": "geo-123",
            "name": "Wing",
            "parent_id": None,
            "status": "processed",
            "type": "Geometry",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "id": "case-123",
            "name": "Case 1",
            "parent_id": "geo-123",
            "status": "completed",
            "type": "Case",
            "updated_at": "2025-01-01T01:00:00Z",
        },
    ]
