import json

from click.testing import CliRunner
import pytest

from flow360.cli import flow360
from flow360.cli.resource_refs import ResourceRefError


def test_root_help_shows_open():
    runner = CliRunner()

    result = runner.invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "open" in result.output


def test_open_help_shows_usage():
    runner = CliRunner()

    result = runner.invoke(flow360, ["open", "--help"])

    assert result.exit_code == 0
    assert "Open a Flow360 resource in the browser." in result.output


def test_open_project_prints_url_and_opens_browser(monkeypatch):
    from flow360.cli import open_resource as open_cli

    runner = CliRunner()
    opened_urls = []
    monkeypatch.setattr(
        open_cli,
        "open_browser_url",
        lambda url: opened_urls.append(url) or True,
    )

    result = runner.invoke(flow360, ["open", "prj-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "prj-123",
        "opened": True,
        "type": "Project",
        "url": "https://flow360.simulation.cloud/workbench/prj-123",
    }
    assert opened_urls == ["https://flow360.simulation.cloud/workbench/prj-123"]


def test_open_case_prints_url_when_browser_does_not_open(monkeypatch):
    from flow360.cli import open_resource as open_cli
    from flow360.cli import browser_links

    runner = CliRunner()
    monkeypatch.setattr(open_cli, "open_browser_url", lambda url: False)
    monkeypatch.setattr(
        browser_links,
        "_get_project_scoped_resource_info",
        lambda resource_type, resource_id: {"projectId": "prj-123"},
    )

    result = runner.invoke(flow360, ["open", "case-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "case-123",
        "opened": False,
        "type": "Case",
        "url": "https://flow360.simulation.cloud/workbench/prj-123?id=case-123&type=Case",
    }


def test_open_respects_root_environment_selection(monkeypatch):
    from flow360.cli import open_resource as open_cli
    from flow360.cli import browser_links

    runner = CliRunner()
    monkeypatch.setattr(open_cli, "open_browser_url", lambda url: True)
    monkeypatch.setattr(
        browser_links,
        "_get_project_scoped_resource_info",
        lambda resource_type, resource_id: {"projectId": "prj-123"},
    )

    result = runner.invoke(flow360, ["--dev", "open", "dft-123"])

    assert result.exit_code == 0
    assert (
        json.loads(result.output)["url"]
        == "https://flow360.dev-simulation.cloud/workbench/prj-123?id=dft-123&type=Draft"
    )


def test_open_folder_infers_workspace_route(monkeypatch):
    from flow360.cli import open_resource as open_cli
    from flow360.cli import browser_links

    runner = CliRunner()
    monkeypatch.setattr(open_cli, "open_browser_url", lambda url: True)
    monkeypatch.setattr(browser_links, "_resolve_folder_workspace_id", lambda resource_id: "private-abc")

    result = runner.invoke(flow360, ["open", "folder-123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "folder-123",
        "opened": True,
        "type": "Folder",
        "url": "https://flow360.simulation.cloud/workspaces?workspaceId=private-abc&folderId=folder-123&activeTabIndex=0",
    }


def test_open_shared_root_folder_uses_inferred_workspace_route(monkeypatch):
    from flow360.cli import open_resource as open_cli
    from flow360.cli import browser_links

    runner = CliRunner()
    monkeypatch.setattr(open_cli, "open_browser_url", lambda url: False)
    monkeypatch.setattr(
        browser_links,
        "_resolve_folder_workspace_id",
        lambda resource_id: "shared-abc",
    )

    result = runner.invoke(flow360, ["open", "ROOT.FLOW360.123"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "id": "ROOT.FLOW360.123",
        "opened": False,
        "type": "Folder",
        "url": "https://flow360.simulation.cloud/workspaces?workspaceId=shared-abc&folderId=ROOT.FLOW360.123&activeTabIndex=0",
    }


def test_resolve_folder_workspace_id_uses_root_folder_workspace_mapping(monkeypatch):
    from flow360.cli import browser_links

    monkeypatch.setattr(browser_links, "_get_root_folder_id", lambda resource_id: "ROOT.FLOW360.123")
    monkeypatch.setattr(
        browser_links,
        "_get_workspace_id_for_root_folder",
        lambda root_folder_id: "shared-abc" if root_folder_id == "ROOT.FLOW360.123" else None,
    )

    assert browser_links._resolve_folder_workspace_id("folder-123") == "shared-abc"


def test_resolve_folder_workspace_id_errors_when_workspace_is_missing(monkeypatch):
    from flow360.cli import browser_links

    monkeypatch.setattr(browser_links, "_get_root_folder_id", lambda resource_id: "ROOT.FLOW360.123")
    monkeypatch.setattr(browser_links, "_get_workspace_id_for_root_folder", lambda root_folder_id: None)

    with pytest.raises(ResourceRefError) as error:
        browser_links._resolve_folder_workspace_id("folder-123")

    assert str(error.value) == (
        "Could not infer a workspace for folder folder-123. "
        "No workspace matched rootFolderId ROOT.FLOW360.123."
    )
