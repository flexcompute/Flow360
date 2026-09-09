import pytest

from flow360.component.simulation.folder import Folder, resolve_parent_folder
from flow360.component.simulation.web.workspace_webapi import WorkspaceWebApi
from flow360.component.workspace import Workspace
from flow360.exceptions import Flow360ValueError
from tests.mock_server import MOCK_WORKSPACE_RECORDS

SHARED_ID = "shared-22222222-2222-2222-2222-222222222222"
SHARED_ROOT = "ROOT.FLOW360.SHAREDROOT"


def test_get_all(mock_response):
    workspaces = Workspace._get_all()
    assert [workspace.id for workspace in workspaces] == [
        record["id"] for record in MOCK_WORKSPACE_RECORDS
    ]


def test_get_shared(mock_response):
    workspace = Workspace.get_shared()
    assert workspace.id == SHARED_ID
    assert workspace.name == "Test Company"
    assert workspace.status == "ENABLED"
    assert workspace.root_folder_id == SHARED_ROOT


def test_get_private(mock_response):
    workspace = Workspace.get_private()
    assert workspace.name == "Home"
    assert workspace.root_folder_id == "ROOT.FLOW360"


def _patch_records(monkeypatch, records):
    monkeypatch.setattr(WorkspaceWebApi, "list_records", classmethod(lambda cls: records))


def test_get_shared_without_company_tenant(monkeypatch):
    _patch_records(monkeypatch, [MOCK_WORKSPACE_RECORDS[0]])
    with pytest.raises(Flow360ValueError, match="found 0.*'Home' \\(PRIVATE\\)"):
        Workspace.get_shared()


def test_get_shared_rejects_duplicates(monkeypatch):
    duplicate = dict(MOCK_WORKSPACE_RECORDS[1], id="shared-duplicate")
    _patch_records(monkeypatch, [MOCK_WORKSPACE_RECORDS[1], duplicate])
    with pytest.raises(Flow360ValueError, match="exactly one SHARED workspace, found 2"):
        Workspace.get_shared()


def test_resolve_parent_folder_passthrough():
    assert resolve_parent_folder(None, {}, "folder") is None

    folder = Folder(id="folder-0000000000000001")
    assert resolve_parent_folder(folder, {}, "folder") is folder


def test_resolve_parent_folder_from_workspace(mock_response):
    workspace = Workspace.get_shared()
    assert resolve_parent_folder(workspace, {}, "folder").id == SHARED_ROOT


def test_resolve_parent_folder_legacy_kwarg():
    folder = Folder(id="folder-0000000000000001")
    assert resolve_parent_folder(None, {"folder": folder}, "folder") is folder


def test_resolve_parent_folder_rejects_conflicting_input(mock_response):
    folder = Folder(id="folder-0000000000000001")
    with pytest.raises(Flow360ValueError, match="not both"):
        resolve_parent_folder(Workspace.get_shared(), {"folder": folder}, "folder")


def test_resolve_parent_folder_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        resolve_parent_folder(None, {"workspace": None}, "folder")


def test_folder_create_in_workspace(mock_response):
    workspace = Workspace.get_shared()
    draft = Folder.create("my-folder", parent_folder_or_workspace=workspace)
    assert draft._parent_folder.id == SHARED_ROOT

    parent = Folder(id="folder-0000000000000001")
    draft = Folder.create("my-folder", parent_folder_or_workspace=parent)
    assert draft._parent_folder is parent

    draft = Folder.create("my-folder", parent_folder=parent)
    assert draft._parent_folder is parent

    with pytest.raises(Flow360ValueError, match="not both"):
        Folder.create("my-folder", parent_folder_or_workspace=workspace, parent_folder=parent)

    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        Folder.create("my-folder", parent=parent)
