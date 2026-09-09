from flow360.component.simulation.web import workspace_webapi
from flow360.component.simulation.web.workspace_webapi import WorkspaceWebApi


def test_workspace_list_records_accepts_bare_list(monkeypatch):
    monkeypatch.setattr(
        workspace_webapi.RestApi,
        "get",
        lambda self: [{"id": "private-abc", "rootFolderId": "ROOT.FLOW360"}],
    )

    assert WorkspaceWebApi.list_records() == [{"id": "private-abc", "rootFolderId": "ROOT.FLOW360"}]


def test_workspace_list_records_accepts_enveloped_data(monkeypatch):
    monkeypatch.setattr(
        workspace_webapi.RestApi,
        "get",
        lambda self: {"data": [{"id": "shared-abc", "rootFolderId": "ROOT.FLOW360.123"}]},
    )

    assert WorkspaceWebApi.list_records() == [
        {"id": "shared-abc", "rootFolderId": "ROOT.FLOW360.123"}
    ]


def test_workspace_get_workspace_id_for_root_folder(monkeypatch):
    monkeypatch.setattr(
        WorkspaceWebApi,
        "list_records",
        classmethod(
            lambda cls: [
                {"id": "shared-abc", "rootFolderId": "ROOT.FLOW360.123"},
                {"id": "private-abc", "rootFolderId": "ROOT.FLOW360"},
            ]
        ),
    )

    assert WorkspaceWebApi.get_workspace_id_for_root_folder("ROOT.FLOW360.123") == "shared-abc"
    assert WorkspaceWebApi.get_workspace_id_for_root_folder("ROOT.FLOW360") == "private-abc"
    assert WorkspaceWebApi.get_workspace_id_for_root_folder("ROOT.FLOW360.missing") is None
