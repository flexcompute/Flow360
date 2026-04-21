from flow360.component.project import Project, ProjectMeta, RootType


def test_from_cloud_lazy_does_not_fetch_until_requested(monkeypatch):
    calls = []
    project_id = "prj-12345678-1234-1234-1234-123456789abc"
    geometry_id = "geo-12345678-1234-1234-1234-123456789abc"

    def fake_get(self, path=None, method=None, json=None, params=None):
        calls.append(method)
        if method == "tree":
            return {
                "records": [
                    {
                        "createdAt": "2025-01-01T00:00:00Z",
                        "displayStatus": "completed",
                            "id": geometry_id,
                        "name": "Wing",
                        "parentCaseId": None,
                        "parentFolderId": "ROOT.FLOW360",
                        "parentId": None,
                        "parentItemName": None,
                        "parentItemProjectId": None,
                        "parentItemTye": None,
                        "postProcessStatus": None,
                        "postProcessedAt": None,
                            "projectId": project_id,
                        "requestStopAt": None,
                        "runSequence": "run-1",
                        "solverFinishAt": None,
                        "solverStartAt": None,
                        "solverVersion": "release-25.2",
                        "status": "processed",
                        "tags": None,
                        "type": "Geometry",
                        "updatedAt": "2025-01-01T00:00:01Z",
                        "userId": "user-1",
                        "viewed": False,
                        "visualizationStatus": None,
                        "visualizedAt": None,
                    }
                ]
            }
        return {
            "userId": "user-1",
            "id": project_id,
            "name": "Wing Study",
            "tags": ["demo"],
            "rootItemId": geometry_id,
            "rootItemType": "Geometry",
        }

    monkeypatch.setattr("flow360.component.project.RestApi.get", fake_get)

    project = Project.from_cloud(project_id, lazy_load=True)

    assert calls == []
    tree = project.get_project_tree()
    assert tree.root.asset_id == geometry_id
    assert calls == ["tree"]


def test_lazy_project_metadata_fetches_only_metadata(monkeypatch):
    calls = []
    project_id = "prj-12345678-1234-1234-1234-123456789abc"
    geometry_id = "geo-12345678-1234-1234-1234-123456789abc"

    def fake_get(self, path=None, method=None, json=None, params=None):
        calls.append(method)
        return {
            "userId": "user-1",
            "id": project_id,
            "name": "Wing Study",
            "tags": ["demo"],
            "rootItemId": geometry_id,
            "rootItemType": "Geometry",
        }

    monkeypatch.setattr("flow360.component.project.RestApi.get", fake_get)

    project = Project.from_cloud(project_id, lazy_load=True)
    meta = project.get_metadata()

    assert isinstance(meta, ProjectMeta)
    assert meta.root_item_type is RootType.GEOMETRY
    assert calls == [None]
