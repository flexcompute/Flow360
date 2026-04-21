import flow360.component.project as project_module

from flow360.component.project import Project


def test_project_from_geometry_passes_description_to_draft_submit(monkeypatch, tmp_path):
    geometry_file = tmp_path / "wing.csm"
    geometry_file.write_text("solid")
    calls = {}

    class FakeDraft:
        def submit(self, description="", run_async=False):
            calls["description"] = description
            calls["run_async"] = run_async
            return type("FakeRootAsset", (), {"project_id": "prj-123"})()

    monkeypatch.setattr(project_module.Geometry, "from_file", lambda *args, **kwargs: FakeDraft())

    project_id = Project.from_geometry(
        str(geometry_file),
        description="demo project",
        run_async=True,
    )

    assert project_id == "prj-123"
    assert calls == {"description": "demo project", "run_async": True}
