import json
import os
import re
from unittest.mock import MagicMock, patch

import pydantic as pd
import pytest
import unyt as u
from flow360_schema.models.entities.surface_entities import ImportedSurface

import flow360 as fl
from flow360 import log
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import Project, ProjectMeta, ProjectTree
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.web import draft as draft_module
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.examples import Cylinder3D
from flow360.exceptions import (
    Flow360ConfigurationError,
    Flow360RuntimeError,
    Flow360ValueError,
    Flow360WebError,
    Flow360WebTimeoutError,
)

log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_from_cloud(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    project.print_project_tree(is_horizontal=True, line_width=15)

    assert project.length_unit == 1 * fl.u.m
    assert isinstance(project._root_asset, fl.Geometry)
    assert len(project.get_surface_mesh_ids()) == 1
    assert len(project.get_volume_mesh_ids()) == 1
    assert len(project.get_case_ids()) == 2

    current_geometry_id = "geo-2877e124-96ff-473d-864b-11eec8648d42"
    current_surface_mesh_id = "sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a"
    current_volume_mesh_id = "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
    current_case_id = "case-84d4604e-f3cd-4c6b-8517-92a80a3346d3"

    assert project.geometry.id == current_geometry_id
    assert project.surface_mesh.id == current_surface_mesh_id
    assert project.volume_mesh.id == current_volume_mesh_id
    assert project.case.id == current_case_id

    assert project.geometry.params
    assert project.surface_mesh.params
    assert project.volume_mesh.params

    for case_id in project.get_case_ids():
        project.project_tree.remove_node(node_id=case_id)
    error_msg = "No Case is available in this project."
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=current_case_id)


def test_failed_root_asset_rejected_before_simulation_json(mock_id, monkeypatch):
    geometry = Geometry(mock_id)
    geometry._webapi._set_meta(
        GeometryMeta(
            **local_metadata_builder(
                id=mock_id,
                name="failed-geometry",
                cloud_path_prefix="--",
                project_id="prj-failed",
                status="error",
            )
        )
    )
    get_simulation_file = MagicMock()
    monkeypatch.setattr(geometry._webapi, "get", get_simulation_file)

    class ProjectApi:
        def __init__(self, *args, **kwargs):
            pass

        def get(self):
            return {"rootItemId": mock_id}

    monkeypatch.setattr("flow360.component.simulation.web.asset_base.RestApi", ProjectApi)
    monkeypatch.setattr(
        "flow360.component.simulation.web.asset_base.get_project_dependency_resource_metadata",
        lambda project_id, resource_type: [],
    )

    with pytest.raises(
        Flow360RuntimeError,
        match=f"Cannot load Geometry {mock_id} because its status is error.",
    ):
        Geometry._get_simulation_json(geometry)
    get_simulation_file.assert_not_called()


def test_simulation_json_downloaded_once_per_asset(mock_id, monkeypatch):
    from flow360_schema import __version__ as schema_version

    geometry = Geometry(mock_id)
    geometry._webapi._set_meta(
        GeometryMeta(
            **local_metadata_builder(
                id=mock_id,
                name="cached-geometry",
                cloud_path_prefix="--",
                project_id="prj-cached",
                status="processed",
            )
        )
    )
    payload = {
        "simulationJson": json.dumps(
            {
                "version": schema_version,
                "unit_system": {"name": "SI"},
                "private_attribute_asset_cache": {
                    "project_length_unit": {"value": 1.0, "display_unit": "m"}
                },
            }
        )
    }
    get_simulation_file = MagicMock(return_value=payload)
    monkeypatch.setattr(geometry._webapi, "get", get_simulation_file)

    class ProjectApi:
        def __init__(self, *args, **kwargs):
            pass

        def get(self):
            return {"rootItemId": "geo-someone-else"}

    monkeypatch.setattr("flow360.component.simulation.web.asset_base.RestApi", ProjectApi)
    monkeypatch.setattr(
        "flow360.component.simulation.web.asset_base.get_project_dependency_resource_metadata",
        lambda project_id, resource_type: [],
    )

    first = Geometry._get_simulation_json(geometry)
    Geometry._get_simulation_json(geometry, clean_front_end_keys=True)
    assert get_simulation_file.call_count == 1

    # Each call re-parses the cached payload: mutations must not leak across reads.
    first["poisoned"] = True
    assert "poisoned" not in Geometry._get_simulation_json(geometry)
    assert get_simulation_file.call_count == 1

    # _load_root_length_unit reads through the same cache: still no extra download.
    project = _make_local_project()
    project._root_asset = geometry
    project._load_root_length_unit()
    assert project.length_unit == 1 * fl.u.m
    assert get_simulation_file.call_count == 1


def test_from_geometry_passes_workflow(monkeypatch):
    Cylinder3D.get_files()
    captured = {}

    class _MockDraft:
        def submit(self, run_async=False, description=""):
            assert run_async is True
            return MagicMock(project_id="prj-test-project-id")

    def _mock_from_file(
        file_names,
        project_name=None,
        solver_version=None,
        length_unit="m",
        tags=None,
        folder=None,
        workflow="standard",
        cad_importer_version=None,
    ):
        captured["file_names"] = file_names
        captured["project_name"] = project_name
        captured["solver_version"] = solver_version
        captured["length_unit"] = length_unit
        captured["workflow"] = workflow
        captured["cad_importer_version"] = cad_importer_version
        return _MockDraft()

    monkeypatch.setattr("flow360.component.project.Geometry.from_file", _mock_from_file)

    project_id = fl.Project.from_geometry(
        Cylinder3D.geometry,
        name="catalyst-project",
        solver_version="release-25.9",
        length_unit="cm",
        run_async=True,
        workflow="catalyst",
    )

    assert project_id == "prj-test-project-id"
    assert captured["file_names"] == Cylinder3D.geometry
    assert captured["project_name"] == "catalyst-project"
    assert captured["solver_version"] == "release-25.9"
    assert captured["length_unit"] == "cm"
    assert captured["workflow"] == "catalyst"


def _fake_geometry_api_response(geo_id: str = "geo-test-0001", prj_id: str = "prj-test-payload"):
    return {
        "id": geo_id,
        "name": "test-geo",
        "userId": "user-test",
        "status": "uploaded",
        "projectId": prj_id,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "deleted": False,
        "tags": [],
    }


def _mock_upload_files(self, *args, **kwargs):
    geo = MagicMock()
    geo.short_description.return_value = "test-geo (geo-test)"
    geo.id = "geo-test-0001"
    geo.project_id = "prj-test-payload"
    return geo


def test_catalyst_workflow_reaches_api_payload(monkeypatch):
    Cylinder3D.get_files()
    captured_payload: dict = {}

    class _FakeRestApi:
        def __init__(self, endpoint, **kwargs):
            pass

        def post(self, json_body):
            captured_payload.update(json_body)
            return _fake_geometry_api_response()

    monkeypatch.setattr("flow360.component.geometry.RestApi", _FakeRestApi)
    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "flow360.component.geometry.GeometryDraft._upload_files", _mock_upload_files
    )

    draft = Geometry.from_file(
        file_names=Cylinder3D.geometry,
        project_name="payload-test",
        solver_version="release-25.9",
        length_unit="cm",
        workflow="catalyst",
    )

    assert draft.workflow == "catalyst"
    draft.submit(run_async=True)

    assert (
        captured_payload.get("useCatalyst") is True
    ), f"Expected Catalyst workflow to set useCatalyst=true, got: {captured_payload}"
    assert set(captured_payload) >= {"useCatalyst"}


def test_standard_workflow_is_default(monkeypatch):
    Cylinder3D.get_files()
    captured_payload: dict = {}

    class _FakeRestApi:
        def __init__(self, endpoint, **kwargs):
            pass

        def post(self, json_body):
            captured_payload.update(json_body)
            return _fake_geometry_api_response(geo_id="geo-test-0002", prj_id="prj-test-default")

    monkeypatch.setattr("flow360.component.geometry.RestApi", _FakeRestApi)
    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "flow360.component.geometry.GeometryDraft._upload_files", _mock_upload_files
    )

    draft = Geometry.from_file(
        file_names=Cylinder3D.geometry,
        project_name="default-test",
        solver_version="release-25.9",
        length_unit="cm",
    )

    assert draft.workflow == "standard"
    draft.submit(run_async=True)

    assert (
        captured_payload.get("useCatalyst") is False
    ), f"Expected standard workflow to keep useCatalyst=false, got: {captured_payload}"
    assert set(captured_payload) >= {"useCatalyst"}


def test_root_asset_entity_change_reflection(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    geo = project.geometry
    geo["wing"].private_attribute_color = "red"

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            outputs=[fl.SurfaceOutput(surfaces=geo["*"], output_fields=["Cp"])],
        )
    params = set_up_params_for_uploading(
        params=params,
        root_asset=project._root_asset,
        length_unit=project.length_unit,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )

    assert (
        params.private_attribute_asset_cache.project_entity_info.grouped_faces[0][
            0
        ].private_attribute_color
        == "red"
    )

    # VolumeMesh: verify zone center/axis modifications flow through set_up_params_for_uploading
    # Simulate the customer scenario where self.volume is a DIFFERENT Python object than project._root_asset
    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")
    separate_vm = VolumeMeshV2.from_cloud(id="vm-bff35714-41b1-4251-ac74-46a40b95a330")
    assert separate_vm is not project_vm._root_asset

    zone = separate_vm["fluid"]
    zone.center = (1.2, 2.3, 3.4) * u.m
    zone.axis = (0, 1, 0)

    with fl.SI_unit_system:
        vm_params = fl.SimulationParams(
            models=[
                fl.Rotation(
                    name="testRotation",
                    volumes=[zone],
                    spec=fl.AngularVelocity(100 * fl.u.rpm),
                ),
            ],
        )
    vm_params = set_up_params_for_uploading(
        params=vm_params,
        root_asset=project_vm._root_asset,
        length_unit=project_vm.length_unit,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )

    entity_info = vm_params.private_attribute_asset_cache.project_entity_info
    fluid_zone = next(z for z in entity_info.zones if z.name == "fluid")
    assert all(fluid_zone.center == [1.2, 2.3, 3.4] * u.m)
    assert fluid_zone.axis == (0, 1, 0)


def test_get_asset_with_id(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    print(f"The correct asset_id: {project.case.id}")

    query_id = "  case-84d4604e-f3cd-4c6b-8517-92a80a3346d3   "
    assert project.get_case(asset_id=query_id)

    query_id = "=case-84d4604e-f3cd-4c6b-8517-92a80a3346d3&"
    assert project.get_case(asset_id=query_id)

    query_id = "case"
    error_msg = (
        r"The supplied ID \(" + query_id + r"\) does not have a proper surffix-ID structure."
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "sm-123456"
    error_msg = r"The input asset ID \(" + query_id + r"\) is not a Case ID."
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "case-b0"
    error_msg = (
        r"The input asset ID \(" + query_id + r"\) is too short to retrieve the correct asset."
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "case-1234567-abcd"
    error_msg = (
        r"This asset does not exist in this project. Please check the input asset ID \("
        + query_id
        + r"\)"
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)


@pytest.mark.usefixtures("s3_download_override")
def test_run(mock_response, capsys):
    capsys.readouterr()
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    parent_case = project.get_case("case-69b8c249")
    case = project.case  # case-84d4604e-f
    params = case.params

    warning_msg = "The VolumeMesh that matches the input already exists in project. No new VolumeMesh will be generated."
    project.generate_volume_mesh(params=params)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text

    warning_msg = (
        "The Case that matches the input already exists in project. No new Case will be generated."
    )
    project.run_case(params=params, fork_from=parent_case)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text

    error_msg = r"Input should be 'SurfaceMesh', 'VolumeMesh' or 'Case'"
    with pytest.raises(pd.ValidationError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Geometry")

    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(Case\) "
        r"cannot be later than 'up_to' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project.generate_volume_mesh(params=params, start_from="Case")

    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")
    params = project_vm.case.params

    error_msg = (
        r"Invalid force creation configuration: 'start_from' \(SurfaceMesh\) "
        r"must be later than 'source_item_type' \(VolumeMesh\)."
    )
    with pytest.raises(ValueError, match=error_msg):
        project_vm.run_case(params=params, start_from="SurfaceMesh")


@pytest.mark.usefixtures("s3_download_override")
def test_uploaded_params_are_the_raw_set_up_output(mock_response, monkeypatch):
    # The uploaded document must be the raw params (set_up output), exactly as the
    # WebUI submits: validator products are derived by the pipeline's own validation
    # at translate time, with the target solver version's schema.
    import flow360.component.project as project_module
    from flow360.component.simulation.web.draft import Draft

    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params
    params.reference_geometry.area = (
        2 * params.reference_geometry.area
    )  # force a genuinely new case

    seen = {}
    real_set_up = project_module.set_up_params_for_uploading

    def capture_set_up(**kwargs):
        seen["set_up"] = real_set_up(**kwargs)
        return seen["set_up"]

    # A sentinel outside the submit path's own error families, so it is not mapped to a
    # submission failure on its way out.
    class _StopAfterCapture(Exception):
        pass

    def capture_upload(self, uploaded_params):
        seen["uploaded"] = uploaded_params
        raise _StopAfterCapture

    monkeypatch.setattr(project_module, "set_up_params_for_uploading", capture_set_up)
    monkeypatch.setattr(Draft, "update_simulation_params", capture_upload)

    with pytest.raises(_StopAfterCapture):
        project.run_case(params=params, name="raw-upload-identity-probe")

    assert seen["uploaded"] is seen["set_up"]


@pytest.mark.usefixtures("s3_download_override")
def test_rejected_draft_is_deleted(mock_response, monkeypatch):
    # A draft the remote validation rejected cannot run; leaving it behind would give a
    # user one dead draft per params iteration.
    from flow360.component.simulation.web.draft import Draft

    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    deleted = []
    monkeypatch.setattr(
        Draft,
        "validate_simulation_json",
        lambda self, **_kwargs: ([{"loc": ["meshing"], "msg": "nope", "type": "missing"}], []),
    )
    monkeypatch.setattr(Draft, "delete_draft", lambda self: deleted.append(self.id))

    with pytest.raises(ValueError, match="Submission terminated due to validation error"):
        project.run_case(params=params, name="rejected-draft-cleanup")

    assert len(deleted) == 1


@pytest.mark.usefixtures("s3_download_override")
def test_draft_is_deleted_when_validation_cannot_be_reached(mock_response, monkeypatch):
    # An unreachable validation service is still "cannot submit", so the draft must not
    # survive either; and the failure has to honour raise_on_error like every other web
    # failure in this path.
    from flow360.component.simulation.web.draft import Draft

    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    deleted = []

    def _unreachable(self, **_kwargs):
        raise Flow360WebError("Draft validation did not return a verdict: {}")

    monkeypatch.setattr(Draft, "validate_simulation_json", _unreachable)
    monkeypatch.setattr(Draft, "delete_draft", lambda self: deleted.append(self.id))

    with pytest.raises(ValueError, match="Submission terminated due to error"):
        project.run_case(params=params, name="unreachable-validation")
    assert len(deleted) == 1

    assert project.run_case(params=params, name="unreachable-soft", raise_on_error=False) is None
    assert len(deleted) == 2


def _submit_with_lost_acknowledgement(
    project, params, monkeypatch, *, document_length, draft_gone, lookups, **run_kwargs
):
    """Submit a case whose run request the gateway abandons, and script what follows.

    ``document_length`` stands in for the uploaded document's size and ``draft_gone`` for
    whether the cloud has already deleted the draft — the two inputs the recovery reads.
    Each draft lookup lands in ``lookups``, and the deadline is now so the unresolved case
    resolves in one poll instead of five real minutes.
    """
    from flow360.component.simulation.web.draft import Draft

    def _timeout(self, *_args, **_kwargs):
        self._uploaded_simulation_json_length = document_length
        raise Flow360WebTimeoutError("Web v2/drafts/x/run: Timed out: 504")

    def _exists(self):
        lookups.append(self.id)
        return not draft_gone

    monkeypatch.setattr(Draft, "run_up_to_target_asset", _timeout)
    monkeypatch.setattr(Draft, "exists_in_cloud", _exists)
    monkeypatch.setattr(draft_module, "_RUN_CONFIRMATION_TIMEOUT_SECONDS", 0)
    return project.run_case(params=params, name="lost-acknowledgement", **run_kwargs)


@pytest.mark.usefixtures("s3_download_override")
def test_lost_run_acknowledgement_resolves_to_the_newest_case(mock_response, monkeypatch, capsys):
    # A run request the gateway abandons is not a failed run: the cloud deletes the draft
    # only after creating the resource, so a vanished draft means it went through, and the
    # newest case of the project is the closest identification available from outside.
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    capsys.readouterr()
    case = _submit_with_lost_acknowledgement(
        project,
        params,
        monkeypatch,
        document_length=60 * 1024 * 1024,
        draft_gone=True,
        lookups=[],
    )

    # Newest by creation time in the project, not the oldest and not the source case.
    assert case.id == "case-84d4604e-f3cd-4c6b-8517-92a80a3346d3"
    reported = " ".join(capsys.readouterr().out.split())
    assert "Returning the newest Case in this project" in reported
    # The gateway's own message, verbatim — a bare "504" would also match the hex of a
    # randomly generated entity id in the debug dump.
    assert "Timed out: 504" not in reported


@pytest.mark.usefixtures("s3_download_override")
def test_lost_run_acknowledgement_stays_unresolved_when_the_draft_remains(
    mock_response, monkeypatch, capsys
):
    # Still there at the deadline means unresolved, not failed — and the gateway status
    # stays out of what the user is told, at every level.
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    lookups = []
    capsys.readouterr()
    with pytest.raises(ValueError, match="did not confirm which resource"):
        _submit_with_lost_acknowledgement(
            project,
            params,
            monkeypatch,
            document_length=60 * 1024 * 1024,
            draft_gone=False,
            lookups=lookups,
        )

    assert lookups  # the draft was actually asked about
    reported = " ".join(capsys.readouterr().out.split())
    # The gateway's own message, verbatim — a bare "504" would also match the hex of a
    # randomly generated entity id in the debug dump.
    assert "Timed out: 504" not in reported

    # And the same outcome without raising, like every other submission failure here.
    assert (
        _submit_with_lost_acknowledgement(
            project,
            params,
            monkeypatch,
            document_length=60 * 1024 * 1024,
            draft_gone=False,
            lookups=lookups,
            raise_on_error=False,
        )
        is None
    )


@pytest.mark.usefixtures("s3_download_override")
def test_small_document_is_not_waited_on(mock_response, monkeypatch):
    # Only documents large enough to outrun the gateway earn the wait; a smaller one timed
    # out for some other reason, and holding the script buys the user nothing.
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    lookups = []
    with pytest.raises(ValueError, match="did not confirm which resource"):
        _submit_with_lost_acknowledgement(
            project, params, monkeypatch, document_length=1024, draft_gone=False, lookups=lookups
        )

    assert lookups == []


@pytest.mark.usefixtures("s3_download_override")
def test_unreadable_project_does_not_escape_the_recovery(mock_response, monkeypatch, capsys):
    # The run is confirmed but the project cannot be read, so the resource cannot be named.
    # That is one more way of not finding it — not a raw error thrown at a caller that
    # asked to be handed a value instead.
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    def _unreadable(*_args, **_kwargs):
        raise Flow360WebError("Web tree: Unexpected response error: 500")

    monkeypatch.setattr(project._project_webapi, "get", _unreadable)

    capsys.readouterr()
    assert (
        _submit_with_lost_acknowledgement(
            project,
            params,
            monkeypatch,
            document_length=60 * 1024 * 1024,
            draft_gone=True,
            lookups=[],
            raise_on_error=False,
        )
        is None
    )

    # And it says which of the two happened: the submission did go through.
    assert "The submission went through" in " ".join(capsys.readouterr().out.split())

    with pytest.raises(ValueError, match="did not confirm which resource"):
        _submit_with_lost_acknowledgement(
            project,
            params,
            monkeypatch,
            document_length=60 * 1024 * 1024,
            draft_gone=True,
            lookups=[],
        )


@pytest.mark.usefixtures("s3_download_override")
def test_uploaded_document_size_is_measured_uncompressed(mock_response, monkeypatch):
    # The size the recovery gates on has to be the document itself, the measure the WebUI
    # uses too — not the compressed body that actually goes on the wire.
    from flow360.component.simulation.web.draft import Draft

    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")

    posted = {}
    monkeypatch.setattr(Draft, "post", lambda self, **kwargs: posted.update(kwargs))

    draft = Draft(draft_id="dft-84b20880-937d-4ef2-983b-7f75089f6dd6")
    draft.update_simulation_params(project.case.params)

    assert draft._uploaded_simulation_json_length == len(posted["json"]["data"])
    assert draft._uploaded_simulation_json_length > 0


@pytest.mark.usefixtures("s3_download_override")
def test_run_warns_on_minor_solver_version(mock_response, capsys):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    params = project.case.params

    capsys.readouterr()
    project.run_case(params=params, solver_version="release-25.8.7")
    captured_text = " ".join(capsys.readouterr().out.split())
    assert "You are running release-25.8.7, which is a minor solver release" in captured_text

    project.run_case(params=params, solver_version="release-25.8")
    captured_text = " ".join(capsys.readouterr().out.split())
    assert "minor solver release" not in captured_text


@pytest.mark.usefixtures("s3_download_override")
def test_fork_cannot_change_release_series(mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    parent_case = project.get_case("case-69b8c249")
    params = project.case.params

    assert parent_case.solver_version == "release-24.11"

    error_msg = r"Cannot change solver version from parent to child"
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.run_case(params=params, fork_from=parent_case, solver_version="release-25.10")

    # a different patch of the parent's release series is allowed
    project.run_case(params=params, fork_from=parent_case, solver_version="release-24.11.2")


def test_conflicting_entity_grouping_tags(mock_response, capsys):
    with open(
        os.path.join(os.path.dirname(__file__), "data", "simulation_by_face_id.json"), "r"
    ) as f:
        params_as_dict = json.load(f)

    # Stamp a LEGACY version: the file is legacy-shaped (inline entities) and must
    # go through the updater's inline-to-reference conversion.
    params_as_dict["version"] = "25.10.0"
    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level=None,
    )

    assert params is not None, print(">>> errors:", errors)

    assert params.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceId"
    assert params.private_attribute_asset_cache.project_entity_info.edge_group_tag == "edgeId"
    assert params.private_attribute_asset_cache.project_entity_info.body_group_tag == "groupByFile"

    geo = Geometry.from_local_storage(
        geometry_id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "geometry_grouped_by_file"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
                name="TEST",
                cloud_path_prefix="/",
                status="processed",
            )
        ),
    )

    assert geo.face_group_tag == "groupByBodyId"

    geo.internal_registry = geo._entity_info.get_persistent_entity_registry(geo.internal_registry)

    new_params = set_up_params_for_uploading(
        geo, 1 * u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )

    assert new_params.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceId"
    assert new_params.private_attribute_asset_cache.project_entity_info.edge_group_tag == "edgeId"
    assert (
        new_params.private_attribute_asset_cache.project_entity_info.body_group_tag == "groupByFile"
    )

    geo.group_faces_by_tag("allInOne")

    # Conflicting grouping in `geo`(allInOne) and in params(faceId)
    new_params = set_up_params_for_uploading(
        geo, 1 * u.m, params, use_beta_mesher=False, use_geometry_AI=False
    )
    assert new_params.private_attribute_asset_cache.project_entity_info.face_group_tag == "faceId"

    with pytest.raises(
        Flow360ConfigurationError,
        match=re.escape(
            "Multiple entity (Surface) grouping tags found in the SimulationParams "
            "(['faceId', 'groupByBodyId'])."
        ),
    ):
        geo.group_faces_by_tag("groupByBodyId")
        params_as_dict["outputs"][0]["entities"]["stored_entities"][
            0
        ].private_attribute_tag_key = "groupByBodyId"
        params, _, _ = validate_model(
            params_as_dict=params_as_dict,
            validated_by=ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level=None,
        )
        new_params = set_up_params_for_uploading(
            geo, 1 * u.m, params, use_beta_mesher=False, use_geometry_AI=False
        )


@pytest.mark.usefixtures("s3_download_override")
def test_interpolate_to_mesh_uses_vm_project_root_asset(mock_response):
    """
    Test that when run_case is called with interpolate_to_mesh,
    the params are set up using the root asset from the project
    that provides the interpolate_to_mesh, not from the current project.
    """

    # Load the geometry-based project (prj-41d2333b-85fd-4bed-ae13-15dcb6da519e)
    project_geo = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")

    # Load the volume mesh-based project (prj-99cc6f96-15d3-4170-973c-a0cced6bf36b)
    project_vm = fl.Project.from_cloud(project_id="prj-99cc6f96-15d3-4170-973c-a0cced6bf36b")

    # Get parent case from geometry project (case-69b8c249)
    parent_case = project_geo.get_case("case-69b8c249")

    # Get params from the VM project's case (case-f7480884) since the params should be defined based on this VM
    params = project_vm.case.params

    # Get volume mesh from the VM project to use as interpolate_to_mesh
    vm_for_interpolation = project_vm.volume_mesh

    # Track which root_asset is passed to set_up_params_for_uploading
    captured_root_asset = None

    def mock_set_up_params_for_uploading(
        params, root_asset, length_unit, use_beta_mesher, use_geometry_AI
    ):
        nonlocal captured_root_asset
        captured_root_asset = root_asset
        return set_up_params_for_uploading(
            root_asset=root_asset,
            length_unit=length_unit,
            params=params,
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=use_geometry_AI,
        )

    with patch(
        "flow360.component.project.set_up_params_for_uploading", mock_set_up_params_for_uploading
    ):
        # The run_case should fail at validation but we just need to verify the root_asset
        # is from the VM project, not the geometry project
        try:
            project_geo.run_case(
                params=params,
                fork_from=parent_case,
                interpolate_to_mesh=vm_for_interpolation,
                raise_on_error=False,
            )
        except Exception:
            # We expect this to fail because the mock doesn't fully support the run flow
            pass

    # Verify that the captured root_asset is from the VM project (VolumeMeshV2)
    # and NOT from the geometry project (Geometry)
    assert captured_root_asset is not None, "set_up_params_for_uploading was not called"

    # The root asset should be a VolumeMeshV2 from the VM project, not a Geometry
    assert isinstance(
        captured_root_asset, VolumeMeshV2
    ), f"Expected VolumeMeshV2 from VM project, got {type(captured_root_asset).__name__}"
    assert not isinstance(
        captured_root_asset, Geometry
    ), "Root asset should NOT be Geometry from current project"

    # Verify the root asset's project_id matches the VM project
    assert (
        captured_root_asset.project_id == project_vm.id
    ), f"Root asset should be from VM project {project_vm.id}, got {captured_root_asset.project_id}"


def _build_params_with_imported_surfaces(imported_surfaces):
    """Build a minimal SimulationParams with imported_surfaces set in asset cache."""
    with fl.SI_unit_system:
        params = fl.SimulationParams()
    params.private_attribute_asset_cache._force_set_attr("imported_surfaces", imported_surfaces)
    return params


class TestImportedSurfaceRequiresDraftContext:
    """Validate that using ImportedSurface without an active DraftContext raises an error."""

    def _call_run_guard(self, project, params):
        """
        Call project._run() with enough mocking to reach the guard clause.
        Everything before the guard (set_up_params_for_uploading, Draft.create)
        is patched out.
        """
        mock_draft = MagicMock()
        mock_draft.submit.return_value = mock_draft
        mock_draft.validate_simulation_json.return_value = (None, [])

        with (
            patch.object(project, "_root_asset", create=True),
            patch(
                "flow360.component.project.set_up_params_for_uploading",
                return_value=params,
            ),
            patch(
                "flow360.component.simulation.web.draft.Draft.create",
                return_value=mock_draft,
            ),
        ):
            project._run(
                params=params,
                target=MagicMock(_cloud_resource_type_name="Case"),
                draft_name="test",
                fork_from=None,
                interpolate_to_mesh=None,
                run_async=True,
                solver_version=None,
                use_beta_mesher=False,
                use_geometry_AI=False,
                raise_on_error=True,
                tags=None,
                draft_only=True,
            )

    def test_imported_surface_without_draft_context_raises(self, mock_id, mock_response):
        project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
        params = _build_params_with_imported_surfaces(
            [
                ImportedSurface(
                    name="test_surface",
                    private_attribute_id="test_surface_defaultBody",
                    surface_mesh_id="sm-123",
                )
            ]
        )

        with pytest.raises(
            Flow360ValueError, match="ImportedSurface feature requires an active DraftContext"
        ):
            self._call_run_guard(project, params)

    def test_no_imported_surface_without_draft_context_passes(self, mock_id, mock_response):
        project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
        params = _build_params_with_imported_surfaces([])

        # Should not raise - no imported surfaces, no draft context is fine
        self._call_run_guard(project, params)

    def test_imported_surface_with_draft_context_passes(
        self, mock_id, mock_response, mock_surface_mesh
    ):
        project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
        params = _build_params_with_imported_surfaces(
            [
                ImportedSurface(
                    name="test_surface",
                    private_attribute_id="test_surface_defaultBody",
                    surface_mesh_id="sm-123",
                )
            ]
        )

        from flow360.component.project import create_draft

        with create_draft(new_run_from=mock_surface_mesh):
            # Should not raise - draft context is active
            self._call_run_guard(project, params)


def test_public_create_methods_forward_description(monkeypatch):
    captured = []

    def _capture(cls, **kwargs):
        captured.append(kwargs.get("description"))
        return "prj-forwarded"

    monkeypatch.setattr(fl.Project, "_create_project_from_files", classmethod(_capture))

    fl.Project.from_geometry("part.csm", description="d-geo")
    fl.Project.from_surface_mesh("mesh.ugrid", description="d-sm")
    fl.Project.from_volume_mesh("mesh.cgns", description="d-vm")

    assert captured == ["d-geo", "d-sm", "d-vm"]


def test_create_helper_passes_description_to_submit(monkeypatch):
    Cylinder3D.get_files()
    captured = {}

    class _MockDraft:
        def submit(self, run_async=False, description=""):
            captured["description"] = description
            return MagicMock(project_id="prj-submit")

    def _mock_from_file(
        file_names,
        project_name=None,
        solver_version=None,
        length_unit="m",
        tags=None,
        folder=None,
        workflow="standard",
        cad_importer_version=None,
    ):
        return _MockDraft()

    monkeypatch.setattr("flow360.component.project.Geometry.from_file", _mock_from_file)

    fl.Project.from_geometry(
        Cylinder3D.geometry,
        solver_version="release-25.9",
        length_unit="cm",
        run_async=True,
        description="submitted description",
    )

    assert captured["description"] == "submitted description"


def test_volume_mesh_clone_forwards_description(monkeypatch):
    captured = {}

    class _FakeRestApi:
        def __init__(self, endpoint, **kwargs):
            pass

        def post(self, json_body, method=None):
            captured.update(json_body)
            return {"projectId": "prj-clone"}

    monkeypatch.setattr("flow360.component.project.RestApi", _FakeRestApi)
    monkeypatch.setattr(
        "flow360.component.project.VolumeMeshMetaV2",
        lambda **resp: MagicMock(project_id=resp.get("projectId")),
    )

    project_id = fl.Project._create_project_from_volume_mesh_clone(
        volume_mesh=MagicMock(id="vm-clone-source"),
        name="clone-proj",
        solver_version="release-25.9",
        length_unit="m",
        tags=[],
        run_async=True,
        folder=None,
        description="clone description",
    )

    assert project_id == "prj-clone"
    assert captured.get("description") == "clone description"


def _make_local_project():
    metadata = ProjectMeta(
        userId="user-test",
        id="prj-test",
        name="orig-name",
        rootItemId="geo-test",
        rootItemType="Geometry",
    )
    return Project(metadata=metadata, project_tree=ProjectTree(), solver_version="release-25.9")


class _FakeProjectApi:
    def __init__(self, refreshed):
        self.patched = []
        self._refreshed = refreshed

    def patch(self, json_body):
        self.patched.append(json_body)

    def get(self):
        return self._refreshed


def _attach_fake_api(project, **refreshed_overrides):
    refreshed = {
        "userId": "user-test",
        "id": "prj-test",
        "name": "orig-name",
        "rootItemId": "geo-test",
        "rootItemType": "Geometry",
    }
    refreshed.update(refreshed_overrides)
    api = _FakeProjectApi(refreshed)
    project._project_webapi = api
    return api


def test_update_sends_only_provided_fields():
    project = _make_local_project()
    api = _attach_fake_api(project)

    project.update(description="hello", tags=["a", "b"])

    assert api.patched == [{"description": "hello", "tags": ["a", "b"]}]


def test_update_forwards_none_and_empty_string_verbatim():
    project = _make_local_project()
    api = _attach_fake_api(project)

    project.update(description=None)
    project.update(description="")

    assert api.patched == [{"description": None}, {"description": ""}]


def test_update_with_no_args_still_issues_one_patch():
    project = _make_local_project()
    api = _attach_fake_api(project)

    project.update()

    assert api.patched == [{}]


def test_update_refreshes_local_metadata():
    project = _make_local_project()
    _attach_fake_api(project, name="new-name", description="new-desc")

    project.update(name="new-name", description="new-desc")

    assert project.metadata.name == "new-name"
    assert project.description == "new-desc"


def test_rename_delegates_to_update_and_refreshes():
    project = _make_local_project()
    api = _attach_fake_api(project, name="renamed")

    project.rename("renamed")

    assert api.patched == [{"name": "renamed"}]
    assert project.metadata.name == "renamed"


def test_metadata_properties_are_read_only():
    project = _make_local_project()

    for field in ("name", "description", "tags"):
        with pytest.raises(AttributeError, match="is read-only"):
            setattr(project, field, "value")
