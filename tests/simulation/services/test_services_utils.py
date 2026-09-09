import json

import flow360_schema.models.simulation.units as u
from flow360_schema.framework.param_utils import AssetCache

import flow360 as fl
from flow360.cloud.rest_api import RestApi
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.web.asset_webapi import DraftWebApi
from flow360.component.simulation.web.draft import Draft


def _build_simulation_params(*, edge_split_layers=None):
    defaults_kwargs = dict(
        boundary_layer_first_layer_thickness=1e-4,
        surface_max_edge_length=1e-2,
    )
    if edge_split_layers is not None:
        defaults_kwargs["edge_split_layers"] = edge_split_layers

    with fl.SI_unit_system:
        return fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(**defaults_kwargs),
                volume_zones=[fl.AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=False, project_length_unit=1 * u.m
            ),
        )


def _validate_uploaded_document(params, monkeypatch):
    """Validate what update_simulation_params actually uploads — the local proxy for
    what the remote draft validation sees."""
    uploaded_payload = {}

    def _capture_post(self, *, json=None, method=None, **_kwargs):
        uploaded_payload["json"] = json
        return {}

    monkeypatch.setattr(Draft, "post", _capture_post, raising=True)
    Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)
    uploaded_dict = json.loads(uploaded_payload["json"]["data"])
    return validate_model(
        params_as_dict=uploaded_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level=["SurfaceMesh", "VolumeMesh"],
    )


def test_uploaded_document_gets_no_warning_for_implicit_default(monkeypatch):
    params = _build_simulation_params()

    _, errors, warnings = _validate_uploaded_document(params, monkeypatch)

    assert errors is None
    assert warnings == []


def test_uploaded_document_warns_for_explicit_default_value(monkeypatch):
    params = _build_simulation_params(edge_split_layers=1)

    _, errors, warnings = _validate_uploaded_document(params, monkeypatch)

    assert errors is None
    assert len(warnings) == 1
    assert warnings[0]["msg"] == (
        "`edge_split_layers` is only supported by the beta mesher; this setting will be ignored."
    )


def test_draft_upload_payload_omits_implicit_default_edge_split_layers(monkeypatch):
    params = _build_simulation_params()
    uploaded_payload = {}

    def _capture_post(self, *, json=None, method=None, **_kwargs):
        uploaded_payload["json"] = json
        uploaded_payload["method"] = method
        return {}

    monkeypatch.setattr(Draft, "post", _capture_post, raising=True)
    Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)

    assert uploaded_payload["method"] == "simulation/file"
    uploaded_dict = json.loads(uploaded_payload["json"]["data"])
    assert "edge_split_layers" not in uploaded_dict["meshing"]["defaults"]


def test_draft_upload_payload_keeps_explicit_default_edge_split_layers(monkeypatch):
    params = _build_simulation_params(edge_split_layers=1)
    uploaded_payload = {}

    def _capture_post(self, *, json=None, method=None, **_kwargs):
        uploaded_payload["json"] = json
        uploaded_payload["method"] = method
        return {}

    monkeypatch.setattr(Draft, "post", _capture_post, raising=True)
    Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)

    assert uploaded_payload["method"] == "simulation/file"
    uploaded_dict = json.loads(uploaded_payload["json"]["data"])
    assert uploaded_dict["meshing"]["defaults"]["edge_split_layers"] == 1


def test_draft_upload_requests_gzip_above_five_mb(monkeypatch):
    """The simulation.json envelope is the one client body that can exceed the gzip
    threshold, so it must ask for compression explicitly."""
    params = _build_simulation_params()
    post_kwargs = {}

    def _capture_post(self, *, json=None, method=None, **kwargs):
        post_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr(Draft, "post", _capture_post, raising=True)
    Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)

    assert post_kwargs["compress_when_larger_than_mb"] == 5


def test_asset_draft_upload_requests_gzip_above_five_mb(monkeypatch):
    post_kwargs = {}

    def _capture_post(self, json, path=None, method=None, **kwargs):
        post_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr(RestApi, "post", _capture_post, raising=True)
    DraftWebApi("00000000-0000-0000-0000-000000000000").set_simulation_params({"version": "x"})

    assert post_kwargs["compress_when_larger_than_mb"] == 5
