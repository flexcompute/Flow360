import json
from pathlib import Path
from types import SimpleNamespace

import flow360 as fl
import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import (
    set_up_params_for_uploading,
    validate_params_with_context,
)
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.meshing_specs import MeshingDefaults
from flow360.component.simulation.services_utils import (
    strip_implicit_edge_split_layers_inplace,
)
from flow360.component.simulation.web.draft import Draft


def _build_dummy_params(defaults: MeshingDefaults):
    return SimpleNamespace(meshing=SimpleNamespace(defaults=defaults))


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


def _build_geometry_grouped_submission_params():
    geometry = Geometry.from_local_storage(
        geometry_id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=str(
            Path(__file__).resolve().parents[1] / "data" / "geometry_grouped_by_file"
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

    with fl.SI_unit_system:
        params = fl.SimulationParams()

    return set_up_params_for_uploading(
        root_asset=geometry,
        length_unit=1 * u.m,
        params=params,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )


def test_strip_implicit_edge_split_layers_removes_default_injected_field():
    with fl.SI_unit_system:
        defaults = MeshingDefaults(boundary_layer_first_layer_thickness=1e-4)

    params = _build_dummy_params(defaults)
    params_dict = {
        "meshing": {"defaults": {"edge_split_layers": 1, "surface_edge_growth_rate": 1.2}}
    }

    out = strip_implicit_edge_split_layers_inplace(params, params_dict)

    assert out is params_dict
    assert "edge_split_layers" not in out["meshing"]["defaults"]
    assert out["meshing"]["defaults"]["surface_edge_growth_rate"] == 1.2


def test_strip_implicit_edge_split_layers_keeps_explicit_default_value():
    with fl.SI_unit_system:
        defaults = MeshingDefaults(boundary_layer_first_layer_thickness=1e-4, edge_split_layers=1)

    params = _build_dummy_params(defaults)
    params_dict = {
        "meshing": {"defaults": {"edge_split_layers": 1, "surface_edge_growth_rate": 1.2}}
    }

    out = strip_implicit_edge_split_layers_inplace(params, params_dict)

    assert out["meshing"]["defaults"]["edge_split_layers"] == 1


def test_strip_implicit_edge_split_layers_keeps_explicit_non_default_value():
    with fl.SI_unit_system:
        defaults = MeshingDefaults(boundary_layer_first_layer_thickness=1e-4, edge_split_layers=3)

    params = _build_dummy_params(defaults)
    params_dict = {
        "meshing": {"defaults": {"edge_split_layers": 3, "surface_edge_growth_rate": 1.2}}
    }

    out = strip_implicit_edge_split_layers_inplace(params, params_dict)

    assert out["meshing"]["defaults"]["edge_split_layers"] == 3


def test_validate_params_with_context_no_warning_for_implicit_default():
    params = _build_simulation_params()

    _, errors, warnings = validate_params_with_context(params, "Geometry", "VolumeMesh")

    assert errors is None
    assert warnings == []


def test_validate_params_with_context_warning_for_explicit_default_value():
    params = _build_simulation_params(edge_split_layers=1)

    _, errors, warnings = validate_params_with_context(params, "Geometry", "VolumeMesh")

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


def test_draft_upload_payload_trims_geometry_entity_info_to_active_groupings(monkeypatch):
    params = _build_geometry_grouped_submission_params()
    uploaded_payload = {}

    def _capture_post(self, *, json=None, method=None, **_kwargs):
        uploaded_payload["json"] = json
        uploaded_payload["method"] = method
        return {}

    monkeypatch.setattr(Draft, "post", _capture_post, raising=True)

    original_entity_info = params.private_attribute_asset_cache.project_entity_info
    assert original_entity_info is not None
    assert original_entity_info.face_attribute_names == ["groupByBodyId", "faceId", "allInOne"]
    assert original_entity_info.body_attribute_names == ["bodyId", "groupByFile"]

    Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)

    assert uploaded_payload["method"] == "simulation/file"
    uploaded_dict = json.loads(uploaded_payload["json"]["data"])
    uploaded_entity_info = uploaded_dict["private_attribute_asset_cache"]["project_entity_info"]

    assert uploaded_entity_info["face_attribute_names"] == ["groupByBodyId"]
    assert len(uploaded_entity_info["grouped_faces"]) == 1
    assert uploaded_entity_info["body_attribute_names"] == ["groupByFile"]
    assert len(uploaded_entity_info["grouped_bodies"]) == 1
    assert uploaded_entity_info["edge_attribute_names"] == ["edgeId"]
    assert len(uploaded_entity_info["grouped_edges"]) == 1

    assert original_entity_info.face_attribute_names == ["groupByBodyId", "faceId", "allInOne"]
    assert len(original_entity_info.grouped_faces) == 3
    assert original_entity_info.body_attribute_names == ["bodyId", "groupByFile"]
    assert len(original_entity_info.grouped_bodies) == 2
