"""
Asset CLI commands.
"""

from __future__ import annotations

from dataclasses import dataclass

from flow360.cli.output import emit_json
from flow360.cli.resource_group import ResourceCommandSpec, make_resource_group
from flow360.cli.resource_state import get_resource_state_for_type
from flow360.component.simulation.web.project_tree import get_project_tree_parent_id


@dataclass(frozen=True)
class _AssetGroupConfig:
    command_name: str
    id_argument: str
    resource_type: str
    webapi_class_name: str
    help_text: str
    label: str


def _get_asset_webapi_class(class_name):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web import asset_webapi

    return getattr(asset_webapi, class_name)


def _serialize_asset_info(info):
    return {
        "id": info.get("id"),
        "name": info.get("name"),
        "project_id": info.get("projectId"),
        "parent_id": info.get("parentId"),
        "solver_version": info.get("solverVersion"),
        "status": info.get("status"),
        "tags": list(info.get("tags") or []),
        "type": info.get("type"),
        "created_at": info.get("createdAt"),
        "updated_at": info.get("updatedAt"),
    }


def _get_asset_info(webapi_cls, asset_id):
    # pylint: disable=import-outside-toplevel
    return webapi_cls(asset_id).get_info()


def _get_asset_simulation_params(webapi_cls, asset_id):
    # pylint: disable=import-outside-toplevel
    return webapi_cls(asset_id).get_simulation_params()


def _summarize_simulation_params(simulation_params):
    # pylint: disable=import-outside-toplevel
    from flow360.cli.simulation_summary import summarize_simulation

    return summarize_simulation(simulation_params)


def _emit_asset_summary(webapi_cls, asset_id):
    emit_json(
        {
            "id": asset_id,
            "summary": _summarize_simulation_params(
                _get_asset_simulation_params(webapi_cls, asset_id)
            ),
        }
    )


def _serialize_case_info(info):
    payload = _serialize_asset_info(info)
    payload["type"] = payload["type"] or "Case"
    payload["parent_id"] = get_project_tree_parent_id(info)
    payload["mesh_id"] = info.get("caseMeshId") or info.get("meshId") or info.get("parentId")
    return payload


def _serialize_info_for_type(resource_type, info):
    if resource_type == "Case":
        return _serialize_case_info(info)
    return _serialize_asset_info(info)


def _emit_asset_info(resource_type, webapi_class_name, asset_id):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    info = _get_asset_info(webapi_cls, asset_id)
    emit_json(_serialize_info_for_type(resource_type, info))


def _emit_asset_simulation_params(webapi_class_name, asset_id):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    emit_json({"simulation_params": _get_asset_simulation_params(webapi_cls, asset_id)})


def _emit_asset_state(resource_type, asset_id):
    emit_json(get_resource_state_for_type(resource_type, asset_id))


def _emit_asset_summary_for_spec(webapi_class_name, asset_id):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    _emit_asset_summary(webapi_cls, asset_id)


def _make_asset_group(config):
    return make_resource_group(
        ResourceCommandSpec(
            command_name=config.command_name,
            id_argument=config.id_argument,
            help_text=config.help_text,
            label=config.label,
            emit_info=lambda asset_id: _emit_asset_info(
                config.resource_type, config.webapi_class_name, asset_id
            ),
            emit_state=lambda asset_id: _emit_asset_state(config.resource_type, asset_id),
            emit_simulation_params=lambda asset_id: _emit_asset_simulation_params(
                config.webapi_class_name, asset_id
            ),
            emit_summary=lambda asset_id: _emit_asset_summary_for_spec(
                config.webapi_class_name, asset_id
            ),
        )
    )


geometry = _make_asset_group(
    _AssetGroupConfig(
        command_name="geometry",
        id_argument="geometry_id",
        resource_type="Geometry",
        webapi_class_name="GeometryWebApi",
        help_text="Inspect Flow360 geometries.",
        label="geometry",
    )
)
surface_mesh = _make_asset_group(
    _AssetGroupConfig(
        command_name="surface-mesh",
        id_argument="surface_mesh_id",
        resource_type="SurfaceMesh",
        webapi_class_name="SurfaceMeshWebApi",
        help_text="Inspect Flow360 surface meshes.",
        label="surface mesh",
    )
)
volume_mesh = _make_asset_group(
    _AssetGroupConfig(
        command_name="volume-mesh",
        id_argument="volume_mesh_id",
        resource_type="VolumeMesh",
        webapi_class_name="VolumeMeshWebApi",
        help_text="Inspect Flow360 volume meshes.",
        label="volume mesh",
    )
)
case = _make_asset_group(
    _AssetGroupConfig(
        command_name="case",
        id_argument="case_id",
        resource_type="Case",
        webapi_class_name="CaseWebApi",
        help_text="Inspect Flow360 cases.",
        label="case",
    )
)
