"""
Asset CLI commands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import click

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


def _rename_asset(webapi_cls, asset_id, name):
    return webapi_cls(asset_id).rename(name)


def _delete_asset(webapi_cls, asset_id):
    return webapi_cls(asset_id).delete()


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


def _emit_asset_rename(webapi_class_name, asset_id, name):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    _rename_asset(webapi_cls, asset_id, name)
    emit_json({"id": asset_id, "name": name})


def _emit_asset_delete(webapi_class_name, asset_id):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    _delete_asset(webapi_cls, asset_id)
    emit_json({"id": asset_id, "deleted": True})


def _emit_asset_summary_for_spec(webapi_class_name, asset_id):
    webapi_cls = _get_asset_webapi_class(webapi_class_name)
    _emit_asset_summary(webapi_cls, asset_id)


def _get_case_result_path(record):
    value = record.get("fileName")
    if not value:
        return None
    parts = value.split("/")
    if "results" in parts:
        return "/".join(parts[parts.index("results") :])
    return value


def _get_case_result_download_path(record):
    return record.get("fileName")


def _serialize_case_result(record):
    path = _get_case_result_path(record)
    return {
        "name": os.path.basename(path) if path else None,
        "path": path,
        "file_type": record.get("fileType"),
        "size_bytes": record.get("length"),
    }


def _list_case_results(case_id):
    webapi_cls = _get_asset_webapi_class("CaseWebApi")
    files = webapi_cls(case_id).list_files()
    result_files = [
        record for record in files if (_get_case_result_path(record) or "").startswith("results/")
    ]
    result_files.sort(key=lambda record: _get_case_result_path(record) or "")
    return result_files


def _resolve_case_result(case_id, result_ref):
    results = _list_case_results(case_id)
    if not results:
        raise click.ClickException(f"No result files are available for case {case_id}.")

    exact_matches = [
        record
        for record in results
        if result_ref in {record.get("fileName"), _get_case_result_path(record)}
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise click.ClickException(f"Multiple results matched '{result_ref}'. Use the full path.")

    basename_matches = [
        record
        for record in results
        if os.path.basename(_get_case_result_path(record) or "") == result_ref
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        matches = ", ".join(
            sorted(_get_case_result_path(record) or "" for record in basename_matches)
        )
        raise click.ClickException(
            f"Multiple results matched '{result_ref}'. Use one of: {matches}"
        )

    raise click.ClickException(f"Result '{result_ref}' was not found for case {case_id}.")


def _download_case_result(case_id, result_path, *, to_path=None, overwrite=False):
    webapi_cls = _get_asset_webapi_class("CaseWebApi")
    if to_path is None:
        return webapi_cls(case_id).download_file(result_path, overwrite=overwrite)
    return webapi_cls(case_id).download_file(
        result_path,
        to_file=to_path,
        overwrite=overwrite,
    )


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
            emit_rename=lambda asset_id, name: _emit_asset_rename(
                config.webapi_class_name, asset_id, name
            ),
            emit_delete=lambda asset_id: _emit_asset_delete(config.webapi_class_name, asset_id),
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


@case.group("results", help="Namespace for case result artifacts.")
def case_results():
    """Manage case result artifacts."""


@case_results.command("list", help="List case result artifacts.")
@click.argument("case_id")
def list_case_results(case_id):
    """List case result artifacts."""
    emit_json(
        {"records": [_serialize_case_result(record) for record in _list_case_results(case_id)]}
    )


@case_results.command("get", help="Download a case result artifact.")
@click.argument("case_id")
@click.argument("result_ref")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output file path. Defaults to the result basename.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing local file.")
def get_case_result(case_id, result_ref, output_path, overwrite):
    """Download a case result artifact."""
    record = _resolve_case_result(case_id, result_ref)
    result_path = _get_case_result_path(record)
    download_path = _get_case_result_download_path(record)
    local_path = _download_case_result(
        case_id,
        download_path,
        to_path=output_path,
        overwrite=overwrite,
    )
    emit_json(
        {
            "id": case_id,
            "name": os.path.basename(result_path) if result_path else None,
            "path": result_path,
            "local_path": local_path,
        }
    )
