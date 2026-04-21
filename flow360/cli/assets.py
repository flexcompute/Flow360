"""
Asset CLI commands.
"""

from __future__ import annotations

import json
import os

import click

from flow360.cli.output import emit_json
from flow360.cli.resource_state import get_resource_state_for_type


def _rename_asset(webapi_cls, asset_id, new_name):
    # pylint: disable=import-outside-toplevel
    from flow360.cloud.flow360_requests import RenameAssetRequestV2

    webapi_cls(asset_id).patch(RenameAssetRequestV2(name=new_name).dict())


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


def _get_asset_simulation_json(webapi_cls, asset_id):
    # pylint: disable=import-outside-toplevel
    simulation_json = webapi_cls(asset_id).get_simulation_json()
    if isinstance(simulation_json, str):
        return json.loads(simulation_json)
    return simulation_json


def _serialize_case_result(record):
    path = _get_case_result_path(record)
    return {
        "name": os.path.basename(path) if path else None,
        "path": path,
        "file_type": record.get("fileType"),
        "size_bytes": record.get("length"),
        "updated_at": record.get("updatedAt"),
    }


def _get_case_result_path(record):
    for value in (record.get("fileName"), record.get("filePath")):
        if not value:
            continue
        if "results/" in value:
            return value[value.index("results/") :]
    return record.get("fileName") or record.get("filePath")


def _list_case_results(case_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    files = CaseWebApi(case_id).list_files()
    result_files = [record for record in files if (_get_case_result_path(record) or "").startswith("results/")]
    result_files.sort(key=lambda record: _get_case_result_path(record) or "")
    return result_files


def _resolve_case_result(case_id, result_ref):
    results = _list_case_results(case_id)
    if not results:
        raise click.ClickException(f"No result files are available for case {case_id}.")

    exact_matches = [
        record
        for record in results
        if result_ref in {record.get("filePath"), record.get("fileName"), _get_case_result_path(record)}
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
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    if to_path is None:
        return CaseWebApi(case_id).download_file(result_path, overwrite=overwrite)

    return CaseWebApi(case_id).download_file(
        result_path,
        to_file=to_path,
        overwrite=overwrite,
    )


@click.group("geometry")
def geometry():
    """Inspect and manage Flow360 geometries."""


def _emit_geometry_info(geometry_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import GeometryWebApi

    info = _get_asset_info(GeometryWebApi, geometry_id)
    emit_json(_serialize_asset_info(info))


@geometry.command("info")
@click.argument("geometry_id")
def info_geometry(geometry_id):
    """Get geometry metadata."""
    _emit_geometry_info(geometry_id)


@geometry.command("get", hidden=True)
@click.argument("geometry_id")
def get_geometry_alias(geometry_id):
    """Backward-compatible alias for geometry info."""
    _emit_geometry_info(geometry_id)


@geometry.command("rename")
@click.argument("geometry_id")
@click.option("--name", required=True, help="New geometry name.")
def rename_geometry(geometry_id, name):
    """Rename a geometry."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import GeometryWebApi

    _rename_asset(GeometryWebApi, geometry_id, name)
    emit_json({"id": geometry_id, "name": name})


@geometry.command("state")
@click.argument("geometry_id")
def state_geometry(geometry_id):
    """Get geometry lifecycle state."""
    emit_json(get_resource_state_for_type("Geometry", geometry_id))


@geometry.group("simulation")
def geometry_simulation():
    """Namespace for geometry simulation commands."""


@geometry_simulation.command("get")
@click.argument("geometry_id")
def get_geometry_simulation(geometry_id):
    """Get geometry simulation JSON."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import GeometryWebApi

    emit_json({"simulation": _get_asset_simulation_json(GeometryWebApi, geometry_id)})


@click.group("surface-mesh")
def surface_mesh():
    """Inspect and manage Flow360 surface meshes."""


def _emit_surface_mesh_info(surface_mesh_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import SurfaceMeshWebApi

    info = _get_asset_info(SurfaceMeshWebApi, surface_mesh_id)
    emit_json(_serialize_asset_info(info))


@surface_mesh.command("info")
@click.argument("surface_mesh_id")
def info_surface_mesh(surface_mesh_id):
    """Get surface mesh metadata."""
    _emit_surface_mesh_info(surface_mesh_id)


@surface_mesh.command("get", hidden=True)
@click.argument("surface_mesh_id")
def get_surface_mesh_alias(surface_mesh_id):
    """Backward-compatible alias for surface mesh info."""
    _emit_surface_mesh_info(surface_mesh_id)


@surface_mesh.command("rename")
@click.argument("surface_mesh_id")
@click.option("--name", required=True, help="New surface mesh name.")
def rename_surface_mesh(surface_mesh_id, name):
    """Rename a surface mesh."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import SurfaceMeshWebApi

    _rename_asset(SurfaceMeshWebApi, surface_mesh_id, name)
    emit_json({"id": surface_mesh_id, "name": name})


@surface_mesh.command("state")
@click.argument("surface_mesh_id")
def state_surface_mesh(surface_mesh_id):
    """Get surface mesh lifecycle state."""
    emit_json(get_resource_state_for_type("SurfaceMesh", surface_mesh_id))


@surface_mesh.group("simulation")
def surface_mesh_simulation():
    """Namespace for surface mesh simulation commands."""


@surface_mesh_simulation.command("get")
@click.argument("surface_mesh_id")
def get_surface_mesh_simulation(surface_mesh_id):
    """Get surface mesh simulation JSON."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import SurfaceMeshWebApi

    emit_json({"simulation": _get_asset_simulation_json(SurfaceMeshWebApi, surface_mesh_id)})


@click.group("volume-mesh")
def volume_mesh():
    """Inspect and manage Flow360 volume meshes."""


def _emit_volume_mesh_info(volume_mesh_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import VolumeMeshWebApi

    info = _get_asset_info(VolumeMeshWebApi, volume_mesh_id)
    emit_json(_serialize_asset_info(info))


@volume_mesh.command("info")
@click.argument("volume_mesh_id")
def info_volume_mesh(volume_mesh_id):
    """Get volume mesh metadata."""
    _emit_volume_mesh_info(volume_mesh_id)


@volume_mesh.command("get", hidden=True)
@click.argument("volume_mesh_id")
def get_volume_mesh_alias(volume_mesh_id):
    """Backward-compatible alias for volume mesh info."""
    _emit_volume_mesh_info(volume_mesh_id)


@volume_mesh.command("rename")
@click.argument("volume_mesh_id")
@click.option("--name", required=True, help="New volume mesh name.")
def rename_volume_mesh(volume_mesh_id, name):
    """Rename a volume mesh."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import VolumeMeshWebApi

    _rename_asset(VolumeMeshWebApi, volume_mesh_id, name)
    emit_json({"id": volume_mesh_id, "name": name})


@volume_mesh.command("state")
@click.argument("volume_mesh_id")
def state_volume_mesh(volume_mesh_id):
    """Get volume mesh lifecycle state."""
    emit_json(get_resource_state_for_type("VolumeMesh", volume_mesh_id))


@volume_mesh.group("simulation")
def volume_mesh_simulation():
    """Namespace for volume mesh simulation commands."""


@volume_mesh_simulation.command("get")
@click.argument("volume_mesh_id")
def get_volume_mesh_simulation(volume_mesh_id):
    """Get volume mesh simulation JSON."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import VolumeMeshWebApi

    emit_json({"simulation": _get_asset_simulation_json(VolumeMeshWebApi, volume_mesh_id)})


@click.group("case")
def case():
    """Inspect and manage Flow360 cases."""


def _serialize_case_info(info):
    payload = _serialize_asset_info(info)
    payload["type"] = payload["type"] or "Case"
    payload["mesh_id"] = info.get("caseMeshId") or info.get("meshId")
    return payload


def _emit_case_info(case_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    info = _get_asset_info(CaseWebApi, case_id)
    emit_json(_serialize_case_info(info))


@case.command("info")
@click.argument("case_id")
def info_case(case_id):
    """Get case metadata."""
    _emit_case_info(case_id)


@case.command("get", hidden=True)
@click.argument("case_id")
def get_case_alias(case_id):
    """Backward-compatible alias for case info."""
    _emit_case_info(case_id)


@case.command("state")
@click.argument("case_id")
def state_case(case_id):
    """Get case lifecycle state."""
    emit_json(get_resource_state_for_type("Case", case_id))


@case.command("rename")
@click.argument("case_id")
@click.option("--name", required=True, help="New case name.")
def rename_case(case_id, name):
    """Rename a case."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    _rename_asset(CaseWebApi, case_id, name)
    emit_json({"id": case_id, "name": name})


@case.group("simulation")
def case_simulation():
    """Namespace for case simulation commands."""


@case_simulation.command("get")
@click.argument("case_id")
def get_case_simulation(case_id):
    """Get case simulation JSON."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    emit_json({"simulation": _get_asset_simulation_json(CaseWebApi, case_id)})


@case.group("results")
def case_results():
    """Namespace for case result artifacts."""


def _emit_case_results_list(case_id):
    emit_json({"records": [_serialize_case_result(record) for record in _list_case_results(case_id)]})


@case_results.command("list")
@click.argument("case_id")
def list_case_results(case_id):
    """List case result artifacts."""
    _emit_case_results_list(case_id)


@case_results.command("ls", hidden=True)
@click.argument("case_id")
def list_case_results_alias(case_id):
    """Backward-compatible alias for case results list."""
    _emit_case_results_list(case_id)


@case_results.command("get")
@click.argument("case_id")
@click.argument("result_ref")
@click.option(
    "--to",
    "to_path",
    default=None,
    type=click.Path(dir_okay=True, file_okay=True, resolve_path=True),
    help="Optional destination file or folder path.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite an existing destination file.")
def get_case_result(case_id, result_ref, to_path, overwrite):
    """Download one case result artifact."""
    result_record = _resolve_case_result(case_id, result_ref)
    result_path = _get_case_result_path(result_record)
    saved_to = _download_case_result(case_id, result_path, to_path=to_path, overwrite=overwrite)
    emit_json(
        {
            "case_id": case_id,
            "result": _serialize_case_result(result_record),
            "saved_to": saved_to,
        }
    )
