"""
Asset CLI commands.
"""

from __future__ import annotations

import json

import click

from flow360.cli.output import emit_json
from flow360.cli.resource_state import get_resource_state_for_type


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


def _summarize_simulation_json(simulation_json):
    # pylint: disable=import-outside-toplevel
    from flow360.cli.simulation_summary import summarize_simulation

    return summarize_simulation(simulation_json)


def _emit_asset_summary(webapi_cls, asset_id):
    emit_json(
        {
            "id": asset_id,
            "summary": _summarize_simulation_json(_get_asset_simulation_json(webapi_cls, asset_id)),
        }
    )


@click.group("geometry")
def geometry():
    """Inspect Flow360 geometries."""


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


@geometry.command("state")
@click.argument("geometry_id")
def state_geometry(geometry_id):
    """Get geometry lifecycle state."""
    emit_json(get_resource_state_for_type("Geometry", geometry_id))


@geometry.command("summary")
@click.argument("geometry_id")
def summary_geometry(geometry_id):
    """Summarize geometry simulation settings."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import GeometryWebApi

    _emit_asset_summary(GeometryWebApi, geometry_id)


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
    """Inspect Flow360 surface meshes."""


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


@surface_mesh.command("state")
@click.argument("surface_mesh_id")
def state_surface_mesh(surface_mesh_id):
    """Get surface mesh lifecycle state."""
    emit_json(get_resource_state_for_type("SurfaceMesh", surface_mesh_id))


@surface_mesh.command("summary")
@click.argument("surface_mesh_id")
def summary_surface_mesh(surface_mesh_id):
    """Summarize surface mesh simulation settings."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import SurfaceMeshWebApi

    _emit_asset_summary(SurfaceMeshWebApi, surface_mesh_id)


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
    """Inspect Flow360 volume meshes."""


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


@volume_mesh.command("state")
@click.argument("volume_mesh_id")
def state_volume_mesh(volume_mesh_id):
    """Get volume mesh lifecycle state."""
    emit_json(get_resource_state_for_type("VolumeMesh", volume_mesh_id))


@volume_mesh.command("summary")
@click.argument("volume_mesh_id")
def summary_volume_mesh(volume_mesh_id):
    """Summarize volume mesh simulation settings."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import VolumeMeshWebApi

    _emit_asset_summary(VolumeMeshWebApi, volume_mesh_id)


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
    """Inspect Flow360 cases."""


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


@case.command("summary")
@click.argument("case_id")
def summary_case(case_id):
    """Summarize case simulation settings."""
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import CaseWebApi

    _emit_asset_summary(CaseWebApi, case_id)


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
