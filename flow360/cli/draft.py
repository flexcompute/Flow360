"""
Draft CLI commands.
"""

from __future__ import annotations

import copy
import json

import click
from flow360.cli.dict_utils import merge_overwrite
from flow360.cli.output import emit_json
from flow360.cli.resource_refs import ResourceRefError, parse_resource_ref, require_resource_type
from flow360.cli.resource_state import (
    WaitTimeoutError,
    get_resource_state as _get_resource_state,
    get_resource_state_for_type,
    wait_for_resource_state as _wait_for_resource_state,
)


RUN_TARGETS = {
    "surface-mesh": "SurfaceMesh",
    "volume-mesh": "VolumeMesh",
    "case": "Case",
}


def _require_typed_id(resource_id, expected_type):
    try:
        return require_resource_type(resource_id, expected_type).id
    except ResourceRefError as error:
        raise click.ClickException(str(error)) from error


def _parse_resource_id(resource_id):
    try:
        return parse_resource_ref(resource_id)
    except ResourceRefError as error:
        raise click.ClickException(str(error)) from error


def _get_draft_info(draft_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    return DraftWebApi(draft_id).get_info()


def _list_drafts(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    return DraftWebApi.list_records(project_id)


def _get_draft_simulation_json(draft_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    simulation_json = DraftWebApi(draft_id).get_simulation_json()
    if isinstance(simulation_json, str):
        return json.loads(simulation_json)
    return simulation_json


def _set_draft_simulation_json(draft_id, simulation_json):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    DraftWebApi(draft_id).set_simulation_json(simulation_json)


def _run_draft(draft_id, up_to):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    return DraftWebApi(draft_id).run(up_to=up_to)


def _rename_draft(draft_id, new_name):
    # pylint: disable=import-outside-toplevel
    from flow360.cloud.flow360_requests import RenameAssetRequestV2
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    DraftWebApi(draft_id).patch(RenameAssetRequestV2(name=new_name).dict())


def _load_simulation_json(simulation_path):
    try:
        with open(simulation_path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise click.ClickException(f"Invalid JSON in {simulation_path}: {error}") from error


def _load_patch_json(patch_path):
    patch_json = _load_simulation_json(patch_path)
    if not isinstance(patch_json, dict):
        raise click.ClickException(f"Patch JSON in {patch_path} must be a JSON object.")
    return patch_json


def _get_asset_info_for_type(resource_type, resource_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import (
        CaseWebApi,
        GeometryWebApi,
        SurfaceMeshWebApi,
        VolumeMeshWebApi,
    )

    webapi_by_type = {
        "Geometry": GeometryWebApi,
        "SurfaceMesh": SurfaceMeshWebApi,
        "VolumeMesh": VolumeMeshWebApi,
        "Case": CaseWebApi,
    }

    webapi_cls = webapi_by_type.get(resource_type)
    if webapi_cls is None:
        raise click.ClickException(f"Unsupported draft source type: {resource_type}.")

    return webapi_cls(resource_id).get_info()


def _get_asset_simulation_json_for_type(resource_type, resource_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import (
        CaseWebApi,
        GeometryWebApi,
        SurfaceMeshWebApi,
        VolumeMeshWebApi,
    )

    webapi_by_type = {
        "Geometry": GeometryWebApi,
        "SurfaceMesh": SurfaceMeshWebApi,
        "VolumeMesh": VolumeMeshWebApi,
        "Case": CaseWebApi,
    }

    webapi_cls = webapi_by_type.get(resource_type)
    if webapi_cls is None:
        raise click.ClickException(f"Unsupported draft source type: {resource_type}.")

    return webapi_cls(resource_id).get_simulation_json()


def _resolve_draft_source(ref_id):
    resource_ref = _parse_resource_id(ref_id)

    if resource_ref.resource_type == "Draft":
        raise click.ClickException(
            "Draft creation requires a project or asset ref, not a draft ID."
        )

    if resource_ref.resource_type == "Project":
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.web.project_webapi import ProjectWebApi

        project_info = ProjectWebApi(resource_ref.id).get_info()
        source_item_id = project_info.get("rootItemId")
        source_item_type = project_info.get("rootItemType")
        if not source_item_id or not source_item_type:
            raise click.ClickException(
                f"Project {resource_ref.id} does not expose a root item for draft creation."
            )
        source_info = _get_asset_info_for_type(source_item_type, source_item_id)
        return {
            "project_id": project_info.get("id") or resource_ref.id,
            "source_item_id": source_item_id,
            "source_item_type": source_item_type,
            "solver_version": source_info.get("solverVersion"),
            "fork_case": source_item_type == "Case",
        }

    if resource_ref.resource_type in {"Geometry", "SurfaceMesh", "VolumeMesh", "Case"}:
        source_info = _get_asset_info_for_type(resource_ref.resource_type, resource_ref.id)
        return {
            "project_id": source_info.get("projectId"),
            "source_item_id": resource_ref.id,
            "source_item_type": resource_ref.resource_type,
            "solver_version": source_info.get("solverVersion"),
            "fork_case": resource_ref.resource_type == "Case",
        }

    raise click.ClickException(
        "Draft creation is only supported from prj-, geo-, sm-, vm-, or case- refs."
    )


def _create_draft_from_ref(ref_id, *, name=None):
    source = _resolve_draft_source(ref_id)
    return _create_draft_from_source(source, name=name)


def _create_draft_from_source(source, *, name=None):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.draft_webapi import DraftWebApi

    created = DraftWebApi.create(
        name=name,
        project_id=source["project_id"],
        source_item_id=source["source_item_id"],
        source_item_type=source["source_item_type"],
        solver_version=source["solver_version"],
        fork_case=source["fork_case"],
    )
    return {
        "id": created.get("id"),
        "name": created.get("name"),
        "projectId": created.get("projectId") or source["project_id"],
        "solverVersion": created.get("solverVersion") or source["solver_version"],
        "sourceItemId": created.get("sourceItemId") or source["source_item_id"],
        "sourceItemType": created.get("sourceItemType") or source["source_item_type"],
        "forkCase": created.get("forkCase", source["fork_case"]),
        "type": created.get("type") or "Draft",
    }


def _apply_patch_to_source_simulation(source, patch_json):
    source_simulation = _get_asset_simulation_json_for_type(
        source["source_item_type"], source["source_item_id"]
    )
    if not isinstance(source_simulation, dict):
        raise click.ClickException("Source simulation JSON must be a JSON object to apply a patch.")
    return merge_overwrite(copy.deepcopy(source_simulation), patch_json)


def _serialize_draft_info(info):
    return {
        "id": info.get("id"),
        "name": info.get("name"),
        "project_id": info.get("projectId"),
        "solver_version": info.get("solverVersion"),
        "source_item_id": info.get("sourceItemId"),
        "source_item_type": info.get("sourceItemType"),
        "fork_case": info.get("forkCase"),
        "type": info.get("type"),
    }


def _serialize_run_result(info):
    payload = {
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
    if payload["type"] == "Case":
        payload["mesh_id"] = (
            info.get("caseMeshId") or info.get("meshId") or info.get("volumeMeshId")
        )
    return payload


def _emit_run_payload(*, draft_info=None, result, state=None, timed_out=False):
    result_payload = _serialize_run_result(result)
    if draft_info is None and state is None and not timed_out:
        emit_json(result_payload)
        return

    payload = {"result": result_payload}
    if draft_info is not None:
        payload["draft"] = _serialize_draft_info(draft_info)
    if state is not None:
        payload["state"] = state
    if timed_out:
        payload["timed_out"] = True
    emit_json(payload)


@click.group("draft")
def draft():
    """Inspect draft resources."""


def _emit_draft_list(project_id):
    emit_json({"records": [_serialize_draft_info(info) for info in _list_drafts(project_id)]})


@draft.command("list")
@click.option("--project-id", required=True, help="Project ID.")
def list_drafts(project_id):
    """List drafts for a project."""
    project_id = _require_typed_id(project_id, "Project")
    _emit_draft_list(project_id)


@draft.command("ls", hidden=True)
@click.option("--project-id", required=True, help="Project ID.")
def list_drafts_alias(project_id):
    """Backward-compatible alias for draft list."""
    project_id = _require_typed_id(project_id, "Project")
    _emit_draft_list(project_id)


@draft.command("create")
@click.argument("ref_id")
@click.option("--name", default=None, help="Optional draft name.")
def create_draft(ref_id, name):
    """Create a draft from a project or asset ref."""
    emit_json(_serialize_draft_info(_create_draft_from_ref(ref_id, name=name)))


@draft.command("run")
@click.argument("ref_id")
@click.argument(
    "simulation_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--patch",
    "patch_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="JSON patch object merged locally into the source simulation before draft run.",
)
@click.option("--name", default=None, help="Optional name for the created draft in one-shot mode.")
@click.option(
    "--up-to",
    "up_to_name",
    required=True,
    type=click.Choice(list(RUN_TARGETS.keys()), case_sensitive=False),
    help="Run the draft up to the selected resource type.",
)
@click.option("--wait", "wait_for_result", is_flag=True, help="Wait for the result to reach a terminal state.")
@click.option(
    "--timeout",
    default=3600,
    show_default=True,
    type=click.FloatRange(min=0.1, min_open=False),
    help="Maximum wait time in seconds when --wait is used.",
)
@click.option(
    "--poll-interval",
    default=2.0,
    show_default=True,
    type=click.FloatRange(min=0.1, min_open=False),
    help="Polling interval in seconds when --wait is used.",
)
def run_draft(ref_id, simulation_path, patch_path, name, up_to_name, wait_for_result, timeout, poll_interval):
    """Run a draft workflow."""
    resource_ref = _parse_resource_id(ref_id)
    up_to = RUN_TARGETS[up_to_name.lower()]

    if resource_ref.resource_type == "Draft":
        if simulation_path is not None or patch_path is not None or name is not None:
            raise click.ClickException(
                "Simulation JSON, patch, or name cannot be passed when running an existing draft. "
                "Use 'flow360 draft simulation set <draft_id> <simulation.json>' first."
            )
        result = _run_draft(resource_ref.id, up_to)
        if not wait_for_result:
            _emit_run_payload(result=result)
            return

        try:
            state = _wait_for_resource_state(
                result["id"], timeout=timeout, poll_interval=poll_interval
            )
        except WaitTimeoutError as error:
            _emit_run_payload(result=result, state=error.state, timed_out=True)
            raise click.exceptions.Exit(124) from error

        _emit_run_payload(result=result, state=state)
        if not state["is_success"]:
            raise click.exceptions.Exit(1)
        return

    if simulation_path is not None and patch_path is not None:
        raise click.ClickException(
            "Provide either a full simulation JSON path or --patch, not both."
        )

    if simulation_path is None and patch_path is None:
        raise click.ClickException(
            "Simulation JSON path or --patch is required when running from a non-draft ref."
        )

    source = _resolve_draft_source(resource_ref.id)
    if simulation_path is not None:
        simulation_json = _load_simulation_json(simulation_path)
    else:
        simulation_json = _apply_patch_to_source_simulation(source, _load_patch_json(patch_path))

    draft_info = _create_draft_from_source(source, name=name)
    _set_draft_simulation_json(draft_info["id"], simulation_json)
    result = _run_draft(draft_info["id"], up_to)
    if not wait_for_result:
        _emit_run_payload(draft_info=draft_info, result=result)
        return

    try:
        state = _wait_for_resource_state(result["id"], timeout=timeout, poll_interval=poll_interval)
    except WaitTimeoutError as error:
        _emit_run_payload(draft_info=draft_info, result=result, state=error.state, timed_out=True)
        raise click.exceptions.Exit(124) from error

    _emit_run_payload(draft_info=draft_info, result=result, state=state)
    if not state["is_success"]:
        raise click.exceptions.Exit(1)


def _emit_draft_info(draft_id):
    info = _get_draft_info(draft_id)
    emit_json(_serialize_draft_info(info))


@draft.command("info")
@click.argument("draft_id")
def show_draft_info(draft_id):
    """Get draft metadata."""
    draft_id = _require_typed_id(draft_id, "Draft")
    _emit_draft_info(draft_id)


@draft.command("get", hidden=True)
@click.argument("draft_id")
def get_draft_alias(draft_id):
    """Backward-compatible alias for draft info."""
    draft_id = _require_typed_id(draft_id, "Draft")
    _emit_draft_info(draft_id)


@draft.command("rename")
@click.argument("draft_id")
@click.option("--name", required=True, help="New draft name.")
def rename_draft(draft_id, name):
    """Rename a draft."""
    draft_id = _require_typed_id(draft_id, "Draft")
    _rename_draft(draft_id, name)
    emit_json({"id": draft_id, "name": name})


@draft.command("state")
@click.argument("draft_id")
def show_draft_state(draft_id):
    """Get draft lifecycle state."""
    draft_id = _require_typed_id(draft_id, "Draft")
    emit_json(get_resource_state_for_type("Draft", draft_id))


@draft.group("simulation")
def draft_simulation():
    """Namespace for draft simulation commands."""


@draft_simulation.command("get")
@click.argument("draft_id")
def get_draft_simulation(draft_id):
    """Get draft simulation JSON."""
    draft_id = _require_typed_id(draft_id, "Draft")
    emit_json({"simulation": _get_draft_simulation_json(draft_id)})


@draft_simulation.command("set")
@click.argument("draft_id")
@click.argument(
    "simulation_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
def set_draft_simulation(draft_id, simulation_path):
    """Replace draft simulation JSON."""
    draft_id = _require_typed_id(draft_id, "Draft")
    simulation_json = _load_simulation_json(simulation_path)
    _set_draft_simulation_json(draft_id, simulation_json)
    emit_json({"id": draft_id, "updated": True})
