"""
Draft CLI commands.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import copy
import json

import click

from flow360.cli.dict_utils import merge_overwrite
from flow360.cli.output import emit_json
from flow360.cli.resource_group import ResourceCommandSpec, make_resource_group
from flow360.cli.resource_refs import (
    ResourceRefError,
    parse_resource_ref,
    require_resource_type,
)
from flow360.cli.resource_state import get_resource_state_for_type
from flow360.cli.resource_state import (
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
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).get_info()


def _list_drafts(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi.list_records(project_id)


def _get_draft_simulation_params(draft_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).get_simulation_params()


def _set_draft_simulation_params(draft_id, simulation_params):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).set_simulation_params(simulation_params)


def _rename_draft(draft_id, name):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).rename(name)


def _delete_draft(draft_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).delete()


def _run_draft(draft_id, *, up_to):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

    return DraftWebApi(draft_id).run(up_to=up_to)


def _create_draft_from_source(source, *, name=None):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import DraftWebApi

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


def _get_asset_webapi_for_type(resource_type, resource_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.asset_webapi import get_resource_webapi_class

    try:
        webapi_cls = get_resource_webapi_class(resource_type)
    except ValueError as error:
        raise click.ClickException(f"Unsupported draft source type: {resource_type}.") from error
    return webapi_cls(resource_id)


def _get_asset_info_for_type(resource_type, resource_id):
    return _get_asset_webapi_for_type(resource_type, resource_id).get_info()


def _get_asset_simulation_params_for_type(resource_type, resource_id):
    return _get_asset_webapi_for_type(resource_type, resource_id).get_simulation_params()


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


def _load_json_file(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise click.ClickException(f"Invalid JSON in {path}: {error}") from error


def _load_patch_json(path):
    patch_json = _load_json_file(path)
    if not isinstance(patch_json, dict):
        raise click.ClickException(f"Patch JSON in {path} must be a JSON object.")
    return patch_json


def _simulation_params_with_patch(base_params, patch_path):
    if not isinstance(base_params, dict):
        raise click.ClickException("SimulationParams JSON must be an object to apply a patch.")
    return merge_overwrite(copy.deepcopy(base_params), _load_patch_json(patch_path))


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
    }
    if payload["type"] == "Case":
        payload["mesh_id"] = info.get("caseMeshId") or info.get("meshId") or info.get("parentId")
    return payload


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


def _emit_draft_list(project_id):
    emit_json({"records": [_serialize_draft_info(info) for info in _list_drafts(project_id)]})


def _emit_draft_info(draft_id):
    info = _get_draft_info(draft_id)
    emit_json(_serialize_draft_info(info))


def _emit_draft_state(draft_id):
    emit_json(get_resource_state_for_type("Draft", draft_id))


def _emit_draft_simulation_params(draft_id):
    emit_json({"simulation_params": _get_draft_simulation_params(draft_id)})


def _emit_draft_rename(draft_id, name):
    _rename_draft(draft_id, name)
    emit_json({"id": draft_id, "name": name})


def _emit_draft_delete(draft_id):
    _delete_draft(draft_id)
    emit_json({"id": draft_id, "deleted": True})


def _normalize_draft_id(draft_id):
    return _require_typed_id(draft_id, "Draft")


draft = make_resource_group(
    ResourceCommandSpec(
        command_name="draft",
        id_argument="draft_id",
        help_text="Inspect draft resources.",
        label="draft",
        normalize_id=_normalize_draft_id,
        emit_info=_emit_draft_info,
        emit_state=_emit_draft_state,
        emit_simulation_params=_emit_draft_simulation_params,
        emit_rename=_emit_draft_rename,
        emit_delete=_emit_draft_delete,
    )
)


@draft.command("list")
@click.option("--project-id", required=True, help="Project ID.")
def list_drafts(project_id):
    """List drafts for a project."""
    project_id = _require_typed_id(project_id, "Project")
    _emit_draft_list(project_id)


@draft.command("create")
@click.argument("source_id")
@click.option("--name", default=None, help="Draft name.")
def create_draft(source_id, name):
    """Create a draft from a project or asset."""
    source = _resolve_draft_source(source_id)
    emit_json(_serialize_draft_info(_create_draft_from_source(source, name=name)))


draft_simulation_params = draft.commands["simulation-params"]


@draft_simulation_params.command(
    "set",
    help="Replace draft SimulationParams from a JSON file.",
)
@click.argument("draft_id")
@click.argument("simulation_params_file", type=click.Path(exists=True, dir_okay=False))
def set_draft_simulation_params(draft_id, simulation_params_file):
    """Replace draft SimulationParams from a JSON file."""
    draft_id = _normalize_draft_id(draft_id)
    _set_draft_simulation_params(draft_id, _load_json_file(simulation_params_file))
    emit_json({"id": draft_id, "updated": True})


@draft_simulation_params.command(
    "patch",
    help=(
        "Apply a small local JSON merge patch to draft SimulationParams. "
        "Recommended only for small edits such as angle of attack or velocity; "
        "use Python and Pydantic models for larger validation-sensitive edits."
    ),
)
@click.argument("draft_id")
@click.argument("patch_file", type=click.Path(exists=True, dir_okay=False))
def patch_draft_simulation_params(draft_id, patch_file):
    """Apply a local JSON merge patch to draft SimulationParams."""
    draft_id = _normalize_draft_id(draft_id)
    patched = _simulation_params_with_patch(_get_draft_simulation_params(draft_id), patch_file)
    _set_draft_simulation_params(draft_id, patched)
    emit_json({"id": draft_id, "updated": True})


@draft.command("run")
@click.argument("source_id")
@click.argument(
    "simulation_params_file",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--up-to",
    type=click.Choice(sorted(RUN_TARGETS), case_sensitive=False),
    default="case",
    show_default=True,
    help="Target asset to run up to.",
)
@click.option("--name", default=None, help="Draft/run name when SOURCE_ID is not already a draft.")
@click.option(
    "--patch",
    "patch_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Apply a small local JSON merge patch before running. "
        "Recommended only for small edits such as angle of attack or velocity."
    ),
)
@click.option("--wait", "wait_for_completion", is_flag=True, help="Wait for the run result.")
@click.option("--timeout", type=float, default=3600.0, show_default=True, help="Wait timeout.")
@click.option(
    "--poll-interval",
    type=float,
    default=15.0,
    show_default=True,
    help="Wait polling interval.",
)
def run_draft(  # pylint: disable=too-many-arguments,too-many-locals
    source_id,
    simulation_params_file,
    up_to,
    name,
    patch_file,
    wait_for_completion,
    timeout,
    poll_interval,
):
    """Create or reuse a draft, optionally set SimulationParams, then run it."""
    resource_ref = _parse_resource_id(source_id)
    created_draft = None
    source = None

    if resource_ref.resource_type == "Draft":
        draft_id = resource_ref.id
    else:
        source = _resolve_draft_source(source_id)
        created_draft = _create_draft_from_source(source, name=name)
        draft_id = created_draft["id"]

    simulation_params = None
    if simulation_params_file is not None:
        simulation_params = _load_json_file(simulation_params_file)

    if patch_file is not None:
        if simulation_params is None:
            if source is None:
                simulation_params = _get_draft_simulation_params(draft_id)
            else:
                simulation_params = _get_asset_simulation_params_for_type(
                    source["source_item_type"], source["source_item_id"]
                )
        simulation_params = _simulation_params_with_patch(simulation_params, patch_file)

    if simulation_params is not None:
        _set_draft_simulation_params(draft_id, simulation_params)

    run_result = _run_draft(draft_id, up_to=RUN_TARGETS[up_to.lower()])
    payload = {
        "draft": (
            _serialize_draft_info(created_draft) if created_draft is not None else {"id": draft_id}
        ),
        "result": _serialize_run_result(run_result),
    }
    result_id = payload["result"]["id"]
    if wait_for_completion and result_id:
        payload["state"] = _wait_for_resource_state(
            result_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    emit_json(payload)
