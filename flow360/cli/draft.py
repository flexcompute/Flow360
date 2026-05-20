"""
Draft CLI commands.
"""

from __future__ import annotations

import click

from flow360.cli.output import emit_json
from flow360.cli.resource_group import ResourceCommandSpec, make_resource_group
from flow360.cli.resource_refs import ResourceRefError, require_resource_type
from flow360.cli.resource_state import get_resource_state_for_type


def _require_typed_id(resource_id, expected_type):
    try:
        return require_resource_type(resource_id, expected_type).id
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
    )
)


@draft.command("list")
@click.option("--project-id", required=True, help="Project ID.")
def list_drafts(project_id):
    """List drafts for a project."""
    project_id = _require_typed_id(project_id, "Project")
    _emit_draft_list(project_id)
