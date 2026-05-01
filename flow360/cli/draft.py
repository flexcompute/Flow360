"""
Draft CLI commands.
"""

from __future__ import annotations

import json

import click

from flow360.cli.output import emit_json
from flow360.cli.resource_refs import ResourceRefError, require_resource_type
from flow360.cli.resource_state import get_resource_state_for_type


def _require_typed_id(resource_id, expected_type):
    try:
        return require_resource_type(resource_id, expected_type).id
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
