"""
Project CLI commands.
"""

from __future__ import annotations

import click

from flow360.cli.output import emit_json, emit_payload
from flow360.cli.project_formatters import format_project_list
from flow360.component.simulation.web.project_tree import (
    build_project_tree,
    get_project_tree_parent_id,
)


def _get_project_records(search=None, limit=25, folder_ids=None, exclude_subfolders=False):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_records import get_project_records

    return get_project_records(
        search_keyword=search or "",
        folder_ids=list(folder_ids or ()) or None,
        exclude_subfolders=exclude_subfolders,
        page_size=limit,
        sort_direction="desc",
    )


def _get_project_info(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    return ProjectWebApi(project_id).get_info()


def _get_project_tree(project_id):
    return _project_tree_from_records(_get_project_tree_records(project_id))


def _get_project_tree_records(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    return ProjectWebApi(project_id).get_tree()["records"]


def _get_project_path(project_id, item_id, item_type):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    resp = ProjectWebApi(project_id).get_path(item_id=item_id, item_type=item_type)
    data = resp.get("data", resp)
    items = []
    for key in ("geometry", "surfaceMesh", "volumeMesh"):
        item = data.get(key)
        if item:
            items.append(item)
    items.extend(data.get("cases") or [])
    return items


def _serialize_project_record(record):
    return {
        "id": record.project_id,
        "name": record.name,
        "root_item_type": record.root_item_type,
        "solver_version": record.solver_version,
        "created_at": record.created_at,
        "tags": record.tags or [],
        "description": record.description,
        "statistics": _serialize_project_statistics(getattr(record, "statistics", None)),
    }


def _serialize_asset_statistics(statistics):
    if statistics is None:
        return None
    return {
        "count": statistics.count,
        "success_count": statistics.successCount,
        "running_count": statistics.runningCount,
        "diverged_count": statistics.divergedCount,
        "error_count": statistics.errorCount,
    }


def _serialize_project_statistics(statistics):
    if statistics is None:
        return {}
    return {
        "geometry": _serialize_asset_statistics(getattr(statistics, "geometry", None)),
        "surface_mesh": _serialize_asset_statistics(getattr(statistics, "surface_mesh", None)),
        "volume_mesh": _serialize_asset_statistics(getattr(statistics, "volume_mesh", None)),
        "case": _serialize_asset_statistics(getattr(statistics, "case", None)),
    }


def _project_tree_from_records(records):
    def create_node(item):
        return {
            "id": item["id"],
            "name": item["name"],
            "type": item["type"],
            "children": [],
        }

    def add_child(parent, child):
        parent["children"].append(child)

    try:
        root, _nodes = build_project_tree(records, create_node=create_node, add_child=add_child)
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    return root


def _project_item_from_record(item):
    return {
        "id": item["id"],
        "name": item["name"],
        "type": item["type"],
        "parent_id": get_project_tree_parent_id(item),
    }


def _project_items_from_records(records):
    return [_project_item_from_record(item) for item in records]


def _serialize_project_item(item):
    return {
        "id": item["id"],
        "name": item["name"],
        "type": item["type"],
        "parent_id": item.get("parentId"),
        "status": item.get("status"),
        "updated_at": item.get("updatedAt"),
    }


def _project_browser_url(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.environment import Env

    return Env.current.get_web_real_url(f"workbench/{project_id}")


@click.group("project")
def project():
    """
    Inspect and manage Flow360 projects.
    """


def build_project_list_payload(search, limit, folder_ids, exclude_subfolders):
    """Build a serialized project list payload for CLI output."""

    records, total = _get_project_records(
        search=search,
        limit=limit,
        folder_ids=folder_ids,
        exclude_subfolders=exclude_subfolders,
    )
    project_records = records.records if hasattr(records, "records") else records
    return {
        "records": [_serialize_project_record(record) for record in project_records],
        "returned": len(project_records),
        "total": total,
    }


def format_project_list_payload(payload):
    """Format a serialized project list payload as text."""

    return format_project_list(payload, project_url_factory=_project_browser_url)


def _emit_project_list(search, limit, folder_ids, exclude_subfolders, output_format="json"):
    payload = build_project_list_payload(search, limit, folder_ids, exclude_subfolders)
    emit_payload(
        payload,
        output_format=output_format,
        text_formatter=format_project_list_payload,
    )


def _emit_project_info(project_id):
    info = _get_project_info(project_id)
    emit_json(
        {
            "id": info["id"],
            "name": info["name"],
            "solver_version": info.get("solverVersion"),
            "tags": list(info.get("tags") or []),
            "root_item": {
                "id": info["rootItemId"],
                "type": info["rootItemType"],
            },
        }
    )


@project.command("list")
@click.option("--search", "--keyword", "-k", default=None, help="Search project names.")
@click.option(
    "--limit",
    "-n",
    type=click.IntRange(1, 1000),
    default=25,
    show_default=True,
    help="Maximum number of projects to return.",
)
@click.option(
    "--folder-id",
    "folder_ids",
    multiple=True,
    help="Filter projects to one or more folder IDs.",
)
@click.option(
    "--exclude-subfolders",
    is_flag=True,
    help="Only search the specified folders, not their subfolders.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Output format.",
)
def list_projects(search, limit, folder_ids, exclude_subfolders, output_format):
    """
    List projects.
    """
    _emit_project_list(search, limit, folder_ids, exclude_subfolders, output_format)


@project.command("info")
@click.argument("project_id")
def info_project(project_id):
    """
    Get project metadata.
    """
    _emit_project_info(project_id)


@project.command("tree")
@click.argument("project_id")
def project_tree(project_id):
    """
    Get the project tree.
    """
    emit_json({"root": _get_project_tree(project_id)})


@project.command("items")
@click.argument("project_id")
def project_items(project_id):
    """
    Get a flat list of project items.
    """
    emit_json({"items": _project_items_from_records(_get_project_tree_records(project_id))})


@project.command("path")
@click.argument("project_id")
@click.option("--item-id", required=True, help="Target item ID.")
@click.option("--item-type", required=True, help="Target item type.")
def project_path(project_id, item_id, item_type):
    """
    Get a path of project items to a target item.
    """
    items = _get_project_path(project_id, item_id=item_id, item_type=item_type)
    emit_json({"items": [_serialize_project_item(item) for item in items]})
