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


def _rename_project(project_id, name):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    return ProjectWebApi(project_id).rename(name)


def _delete_project(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    return ProjectWebApi(project_id).delete()


def _create_project_from_files(  # pylint: disable=too-many-arguments
    source_type,
    files,
    *,
    name=None,
    solver_version=None,
    length_unit="m",
    tags=None,
    folder_id=None,
    workflow="standard",
    run_async=True,
):
    # pylint: disable=import-outside-toplevel
    from flow360.component.project import Project

    folder = None
    if folder_id is not None:
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.folder import Folder

        folder = Folder(folder_id)

    kwargs = {
        "name": name,
        "length_unit": length_unit,
        "tags": list(tags or []),
        "run_async": run_async,
        "folder": folder,
    }
    if solver_version is not None:
        kwargs["solver_version"] = solver_version

    if source_type == "geometry":
        return Project.from_geometry(
            list(files) if len(files) > 1 else files[0],
            workflow=workflow,
            **kwargs,
        )
    if len(files) != 1:
        raise click.ClickException(f"Project creation from {source_type} expects exactly one file.")
    if source_type == "surface-mesh":
        return Project.from_surface_mesh(files[0], **kwargs)
    if source_type == "volume-mesh":
        return Project.from_volume_mesh(files[0], **kwargs)

    raise click.ClickException(f"Unsupported project source type: {source_type}.")


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


def _serialize_created_project(result):
    project_id = result if isinstance(result, str) else getattr(result, "id", None)
    return {"id": project_id, "type": "Project"}


@project.command("create")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--from",
    "source_type",
    required=True,
    type=click.Choice(["geometry", "surface-mesh", "volume-mesh"], case_sensitive=False),
    help="Input file type used to create the project.",
)
@click.option("--name", default=None, help="Project name.")
@click.option("--solver-version", default=None, help="Solver version.")
@click.option("--unit", "length_unit", default="m", show_default=True, help="Project length unit.")
@click.option("--tag", "tags", multiple=True, help="Project tag. Repeat for multiple tags.")
@click.option("--folder-id", default=None, help="Destination folder ID.")
@click.option(
    "--workflow",
    type=click.Choice(["standard", "catalyst"], case_sensitive=False),
    default="standard",
    show_default=True,
    help="Geometry workflow. Only used with --from geometry.",
)
@click.option(
    "--sync",
    "run_sync",
    is_flag=True,
    help="Wait for project root processing before returning.",
)
def create_project(  # pylint: disable=too-many-arguments
    files,
    source_type,
    name,
    solver_version,
    length_unit,
    tags,
    folder_id,
    workflow,
    run_sync,
):
    """
    Create a new project from geometry, surface mesh, or volume mesh files.
    """
    result = _create_project_from_files(
        source_type.lower(),
        files,
        name=name,
        solver_version=solver_version,
        length_unit=length_unit,
        tags=tags,
        folder_id=folder_id,
        workflow=workflow.lower(),
        run_async=not run_sync,
    )
    emit_json(_serialize_created_project(result))


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


@project.command("rename")
@click.argument("project_id")
@click.option("--name", required=True, help="New project name.")
def rename_project(project_id, name):
    """
    Rename a project.
    """
    _rename_project(project_id, name)
    emit_json({"id": project_id, "name": name})


@project.command("delete")
@click.argument("project_id")
@click.option("--yes", is_flag=True, help="Confirm project deletion.")
def delete_project(project_id, yes):
    """
    Delete a project.
    """
    if not yes:
        raise click.ClickException("Pass --yes to confirm project deletion.")
    _delete_project(project_id)
    emit_json({"id": project_id, "deleted": True})


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
