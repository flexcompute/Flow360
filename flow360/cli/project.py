"""
Project CLI commands.
"""

from __future__ import annotations

import click
from flow360.cli.output import emit_json


def _get_project_records(search=None, limit=25, folder_ids=None, exclude_subfolders=False):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_records import get_project_records

    return get_project_records(
        search_keyword=search or "",
        folder_ids=list(folder_ids) or None,
        exclude_subfolders=exclude_subfolders,
        page_size=limit,
        sort_direction="desc",
    )


def _get_project_info(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    return ProjectWebApi(project_id).get_info()


def _get_project_tree(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi
    from flow360.component.simulation.web.project_tree import ProjectTree

    records = _get_project_tree_records(project_id)
    tree = ProjectTree()
    tree.construct_tree(asset_records=records)
    return tree


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
        "tags": list(record.tags),
        "description": record.description,
    }


def _serialize_tree_node(node):
    return {
        "id": node.asset_id,
        "name": node.asset_name,
        "type": node.asset_type,
        "children": [_serialize_tree_node(child) for child in node.children],
    }


def _project_items_from_records(records):
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "type": item["type"],
            "parent_id": item.get("parentCaseId") or item.get("parentId"),
        }
        for item in records
    ]


def _serialize_project_item(item):
    return {
        "id": item["id"],
        "name": item["name"],
        "type": item["type"],
        "parent_id": item.get("parentId"),
        "status": item.get("status"),
        "updated_at": item.get("updatedAt"),
    }


def _rename_project(project_id, new_name):
    # pylint: disable=import-outside-toplevel
    from flow360.cloud.flow360_requests import RenameAssetRequestV2
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    ProjectWebApi(project_id).patch(RenameAssetRequestV2(name=new_name).dict())


def _delete_project(project_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_webapi import ProjectWebApi

    ProjectWebApi(project_id).delete()


def _create_project(
    source,
    files,
    name=None,
    solver_version=None,
    length_unit="m",
    description="",
    tags=None,
    folder_id=None,
    run_async=False,
):
    # pylint: disable=import-outside-toplevel
    from flow360.component.project import Project
    from flow360.component.simulation.folder import Folder
    from flow360.version import __solver_version__

    resolved_solver_version = solver_version or __solver_version__
    folder = Folder(folder_id) if folder_id else None
    tags = list(tags or [])

    if source == "geometry":
        return Project.from_geometry(
            list(files),
            name=name,
            solver_version=resolved_solver_version,
            length_unit=length_unit,
            tags=tags,
            description=description,
            run_async=run_async,
            folder=folder,
        )

    file_path = files[0]
    if source == "surface-mesh":
        return Project.from_surface_mesh(
            file_path,
            name=name,
            solver_version=resolved_solver_version,
            length_unit=length_unit,
            tags=tags,
            description=description,
            run_async=run_async,
            folder=folder,
        )

    if source == "volume-mesh":
        return Project.from_volume_mesh(
            file_path,
            name=name,
            solver_version=resolved_solver_version,
            length_unit=length_unit,
            tags=tags,
            description=description,
            run_async=run_async,
            folder=folder,
        )

    raise ValueError(f"Unsupported project source: {source}")


def _serialize_created_project(result, run_async):
    if isinstance(result, str):
        return {"id": result, "async": True}

    metadata = result.get_metadata()
    root_item_type = (
        metadata.root_item_type.value
        if hasattr(metadata.root_item_type, "value")
        else metadata.root_item_type
    )
    return {
        "id": result.id,
        "name": metadata.name,
        "tags": list(metadata.tags or []),
        "root_item": {
            "id": metadata.root_item_id,
            "type": root_item_type,
        },
        "async": bool(run_async),
    }


@click.group("project")
def project():
    """
    Inspect and manage Flow360 projects.
    """


def _emit_project_list(search, limit, folder_ids, exclude_subfolders):
    records, total = _get_project_records(
        search=search,
        limit=limit,
        folder_ids=folder_ids,
        exclude_subfolders=exclude_subfolders,
    )
    project_records = records.records if hasattr(records, "records") else records
    emit_json(
        {
            "records": [_serialize_project_record(record) for record in project_records],
            "returned": len(project_records),
            "total": total,
        }
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


@project.command("create")
@click.option(
    "--from",
    "source",
    required=True,
    type=click.Choice(["geometry", "surface-mesh", "volume-mesh"], case_sensitive=False),
    help="Initial project source.",
)
@click.option(
    "--file",
    "files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Input file. Repeat for multi-file geometry uploads.",
)
@click.option("--name", default=None, help="Project name.")
@click.option("--solver-version", default=None, help="Solver version to use.")
@click.option(
    "--length-unit",
    type=click.Choice(["m", "mm", "cm", "inch", "ft"], case_sensitive=False),
    default="m",
    show_default=True,
    help="Project length unit.",
)
@click.option("--description", default="", show_default=True, help="Project description.")
@click.option("--tag", "tags", multiple=True, help="Project tag. Repeat to pass multiple tags.")
@click.option("--folder-id", default=None, help="Parent folder ID.")
@click.option("--async", "run_async", is_flag=True, help="Return after upload submission.")
def create_project(source, files, name, solver_version, length_unit, description, tags, folder_id, run_async):
    """
    Create a new project from uploaded files.
    """
    if source == "geometry" and len(files) < 1:
        raise click.ClickException("Geometry projects require at least one --file.")
    if source in {"surface-mesh", "volume-mesh"} and len(files) != 1:
        raise click.ClickException(f"{source} projects require exactly one --file.")

    result = _create_project(
        source=source,
        files=files,
        name=name,
        solver_version=solver_version,
        length_unit=length_unit,
        description=description,
        tags=tags,
        folder_id=folder_id,
        run_async=run_async,
    )
    emit_json(_serialize_created_project(result, run_async=run_async))


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
def list_projects(search, limit, folder_ids, exclude_subfolders):
    """
    List projects.
    """
    _emit_project_list(search, limit, folder_ids, exclude_subfolders)


@project.command("ls", hidden=True)
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
def list_projects_alias(search, limit, folder_ids, exclude_subfolders):
    """Backward-compatible alias for project list."""
    _emit_project_list(search, limit, folder_ids, exclude_subfolders)


@project.command("info")
@click.argument("project_id")
def info_project(project_id):
    """
    Get project metadata.
    """
    _emit_project_info(project_id)


@project.command("get", hidden=True)
@click.argument("project_id")
def get_project_alias(project_id):
    """Backward-compatible alias for project info."""
    _emit_project_info(project_id)


@project.command("tree")
@click.argument("project_id")
def project_tree(project_id):
    """
    Get the project tree.
    """
    tree = _get_project_tree(project_id)
    emit_json({"root": _serialize_tree_node(tree.root)})


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


@project.command("rename")
@click.argument("project_id")
@click.option("--name", required=True, help="New project name.")
def project_rename(project_id, name):
    """
    Rename a project.
    """
    _rename_project(project_id, new_name=name)
    click.echo(f"Renamed project {project_id} to {name}.")


@project.command("delete")
@click.argument("project_id")
@click.option("--yes", is_flag=True, help="Confirm project deletion.")
def project_delete(project_id, yes):
    """
    Delete a project.
    """
    if not yes:
        raise click.ClickException("Pass --yes to confirm project deletion.")
    _delete_project(project_id)
    click.echo(f"Deleted project {project_id}.")
