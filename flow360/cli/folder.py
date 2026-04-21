"""
Folder CLI commands.
"""

from __future__ import annotations

import click

from flow360.cli.output import emit_json
from flow360.component.simulation.web.folder_webapi import ROOT_FOLDER_ID


def _get_folder_info(folder_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    return FolderWebApi(folder_id).get_info()


def _get_folder_tree(folder_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    return FolderWebApi.get_tree(root_folder_id=folder_id)


def _create_folder(name, parent_folder_id=ROOT_FOLDER_ID, tags=None):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    return FolderWebApi.create(name=name, parent_folder_id=parent_folder_id, tags=tags)


def _rename_folder(folder_id, new_name):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    FolderWebApi(folder_id).rename(new_name)


def _move_folder(folder_id, parent_folder_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    FolderWebApi(folder_id).move(parent_folder_id)


def _serialize_folder_info(info):
    return {
        "id": info.get("id"),
        "name": info.get("name"),
        "parent_id": info.get("parentFolderId"),
        "type": info.get("type") or "Folder",
        "tags": list(info.get("tags") or []),
        "created_at": info.get("createdAt"),
        "updated_at": info.get("updatedAt"),
        "parent_folders": [
            {
                "id": parent.get("id"),
                "name": parent.get("name"),
            }
            for parent in (info.get("parentFolders") or [])
        ],
    }


@click.group("folder")
def folder():
    """Inspect Flow360 folders."""


@folder.command("get")
@click.argument("folder_id")
def get_folder(folder_id):
    """Get folder metadata."""
    emit_json(_serialize_folder_info(_get_folder_info(folder_id)))


@folder.command("tree")
@click.argument("folder_id", required=False, default=ROOT_FOLDER_ID)
def folder_tree(folder_id):
    """Get the folder tree."""
    tree = _get_folder_tree(folder_id)
    if tree is None:
        raise click.ClickException(f"Folder {folder_id} was not found.")
    emit_json({"root": tree})


@folder.command("create")
@click.option("--name", required=True, help="Folder name.")
@click.option(
    "--parent-folder-id",
    default=ROOT_FOLDER_ID,
    show_default=True,
    help="Parent folder ID.",
)
@click.option("--tag", "tags", multiple=True, help="Folder tag. Repeat to pass multiple tags.")
def create_folder(name, parent_folder_id, tags):
    """Create a folder."""
    emit_json(_serialize_folder_info(_create_folder(name, parent_folder_id=parent_folder_id, tags=tags)))


@folder.command("rename")
@click.argument("folder_id")
@click.option("--name", required=True, help="New folder name.")
def rename_folder(folder_id, name):
    """Rename a folder."""
    _rename_folder(folder_id, name)
    emit_json({"id": folder_id, "name": name})


@folder.command("move")
@click.argument("folder_id")
@click.option("--parent-folder-id", required=True, help="Destination parent folder ID.")
def move_folder(folder_id, parent_folder_id):
    """Move a folder."""
    _move_folder(folder_id, parent_folder_id)
    emit_json({"id": folder_id, "parent_id": parent_folder_id})
