"""
Folder CLI commands.
"""

from __future__ import annotations

import click

from flow360.cli.output import emit_json


def _get_folder_info(folder_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.folder_webapi import FolderWebApi

    return FolderWebApi(folder_id).get_info()


def _get_folder_tree(folder_id):
    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.folder import Folder

    return Folder(folder_id).get_folder_tree()


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
@click.argument("folder_id", required=False)
def folder_tree(folder_id):
    """Get the folder tree."""
    if folder_id is None:
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.folder import ROOT_FOLDER

        folder_id = ROOT_FOLDER

    tree = _get_folder_tree(folder_id)
    if tree is None:
        raise click.ClickException(f"Folder {folder_id} was not found.")
    emit_json({"root": tree})
