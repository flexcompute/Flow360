"""
Thin folder web API wrapper.
"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import FolderInterface, FolderInterfaceV2
from flow360.component.simulation.web.folder_constants import (
    ROOT_FOLDER_ID,
    ROOT_FOLDER_NAME,
)


class FolderWebApi:
    """Thin wrapper around folder endpoints."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self._api = RestApi(FolderInterfaceV2.endpoint, id=folder_id)

    def get_info(self):
        """Get folder metadata."""

        return self._api.get()

    @classmethod
    def create(cls, name: str, tags=None, parent_folder_id: str = ROOT_FOLDER_ID):
        """Create a folder."""

        return RestApi(FolderInterface.endpoint).post(
            json={
                "name": name,
                "tags": list(tags or []),
                "parentFolderId": parent_folder_id,
                "type": "folder",
            }
        )

    def rename(self, name: str):
        """Rename folder."""

        return self._api.patch(json={"name": name})

    def move(self, parent_folder_id: str):
        """Move folder to another parent folder."""

        return self._api.patch(json={"parentFolderId": parent_folder_id})

    def delete(self):
        """Delete folder."""

        return self._api.delete()

    @classmethod
    def list_records(cls, include_subfolders: bool = True, page: int = 0, size: int = 1000):
        """List folder records."""

        response = RestApi(FolderInterfaceV2.endpoint).get(
            params={
                "includeSubfolders": include_subfolders,
                "page": page,
                "size": size,
            }
        )
        return response.get("records", [])

    @classmethod
    def get_tree(cls, root_folder_id: str = ROOT_FOLDER_ID):
        """Build a folder tree from folder records."""

        records = cls.list_records(include_subfolders=True)
        return cls._build_tree(records, root_folder_id=root_folder_id)

    @classmethod
    def _build_tree(cls, records, root_folder_id: str):
        folder_dict = {record["id"]: dict(record) for record in records}
        folder_dict[ROOT_FOLDER_ID] = {
            "id": ROOT_FOLDER_ID,
            "name": ROOT_FOLDER_NAME,
            "parentFolderId": None,
        }

        for folder in folder_dict.values():
            folder["subfolders"] = []

        for record in records:
            parent_id = record.get("parentFolderId")
            parent = folder_dict.get(parent_id)
            if parent is not None:
                parent["subfolders"].append(record["id"])

        def build(folder_id):
            folder = folder_dict.get(folder_id)
            if folder is None:
                return None

            return {
                "id": folder["id"],
                "name": folder["name"],
                "subfolders": [build(child_id) for child_id in folder["subfolders"]],
            }

        return build(root_folder_id)
