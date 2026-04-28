"""
Thin folder web API wrapper.
"""

from __future__ import annotations

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import FolderInterfaceV2

ROOT_FOLDER_ID = "ROOT.FLOW360"
ROOT_FOLDER_NAME = "My workspace"


class FolderWebApi:
    """Thin wrapper around folder endpoints."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self._api = RestApi(FolderInterfaceV2.endpoint, id=folder_id)

    def get_info(self):
        """Get folder metadata."""

        return self._api.get()

    @classmethod
    def list_records(cls, include_subfolders: bool = True, page: int = 0, size: int = 1000):
        """List folder records."""

        api = RestApi(FolderInterfaceV2.endpoint)
        response = api.get(
            params={
                "includeSubfolders": include_subfolders,
                "page": page,
                "size": size,
            }
        )
        return response.get("records", [])

    @classmethod
    def get_tree(cls, root_folder_id: str = ROOT_FOLDER_ID):
        """Build a folder tree rooted at the requested folder."""

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
