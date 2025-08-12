"""
Folder component
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import List, Optional, Union

import pydantic as pd

from ...cloud.flow360_requests import (
    MoveToFolderRequestV2,
    NewFolderRequest,
    RenameAssetRequestV2,
)
from ...cloud.rest_api import RestApi
from ...exceptions import Flow360ValueError
from ...log import log
from ..interfaces import FolderInterface, FolderInterfaceV2
from ..resource_base import AssetMetaBaseModel, Flow360Resource, ResourceDraft
from ..utils import (
    shared_account_confirm_proceed,
    storage_size_formatter,
    validate_type,
)

ROOT_FOLDER = "ROOT.FLOW360"


class FolderMeta(AssetMetaBaseModel, extra="allow"):
    """
    FolderMeta component
    """

    parent_folder_id: Union[str, None] = pd.Field(alias="parentFolderId")
    status: Optional[str] = pd.Field()
    deleted: Optional[bool]
    user_id: Optional[str] = pd.Field(alias="userId")
    parent_folders: Optional[List[FolderMeta]] = pd.Field(alias="parentFolders")


class FolderDraft(ResourceDraft):
    """
    Folder Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(self, name: str = None, tags: List[str] = None, parent_folder: Folder = None):
        self.name = name
        self.tags = tags
        self._id = None
        self._parent_folder = parent_folder
        ResourceDraft.__init__(self)

    # pylint: disable=protected-access
    def submit(self) -> Folder:
        """create folder in cloud

        Returns
        -------
        Folder
            Folder object with id
        """

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        req = NewFolderRequest(name=self.name, tags=self.tags)
        if self._parent_folder:
            req.parent_folder_id = self._parent_folder.id
        resp = RestApi(FolderInterface.endpoint).post(req.dict())
        info = FolderMeta(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        submitted_folder = Folder(self.id)
        log.info(f"Folder successfully created: {info.name}, {info.id}")
        return submitted_folder


class Folder(Flow360Resource):
    """
    Folder component
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=FolderInterface,
            meta_class=FolderMeta,
            id=id,
        )

    @classmethod
    def _from_meta(cls, meta: FolderMeta):
        validate_type(meta, "meta", FolderMeta)
        folder = cls(id=meta.id)
        folder._set_meta(meta)
        return folder

    @property
    def info(self) -> FolderMeta:
        return super().info

    def get_info(self, force=False) -> FolderMeta:
        """
        returns metadata info for resource
        """

        if self._info is None or force:
            self._info = self.meta_class(**RestApi(f"v2/folders/{self.id}").get())
        return self._info

    def move_to_folder(self, folder: Folder):
        """
        Move the current folder to the specified folder.

        Parameters
        ----------
        folder : Folder
            The destination folder where the item will be moved.

        Returns
        -------
        self
            Returns the modified item after it has been moved to the new folder.

        Notes
        -----
        This method sends a REST API request to move the current item to the specified folder.
        The `folder` parameter should be an instance of the `Folder` class with a valid ID.
        """
        RestApi(FolderInterfaceV2.endpoint).patch(
            MoveToFolderRequestV2(parent_folder_id=folder.id).dict(),
            method=f"{self.id}",
        )
        return self

    def rename(self, new_name: str):
        """
        Rename the current folder.

        Parameters
        ----------
        new_name : str
            The new name for the folder.
        """
        RestApi(FolderInterfaceV2.endpoint).patch(
            RenameAssetRequestV2(name=new_name).dict(), method=self.id
        )

    @classmethod
    def _interface(cls):
        return FolderInterface

    @classmethod
    def _meta_class(cls):
        """
        returns folder mesh meta info class: FolderMeta
        """
        return FolderMeta

    @classmethod
    def create(cls, name: str, tags: List[str] = None, parent_folder: Folder = None) -> FolderDraft:
        """ "Create a new folder"

        Parameters
        ----------
        name : str
            name of the folder
        tags : List[str], optional
            tags for the folder, by default None
        parent_folder : Folder, optional
            parent folder object, by default folder is created at root level, by default None

        Returns
        -------
        FolderDraft
            _description_
        """
        new_folder = FolderDraft(
            name=name,
            tags=tags,
            parent_folder=parent_folder,
        )
        return new_folder

    def get_items(self):
        """
        Fetch all items within the current folder, handling pagination if needed.

        Returns
        -------
        list
            A list of all items found in the folder, sorted by storage size in descending order.
        """

        all_records = []
        page = 0
        size = 1000  # Page size
        total_record_count = size

        # Loop until all pages are fetched
        while len(all_records) < total_record_count:
            payload = {
                "page": page,
                "size": size,
                "filterFolderIds": self.id,
                "filterExcludeSubfolders": True,
                "sortFields": ["storageSize"],
                "sortDirections": ["desc"],
                "expandFields": ["contentInfo"],
            }

            data = RestApi("/v2/items").get(params=payload)
            records = data.get("records", [])
            all_records.extend(records)
            total_record_count = data.get("total", 0)
            page += 1

        return all_records

    def _build_folder_tree(self, folders):
        """
        Build a hierarchical folder tree starting from the current folder.

        Parameters
        ----------
        folders : list
            A list of folder records.

        Returns
        -------
        dict
            A dictionary representing the folder hierarchy with nested subfolders.
        """

        folder_dict = {folder["id"]: folder for folder in folders}
        folder_dict[ROOT_FOLDER] = {"id": ROOT_FOLDER, "name": "My workspace"}

        for folder in folder_dict.values():
            folder["subfolders"] = []

        for folder in folders:
            parent_id = folder.get("parentFolderId")
            if parent_id is not None:
                parent_folder = folder_dict.get(parent_id)
                if parent_folder:
                    parent_folder["subfolders"].append(
                        {"name": folder["name"], "id": folder["id"], "subfolders": []}
                    )

        def build_hierarchy(folder_id):
            folder = folder_dict.get(folder_id)
            if not folder:
                return None

            return {
                "name": folder["name"],
                "id": folder["id"],
                "subfolders": [
                    build_hierarchy(subfolder["id"]) for subfolder in folder["subfolders"]
                ],
            }

        return build_hierarchy(self.id)

    def get_folder_tree(self):
        """
        Retrieve the folder tree including subfolders from the API.

        Returns
        -------
        dict
            A hierarchical representation of the folder tree starting from the current folder.
        """

        payload = {
            "includeSubfolders": True,
            "page": 0,
            "size": 1000,
        }  # it assumes user will not have more than 1000 folders
        data = RestApi("/v2/folders").get(params=payload)
        folder_tree = self._build_folder_tree(data["records"])
        return folder_tree

    def _print_storage(self, tree, indent: int, n_display: int):
        """
        Recursively print the folder tree along with its contents and total storage usage.

        Parameters
        ----------
        tree : dict
            The current folder tree to display.
        indent : int
            The indentation level for pretty-printing.
        n_display : int
            The number of items to display before summarizing the remaining items.

        Returns
        -------
        int
            The total storage size of the current folder and its subfolders.
        """

        log.info("  " * indent + f"- [FOLDER] {tree['name']}")
        total_storage = 0
        for subfolder in tree["subfolders"]:
            # pylint: disable=protected-access
            total_storage += Folder(subfolder["id"])._print_storage(
                subfolder, indent + 1, n_display
            )

        items = self.get_items()
        displayed_items = items[:n_display]
        remaining_items = items[n_display:]

        for item in displayed_items:
            if item["type"] != "Folder":
                storage_size = item.get("storageSize", 0)
                total_storage += storage_size
                log.info(
                    "  " * (indent + 1)
                    + f"- [{item['type']}] {item['name']} (Size: {storage_size_formatter(storage_size)})"
                )

        if len(remaining_items) > 0:
            total_remaining_size = sum(item.get("storageSize", 0) for item in remaining_items)
            log.info(
                "  " * (indent + 1)
                + f"+{len(remaining_items)} more (total {storage_size_formatter(total_remaining_size)})"
            )
            total_storage += total_remaining_size

        log.info("  " * (indent + 1) + f"Total Storage: {storage_size_formatter(total_storage)}")
        return total_storage

    @classmethod
    def print_storage(cls, folder_id: str = "ROOT.FLOW360", n_display: int = 10) -> None:
        """
        Display the storage details of a folder, including subfolders and a summary of all items.

        Parameters
        ----------
        folder_id : str, optional
            The ID of the folder to print storage details for. Defaults to "ROOT.FLOW360".
        n_display : int, optional
            The number of items to display before summarizing the remaining items. Defaults to 10.
        """
        folder = cls(id=folder_id)
        tree = folder.get_folder_tree()
        folder._print_storage(tree, 0, n_display)
