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
from ...environment import current_environment
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
ROOT_FOLDER_NAME = "My workspace"

# The item search endpoint is backed by OpenSearch, whose default max_result_window stops a query
# at its first 10000 hits. Asking for a page past that fails the whole request with an opaque
# HTTP 409 rather than returning an empty page, so the walk has to stop itself. A folder holding
# up to twice this can still be listed in full by walking both sort directions - see
# _fetch_items. Server-side fix tracked in SCFD-10070.
ITEM_SEARCH_RESULT_LIMIT = 10000

# Appended to any storage total that an incomplete listing fed into, so the number is not read as
# exact at the line where it is printed.
INCOMPLETE_TOTAL_NOTE = (
    f" (lower bound: folders over {2 * ITEM_SEARCH_RESULT_LIMIT} items cannot be listed in full)"
)


def build_folder_tree(folders, root_folder_id: str = ROOT_FOLDER):
    """
    Build a hierarchical folder tree from folder records.

    Parameters
    ----------
    folders : list
        A list of folder records.
    root_folder_id : str
        The folder ID to use as the tree root.

    Returns
    -------
    dict
        A dictionary representing the folder hierarchy with nested subfolders.
    """

    folder_dict = {folder["id"]: dict(folder) for folder in folders}
    folder_dict[ROOT_FOLDER] = {"id": ROOT_FOLDER, "name": ROOT_FOLDER_NAME}

    for folder in folder_dict.values():
        folder["subfolders"] = []

    for folder in folders:
        child_folder = folder_dict.get(folder.get("id"))
        parent_id = folder.get("parentFolderId")
        if child_folder is not None and parent_id is not None:
            parent_folder = folder_dict.get(parent_id)
            if parent_folder:
                parent_folder["subfolders"].append(child_folder)

    def build_hierarchy(folder_id):
        folder = folder_dict.get(folder_id)
        if not folder:
            return None

        subfolders = []
        for subfolder in folder["subfolders"]:
            child_tree = build_hierarchy(subfolder["id"])
            if child_tree is not None:
                subfolders.append(child_tree)

        return {
            "name": folder["name"],
            "id": folder["id"],
            "subfolders": subfolders,
        }

    return build_hierarchy(root_folder_id)


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
        resp = RestApi(FolderInterface.endpoint, environment_provider=current_environment).post(
            req.dict()
        )
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
            self._info = self.meta_class(
                **RestApi(f"v2/folders/{self.id}", environment_provider=current_environment).get()
            )
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
        RestApi(FolderInterfaceV2.endpoint, environment_provider=current_environment).patch(
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
        RestApi(FolderInterfaceV2.endpoint, environment_provider=current_environment).patch(
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

    def get_projects(
        self,
        search_keyword: str = "",
        tags: Optional[List[str]] = None,
        exclude_subfolders: bool = False,
    ) -> list[dict]:
        """Get projects within this folder.

        Parameters
        ----------
        search_keyword : str
            Keyword to filter projects by name. Defaults to "" (all projects).
        tags : Optional[List[str]]
            Tags to filter projects.
        exclude_subfolders : bool
            If True, only search this folder, not its subfolders. Defaults to False.

        Returns
        -------
        list
            A list of project dictionaries found in the folder.
        """

        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.web.project_records import get_project_records

        records, _ = get_project_records(
            search_keyword=search_keyword,
            tags=tags,
            folder_ids=[self.id],
            exclude_subfolders=exclude_subfolders,
        )
        return [record.model_dump() for record in records.records]

    def _walk_items(self, sort_direction: str):
        """
        Page one sort direction of the folder listing, up to the item search limit.

        Parameters
        ----------
        sort_direction : str
            "desc" to walk from the largest item, "asc" to walk from the smallest.

        Returns
        -------
        tuple
            The records retrieved keyed by item id, and the raw hit count the server reported.
        """

        records = {}
        page = 0
        size = 1000  # Page size

        # The reported "total" counts raw search hits, including ones the server drops from the
        # page it returns, so accumulating records never reaches it and any page can come back
        # short. Advance by page index over the raw hits instead.
        while True:
            payload = {
                "page": page,
                "size": size,
                "filterFolderIds": self.id,
                "filterExcludeSubfolders": True,
                # createdAt breaks storageSize ties, making the two directions exact reverses of
                # each other. Without it tied items are ordered arbitrarily and independently per
                # query, so a tie group straddling both window edges could fall outside both.
                "sortFields": ["storageSize", "createdAt"],
                "sortDirections": [sort_direction, sort_direction],
                "expandFields": ["contentInfo"],
            }

            data = RestApi("/v2/items", environment_provider=current_environment).get(
                params=payload
            )
            records.update({record["id"]: record for record in data.get("records", [])})
            total = data.get("total", 0)
            page += 1
            if page * size >= min(total, ITEM_SEARCH_RESULT_LIMIT):
                return records, total

    def _fetch_items(self):
        """
        Fetch all items within the current folder, handling pagination if needed.

        Returns
        -------
        tuple
            The items found in the folder sorted by storage size in descending order, and whether
            the listing is still short of the folder because of the item search limit.
        """

        largest, total = self._walk_items("desc")
        if total <= ITEM_SEARCH_RESULT_LIMIT:
            return list(largest.values()), False

        # A single window reaches only the largest ITEM_SEARCH_RESULT_LIMIT hits. Walking the
        # opposite sort direction reaches the smallest just as many, so the two windows overlap -
        # and therefore cover the folder completely - unless it holds more than twice the limit.
        # Their intersection being non-empty is the proof that they met. The proof needs the sort
        # to be a total order, which is why _walk_items sorts on a tiebreaker as well: it makes an
        # item's rank from one end determine its rank from the other.
        smallest, _ = self._walk_items("asc")
        merged = {**smallest, **largest}
        truncated = not largest.keys() & smallest.keys()
        if truncated:
            log.warning(
                f"Folder {self.id} holds {total} items, over twice the {ITEM_SEARCH_RESULT_LIMIT} "
                f"the search API returns per query. Only the {len(merged)} largest and smallest "
                "were retrieved."
            )
        return (
            sorted(merged.values(), key=lambda record: record.get("storageSize", 0), reverse=True),
            truncated,
        )

    def get_items(self):
        """
        Fetch all items within the current folder, handling pagination if needed.

        Returns
        -------
        list
            A list of all items found in the folder, sorted by storage size in descending order.
        """

        items, _ = self._fetch_items()
        return items

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

        return build_folder_tree(folders, root_folder_id=self.id)

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
        data = RestApi("v2/folders", environment_provider=current_environment).get(params=payload)
        folder_tree = self._build_folder_tree(data["records"])
        return folder_tree

    def _collect_storage(self, tree):
        """
        Recursively total the storage of a folder tree.

        Subfolder totals have to be known before anything is printed, so that the biggest ones
        can be shown and the rest summarised.

        Parameters
        ----------
        tree : dict
            The current folder tree to total.

        Returns
        -------
        dict
            The folder name, its items, the storage total for the whole subtree, whether that
            total is a lower bound, and the same for each subfolder, largest first.
        """

        # pylint: disable=protected-access
        subfolders = sorted(
            (
                Folder(subfolder["id"])._collect_storage(subfolder)
                for subfolder in tree["subfolders"]
            ),
            key=lambda subfolder: subfolder["total"],
            reverse=True,
        )
        items, truncated = self._fetch_items()
        return {
            "name": tree["name"],
            "items": items,
            "total": sum(item.get("storageSize", 0) for item in items)
            + sum(subfolder["total"] for subfolder in subfolders),
            "incomplete": truncated or any(subfolder["incomplete"] for subfolder in subfolders),
            "subfolders": subfolders,
        }

    @classmethod
    def _render_storage(cls, node, indent: int, n_display: int, n_subfolders: int):
        """
        Print one collected folder and its descendants, biggest first.

        Parameters
        ----------
        node : dict
            A folder as returned by :func:`_collect_storage`.
        indent : int
            The indentation level for pretty-printing.
        n_display : int
            The number of items to show before summarizing the remaining items.
        n_subfolders : int
            The number of subfolders to show before summarizing the remaining subfolders.
        """

        log.info("  " * indent + f"- [FOLDER] {node['name']}")

        for subfolder in node["subfolders"][:n_subfolders]:
            cls._render_storage(subfolder, indent + 1, n_display, n_subfolders)

        hidden_subfolders = node["subfolders"][n_subfolders:]
        if hidden_subfolders:
            hidden_storage = sum(subfolder["total"] for subfolder in hidden_subfolders)
            log.info(
                "  " * (indent + 1)
                + f"+{len(hidden_subfolders)} more folders "
                + f"(total {storage_size_formatter(hidden_storage)})"
            )

        for item in node["items"][:n_display]:
            log.info(
                "  " * (indent + 1)
                + f"- [{item['type']}] {item['name']} "
                + f"(Size: {storage_size_formatter(item.get('storageSize', 0))})"
            )

        hidden_items = node["items"][n_display:]
        if hidden_items:
            hidden_storage = sum(item.get("storageSize", 0) for item in hidden_items)
            log.info(
                "  " * (indent + 1)
                + f"+{len(hidden_items)} more (total {storage_size_formatter(hidden_storage)})"
            )

        log.info(
            "  " * (indent + 1)
            + f"Total Storage: {storage_size_formatter(node['total'])}"
            + (INCOMPLETE_TOTAL_NOTE if node["incomplete"] else "")
        )

    @classmethod
    def print_storage(
        cls, folder_id: str = "ROOT.FLOW360", n_display: int = 10, n_subfolders: int = 10
    ) -> None:
        """
        Display the storage details of a folder, including subfolders and a summary of all items.

        Storage totals always cover the whole folder; the two limits only decide how much of it
        is spelled out line by line, biggest first.

        Parameters
        ----------
        folder_id : str, optional
            The ID of the folder to print storage details for. Defaults to "ROOT.FLOW360".
        n_display : int, optional
            The number of items to display per folder before summarizing the remaining items.
            Defaults to 10.
        n_subfolders : int, optional
            The number of subfolders to display per folder before summarizing the remaining
            subfolders. Defaults to 10.
        """
        folder = cls(id=folder_id)
        tree = folder.get_folder_tree()
        # pylint: disable=protected-access
        cls._render_storage(folder._collect_storage(tree), 0, n_display, n_subfolders)
