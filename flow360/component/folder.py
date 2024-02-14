"""
Folder component
"""

from __future__ import annotations

from typing import List, Optional, Union

import pydantic as pd

from ..cloud.requests import MoveFolderItem, MoveToFolderRequest, NewFolderRequest
from ..cloud.rest_api import RestApi
from ..exceptions import Flow360ValueError
from ..log import log
from .interfaces import FolderInterface
from .resource_base import Flow360Resource, Flow360ResourceBaseModel, ResourceDraft
from .utils import shared_account_confirm_proceed, validate_type


# pylint: disable=E0213
class FolderMeta(Flow360ResourceBaseModel, extra=pd.Extra.allow):
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
            info_type_class=FolderMeta,
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
            self._info = self.info_type_class(
                **self.get(f"{self._endpoint}/items/{self.id}/metadata")
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
        RestApi(FolderInterface.endpoint).put(
            MoveToFolderRequest(
                dest_folder_id=folder.id, items=[MoveFolderItem(id=self.id)]
            ).dict(),
            method="move",
        )
        return self

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


# FOLDER LIST uses different endpoint, requires separate implementation

# class FolderList(Flow360ResourceListBase):
#     """
#     FolderList List component
#     """

#     def __init__(
#         self,
#         from_cloud: bool = True,
#         include_deleted: bool = False,
#         limit=100,
#     ):
#         super().__init__(
#             ancestor_id=None,
#             from_cloud=from_cloud,
#             include_deleted=include_deleted,
#             limit=limit,
#             resourceClass=Folder,
#         )

#     # pylint: disable=useless-parent-delegation
#     def __getitem__(self, index) -> Folder:
#         """
#         returns Folder item of the list
#         """
#         return super().__getitem__(index)

#     # pylint: disable=useless-parent-delegation
#     def __iter__(self) -> Iterator[Folder]:
#         return super().__iter__()
