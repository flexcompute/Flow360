"""Requests module"""

from typing import List, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..component.flow360_params.flow360_params import Flow360MeshParams


class Flow360Requests(pd.BaseModel):
    """
    Represents a request for Flow360 WEBAPI.

    This class extends `pd.BaseModel` and provides a method for converting the request
    object into a dictionary representation.
    """

    def dict(self, *args, **kwargs) -> dict:
        """returns dict representation of request"""
        return super().dict(*args, by_alias=True, exclude_none=True, **kwargs)

    class Config:  # pylint: disable=too-few-public-methods
        """config"""

        allow_population_by_field_name = True


class NewVolumeMeshRequest(Flow360Requests):
    """request for new volume mesh"""

    name: str = pd.Field(alias="meshName")
    file_name: str = pd.Field(alias="fileName")
    tags: Optional[List[str]] = pd.Field(alias="meshTags")
    format: Literal["aflr3", "cgns"] = pd.Field(alias="meshFormat")
    endianness: Optional[Literal["little", "big"]] = pd.Field(alias="meshEndianness")
    compression: Optional[Literal["gz", "bz2", "zst"]] = pd.Field(alias="meshCompression")
    mesh_params: Optional[Flow360MeshParams] = pd.Field(alias="meshParams")
    solver_version: Optional[str] = pd.Field(alias="solverVersion")

    # pylint: disable=no-self-argument
    @pd.validator("mesh_params")
    def set_mesh_params(cls, value: Union[Flow360MeshParams, None]):
        """validate mesh params"""
        if value:
            return value.flow360_json()
        return value


class CopyExampleVolumeMeshRequest(Flow360Requests):
    """request for new volume mesh"""

    name: str = pd.Field()
    example_id: str = pd.Field(alias="meshId")


class NewFolderRequest(Flow360Requests):
    """request for new folder"""

    name: str = pd.Field()
    tags: Optional[List[str]] = pd.Field(alias="tags")
    parent_folder_id: Optional[str] = pd.Field(alias="parentFolderId", default="ROOT.FLOW360")
    type: Literal["folder"] = pd.Field("folder", const=True)


class MoveCaseItem(pd.BaseModel):
    """move case item"""

    id: str
    type: Literal["case"] = pd.Field("case", const=True)


class MoveFolderItem(pd.BaseModel):
    """move folder item"""

    id: str
    type: Literal["folder"] = pd.Field("folder", const=True)


class MoveToFolderRequest(Flow360Requests):
    """request for move to folder"""

    dest_folder_id: str = pd.Field(alias="destFolderId")
    items: List[Union[MoveCaseItem, MoveFolderItem]] = pd.Field()
